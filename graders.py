"""
graders.py
----------
Deterministic graders for all three tasks.

Each grader receives the agent's current working DataFrame and the
TaskDataset produced by dataset_factory, and returns a GradeResult
with a scalar score in [0.0, 1.0] plus a human-readable breakdown.

Public API
----------
grade(task_id, agent_df, dataset) -> GradeResult

    Dispatches to the correct grader. Call this from step().

GradeResult
    .score          float 0.0–1.0  (the number that feeds the reward)
    .breakdown      dict           (sub-scores, useful for logging/debugging)
    .issues_remaining int          (how many cells still need fixing)
    .detail         str            (one-line human summary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    score: float                              # 0.0 – 1.0, fed into reward
    breakdown: Dict[str, float] = field(default_factory=dict)
    issues_remaining: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        self.score = round(float(np.clip(self.score, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def grade(
    task_id: str,
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    metadata: Dict[str, Any],
    initial_dirty_cells: int,
) -> GradeResult:
    """
    Route to the correct grader and return a GradeResult.

    Parameters
    ----------
    task_id
        One of "easy", "medium", "hard".
    agent_df
        The agent's current working DataFrame (may still be dirty).
    clean_df
        Ground-truth clean DataFrame from TaskDataset.
    metadata
        TaskDataset.metadata dict (grader-specific ground truth).
    initial_dirty_cells
        Dirty cell count at episode start; used to compute issues_remaining
        for easy/medium tasks.
    """
    if agent_df is None or len(agent_df) == 0:
        return GradeResult(score=0.0, detail="Empty DataFrame — no score.")

    if task_id == "easy":
        return _grade_easy(agent_df, clean_df, metadata, initial_dirty_cells)
    elif task_id == "medium":
        return _grade_medium(agent_df, clean_df, metadata, initial_dirty_cells)
    elif task_id == "hard":
        return _grade_hard(agent_df, clean_df, metadata)
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — easy: cell-level match against ground truth
# ─────────────────────────────────────────────────────────────────────────────
#
# Score = (cells matching ground truth) / (total cells)
#
# "Matching" is defined after normalisation:
#   - strip leading/trailing whitespace
#   - numeric columns: round to 2dp, compare as float strings
#   - date column: accept YYYY-MM-DD only
#   - string columns: case-sensitive exact match after strip
#   - NaN vs NaN → always mismatch (agent must fill or fix them)

def _grade_easy(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    metadata: Dict[str, Any],
    initial_dirty_cells: int,
) -> GradeResult:

    # Align shape — agent might have different row count if they accidentally
    # dropped rows; penalise by treating missing rows as all-wrong.
    agent_norm  = _normalise_easy(agent_df,  clean_df)
    clean_norm  = _normalise_easy(clean_df, clean_df)

    total_cells = clean_norm.size

    # Pad or truncate agent rows to match clean row count
    if len(agent_norm) < len(clean_norm):
        pad = pd.DataFrame(
            [["__MISSING__"] * len(clean_norm.columns)] * (len(clean_norm) - len(agent_norm)),
            columns=clean_norm.columns,
        )
        agent_norm = pd.concat([agent_norm, pad], ignore_index=True)
    elif len(agent_norm) > len(clean_norm):
        agent_norm = agent_norm.iloc[: len(clean_norm)].copy()

    matches = (agent_norm == clean_norm).sum().sum()
    score   = matches / total_cells

    # Issues remaining: number of cells that still differ
    mismatches = int((agent_norm != clean_norm).sum().sum())

    breakdown = {
        "cell_match_ratio":   round(score, 4),
        "cells_matched":      int(matches),
        "total_cells":        int(total_cells),
        "cells_mismatched":   mismatches,
    }

    detail = (
        f"{int(matches)}/{total_cells} cells correct "
        f"({100*score:.1f}%) — {mismatches} still need fixing."
    )

    return GradeResult(
        score=score,
        breakdown=breakdown,
        issues_remaining=mismatches,
        detail=detail,
    )


def _normalise_easy(df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring a DataFrame to a canonical string form for cell-level comparison.

    Rules applied per column based on clean_df's dtype:
      - Numeric (price, quantity): round to 2 decimal places → string
      - Date (order_date):         parse and reformat to YYYY-MM-DD
      - String (all others):       strip whitespace, leave case unchanged
      - NaN / unparseable:         normalise to the sentinel "__NAN__"
    """
    out = {}
    NUMERIC_COLS = {"price", "quantity"}
    DATE_COLS    = {"order_date"}

    for col in clean_df.columns:
        if col not in df.columns:
            # Agent removed or renamed the column — all cells wrong
            out[col] = pd.Series(["__MISSING_COL__"] * len(df))
            continue

        series = df[col].copy()

        if col in NUMERIC_COLS:
            out[col] = series.apply(_to_numeric_str)
        elif col in DATE_COLS:
            out[col] = series.apply(_to_date_str)
        else:
            out[col] = series.apply(
                lambda x: "__NAN__" if _is_missing(x) else str(x).strip()
            )

    return pd.DataFrame(out, dtype=str)


def _to_numeric_str(x: Any) -> str:
    if _is_missing(x):
        return "__NAN__"
    try:
        return f"{float(str(x).strip().replace(',', '')):.2f}"
    except (ValueError, TypeError):
        return "__INVALID__"


def _to_date_str(x: Any) -> str:
    if _is_missing(x):
        return "__NAN__"
    s = str(x).strip()
    # Reject obviously wrong dates (e.g. year 2099)
    try:
        parsed = pd.to_datetime(s, dayfirst=False)
        if parsed.year > 2030 or parsed.year < 2000:
            return "__BAD_DATE__"
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return "__INVALID_DATE__"


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — medium: F1 on outlier detection + typo correction
# ─────────────────────────────────────────────────────────────────────────────
#
# Two independent sub-scores, equally weighted:
#
#   outlier_f1   — precision/recall on which rows were fixed or removed
#   typo_score   — fraction of category typo-cells correctly fixed
#
# Final score = 0.50 * outlier_f1 + 0.50 * typo_score
#
# Outlier logic:
#   A true-outlier row is "correctly handled" if:
#     (a) the row still exists AND amount is now in [5, 800], OR
#     (b) the row was dropped entirely
#   A valid-extreme row is a "false positive" if it was dropped OR
#     its amount was changed to something outside [900, 2000].
#
# The thresholds match the schema_hint the agent was given.

_VALID_AMOUNT_MIN    = 5.0
_VALID_AMOUNT_MAX    = 800.0
_EXTREME_AMOUNT_MIN  = 900.0
_EXTREME_AMOUNT_MAX  = 2000.0


def _grade_medium(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    metadata: Dict[str, Any],
    initial_dirty_cells: int,
) -> GradeResult:

    outlier_rows:       List[int] = metadata.get("outlier_rows", [])
    valid_extreme_rows: List[int] = metadata.get("valid_extreme_rows", [])
    typo_cells: List[Tuple[int, str, str]] = metadata.get("typo_cells", [])

    # ── Outlier sub-score ────────────────────────────────────────────────────
    # Detect which of the original row indices are still present in agent_df.
    # We track by tx_id (which is stable and unique) rather than df index,
    # since the agent may reset the index after dropping rows.
    agent_tx_ids: set = set()
    if "tx_id" in agent_df.columns:
        agent_tx_ids = set(agent_df["tx_id"].dropna().astype(int).tolist())

    tp = 0  # outlier rows that were correctly handled
    fn = 0  # outlier rows still wrong (extreme amount still present)
    fp = 0  # valid-extreme rows wrongly removed or damaged

    # True-positive check
    for orig_idx in outlier_rows:
        tx_id_val = int(clean_df.iloc[orig_idx]["tx_id"]) if orig_idx < len(clean_df) else None
        if tx_id_val is None:
            continue
        if tx_id_val not in agent_tx_ids:
            # Row was dropped — counts as correctly handled (outlier removed)
            tp += 1
        else:
            # Row still present — check if amount was fixed
            agent_row = agent_df[agent_df["tx_id"].astype(int) == tx_id_val]
            if len(agent_row) == 0:
                tp += 1  # dropped after all
            else:
                amt = _safe_float(agent_row.iloc[0].get("amount"))
                if amt is not None and _VALID_AMOUNT_MIN <= amt <= _VALID_AMOUNT_MAX:
                    tp += 1
                else:
                    fn += 1

    # False-positive check (valid extremes must survive untouched)
    for orig_idx in valid_extreme_rows:
        if orig_idx >= len(clean_df):
            continue
        tx_id_val = int(clean_df.iloc[orig_idx]["tx_id"])
        clean_amt = float(clean_df.iloc[orig_idx]["amount"])

        if tx_id_val not in agent_tx_ids:
            fp += 1  # wrongly dropped a valid row
        else:
            agent_row = agent_df[agent_df["tx_id"].astype(int) == tx_id_val]
            if len(agent_row) == 0:
                fp += 1
            else:
                amt = _safe_float(agent_row.iloc[0].get("amount"))
                # Accept if amount is within ±5% of original clean value
                if amt is None or not (clean_amt * 0.95 <= amt <= clean_amt * 1.05):
                    fp += 1

    n_outliers  = len(outlier_rows)
    precision   = tp / (tp + fp + 1e-9)
    recall      = tp / (n_outliers + 1e-9)
    outlier_f1  = (2 * precision * recall) / (precision + recall + 1e-9)

    # ── Typo sub-score ───────────────────────────────────────────────────────
    typo_correct = 0
    for (row_idx, dirty_val, clean_val) in typo_cells:
        if "tx_id" not in clean_df.columns or row_idx >= len(clean_df):
            continue
        tx_id_val = int(clean_df.iloc[row_idx]["tx_id"])
        agent_rows = agent_df[agent_df["tx_id"].astype(int) == tx_id_val] \
            if "tx_id" in agent_df.columns else pd.DataFrame()
        if len(agent_rows) == 0:
            continue  # row dropped; neither credit nor penalty
        agent_cat = str(agent_rows.iloc[0].get("category", "")).strip()
        if agent_cat == clean_val:
            typo_correct += 1

    typo_score = typo_correct / max(len(typo_cells), 1)

    # ── Combined score ───────────────────────────────────────────────────────
    score = 0.50 * outlier_f1 + 0.50 * typo_score

    # Approximate issues remaining: unsolved outliers + unsolved typos
    issues_remaining = fn + (len(typo_cells) - typo_correct)

    breakdown = {
        "outlier_f1":    round(outlier_f1, 4),
        "outlier_tp":    tp,
        "outlier_fn":    fn,
        "outlier_fp":    fp,
        "precision":     round(precision, 4),
        "recall":        round(recall, 4),
        "typo_score":    round(typo_score, 4),
        "typos_fixed":   typo_correct,
        "typos_total":   len(typo_cells),
        "combined":      round(score, 4),
    }

    detail = (
        f"Outlier F1={outlier_f1:.3f} (TP={tp}, FP={fp}, FN={fn}) | "
        f"Typos {typo_correct}/{len(typo_cells)} fixed → score={score:.3f}"
    )

    return GradeResult(
        score=score,
        breakdown=breakdown,
        issues_remaining=issues_remaining,
        detail=detail,
    )


def _safe_float(x: Any) -> Optional[float]:
    if _is_missing(x):
        return None
    try:
        return float(str(x).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — hard: schema normalisation + deduplication + date formatting
# ─────────────────────────────────────────────────────────────────────────────
#
# Three independent sub-scores:
#
#   schema_score  (weight 0.40)
#       Fraction of canonical column names present in agent_df.
#       Bonus: all 9 canonical columns present AND no extra columns → +0.1
#
#   dedup_score   (weight 0.35)
#       How many of the 30 true duplicate tx records were removed.
#       Penalises over-deletion (removing rows that were not duplicates).
#       dedup_precision = removed_true_dups / (rows_removed + ε)
#       dedup_recall    = removed_true_dups / n_duplicate_pairs
#       dedup_f1        = harmonic mean
#
#   format_score  (weight 0.25)
#       Fraction of values in the purchase_date column (or canonical alias)
#       that are valid YYYY-MM-DD strings.
#
# Final score = 0.40 * schema_score + 0.35 * dedup_score + 0.25 * format_score

_CANONICAL_COLS = [
    "record_id", "customer_id", "full_name", "email",
    "amount", "currency", "purchase_date", "product_name", "region",
]

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _grade_hard(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> GradeResult:

    canonical_lookup: Dict[str, str] = metadata.get("canonical_lookup", {})
    n_clean_rows: int                = metadata.get("n_clean_rows", len(clean_df))

    # ── 1. Schema score ──────────────────────────────────────────────────────
    schema_score, schema_detail = _grade_schema(agent_df, canonical_lookup)

    # ── 2. Deduplication score ───────────────────────────────────────────────
    dedup_score, dedup_detail = _grade_deduplication(
        agent_df, clean_df, n_clean_rows, canonical_lookup
    )

    # ── 3. Date format score ─────────────────────────────────────────────────
    format_score, format_detail = _grade_date_format(agent_df, canonical_lookup)

    # ── Combined ─────────────────────────────────────────────────────────────
    score = 0.40 * schema_score + 0.35 * dedup_score + 0.25 * format_score

    # issues_remaining: rough proxy (unresolved column aliases + excess rows)
    n_canonical_present = sum(
        1 for c in _CANONICAL_COLS if c in agent_df.columns
    )
    issues_remaining = (
        (len(_CANONICAL_COLS) - n_canonical_present)   # missing canonical cols
        + max(0, len(agent_df) - n_clean_rows)          # excess rows (dups not removed)
    )

    breakdown = {
        "schema_score":  round(schema_score,  4),
        "dedup_score":   round(dedup_score,   4),
        "format_score":  round(format_score,  4),
        "combined":      round(score,          4),
        **{f"schema_{k}": v for k, v in schema_detail.items()},
        **{f"dedup_{k}":  v for k, v in dedup_detail.items()},
        **{f"fmt_{k}":    v for k, v in format_detail.items()},
    }

    detail = (
        f"Schema={schema_score:.3f} | "
        f"Dedup={dedup_score:.3f} | "
        f"DateFmt={format_score:.3f} → score={score:.3f}"
    )

    return GradeResult(
        score=score,
        breakdown=breakdown,
        issues_remaining=issues_remaining,
        detail=detail,
    )


def _grade_schema(
    agent_df: pd.DataFrame,
    canonical_lookup: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score how well the agent normalised column names.

    Strategy:
      - Build a set of "recognised" columns: canonical names + their aliases.
      - For each canonical column, check if the agent has it (by canonical name).
      - Partial credit per canonical column found.
      - Small bonus if ALL 9 are present and no unrecognised extra columns remain.
    """
    agent_cols = set(agent_df.columns)
    canonical_set = set(_CANONICAL_COLS)

    # All known column names (canonical + every alias)
    all_known = canonical_set | set(canonical_lookup.keys())

    # Count canonical columns present
    found    = [c for c in _CANONICAL_COLS if c in agent_cols]
    n_found  = len(found)
    base     = n_found / len(_CANONICAL_COLS)

    # Bonus: all canonical present AND no leftover alias columns
    leftover_aliases = [c for c in agent_cols if c not in canonical_set]
    all_present      = n_found == len(_CANONICAL_COLS)
    clean_rename     = len(leftover_aliases) == 0

    bonus = 0.10 if (all_present and clean_rename) else 0.0

    score = min(1.0, base + bonus)

    detail: Dict[str, Any] = {
        "canonical_found":    n_found,
        "canonical_total":    len(_CANONICAL_COLS),
        "leftover_aliases":   len(leftover_aliases),
        "rename_bonus":       bonus,
    }
    return score, detail


def _grade_deduplication(
    agent_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    n_clean_rows: int,
    canonical_lookup: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score how well the agent removed duplicate rows.

    We compare row counts and detect near-duplicate detection quality:
      - n_injected_dups: 30 (hardcoded from dataset_factory)
      - expected_final_rows: n_clean_rows (400)
      - rows_removed: (raw dirty rows = 430) - len(agent_df)
      - true_dups_removed: min(rows_removed, 30) if rows_removed ≤ 35
        (we're lenient — removing 1–35 rows likely targets dups)
      - over_deletion: max(0, rows_removed - 30) rows beyond the dup count
        penalises removing valid data.

    Precision = true_dups_removed / (rows_removed + ε)
    Recall    = true_dups_removed / 30
    F1        = harmonic mean
    """
    N_INJECTED_DUPS = 30
    N_DIRTY_ROWS    = n_clean_rows + N_INJECTED_DUPS  # 430

    rows_removed = max(0, N_DIRTY_ROWS - len(agent_df))

    # Heuristic: any removal ≤ 35 rows is probably targeting dups
    true_dups_removed = min(rows_removed, N_INJECTED_DUPS)

    # Penalise over-removal (agent deleted valid rows beyond dups)
    over_deletion = max(0, rows_removed - N_INJECTED_DUPS)
    # Each over-deleted row reduces precision
    effective_true = max(0, true_dups_removed - over_deletion)

    precision = effective_true / (rows_removed + 1e-9)
    recall    = true_dups_removed / (N_INJECTED_DUPS + 1e-9)
    f1        = (2 * precision * recall) / (precision + recall + 1e-9)

    detail: Dict[str, Any] = {
        "rows_removed":       rows_removed,
        "true_dups_removed":  true_dups_removed,
        "over_deletion":      over_deletion,
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
    }
    return f1, detail


def _grade_date_format(
    agent_df: pd.DataFrame,
    canonical_lookup: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Fraction of purchase_date values matching YYYY-MM-DD.

    Looks for the canonical name "purchase_date" first; falls back to
    known aliases ("date", "PurchaseDate") if the agent hasn't renamed yet.
    """
    DATE_ALIASES = {"purchase_date", "date", "PurchaseDate"}

    date_col = None
    # Prefer canonical name
    if "purchase_date" in agent_df.columns:
        date_col = "purchase_date"
    else:
        for alias in DATE_ALIASES:
            if alias in agent_df.columns:
                date_col = alias
                break

    if date_col is None:
        return 0.0, {"date_col_found": False, "valid_ratio": 0.0}

    # Guard: duplicate column names after rename produce a DataFrame, not Series.
    # Take the first occurrence.
    col_data = agent_df[date_col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]

    # Force object dtype so .sum() always returns a numeric 0, not '' (the
    # StringDtype identity).  Python 3.14 + pandas 2.2+ infer StringDtype
    # from .astype(str), which makes .sum() on an empty Series return ''.
    series   = col_data.dropna().astype(object).apply(str).str.strip()
    n_total  = len(series)
    if n_total == 0:
        return 0.0, {"date_col_found": True, "valid_ratio": 0.0, "n_total": 0}

    # Combined check: ISO pattern match AND year in plausible range
    def _is_valid_iso(s: str) -> bool:
        if not _ISO_DATE_PATTERN.match(s):
            return False
        try:
            return 2000 <= int(s[:4]) <= 2030
        except Exception:
            return False

    valid_flags = series.apply(_is_valid_iso)
    n_valid    = int(valid_flags.sum())   # int() guards against numpy/pandas scalar types
    n_year_ok  = n_valid                  # same condition — kept for breakdown detail
    valid_ratio = n_year_ok / n_total

    detail: Dict[str, Any] = {
        "date_col_found": True,
        "date_col_used":  date_col,
        "n_total":        int(n_total),
        "n_valid_iso":    int(n_valid),
        "n_year_ok":      int(n_year_ok),
        "valid_ratio":    round(valid_ratio, 4),
    }
    return valid_ratio, detail


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset_factory import make_dataset

    SEP = "─" * 62

    # ── Task 1: easy ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\nTASK: easy\n{SEP}")
    ds = make_dataset("easy")

    # Baseline: grade dirty df (should be low)
    r_dirty = grade("easy", ds.dirty_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[dirty]  score={r_dirty.score:.4f}  {r_dirty.detail}")

    # Perfect: grade clean df (should be 1.0)
    r_clean = grade("easy", ds.clean_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[clean]  score={r_clean.score:.4f}  {r_clean.detail}")

    # Partial: fix half the injected cells
    partial = ds.dirty_df.copy()
    injected = ds.metadata.get("injected_cells", [])
    for (row, col) in injected[:len(injected)//2]:
        partial.at[row, col] = ds.clean_df.at[row, col]
    r_partial = grade("easy", partial, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[half]   score={r_partial.score:.4f}  {r_partial.detail}")

    print(f"Breakdown: {r_partial.breakdown}")

    # ── Task 2: medium ────────────────────────────────────────────────────────
    print(f"\n{SEP}\nTASK: medium\n{SEP}")
    ds = make_dataset("medium")

    r_dirty = grade("medium", ds.dirty_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[dirty]  score={r_dirty.score:.4f}  {r_dirty.detail}")

    r_clean = grade("medium", ds.clean_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[clean]  score={r_clean.score:.4f}  {r_clean.detail}")

    # Simulate agent fixing all outliers (set amount to 150.0) + all typos
    fixed = ds.dirty_df.copy()
    for row in ds.metadata["outlier_rows"]:
        if "tx_id" in ds.clean_df.columns:
            fixed.at[row, "amount"] = 150.0
    for (row, dirty_val, clean_val) in ds.metadata["typo_cells"]:
        fixed.at[row, "category"] = clean_val
    r_fixed = grade("medium", fixed, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[fixed]  score={r_fixed.score:.4f}  {r_fixed.detail}")

    print(f"Breakdown: {r_fixed.breakdown}")

    # ── Task 3: hard ──────────────────────────────────────────────────────────
    print(f"\n{SEP}\nTASK: hard\n{SEP}")
    ds = make_dataset("hard")

    r_dirty = grade("hard", ds.dirty_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[dirty]  score={r_dirty.score:.4f}  {r_dirty.detail}")

    r_clean = grade("hard", ds.clean_df, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[clean]  score={r_clean.score:.4f}  {r_clean.detail}")

    # Simulate partial fix: rename columns only, don't dedup or fix dates
    partial_hard = ds.dirty_df.copy()
    rename_map = ds.metadata.get("canonical_lookup", {})
    partial_hard = partial_hard.rename(columns=rename_map)
    # Keep only canonical columns that exist
    canonical_present = [c for c in _CANONICAL_COLS if c in partial_hard.columns]
    partial_hard = partial_hard[canonical_present]
    r_renamed = grade("hard", partial_hard, ds.clean_df, ds.metadata, ds.total_dirty_cells)
    print(f"[rename] score={r_renamed.score:.4f}  {r_renamed.detail}")

    print(f"Breakdown: {r_renamed.breakdown}")