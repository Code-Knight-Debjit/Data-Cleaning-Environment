"""
inference.py  —  Data Cleaning Pipeline
----------------------------------------
Now that the server populates obs.remaining_issues on every step,
the agent has a machine-readable, authoritative list of what is still
broken.  Client-side analysis is used only as a fallback if the server
field is empty (backward compatibility).

Environment variables (all free):
  API_BASE_URL      HF router or local endpoint
  MODEL_NAME        Any HF-hosted model
  HF_TOKEN          HuggingFace read token
  LOCAL_IMAGE_NAME  Docker image (optional)
  ENV_BASE_URL      Server URL  (default http://localhost:8000)

STDOUT (evaluator reads these exact lines):
  [START] task=<id> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# ── env / model imports ────────────────────────────────────────────────────────
try:
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD

from openai import OpenAI

# ── configuration ──────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK   = "data_cleaning_env"
TASK_IDS    = ["easy", "medium", "hard"]
STEP_LIMITS = {"easy": 25, "medium": 50, "hard": 80}

# ── official log helpers ───────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action[:80].replace(chr(10),' ')} "
          f"reward={reward:.2f} done={str(done).lower()} "
          f"error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} "
          f"rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# REMAINING-ISSUES READER
# Reads obs.remaining_issues (server-authoritative).
# Falls back to client-side analysis if the field is missing/empty.
# ══════════════════════════════════════════════════════════════════════════════

def get_remaining_issues(obs, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Return the authoritative remaining-issues dict.

    Priority:
      1. obs.remaining_issues   — computed server-side from the live DataFrame
      2. client-side fallback   — computed from obs.dirty_csv
    """
    ri = getattr(obs, "remaining_issues", None) or {}

    # If server populated it, use it directly
    if ri and isinstance(ri, dict):
        return ri

    # ── Fallback: analyse dirty_csv client-side ───────────────────────────────
    result: Dict[str, Any] = {
        "missing_values":          {},
        "format_issues":           {},
        "invalid_category_values": {},
        "duplicate_rows":          0,
        "unstandardized_columns":  [],
        "missing_canonical_cols":  [],
        "alias_columns":           {},
        "is_clean":                False,
    }

    if df is None:
        return result

    numeric_kw = {"price", "amount", "sale_amount", "value", "Amount", "quantity"}
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            result["missing_values"][col] = n
        if any(kw in col.lower() for kw in numeric_kw):
            series  = df[col].dropna()
            coerced = pd.to_numeric(series, errors="coerce")
            n_bad   = int(coerced.isna().sum())
            if n_bad:
                result["format_issues"][col] = {
                    "count":    n_bad,
                    "examples": series[coerced.isna()].astype(str).unique()[:4].tolist(),
                }
        if "date" in col.lower():
            series = df[col].dropna().astype(str)
            if len(series) and int((~series.str.match(r"^\d{4}-\d{2}-\d{2}$")).sum()) > 0:
                result["unstandardized_columns"].append(col)

    result["duplicate_rows"] = int(df.duplicated().sum())

    result["is_clean"] = (
        not result["missing_values"]
        and not result["format_issues"]
        and not result["invalid_category_values"]
        and result["duplicate_rows"] == 0
        and not result["unstandardized_columns"]
        and not result["missing_canonical_cols"]
        and not result["alias_columns"]
        and obs.current_score >= DONE_THRESHOLD.get(obs.task_id, 0.85)
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN TRACKER  —  completion memory + brute-force detection
# ══════════════════════════════════════════════════════════════════════════════

class ColumnTracker:
    def __init__(self) -> None:
        self.standardized:   Set[str]          = set()
        self.filled:         Set[str]          = set()
        self.set_value_rows: Dict[str, Set[int]] = {}
        self._consec_sv:     int               = 0
        self._prev_sv:       bool              = False

    def record(self, action: CleanAction, score_delta: float) -> None:
        improved = score_delta > 0.001
        cmd = action.command
        if cmd == "STANDARDIZE_COL" and action.column and improved:
            self.standardized.add(action.column)
            self._reset()
        elif cmd == "FILL_MISSING" and action.column and improved:
            self.filled.add(action.column)
            self._reset()
        elif cmd == "SET_VALUE" and action.column and action.row_index is not None:
            self.set_value_rows.setdefault(action.column, set()).add(action.row_index)
            self._consec_sv = (self._consec_sv + 1) if self._prev_sv else 1
            self._prev_sv = True
            return
        else:
            self._reset()

    def _reset(self) -> None:
        self._prev_sv   = False
        self._consec_sv = 0

    def is_blocked(self, cmd: str, col: Optional[str] = None,
                   row: Optional[int] = None) -> Tuple[bool, str]:
        if cmd == "STANDARDIZE_COL" and col in self.standardized:
            return True, f"already standardized '{col}'"
        if cmd == "FILL_MISSING" and col in self.filled:
            return True, f"already filled '{col}'"
        if cmd == "SET_VALUE" and col and row is not None:
            if row in self.set_value_rows.get(col, set()):
                return True, f"row {row}/'{col}' already set"
        return False, ""

    def brute_force_alert(self) -> Optional[str]:
        if self._consec_sv >= 3:
            return (
                f"🚨 BRUTE FORCE ({self._consec_sv} consecutive SET_VALUEs). "
                "STOP. Use FILL_MISSING or STANDARDIZE_COL to fix the whole column."
            )
        return None

    def ledger(self) -> str:
        lines = ["COMPLETION LEDGER (never redo completed work):"]
        if self.standardized:
            lines.append(f"  STANDARDIZED : {', '.join(sorted(self.standardized))}")
        if self.filled:
            lines.append(f"  FILLED       : {', '.join(sorted(self.filled))}")
        for col, rows in self.set_value_rows.items():
            lines.append(f"  SET_VALUE '{col}' rows: {sorted(rows)}")
        if len(lines) == 1:
            lines.append("  (nothing completed yet)")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STATE BLOCK  —  built from obs.remaining_issues (authoritative)
# ══════════════════════════════════════════════════════════════════════════════

def _state_block(ri: Dict[str, Any], obs, df: Optional[pd.DataFrame]) -> str:
    threshold = DONE_THRESHOLD.get(obs.task_id, 0.85)
    L = [
        f"=== REMAINING ISSUES  "
        f"(score={obs.current_score:.4f}  need≥{threshold}  "
        f"step={obs.step_number}/{obs.max_steps}) ===",
        f"is_clean: {'✅ YES — ready for DONE' if ri.get('is_clean') else '❌ NO — do NOT call DONE'}",
    ]

    # Missing canonical columns (hard task)
    if ri.get("missing_canonical_cols"):
        L.append(f"\nMISSING CANONICAL COLUMNS (must rename/create):")
        for col in ri["missing_canonical_cols"]:
            L.append(f"  ❌ {col}")

    # Alias columns (hard task)
    if ri.get("alias_columns"):
        L.append(f"\nCOLUMNS NEEDING RENAME (use STANDARDIZE_COL):")
        for alias, canonical in ri["alias_columns"].items():
            L.append(f"  '{alias}'  →  '{canonical}'")

    # Missing values — THE most important section
    if ri.get("missing_values"):
        L.append("\nMISSING VALUES — fix each with FILL_MISSING (1 action = whole column):")
        for col, n in ri["missing_values"].items():
            L.append(f"  {col:22s}  {n:3d} cells missing")
    else:
        L.append("\nMISSING VALUES:    ✅ none")

    # Format issues
    if ri.get("format_issues"):
        L.append("\nFORMAT ISSUES — fix each with STANDARDIZE_COL (1 action = whole column):")
        for col, info in ri["format_issues"].items():
            L.append(f"  {col:22s}  {info['count']:3d} bad values e.g. {info['examples']}")
    else:
        L.append("\nFORMAT ISSUES:     ✅ none")

    # Invalid categories
    if ri.get("invalid_category_values"):
        L.append("\nINVALID CATEGORY VALUES — fix with FILL_MISSING mode:")
        for col, info in ri["invalid_category_values"].items():
            L.append(f"  {col:22s}  bad: {info['bad']}  allowed: {info['allowed']}")
    else:
        L.append("\nINVALID CATEGORIES:✅ none")

    # Unstandardized date columns
    if ri.get("unstandardized_columns"):
        L.append(f"\nUNSTANDARDIZED DATE COLUMNS — use STANDARDIZE_COL:")
        for col in ri["unstandardized_columns"]:
            L.append(f"  {col}")
    else:
        L.append("\nDATE FORMATS:      ✅ all ISO 8601")

    # Duplicates
    n_dup = ri.get("duplicate_rows", 0)
    if n_dup:
        L.append(f"\nDUPLICATE ROWS: {n_dup}  — use DROP_ROW (find index in preview)")
    else:
        L.append("\nDUPLICATE ROWS:    ✅ none")

    # Data preview
    if df is not None:
        L.append("\nDATA PREVIEW (row_index = leftmost int):")
        L.append(df.head(8).to_string(max_cols=12))

    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
# ACTION MENU  —  allowed / blocked with fill-strategy enforcement
# ══════════════════════════════════════════════════════════════════════════════

_FILL_STRATEGIES = {
    # numeric
    "price": "median", "amount": "median", "quantity": "median",
    "sale_amount": "median", "value": "median",
    # categorical
    "customer": "mode", "product": "mode", "category": "mode",
    "region": "mode", "country": "mode", "status": "mode",
    "currency": "mode",
}

def _fill_strategy(col: str) -> str:
    direct = _FILL_STRATEGIES.get(col)
    if direct:
        return direct
    numeric_kw = {"price","amount","qty","quantity","value","count"}
    if any(kw in col.lower() for kw in numeric_kw):
        return "median"
    return "mode"


def _action_menu(ri: Dict[str, Any], tracker: ColumnTracker,
                 obs, df: Optional[pd.DataFrame]) -> str:
    allowed = []
    blocked = []
    score   = obs.current_score

    # Hard: rename aliases
    for alias, canonical in ri.get("alias_columns", {}).items():
        blk, reason = tracker.is_blocked("STANDARDIZE_COL", alias)
        if blk:
            blocked.append(f'STANDARDIZE_COL "{alias}"  [SKIP: {reason}]')
        else:
            allowed.append(
                f'STANDARDIZE_COL  col="{alias}"   '
                f'← renames alias to canonical "{canonical}"')

    # Unstandardized date columns
    for col in ri.get("unstandardized_columns", []):
        blk, reason = tracker.is_blocked("STANDARDIZE_COL", col)
        if blk:
            blocked.append(f'STANDARDIZE_COL "{col}"  [SKIP: {reason}]')
        else:
            allowed.append(
                f'STANDARDIZE_COL  col="{col}"   '
                f'← normalises all dates to YYYY-MM-DD')

    # Format issues
    for col, info in ri.get("format_issues", {}).items():
        blk, reason = tracker.is_blocked("STANDARDIZE_COL", col)
        if blk:
            blocked.append(f'STANDARDIZE_COL "{col}"  [SKIP: {reason}]')
        else:
            allowed.append(
                f'STANDARDIZE_COL  col="{col}"   '
                f'← fixes {info["count"]} bad values {info["examples"]}')

    # Missing values
    for col, n in ri.get("missing_values", {}).items():
        blk, reason = tracker.is_blocked("FILL_MISSING", col)
        if blk:
            blocked.append(f'FILL_MISSING "{col}"  [SKIP: {reason}]')
            continue
        strategy = _fill_strategy(col)
        allowed.append(
            f'FILL_MISSING  col="{col}"  fill_strategy="{strategy}"   '
            f'← fills ALL {n} missing values in one step')

    # Invalid categories
    for col, info in ri.get("invalid_category_values", {}).items():
        blk, reason = tracker.is_blocked("FILL_MISSING", col)
        if not blk:
            allowed.append(
                f'FILL_MISSING  col="{col}"  fill_strategy="mode"   '
                f'← fixes typos/invalid values {info["bad"]}  '
                f'(allowed: {info["allowed"]})')

    # Duplicates
    n_dup = ri.get("duplicate_rows", 0)
    if n_dup > 0 and df is not None:
        dup_indices = df[df.duplicated(keep="first")].index.tolist()
        for idx in dup_indices[:3]:
            allowed.append(f'DROP_ROW  row_index={idx}   ← duplicate')

    # SET_VALUE: only if no column-level fix is available
    has_col_fix = (
        ri.get("missing_values")
        or ri.get("format_issues")
        or ri.get("invalid_category_values")
        or ri.get("unstandardized_columns")
        or ri.get("alias_columns")
    )
    if not has_col_fix:
        allowed.append('SET_VALUE  row_index=<int>  column="<col>"  value="<str>"   ← isolated cell fix only')

    # DONE — gated by is_clean
    if ri.get("is_clean"):
        allowed.append(f'DONE   ← is_clean=True, score={score:.3f}')
    else:
        remaining_summary = _summarise_ri(ri)
        blocked.append(
            f'DONE   [HARD BLOCKED by server. is_clean=False. '
            f'Outstanding: {remaining_summary}]')

    lines = ["ALLOWED ACTIONS (pick the single most impactful one):"]
    for a in allowed[:8]:
        lines.append(f"  ✅ {a}")
    if blocked:
        lines.append("BLOCKED — do NOT use:")
        for b in blocked[:8]:
            lines.append(f"  🚫 {b}")
    return "\n".join(lines)


def _summarise_ri(ri: Dict[str, Any]) -> str:
    parts = []
    if ri.get("missing_values"):
        parts.append(f"missing({list(ri['missing_values'].keys())})")
    if ri.get("format_issues"):
        parts.append(f"format({list(ri['format_issues'].keys())})")
    if ri.get("invalid_category_values"):
        parts.append(f"invalid_cats({list(ri['invalid_category_values'].keys())})")
    if ri.get("duplicate_rows"):
        parts.append(f"dups={ri['duplicate_rows']}")
    if ri.get("unstandardized_columns"):
        parts.append(f"bad_dates={ri['unstandardized_columns']}")
    if ri.get("missing_canonical_cols"):
        parts.append(f"missing_cols={ri['missing_canonical_cols']}")
    if ri.get("alias_columns"):
        parts.append(f"aliases={list(ri['alias_columns'].keys())}")
    return " | ".join(parts) if parts else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def _trajectory(history: list) -> str:
    if not history:
        return "  (none yet)"
    lines = []
    for i, e in enumerate(history[-8:], 1):
        r     = f"{e['reward']:+.4f}" if e["reward"] is not None else "n/a"
        delta = f"  Δscore={e['score_delta']:+.4f}" if "score_delta" in e else ""
        err   = f"  ⚠ {e['error']}"                 if e.get("error")  else ""
        lines.append(f"  {i}. {e['action_str']:50s} r={r}{delta}{err}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a cost-aware data cleaning optimizer.
Fix a dirty CSV dataset in as few steps as possible.

━━━ MANDATORY PROCESS (every turn) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. READ  "REMAINING ISSUES"    — the server tells you EXACTLY what is broken.
2. VERIFY  is_clean field      — if ❌ NO, you MUST NOT call DONE.
3. READ  "COMPLETION LEDGER"   — never repeat work you already did.
4. READ  "ALLOWED ACTIONS"     — pick the action that closes the most issues.
5. OUTPUT  one JSON object     — nothing else.

━━━ DONE CONTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONE is only valid when is_clean = ✅ YES.
If you call DONE when is_clean = ❌ NO:
  • The server will reject it (done=false, episode continues)
  • You receive reward = −1.0
  • The error message will tell you exactly what still needs fixing
There is NO exception to this rule.

━━━ EFFICIENCY LAWS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• FILL_MISSING     fixes ALL missing values in one column in ONE step.
• STANDARDIZE_COL  fixes ALL format errors in one column in ONE step.
• Never use SET_VALUE when a column-level fix applies.
• Never repeat the same command on the same column.
• 3+ consecutive SET_VALUE calls → brute force → near-zero reward.

━━━ TYPE RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Numeric  (price, amount, quantity): fill_strategy = "mean" or "median"
• Category (region, category, status, country, currency): fill_strategy = "mode"
• Date     (order_date, tx_date, purchase_date): use STANDARDIZE_COL

━━━ COMMANDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"command":"FILL_MISSING",    "column":"<col>", "fill_strategy":"mean|median|mode|drop"}
{"command":"STANDARDIZE_COL", "column":"<col>"}
{"command":"DROP_ROW",        "row_index":<int>}
{"command":"SET_VALUE",       "row_index":<int>, "column":"<col>", "value":"<str>"}
{"command":"DONE"}

Output: ONE JSON object, one line, no markdown, no explanation."""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(obs, ri: Dict[str, Any], df: Optional[pd.DataFrame],
                  tracker: ColumnTracker, history: list) -> str:
    parts = [
        f"TASK: {obs.task_id}",
        f"SCHEMA: {obs.schema_hint[:400]}",
    ]

    alert = tracker.brute_force_alert()
    if alert:
        parts.append(f"\n{alert}\n")

    if obs.last_action_error:
        parts.append(f"\nLAST ACTION FAILED: {obs.last_action_error}")
        parts.append("→ Pick a DIFFERENT action.\n")

    parts.append(_state_block(ri, obs, df))
    parts += ["", tracker.ledger(), ""]
    parts.append(_action_menu(ri, tracker, obs, df))
    parts += ["", "PREVIOUS STEPS:"]
    parts.append(_trajectory(history))
    parts += ["", "Your JSON action:"]
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ACTION PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse(raw: str) -> CleanAction:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()
    try:
        return CleanAction(**json.loads(text))
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return CleanAction(**json.loads(m.group()))
        except Exception:
            pass
    return CleanAction(command="DONE")


def _astr(a: CleanAction) -> str:
    p = [a.command]
    if a.column:        p.append(f"col='{a.column}'")
    if a.row_index is not None: p.append(f"row={a.row_index}")
    if a.value is not None:     p.append(f"val={a.value!r}")
    if a.fill_strategy: p.append(f"strategy={a.fill_strategy}")
    return "  ".join(p)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(client: OpenAI, messages: list) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: (
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.05,
            ).choices[0].message.content or ""
        ).strip(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE LOOP
# ══════════════════════════════════════════════════════════════════════════════

async def run_episode(env, client: OpenAI, task_id: str) -> dict:
    max_steps  = STEP_LIMITS[task_id]
    threshold  = DONE_THRESHOLD[task_id]
    rewards:   List[float] = []
    steps_done = 0
    score      = 0.0
    prev_score = 0.0
    success    = False
    history:   list = []
    tracker    = ColumnTracker()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env.reset(task_id=task_id)
        obs        = result.observation
        score      = obs.current_score
        prev_score = score

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_done = step

            df = None
            try:
                df = pd.read_csv(io.StringIO(obs.dirty_csv), index_col=0)
            except Exception:
                pass

            # Get authoritative remaining-issues from server (with fallback)
            ri = get_remaining_issues(obs, df)

            user_msg = _build_prompt(obs, ri, df, tracker, history)
            messages.append({"role": "user", "content": user_msg})

            try:
                raw    = await _llm(client, messages)
                action = _parse(raw)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Rolling window: system + last 6 exchanges
            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            result      = await env.step(action)
            obs         = result.observation
            reward      = result.reward or 0.0
            score       = obs.current_score
            score_delta = score - prev_score
            prev_score  = score
            rewards.append(reward)

            astr = _astr(action)
            tracker.record(action, score_delta)
            history.append({
                "action_str":  astr,
                "reward":      reward,
                "score_delta": score_delta,
                "error":       obs.last_action_error,
            })

            log_step(step=step, action=astr, reward=reward,
                     done=obs.done, error=obs.last_action_error)

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    finally:
        log_end(success=success, steps=steps_done,
                score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "reward": sum(rewards), "steps": steps_done, "success": success}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN not set.\n"
            "1. https://huggingface.co/settings/tokens → New token → Read\n"
            "2. PowerShell: $env:HF_TOKEN='hf_...'\n"
            "3. python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL     : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME       : {MODEL_NAME}", flush=True)
    print(f"LOCAL_IMAGE_NAME : {LOCAL_IMAGE_NAME or '(not set — using ENV_BASE_URL)'}", flush=True)
    print(f"ENV_BASE_URL     : {ENV_BASE_URL}", flush=True)
    print("", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    if LOCAL_IMAGE_NAME:
        env = await DataCleaningEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = DataCleaningEnv(base_url=ENV_BASE_URL)
        await env.connect()

    results = []
    try:
        for task_id in TASK_IDS:
            summary = await run_episode(env, llm, task_id)
            results.append(summary)
            print("", flush=True)
    finally:
        await env.close()

    print("=" * 58, flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Reward':>9} {'Steps':>6} {'Pass':>5}")
    print("-" * 58, flush=True)
    for r in results:
        print(f"{r['task_id']:<10} {r['score']:>7.4f} {r['reward']:>9.4f} "
              f"{r['steps']:>6} {'YES' if r['success'] else 'NO':>4}",
              flush=True)
    print("=" * 58, flush=True)


if __name__ == "__main__":
    asyncio.run(main())