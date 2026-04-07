"""
server/data_cleaning_env.py
---------------------------
DataCleaningEnvironment — the heart of the environment.

Implements the three abstract methods from openenv.core.env_server.interfaces.Environment:
    reset(seed, episode_id, **kwargs) -> CleanObservation
    step(action, timeout_s, **kwargs) -> CleanObservation
    state  (property)                -> CleanState

Architecture
------------
Live DataFrames (_dirty_df, _clean_df) live as instance variables for speed.
CleanState holds lightweight CSV snapshots used only for WebSocket state()
responses — not for every step. This avoids serialising a 400-row DataFrame
on every call.

Action dispatch
---------------
Each CleanAction.command routes to a private _apply_* method that mutates
_dirty_df in place. Errors in those methods (bad column name, out-of-bounds
row) are caught and returned as (success=False, error_msg=...) so the agent
gets corrective feedback instead of a 500.

Reward
------
compute_reward() implements the dense reward formula designed in the plan:
    progress term      — grader score delta (main signal)
    efficiency bonus   — small reward for early completion
    false-positive penalty — for dropping a valid-extreme row (medium task)
    early-DONE penalty — for calling DONE with a low score
    step cost          — -0.005 every step to discourage padding
"""

from __future__ import annotations

import sys
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

# ── OpenEnv imports (try relative → absolute) ─────────────────────────────────
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

# ── Local imports (try relative → absolute for both server and standalone) ───
try:
    from ..models import (
        CleanAction, CleanObservation, CleanState,
        MAX_STEPS, DONE_THRESHOLD,
    )
    from ..dataset_factory import make_dataset, TaskDataset
    from ..graders import grade, GradeResult
except ImportError:
    try:
        from models import (
            CleanAction, CleanObservation, CleanState,
            MAX_STEPS, DONE_THRESHOLD,
        )
        from dataset_factory import make_dataset, TaskDataset
        from graders import grade, GradeResult
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from models import (
            CleanAction, CleanObservation, CleanState,
            MAX_STEPS, DONE_THRESHOLD,
        )
        from dataset_factory import make_dataset, TaskDataset
        from graders import grade, GradeResult


# ── Constants ─────────────────────────────────────────────────────────────────

# Per-step cost that discourages infinite loops / padding
STEP_COST = -0.005

# Penalty for calling DONE before the score is reasonable
EARLY_DONE_PENALTY = -0.20
EARLY_DONE_THRESHOLD = 0.60   # DONE below this score triggers the penalty

# Penalty for removing a valid-extreme row in the medium task
FALSE_POSITIVE_PENALTY = -0.15

# Efficiency bonus multiplier (only awarded when episode is solved)
EFFICIENCY_BONUS_WEIGHT = 0.10

# Date formats the STANDARDIZE_COL handler will try, in priority order
_DATE_PARSE_FORMATS = [
    "%Y-%m-%d",   # ISO — most reliable, try first
    "%m/%d/%Y",   # US
    "%d.%m.%Y",   # EU
    "%d/%m/%Y",   # EU alt
    "%Y/%m/%d",   # Asian
]


# ─────────────────────────────────────────────────────────────────────────────
# DataCleaningEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaningEnvironment(Environment):
    """
    Gym-style environment for the data cleaning pipeline task.

    Each episode:
      1. reset(task_id="easy"|"medium"|"hard") loads a dirty/clean CSV pair.
      2. The agent calls step() repeatedly, each time sending a CleanAction.
      3. The episode ends when the agent sends DONE, the score crosses the
         task threshold, or the step budget is exhausted.

    The environment is fully stateless between sessions — all mutable state
    lives in instance variables, so concurrent sessions each get their own
    isolated copy (SUPPORTS_CONCURRENT_SESSIONS = True).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()

        # Live DataFrames — mutated by each step()
        self._dirty_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None

        # Full task dataset from dataset_factory (holds metadata for grader)
        self._dataset: Optional[TaskDataset] = None

        # Pydantic state (lightweight; updated on demand)
        self._state: Optional[CleanState] = None

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> CleanObservation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        seed
            Ignored — datasets use fixed seeds per task for reproducibility.
        episode_id
            Optional; auto-generated if not provided.
        task_id
            Which task to load: "easy", "medium", or "hard".
        """
        if task_id not in MAX_STEPS:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Must be one of: {list(MAX_STEPS)}"
            )

        # Load dataset (always deterministic via fixed seed in dataset_factory)
        self._dataset  = make_dataset(task_id)
        self._dirty_df = self._dataset.dirty_df.copy(deep=True)
        self._clean_df = self._dataset.clean_df.copy(deep=True)

        max_steps = MAX_STEPS[task_id]

        # Run grader on the initial dirty state so we have a starting score
        initial_result = grade(
            task_id=task_id,
            agent_df=self._dirty_df,
            clean_df=self._clean_df,
            metadata=self._dataset.metadata,
            initial_dirty_cells=self._dataset.total_dirty_cells,
        )

        self._state = CleanState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            dirty_csv_snapshot=self._df_to_csv(self._dirty_df),
            clean_csv_snapshot=self._df_to_csv(self._clean_df),
            initial_dirty_cells=self._dataset.total_dirty_cells,
            current_score=initial_result.score,
            previous_score=0.0,
            task_metadata=self._dataset.metadata,
            schema_hint=self._dataset.schema_hint,
            max_steps=max_steps,
        )

        return self._build_observation(
            reward=None,
            done=False,
            last_action_success=True,
            last_action_error=None,
            grader_result=initial_result,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: CleanAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CleanObservation:
        """
        Apply one CleanAction and return the resulting observation.

        Never raises for bad action inputs — instead returns
        last_action_success=False with a descriptive error message so the
        agent can self-correct on the next step.
        """
        if self._state is None or self._dirty_df is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._state.step_count += 1

        # ── Save previous score before mutating ──────────────────────────────
        prev_score = self._state.current_score
        self._state.previous_score = prev_score

        # ── DONE shortcut ────────────────────────────────────────────────────
        if action.command == "DONE":
            reward = self._compute_reward(
                action=action,
                prev_score=prev_score,
                curr_score=prev_score,   # score doesn't change on DONE
                action_success=True,
                was_false_positive=False,
            )
            done = True
            self._state.dirty_csv_snapshot = self._df_to_csv(self._dirty_df)
            return self._build_observation(
                reward=reward,
                done=done,
                last_action_success=True,
                last_action_error=None,
                grader_result=GradeResult(
                    score=prev_score,
                    issues_remaining=self._state.initial_dirty_cells
                        - int(prev_score * self._state.initial_dirty_cells),
                    detail="Agent signalled DONE.",
                ),
            )

        # ── Apply action to _dirty_df ────────────────────────────────────────
        action_success, error_msg, was_false_positive = self._apply_action(action)

        # ── Grade the result ──────────────────────────────────────────────────
        grader_result = grade(
            task_id=self._state.task_id,
            agent_df=self._dirty_df,
            clean_df=self._clean_df,
            metadata=self._state.task_metadata,
            initial_dirty_cells=self._state.initial_dirty_cells,
        )
        curr_score = grader_result.score
        self._state.current_score = curr_score

        # ── Compute reward ────────────────────────────────────────────────────
        reward = self._compute_reward(
            action=action,
            prev_score=prev_score,
            curr_score=curr_score,
            action_success=action_success,
            was_false_positive=was_false_positive,
        )

        # ── Check termination ────────────────────────────────────────────────
        done = (
            curr_score >= DONE_THRESHOLD[self._state.task_id]
            or self._state.step_count >= self._state.max_steps
        )

        # ── Sync state snapshot ──────────────────────────────────────────────
        self._state.dirty_csv_snapshot = self._df_to_csv(self._dirty_df)

        return self._build_observation(
            reward=reward,
            done=done,
            last_action_success=action_success,
            last_action_error=error_msg,
            grader_result=grader_result,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # state (property)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> CleanState:
        """Return the current environment state (serialisable snapshot)."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        # Keep snapshot fresh in case step() was called without triggering a sync
        if self._dirty_df is not None:
            self._state.dirty_csv_snapshot = self._df_to_csv(self._dirty_df)
        return self._state

    # ─────────────────────────────────────────────────────────────────────────
    # Action dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(
        self, action: CleanAction
    ) -> tuple[bool, Optional[str], bool]:
        """
        Mutate self._dirty_df according to the action.

        Returns
        -------
        (success, error_msg, was_false_positive)
            success           — True if action applied without error
            error_msg         — human-readable description if success=False
            was_false_positive — True if a DROP_ROW removed a valid-extreme row
        """
        cmd = action.command

        if cmd == "SET_VALUE":
            return self._apply_set_value(action)

        elif cmd == "DROP_ROW":
            return self._apply_drop_row(action)

        elif cmd == "STANDARDIZE_COL":
            return self._apply_standardize_col(action)

        elif cmd == "FILL_MISSING":
            return self._apply_fill_missing(action)

        else:
            return False, f"Unknown command: {cmd!r}", False

    # ── SET_VALUE ─────────────────────────────────────────────────────────────

    def _apply_set_value(
        self, action: CleanAction
    ) -> tuple[bool, Optional[str], bool]:
        df = self._dirty_df
        row_idx = action.row_index
        col     = action.column
        val     = action.value

        # Validate column
        if col not in df.columns:
            return (
                False,
                f"Column {col!r} not found. Available: {list(df.columns)}",
                False,
            )

        # Validate row index (positional)
        if row_idx < 0 or row_idx >= len(df):
            return (
                False,
                f"Row index {row_idx} out of range. DataFrame has {len(df)} rows (0–{len(df)-1}).",
                False,
            )

        # Try to cast value to the column's expected type
        cast_val, cast_err = self._cast_value(val, df, col)
        if cast_err:
            return False, cast_err, False

        df.iloc[row_idx, df.columns.get_loc(col)] = cast_val
        return True, None, False

    # ── DROP_ROW ──────────────────────────────────────────────────────────────

    def _apply_drop_row(
        self, action: CleanAction
    ) -> tuple[bool, Optional[str], bool]:
        df = self._dirty_df
        row_idx = action.row_index

        if row_idx < 0 or row_idx >= len(df):
            return (
                False,
                f"Row index {row_idx} out of range. DataFrame has {len(df)} rows.",
                False,
            )

        # Detect false positive for medium task: is this a valid-extreme row?
        was_false_positive = self._is_valid_extreme_row(row_idx)

        # Drop the row and reset positional index so future iloc references stay valid
        self._dirty_df = df.drop(df.index[row_idx]).reset_index(drop=True)
        return True, None, was_false_positive

    def _is_valid_extreme_row(self, iloc_idx: int) -> bool:
        """
        Return True if dropping this row would be a false positive.
        Only applies to the medium task, which tracks valid_extreme_rows
        by their original tx_id.
        """
        if self._state is None or self._state.task_id != "medium":
            return False

        valid_extreme_rows: list = self._state.task_metadata.get(
            "valid_extreme_rows", []
        )
        if not valid_extreme_rows or self._clean_df is None:
            return False

        df = self._dirty_df
        if "tx_id" not in df.columns:
            return False

        # Get the tx_id of the row being dropped
        try:
            tx_id_to_drop = int(df.iloc[iloc_idx]["tx_id"])
        except (IndexError, ValueError, KeyError):
            return False

        # Check if any valid-extreme row in clean_df has this tx_id
        for orig_idx in valid_extreme_rows:
            if orig_idx >= len(self._clean_df):
                continue
            if int(self._clean_df.iloc[orig_idx]["tx_id"]) == tx_id_to_drop:
                return True

        return False

    # ── STANDARDIZE_COL ───────────────────────────────────────────────────────

    def _apply_standardize_col(
        self, action: CleanAction
    ) -> tuple[bool, Optional[str], bool]:
        df  = self._dirty_df
        col = action.column

        if col not in df.columns:
            return (
                False,
                f"Column {col!r} not found. Available: {list(df.columns)}",
                False,
            )

        series = df[col].copy()

        # ── Try date normalisation first ──────────────────────────────────────
        if self._looks_like_date_column(col, series):
            normalised, err = self._normalise_dates(series)
            if err:
                return False, f"Date normalisation failed for column {col!r}: {err}", False
            self._dirty_df[col] = normalised
            return True, None, False

        # ── Try numeric coercion ──────────────────────────────────────────────
        if self._looks_like_numeric_column(col, series):
            numeric = pd.to_numeric(series, errors="coerce")
            # Only apply if we didn't lose more than 20% of non-null values
            original_non_null = series.notna().sum()
            coerced_non_null  = numeric.notna().sum()
            if original_non_null == 0 or coerced_non_null / original_non_null >= 0.8:
                self._dirty_df[col] = numeric
                return True, None, False

        # ── String normalisation: strip whitespace ───────────────────────────
        self._dirty_df[col] = series.apply(
            lambda x: str(x).strip() if not _is_nan(x) else x
        )
        return True, None, False

    def _looks_like_date_column(self, col: str, series: pd.Series) -> bool:
        """Heuristic: column name contains 'date' or most non-null values parse as dates."""
        if "date" in col.lower():
            return True
        sample = series.dropna().astype(str).head(5)
        parsed = 0
        for s in sample:
            for fmt in _DATE_PARSE_FORMATS:
                try:
                    pd.to_datetime(s, format=fmt)
                    parsed += 1
                    break
                except Exception:
                    pass
        return parsed >= max(1, len(sample) // 2)

    def _looks_like_numeric_column(self, col: str, series: pd.Series) -> bool:
        """Heuristic: column name or majority of values suggests numeric data."""
        numeric_keywords = {"price", "amount", "value", "quantity", "qty", "count", "id", "num"}
        if any(kw in col.lower() for kw in numeric_keywords):
            return True
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        convertible = pd.to_numeric(sample, errors="coerce").notna().sum()
        return convertible / len(sample) >= 0.7

    def _normalise_dates(self, series: pd.Series) -> tuple[pd.Series, Optional[str]]:
        """Parse dates in any supported format and reformat as YYYY-MM-DD."""
        def _parse_one(x: Any) -> Any:
            if _is_nan(x):
                return x
            s = str(x).strip()
            for fmt in _DATE_PARSE_FORMATS:
                try:
                    return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
                except Exception:
                    pass
            # Last resort: let pandas guess
            try:
                parsed = pd.to_datetime(s, dayfirst=False)
                if 2000 <= parsed.year <= 2030:
                    return parsed.strftime("%Y-%m-%d")
            except Exception:
                pass
            return x  # leave unchanged if unparseable

        return series.apply(_parse_one), None

    # ── FILL_MISSING ──────────────────────────────────────────────────────────

    def _apply_fill_missing(
        self, action: CleanAction
    ) -> tuple[bool, Optional[str], bool]:
        df  = self._dirty_df
        col = action.column
        strategy = action.fill_strategy

        if col not in df.columns:
            return (
                False,
                f"Column {col!r} not found. Available: {list(df.columns)}",
                False,
            )

        series  = df[col].copy()
        numeric = pd.to_numeric(series, errors="coerce")
        has_numeric = numeric.notna().sum() > 0

        if strategy == "mean":
            if not has_numeric:
                return False, f"Cannot compute mean for non-numeric column {col!r}.", False
            fill_val = numeric.mean()
            self._dirty_df[col] = numeric.fillna(round(fill_val, 2))

        elif strategy == "median":
            if not has_numeric:
                return False, f"Cannot compute median for non-numeric column {col!r}.", False
            fill_val = numeric.median()
            self._dirty_df[col] = numeric.fillna(round(fill_val, 2))

        elif strategy == "mode":
            mode_result = series.mode(dropna=True)
            if mode_result.empty:
                return False, f"No mode found for column {col!r} (all values missing?).", False
            self._dirty_df[col] = series.fillna(mode_result.iloc[0])

        elif strategy == "drop":
            before = len(self._dirty_df)
            self._dirty_df = self._dirty_df.dropna(subset=[col]).reset_index(drop=True)
            after = len(self._dirty_df)
            return True, None, False

        else:
            return False, f"Unknown fill_strategy: {strategy!r}", False

        return True, None, False

    # ─────────────────────────────────────────────────────────────────────────
    # Reward computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        action: CleanAction,
        prev_score: float,
        curr_score: float,
        action_success: bool,
        was_false_positive: bool,
    ) -> float:
        """
        Dense per-step reward in the range [-0.5, +1.0].

        Components
        ----------
        progress          score delta (main learning signal)
        efficiency bonus  small reward for solving with steps to spare
        fp_penalty        penalise removing a valid-extreme row (medium task)
        early_done_penalty penalise calling DONE with a very low score
        step_cost         tiny constant cost to discourage padding
        """
        if self._state is None:
            return 0.0

        max_steps   = self._state.max_steps
        step_count  = self._state.step_count

        # 1. Progress term
        progress = curr_score - prev_score

        # 2. Efficiency bonus (only when task is solved this step)
        threshold = DONE_THRESHOLD[self._state.task_id]
        just_solved = prev_score < threshold <= curr_score
        step_fraction = step_count / max_steps
        efficiency = EFFICIENCY_BONUS_WEIGHT * (1.0 - step_fraction) if just_solved else 0.0

        # 3. False-positive penalty
        fp_penalty = FALSE_POSITIVE_PENALTY if was_false_positive else 0.0

        # 4. Early-DONE penalty
        early_done = (
            EARLY_DONE_PENALTY
            if action.command == "DONE" and curr_score < EARLY_DONE_THRESHOLD
            else 0.0
        )

        # 5. Step cost
        step_cost = STEP_COST

        reward = progress + efficiency + fp_penalty + early_done + step_cost
        return round(float(np.clip(reward, -0.5, 1.0)), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        last_action_success: bool,
        last_action_error: Optional[str],
        grader_result: GradeResult,
    ) -> CleanObservation:
        if self._state is None:
            raise RuntimeError("State not initialised.")

        return CleanObservation(
            # Inherited from Observation base
            done=done,
            reward=reward,
            # Task context
            task_id=self._state.task_id,
            schema_hint=self._state.schema_hint,
            initial_dirty_cells=self._state.initial_dirty_cells,
            # Per-step state
            dirty_csv=self._df_to_csv(self._dirty_df),
            current_score=grader_result.score,
            issues_remaining=grader_result.issues_remaining,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            # Last-action feedback
            last_action_success=last_action_success,
            last_action_error=last_action_error,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _df_to_csv(df: Optional[pd.DataFrame]) -> str:
        """Serialise DataFrame to CSV string with the integer position index."""
        if df is None:
            return ""
        return df.to_csv(index=True, index_label="row_index")

    @staticmethod
    def _cast_value(
        val: str, df: pd.DataFrame, col: str
    ) -> tuple[Any, Optional[str]]:
        """
        Try to cast a string value to the appropriate type for `col`.

        Returns (cast_value, error_message). error_message is None on success.
        """
        # Determine target type from the clean (non-null, non-text) column values
        sample = pd.to_numeric(
            df[col].dropna().astype(str).str.strip(), errors="coerce"
        )
        majority_numeric = sample.notna().sum() / max(len(df[col].dropna()), 1) >= 0.5

        if majority_numeric:
            try:
                float_val = float(val.strip().replace(",", ""))
                # If all sample values are whole numbers, keep as int
                if (sample.dropna() % 1 == 0).all() and float_val % 1 == 0:
                    return int(float_val), None
                return round(float_val, 2), None
            except (ValueError, AttributeError):
                return (
                    None,
                    f"Cannot cast {val!r} to numeric for column {col!r}. "
                    f"Provide a plain number (e.g. '29.99').",
                )

        # String column — accept as-is (strip whitespace)
        return val.strip(), None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._dirty_df = None
        self._clean_df = None
        self._dataset  = None
        self._state    = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="data_cleaning_env",
            description=(
                "Data cleaning pipeline: the agent receives a dirty CSV "
                "and must fix type errors, outliers, missing values, and "
                "schema inconsistencies to match a hidden ground truth."
            ),
            version="1.0.0",
            author="hackathon",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_nan(x: Any) -> bool:
    """Return True if x is any flavour of missing value."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "─" * 64

    for task_id in ("easy", "medium", "hard"):
        print(f"\n{SEP}\nTASK: {task_id.upper()}\n{SEP}")

        env = DataCleaningEnvironment()

        # ── reset ────────────────────────────────────────────────────────────
        obs = env.reset(task_id=task_id)
        print(f"reset()  → score={obs.current_score:.4f}  "
              f"issues={obs.issues_remaining}  done={obs.done}")
        assert obs.reward is None,  "reward must be None after reset"
        assert obs.done   is False, "done must be False after reset"

        lines = obs.dirty_csv.strip().split("\n")
        print(f"  CSV:  {len(lines)} rows, {len(lines[0].split(','))} cols")
        print(f"  Hint: {obs.schema_hint[:70]}…")

        # ── state() ──────────────────────────────────────────────────────────
        st = env.state
        print(f"state()  → episode_id={st.episode_id[:8]}…  step_count={st.step_count}")

        # ── step: bad column (should give feedback, not crash) ───────────────
        bad_action = CleanAction(
            command="SET_VALUE", row_index=0, column="DOES_NOT_EXIST", value="0"
        )
        obs2 = env.step(bad_action)
        assert obs2.last_action_success is False
        print(f"step (bad col) → success={obs2.last_action_success}  "
              f"error='{obs2.last_action_error[:50]}…'")

        # ── step: out-of-bounds row ──────────────────────────────────────────
        bad_row = CleanAction(
            command="SET_VALUE", row_index=9999, column="price", value="10.0"
        )
        obs3 = env.step(bad_row)
        assert obs3.last_action_success is False
        print(f"step (bad row) → success={obs3.last_action_success}  "
              f"error='{obs3.last_action_error[:50]}…'")

        # ── step: valid fix ──────────────────────────────────────────────────
        if task_id == "easy":
            # Find the first injected dirty cell and fix it
            injected = env._dataset.metadata.get("injected_cells", [])
            if injected:
                row, col = injected[0]
                clean_val = str(env._clean_df.iloc[row][col])
                fix_action = CleanAction(
                    command="SET_VALUE", row_index=row, column=col, value=clean_val
                )
                obs4 = env.step(fix_action)
                print(f"step (fix row={row} col={col!r}) → "
                      f"success={obs4.last_action_success}  "
                      f"score={obs4.current_score:.4f}  "
                      f"reward={obs4.reward:.4f}")
                assert obs4.last_action_success is True
                assert obs4.reward is not None

        elif task_id == "medium":
            # Fix one outlier row via FILL_MISSING on amount
            obs4 = env.step(CleanAction(
                command="FILL_MISSING", column="amount", fill_strategy="median"
            ))
            print(f"step (FILL_MISSING amount/median) → "
                  f"score={obs4.current_score:.4f}  reward={obs4.reward:.4f}")

        elif task_id == "hard":
            # Standardize the date column
            obs4 = env.step(CleanAction(
                command="STANDARDIZE_COL", column="date"
            ))
            print(f"step (STANDARDIZE_COL date) → "
                  f"success={obs4.last_action_success}  "
                  f"score={obs4.current_score:.4f}  reward={obs4.reward:.4f}")

        # ── DONE action ───────────────────────────────────────────────────────
        done_obs = env.step(CleanAction(command="DONE"))
        assert done_obs.done is True
        print(f"step (DONE)    → done={done_obs.done}  "
              f"reward={done_obs.reward:.4f}  score={done_obs.current_score:.4f}")

        env.close()

    print(f"\n{SEP}\nAll smoke tests passed.\n{SEP}")