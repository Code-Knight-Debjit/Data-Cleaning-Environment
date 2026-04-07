"""
models.py
---------
Pydantic models for the Data Cleaning Pipeline environment.

Three models define the full agent↔environment contract:

  CleanAction      — what the agent sends on each step
  CleanObservation — what the agent receives back
  CleanState       — internal server state (not sent to agent directly)

Inheritance chain (confirmed from OpenEnv source):
  Action      → extra="forbid", has: metadata: Dict[str, Any]
  Observation → extra="forbid", has: done: bool, reward: float|None, metadata: Dict[str, Any]
  State       → extra="allow",  has: episode_id: Optional[str], step_count: int
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for local development without the full OpenEnv install
    from openenv.core.env_server import Action, Observation, State


# ── Valid values (used by validators + schema hints) ──────────────────────────

VALID_COMMANDS = Literal[
    "SET_VALUE",       # Fix a specific cell: (row_index, column, value)
    "DROP_ROW",        # Remove an entire row: (row_index,)
    "STANDARDIZE_COL", # Normalize an entire column's format: (column,)
    "FILL_MISSING",    # Fill NaN values in a column: (column, fill_strategy)
    "DONE",            # Agent signals episode is complete: ()
]

VALID_FILL_STRATEGIES = Literal["mean", "median", "mode", "drop"]

VALID_TASK_IDS = Literal["easy", "medium", "hard"]


# ─────────────────────────────────────────────────────────────────────────────
# CleanAction
# ─────────────────────────────────────────────────────────────────────────────

class CleanAction(Action):
    """Action sent by the agent each step.

    The ``command`` field selects the operation. Depending on command,
    only a subset of the remaining fields are required:

    +-----------------+------------+--------+-------+---------------+
    | command         | row_index  | column | value | fill_strategy |
    +=================+============+========+=======+===============+
    | SET_VALUE       | required   | req    | req   | —             |
    | DROP_ROW        | required   | —      | —     | —             |
    | STANDARDIZE_COL | —          | req    | —     | —             |
    | FILL_MISSING    | —          | req    | —     | required      |
    | DONE            | —          | —      | —     | —             |
    +-----------------+------------+--------+-------+---------------+

    Example (fix a single cell)::

        CleanAction(
            command="SET_VALUE",
            row_index=3,
            column="price",
            value="29.99",
        )

    Example (drop a whole row)::

        CleanAction(command="DROP_ROW", row_index=17)

    Example (fill all NaN in a column with the median)::

        CleanAction(
            command="FILL_MISSING",
            column="quantity",
            fill_strategy="median",
        )
    """

    command: VALID_COMMANDS = Field(
        ...,
        description=(
            "Operation to perform. One of: SET_VALUE, DROP_ROW, "
            "STANDARDIZE_COL, FILL_MISSING, DONE."
        ),
    )

    row_index: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based row index to target. "
            "Required for SET_VALUE and DROP_ROW."
        ),
    )

    column: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Name of the column to target. "
            "Required for SET_VALUE, STANDARDIZE_COL, and FILL_MISSING."
        ),
    )

    value: Optional[str] = Field(
        default=None,
        description=(
            "New cell value as a string. "
            "Required for SET_VALUE. The environment casts this to the "
            "column's expected dtype (e.g. '29.99' → float for a price column)."
        ),
    )

    fill_strategy: Optional[VALID_FILL_STRATEGIES] = Field(
        default=None,
        description=(
            "Strategy for FILL_MISSING. One of: mean, median, mode, drop. "
            "'drop' removes rows where the column is NaN."
        ),
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "CleanAction":
        """Ensure each command has exactly the fields it needs."""
        cmd = self.command

        if cmd == "SET_VALUE":
            missing = []
            if self.row_index is None:
                missing.append("row_index")
            if self.column is None:
                missing.append("column")
            if self.value is None:
                missing.append("value")
            if missing:
                raise ValueError(
                    f"SET_VALUE requires: {', '.join(missing)}"
                )

        elif cmd == "DROP_ROW":
            if self.row_index is None:
                raise ValueError("DROP_ROW requires row_index")

        elif cmd == "STANDARDIZE_COL":
            if self.column is None:
                raise ValueError("STANDARDIZE_COL requires column")

        elif cmd == "FILL_MISSING":
            missing = []
            if self.column is None:
                missing.append("column")
            if self.fill_strategy is None:
                missing.append("fill_strategy")
            if missing:
                raise ValueError(
                    f"FILL_MISSING requires: {', '.join(missing)}"
                )

        # DONE requires nothing — always valid

        return self

    @field_validator("row_index")
    @classmethod
    def _non_negative_row(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError(f"row_index must be >= 0, got {v}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# CleanObservation
# ─────────────────────────────────────────────────────────────────────────────

class CleanObservation(Observation):
    """Observation returned to the agent after each step (and at reset).

    The agent sees the full current state of the dirty CSV at every step
    so it can decide what to fix next. This is intentionally verbose —
    passing the whole CSV string keeps the environment stateless from the
    agent's perspective (no hidden memory needed).

    Inherited from Observation (do NOT redeclare these):
      done:     bool           — True when the episode has ended
      reward:   float | None   — per-step reward (None at reset)
      metadata: Dict[str, Any] — extra info (unused by core loop)
    """

    # ── Task context (set at reset, constant for the episode) ────────────────

    task_id: VALID_TASK_IDS = Field(
        ...,
        description="Which task is active: 'easy', 'medium', or 'hard'.",
    )

    schema_hint: str = Field(
        ...,
        description=(
            "Plain-English description of the target schema. "
            "Tells the agent what the clean data should look like."
        ),
    )

    initial_dirty_cells: int = Field(
        ...,
        ge=0,
        description=(
            "Total number of cells that differed from ground truth at episode start. "
            "Used to compute a normalised progress score."
        ),
    )

    # ── Per-step state ───────────────────────────────────────────────────────

    dirty_csv: str = Field(
        ...,
        description=(
            "Full current state of the working DataFrame serialised as a CSV string. "
            "This reflects all changes the agent has made so far this episode."
        ),
    )

    current_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Grader score after the last action (0.0 = no cells correct, "
            "1.0 = perfect match with ground truth)."
        ),
    )

    issues_remaining: int = Field(
        default=0,
        ge=0,
        description=(
            "Approximate count of cells still differing from ground truth. "
            "Convenience field — agents can also derive this from the CSV."
        ),
    )

    step_number: int = Field(
        default=0,
        ge=0,
        description="How many steps have been taken in this episode so far.",
    )

    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum steps allowed for this task before forced termination.",
    )

    # ── Last-action feedback ────────────────────────────────────────────────

    last_action_success: bool = Field(
        default=True,
        description=(
            "Whether the last action was applied without errors. "
            "False if the column/row didn't exist, value couldn't be cast, etc."
        ),
    )

    last_action_error: Optional[str] = Field(
        default=None,
        description=(
            "Error message if last_action_success is False, else None. "
            "Helps the agent self-correct."
        ),
    )

    @field_validator("current_score")
    @classmethod
    def _round_score(cls, v: float) -> float:
        return round(v, 4)


# ─────────────────────────────────────────────────────────────────────────────
# CleanState
# ─────────────────────────────────────────────────────────────────────────────

class CleanState(State):
    """Internal server-side state. Never sent to the agent directly.

    Holds the live DataFrames, ground truth, and grader metadata.
    Because State uses extra="allow", we can store arbitrary fields
    without listing them in the JSON schema.

    Inherited from State:
      episode_id: Optional[str]   — unique episode identifier
      step_count: int             — steps taken this episode (ge=0)
    """

    # ── Task identity ────────────────────────────────────────────────────────

    task_id: str = Field(
        default="easy",
        description="Active task: 'easy', 'medium', or 'hard'.",
    )

    # ── DataFrame snapshots (stored as CSV strings for serialisation) ────────
    # NOTE: The environment keeps live pd.DataFrame objects in instance vars.
    # These string fields are the serialised snapshots used by state() calls
    # and for WebSocket state responses.

    dirty_csv_snapshot: str = Field(
        default="",
        description="Current working DataFrame serialised to CSV string.",
    )

    clean_csv_snapshot: str = Field(
        default="",
        description="Ground-truth clean DataFrame serialised to CSV string.",
    )

    # ── Scoring ──────────────────────────────────────────────────────────────

    initial_dirty_cells: int = Field(
        default=0,
        ge=0,
        description="Dirty cell count at episode start (denominator for progress).",
    )

    current_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Grader score after the last step.",
    )

    previous_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Grader score before the last step (for reward delta).",
    )

    # ── Task metadata (passed through from TaskDataset.metadata) ─────────────
    # Contains grader-specific ground truth: outlier_rows, canonical_lookup, etc.

    task_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-specific metadata from dataset_factory.TaskDataset.metadata. "
            "Contains grader ground truth (outlier_rows, duplicate_pairs, etc.)."
        ),
    )

    # ── Schema hint (echoed in observations) ────────────────────────────────

    schema_hint: str = Field(
        default="",
        description="Plain-English schema description for this task.",
    )

    # ── Per-task step budget ─────────────────────────────────────────────────

    max_steps: int = Field(
        default=40,
        ge=1,
        description="Maximum steps for this task (40 / 80 / 150 for easy/medium/hard).",
    )

    @field_validator("current_score", "previous_score")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)


# ── Step budget constants ─────────────────────────────────────────────────────

MAX_STEPS: Dict[str, int] = {
    "easy":   40,
    "medium": 80,
    "hard":   150,
}

# Done threshold: score at which the agent is considered successful
DONE_THRESHOLD: Dict[str, float] = {
    "easy":   0.95,
    "medium": 0.85,
    "hard":   0.80,
}


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("── CleanAction examples ──────────────────────────────────────")

    a1 = CleanAction(command="SET_VALUE", row_index=3, column="price", value="29.99")
    print("SET_VALUE:      ", a1.model_dump())

    a2 = CleanAction(command="DROP_ROW", row_index=17)
    print("DROP_ROW:       ", a2.model_dump())

    a3 = CleanAction(command="FILL_MISSING", column="quantity", fill_strategy="median")
    print("FILL_MISSING:   ", a3.model_dump())

    a4 = CleanAction(command="STANDARDIZE_COL", column="order_date")
    print("STANDARDIZE_COL:", a4.model_dump())

    a5 = CleanAction(command="DONE")
    print("DONE:           ", a5.model_dump())

    # Validation: SET_VALUE without row_index should fail
    print("\n── Validation ────────────────────────────────────────────────")
    try:
        bad = CleanAction(command="SET_VALUE", column="price", value="10.0")
    except Exception as e:
        print(f"Expected error (missing row_index): {e}")

    try:
        bad = CleanAction(command="FILL_MISSING", column="price")
    except Exception as e:
        print(f"Expected error (missing fill_strategy): {e}")

    print("\n── CleanObservation ──────────────────────────────────────────")
    obs = CleanObservation(
        task_id="easy",
        schema_hint="Sales orders dataset. price must be float.",
        initial_dirty_cells=29,
        dirty_csv="order_id,price\n1001,N/A\n1002,19.99",
        current_score=0.0,
        issues_remaining=29,
        step_number=0,
        max_steps=40,
        done=False,
        reward=None,
    )
    print(json.dumps(obs.model_dump(), indent=2))

    print("\n── CleanState ────────────────────────────────────────────────")
    state = CleanState(
        episode_id="ep-001",
        step_count=0,
        task_id="easy",
        dirty_csv_snapshot="order_id,price\n1001,N/A",
        clean_csv_snapshot="order_id,price\n1001,14.99",
        initial_dirty_cells=29,
        current_score=0.0,
        previous_score=0.0,
        task_metadata={"injected_cells": [(0, "price")]},
        schema_hint="Sales orders dataset.",
        max_steps=40,
    )
    print(json.dumps(state.model_dump(), indent=2))

    print("\n── JSON schemas ──────────────────────────────────────────────")
    print("Action schema keys:     ", list(CleanAction.model_json_schema()["properties"].keys()))
    print("Observation schema keys:", list(CleanObservation.model_json_schema()["properties"].keys()))
    print("State schema keys:      ", list(CleanState.model_json_schema()["properties"].keys()))    