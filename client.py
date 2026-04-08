"""
client.py
---------
DataCleaningEnv — the typed WebSocket client for the data cleaning pipeline.

This module contains exactly one public class: ``DataCleaningEnv``.
It extends ``EnvClient`` from OpenEnv core and implements the three abstract
translation methods that bridge Python objects and the server's JSON wire format:

    _step_payload(action)      CleanAction   → dict   (outbound)
    _parse_result(payload)     dict          → StepResult[CleanObservation]  (inbound)
    _parse_state(payload)      dict          → CleanState  (inbound)

Everything else — WebSocket lifecycle, connect/disconnect, async context
manager, the `.sync()` wrapper — is handled by the base class.

Usage (async)
-------------
    import asyncio
    from data_cleaning_env.client import DataCleaningEnv
    from data_cleaning_env.models import CleanAction

    async def main():
        async with DataCleaningEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(task_id="easy")
            print(result.observation.schema_hint)

            result = await env.set_value(row_index=3, column="price", value="29.99")
            print(result.reward, result.observation.current_score)

            result = await env.done()

    asyncio.run(main())

Usage (sync wrapper)
--------------------
    env = DataCleaningEnv(base_url="http://localhost:7860").sync()
    with env:
        result = env.reset(task_id="medium")
        result = env.fill_missing(column="amount", fill_strategy="median")
        result = env.done()
"""

from __future__ import annotations

from typing import Any, Optional

# ── OpenEnv core imports ──────────────────────────────────────────────────────
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    from openenv.core.client_types import StepResult  # type: ignore[no-redef]
    from openenv.core.env_client import EnvClient     # type: ignore[no-redef]
#7860

# ── Local model imports (try relative then absolute) ──────────────────────────
try:
    from .models import (
        CleanAction,
        CleanObservation,
        CleanState,
        MAX_STEPS,
        DONE_THRESHOLD,
    )
except ImportError:
    from models import (                              # type: ignore[no-redef]
        CleanAction,
        CleanObservation,
        CleanState,
        MAX_STEPS,
        DONE_THRESHOLD,
    )


class DataCleaningEnv(EnvClient[CleanAction, CleanObservation, CleanState]):
    """
    Async WebSocket client for the Data Cleaning Pipeline environment.

    Connects to a running ``DataCleaningEnvironment`` server and exposes the
    standard OpenEnv interface (``reset``, ``step``, ``state``) plus typed
    convenience helpers for each command.

    All methods are async. For synchronous use, call ``.sync()`` to get a
    ``SyncEnvClient`` wrapper:

        with DataCleaningEnv(base_url="http://localhost:7860").sync() as env:
            result = env.reset(task_id="easy")
            result = env.set_value(row_index=0, column="price", value="9.99")

    Connecting to different backends
    ---------------------------------
    Local dev server (after ``openenv serve``):
        env = DataCleaningEnv(base_url="http://localhost:7860")

    Local Docker image (after ``openenv build``):
        env = await DataCleaningEnv.from_docker_image("data-cleaning-env:latest")

    Hugging Face Space (after ``openenv push``):
        env = await DataCleaningEnv.from_env("your-org/data-cleaning-env")
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract method implementations — the three translation methods
    # ─────────────────────────────────────────────────────────────────────────

    def _step_payload(self, action: CleanAction) -> dict[str, Any]:
        """
        Serialise a CleanAction to the JSON dict the server expects.

        The server's ``step()`` endpoint receives this dict, validates it
        against ``CleanAction``, and dispatches to the correct handler.

        We use ``model_dump(exclude_none=True)`` to omit fields the agent
        left as ``None`` — this keeps the wire message minimal and avoids
        triggering Pydantic's ``extra="forbid"`` validator on the server side
        for fields that weren't set.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CleanObservation]:
        """
        Parse the server's step/reset response into a ``StepResult``.

        Wire format (what the server sends back):
        ::
            {
              "observation": {
                "done": false,
                "reward": -0.005,
                "metadata": {},
                "task_id": "easy",
                "schema_hint": "Sales orders...",
                "initial_dirty_cells": 29,
                "dirty_csv": "row_index,order_id,...\\n0,1001,...",
                "current_score": 0.9550,
                "issues_remaining": 18,
                "step_number": 1,
                "max_steps": 40,
                "last_action_success": true,
                "last_action_error": null
              },
              "reward": -0.005,
              "done": false
            }

        Note: ``reward`` and ``done`` appear both at the top level (for
        convenience) and inside ``observation`` (because ``Observation`` base
        carries them).  We use the top-level copies for ``StepResult`` so the
        caller doesn't have to dig into the observation.
        """
        obs_data = payload.get("observation", {})

        observation = CleanObservation(
            # ── inherited from Observation base ──────────────────────────────
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),

            # ── task context (constant for the episode) ───────────────────────
            task_id=obs_data["task_id"],
            schema_hint=obs_data["schema_hint"],
            initial_dirty_cells=obs_data["initial_dirty_cells"],

            # ── per-step state ────────────────────────────────────────────────
            dirty_csv=obs_data["dirty_csv"],
            current_score=obs_data.get("current_score", 0.0),
            issues_remaining=obs_data.get("issues_remaining", 0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data["max_steps"],

            # ── last-action feedback ──────────────────────────────────────────
            last_action_success=obs_data.get("last_action_success", True),
            last_action_error=obs_data.get("last_action_error"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CleanState:
        """
        Parse the server's state response into a ``CleanState``.

        The server serialises ``CleanState`` via Pydantic's ``model_dump()``,
        so the wire keys match our field names exactly.  We use ``.get()``
        with sensible defaults everywhere so a partially-initialised state
        (e.g. before the first reset) doesn't crash the client.
        """
        return CleanState(
            # ── inherited from State base ─────────────────────────────────────
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),

            # ── task identity ─────────────────────────────────────────────────
            task_id=payload.get("task_id", "easy"),

            # ── DataFrame snapshots ───────────────────────────────────────────
            dirty_csv_snapshot=payload.get("dirty_csv_snapshot", ""),
            clean_csv_snapshot=payload.get("clean_csv_snapshot", ""),

            # ── scoring ───────────────────────────────────────────────────────
            initial_dirty_cells=payload.get("initial_dirty_cells", 0),
            current_score=payload.get("current_score", 0.0),
            previous_score=payload.get("previous_score", 0.0),

            # ── grader metadata ───────────────────────────────────────────────
            task_metadata=payload.get("task_metadata", {}),

            # ── schema ────────────────────────────────────────────────────────
            schema_hint=payload.get("schema_hint", ""),

            # ── step budget ───────────────────────────────────────────────────
            max_steps=payload.get("max_steps", 40),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Typed convenience helpers — one per CleanAction command
    # ─────────────────────────────────────────────────────────────────────────
    # These methods exist purely for ergonomics: they let callers write
    #
    #     await env.set_value(row_index=3, column="price", value="29.99")
    #
    # instead of the more verbose:
    #
    #     await env.step(CleanAction(
    #         command="SET_VALUE", row_index=3, column="price", value="29.99"
    #     ))
    #
    # The baseline inference script can use either form.

    async def set_value(
        self,
        row_index: int,
        column: str,
        value: str,
    ) -> StepResult[CleanObservation]:
        """Fix a single cell. ``value`` is always passed as a string; the
        server casts it to the column's target dtype automatically."""
        return await self.step(
            CleanAction(
                command="SET_VALUE",
                row_index=row_index,
                column=column,
                value=value,
            )
        )

    async def drop_row(self, row_index: int) -> StepResult[CleanObservation]:
        """Remove an entire row (e.g. a true outlier in the medium task)."""
        return await self.step(
            CleanAction(command="DROP_ROW", row_index=row_index)
        )

    async def standardize_col(self, column: str) -> StepResult[CleanObservation]:
        """Normalise a whole column's format.

        The server auto-detects what to do:
        - Date columns → parse any format, reformat as ``YYYY-MM-DD``
        - Numeric columns → coerce to float/int, drop unit strings
        - String columns → strip leading/trailing whitespace
        """
        return await self.step(
            CleanAction(command="STANDARDIZE_COL", column=column)
        )

    async def fill_missing(
        self,
        column: str,
        fill_strategy: str,
    ) -> StepResult[CleanObservation]:
        """Fill ``NaN`` values in ``column``.

        Args:
            column: Column name to fill.
            fill_strategy: One of ``"mean"``, ``"median"``, ``"mode"``, ``"drop"``.
                ``"drop"`` removes rows where the column is ``NaN``.
        """
        return await self.step(
            CleanAction(
                command="FILL_MISSING",
                column=column,
                fill_strategy=fill_strategy,
            )
        )

    async def done(self) -> StepResult[CleanObservation]:
        """Signal that the agent believes the CSV is clean.

        This ends the episode immediately.  If the current score is below
        ``EARLY_DONE_THRESHOLD`` (0.60) a penalty of -0.20 is applied.
        """
        return await self.step(CleanAction(command="DONE"))

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def current_score(self) -> float:
        """Return the grader score from the last step (0.0–1.0)."""
        st = await self.state()
        return st.current_score

    async def task_id(self) -> str:
        """Return the active task ID (``"easy"``, ``"medium"``, or ``"hard"``)."""
        st = await self.state()
        return st.task_id

    async def steps_remaining(self) -> int:
        """Return the number of steps left before forced termination."""
        st = await self.state()
        return max(0, st.max_steps - st.step_count)

    async def is_solved(self) -> bool:
        """Return ``True`` if the current score meets the task's done threshold."""
        st = await self.state()
        threshold = DONE_THRESHOLD.get(st.task_id, 0.95)
        return st.current_score >= threshold