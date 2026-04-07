# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Cleaning Env Environment Client."""
from pydantic import BaseModel
from typing import Dict

from openenv.core import EnvClient, Action, Observation
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CleanAction, CleanObservation
except ImportError:
    from models import CleanAction, CleanObservation


# ✅ ACTION model — what the agent sends TO the environment
class CleanAction(Action):
    command: str          # e.g. "drop_column", "fill_missing", "rename_column"
    column: str | None = None
    value: str | None = None

# ✅ OBSERVATION model — what the environment sends BACK to the agent
class CleanObservation(Observation):
    echoed_message: str = ""
    message_length: int = 0
    done: bool = False
    reward: float | None = None
    metadata: dict = {}

class DataCleaningEnvClient(EnvClient[CleanAction, CleanObservation]):
    def _parse_result(self, data: dict) -> CleanObservation:
        # ✅ Parse into Observation, NOT CleanAction
        return CleanObservation(**data)

class DataCleaningEnv(
    EnvClient[CleanAction, CleanObservation, State]
):
    """
    Client for the Data Cleaning Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DataCleaningEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CleanAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DataCleaningEnv.from_docker_image("data_cleaning_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CleanAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CleanAction) -> Dict:
        """
        Convert CleanAction to JSON payload for step message.

        Args:
            action: CleanAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CleanAction]:
        """
        Parse server response into StepResult[CleanAction].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CleanAction
        """
        obs_data = payload.get("observation", {})
        observation = CleanAction(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
