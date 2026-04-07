try:
    from openenv.core.env_server import create_app
    from ..models import CleanAction, CleanObservation
    from .data_cleaning_env import DataCleaningEnvironment
except ImportError:
    from openenv.core.env_server import create_app
    from models import CleanAction, CleanObservation
    from server.data_cleaning_env import DataCleaningEnvironment

app = create_app(
    DataCleaningEnvironment,   # class, not instance
    CleanAction,
    CleanObservation,
    env_name="data_cleaning_env",
)


def main() -> None:
    """Entry point for openenv serve / uv run / python -m."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()