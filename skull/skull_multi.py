try:
    from env.skull_env import *  # noqa: F403
    from env.skull_env import main
except ModuleNotFoundError:
    from .env.skull_env import *  # type: ignore # noqa: F403
    from .env.skull_env import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
