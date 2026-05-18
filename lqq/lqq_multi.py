try:
    from env.lqq_env import *  # noqa: F403
    from env.lqq_env import main
except ModuleNotFoundError:
    from .env.lqq_env import *  # type: ignore # noqa: F403
    from .env.lqq_env import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
