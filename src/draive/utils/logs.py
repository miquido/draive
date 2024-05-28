from logging.config import dictConfig

from draive.utils.env import getenv_bool

__all__ = [
    "setup_logging",
]


def setup_logging(
    *loggers: str,
    debug: bool = getenv_bool("DEBUG_LOGGING", __debug__),
) -> None:
    """\
    Setup logging configuration and prepare specified loggers.

    Parameters
    ----------
    loggers: *str
        names of additional loggers to configure

    NOTE: this function should be run only once on application start
    """

    dictConfig(
        config={
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)-4s] [%(name)s] %(message)s",
                    "datefmt": "%d/%b/%Y:%H:%M:%S +0000",
                },
            },
            "handlers": {
                "console": {
                    "level": "DEBUG" if debug else "INFO",
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                name: {
                    "handlers": ["console"],
                    "level": "DEBUG" if debug else "INFO",
                    "propagate": False,
                }
                for name in loggers
            },
            "root": {  # root logger
                "handlers": ["console"],
                "level": "DEBUG" if debug else "INFO",
                "propagate": False,
            },
        },
    )
