__all__ = ("GenerationOutputMissing",)


class GenerationOutputMissing(Exception):
    """Raised when a generation completes without the requested output kind.

    Parameters
    ----------
    *args
        Exception message parts passed to ``Exception``.
    """
