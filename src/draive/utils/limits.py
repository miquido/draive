__all__ = ("verify_max_iterations",)


def verify_max_iterations(
    max_iterations: int | None,
    /,
) -> int | None:
    """Verify a limit of loop iterations.

    Parameters
    ----------
    max_iterations : int | None
        Maximum number of iterations, ``None`` stands for no limit.

    Returns
    -------
    int | None
        The verified limit.

    Raises
    ------
    ValueError
        When the limit is lower than 1 - a loop which can't run once is never intended.
    """
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations has to be at least 1")

    return max_iterations
