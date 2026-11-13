from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Literal

from haiway import ctx

from draive.guardrails.types import GuardrailsException, GuardrailsFailure

__all__ = ("guardrails_verdict",)

type _Outcome = Literal["passed", "blocked", "failed", "cancelled"]


def guardrails_verdict[**Args, Result](
    kind: str,
    /,
) -> Callable[[Callable[Args, Awaitable[Result]]], Callable[Args, Awaitable[Result]]]:
    """Record the verdict of a guardrails check as a counter.

    Parameters
    ----------
    kind : str
        Guardrails kind used to label the verdict, i.e. ``"moderation.input"``.

    Returns
    -------
    Callable
        Decorator recording ``guardrails.verdicts`` with the outcome of each call -
        ``passed``, ``blocked`` when guardrails rejected the content, ``failed`` when
        the check itself broke and ``cancelled``. Content is never recorded.
    """

    def decorator(
        function: Callable[Args, Awaitable[Result]],
        /,
    ) -> Callable[Args, Awaitable[Result]]:
        @wraps(function)
        async def verifying(
            *args: Args.args,
            **kwargs: Args.kwargs,
        ) -> Result:
            outcome: _Outcome = "passed"
            try:
                return await function(*args, **kwargs)

            except GuardrailsFailure:
                outcome = "failed"
                raise

            except GuardrailsException:
                outcome = "blocked"
                raise

            except Exception:
                outcome = "failed"
                raise

            except BaseException:
                outcome = "cancelled"
                raise

            finally:
                ctx.record_info(
                    metric="guardrails.verdicts",
                    value=1,
                    unit="count",
                    kind="counter",
                    attributes={
                        "guardrails.kind": kind,
                        "guardrails.outcome": outcome,
                    },
                )

        return verifying

    return decorator
