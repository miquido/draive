import asyncio
import contextvars
import fcntl
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from typing import Any, Final, cast, final

from haiway import ValidationError, ctx

from draive.claude.credentials import (
    DEFAULT_LOGIN_DIRECTORY,
    ClaudeCodeAccessToken,
    ClaudeCodeApiKey,
    ClaudeCodeCredential,
    ClaudeCodeLocalLogin,
    ClaudeCodeOAuthToken,
    ClaudeCodeResolvedCredential,
)
from draive.claude.exceptions import ClaudeCodeUnavailable
from draive.claude.launch import inherited_environment

__all__ = ("ClaudeCodeLogins",)

_CREDENTIALS_FILE: Final[str] = ".credentials.json"
_LOCK_FILE: Final[str] = ".login-refresh.lock"
_LOCK_POLL: Final[float] = 0.05
# a credentials file is a few hundred bytes, anything far larger is not one
_MAX_CREDENTIALS_BYTES: Final[int] = 64 * 1024


@final
class ClaudeCodeLogins:
    """Resolves local Claude Code logins into access tokens and owns their refreshes.

    Prepared by ``ClaudeCode`` for the scope it enters, which waits for the refreshes still in
    flight when it exits. A login has at most one refresh in flight, shared by every run which
    needs it meanwhile, and a refresh which did not help or failed is not attempted again during
    a cooldown. Refreshes run within the context the instance was made in, so what they report
    reaches that scope also after the run which started them has ended. Across processes
    refreshes are serialized with a file lock (``.login-refresh.lock`` in the configuration
    directory of the login, which has to be writable), which coordinates draive processes only -
    not an interactive Claude Code of the user.

    A refresh runs on its own task: a run which is cancelled, or which stops waiting for it, does
    not stop it, so a rotated refresh token is still written. A run whose token is still valid
    waits for a refresh at most ``refresh_timeout`` (and never past the expiry of the token) and
    then goes on with that token, also when the refresh failed; a run whose token has expired
    waits for the refresh to end.

    Parameters
    ----------
    lock_timeout
        Limit in seconds of waiting for the lock of another process.
    refresh_timeout
        Limit in seconds of waiting for a refresh by a run whose token is still valid.
    refresh_limit
        Limit in seconds of a single ``claude auth status`` refresh, after which its process is
        killed - only a refresh which hangs gets there.
    cooldown
        Time in seconds after a refresh which did not extend the token or failed, in which no
        other refresh of the same login is attempted.
    """

    __slots__ = (
        "_attempts",
        "_context",
        "_cooldown",
        "_lock_timeout",
        "_refresh_limit",
        "_refresh_timeout",
        "_refreshes",
    )

    def __init__(
        self,
        *,
        lock_timeout: float = 30.0,
        refresh_timeout: float = 15.0,
        refresh_limit: float = 120.0,
        cooldown: float = 60.0,
    ) -> None:
        self._lock_timeout: float = lock_timeout
        self._refresh_timeout: float = refresh_timeout
        self._refresh_limit: float = refresh_limit
        self._cooldown: float = cooldown
        # refreshes outlive the runs which start them, they report to the scope of the instance
        self._context: contextvars.Context = contextvars.copy_context()
        self._attempts: dict[str, float] = {}
        self._refreshes: dict[str, asyncio.Task[ClaudeCodeAccessToken]] = {}

    def __copy__(self) -> ClaudeCodeLogins:
        return self  # a service owning tasks and locks, copies of a state share it

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> ClaudeCodeLogins:
        return self

    @staticmethod
    async def verify(
        credential: ClaudeCodeCredential,
        /,
    ) -> None:
        """Verify the credentials file of a login is there and readable, nothing is cached.

        Parameters
        ----------
        credential
            Credential to verify, only a local login has anything to verify.

        Raises
        ------
        ClaudeCodeUnavailable
            When the file of a login is missing or malformed.
        """
        match credential:
            case ClaudeCodeLocalLogin() as login:
                await asyncio.to_thread(_read_login, _credentials_path(login))

            case ClaudeCodeApiKey() | ClaudeCodeOAuthToken():
                pass

    async def drain(self) -> None:
        """Wait for the refreshes still in flight, so none is cut off when the scope ends.

        Bounded by ``lock_timeout`` and ``refresh_limit`` of each refresh. Their failures were
        reported to the runs which waited for them and are not raised again.
        """
        while self._refreshes:
            # not gather - a cancelled drain must not cancel the refreshes it waits for
            await asyncio.wait(tuple(self._refreshes.values()))

    async def resolve(
        self,
        credential: ClaudeCodeCredential,
        /,
        *,
        binary: str,
    ) -> ClaudeCodeResolvedCredential:
        """Resolve the credential for a single run - a login becomes its current access token.

        Parameters
        ----------
        credential
            Credential given to the client.
        binary
            Claude Code executable, used to refresh a login about to expire.

        Returns
        -------
        ClaudeCodeResolvedCredential
            The credential the run carries.

        Raises
        ------
        ClaudeCodeUnavailable
            When the login has no usable credentials, or its token has expired and could not be
            refreshed.
        """
        match credential:
            case ClaudeCodeLocalLogin() as login:
                return await self._resolve(login, binary=binary)

            case ClaudeCodeApiKey() | ClaudeCodeOAuthToken() as value:
                return value

    async def _resolve(
        self,
        login: ClaudeCodeLocalLogin,
        /,
        *,
        binary: str,
    ) -> ClaudeCodeAccessToken:
        path: str = _credentials_path(login)
        token: ClaudeCodeAccessToken = await asyncio.to_thread(_read_login, path)
        if _remaining(token) < login.refresh_margin:
            token = await self._refreshing(login, token, path=path, binary=binary)

        remaining: float = _remaining(token)
        if remaining <= 0:
            raise ClaudeCodeUnavailable(
                "the access token of the login has expired and could not be refreshed,"
                " log in again with `claude /login` as the user of the login"
            )

        if remaining < login.refresh_margin:
            ctx.log_warning(
                f"The access token of the Claude Code login expires in {remaining:.0f} seconds,"
                " a run which outlives it fails"
            )

        return token

    async def _refreshing(
        self,
        login: ClaudeCodeLocalLogin,
        token: ClaudeCodeAccessToken,
        /,
        *,
        path: str,
        binary: str,
    ) -> ClaudeCodeAccessToken:
        key: str = await asyncio.to_thread(os.path.realpath, os.path.dirname(path))
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        task: asyncio.Task[ClaudeCodeAccessToken] | None = self._refreshes.get(key)
        if task is None:
            if (attempt := self._attempts.get(key)) is not None and (
                loop.time() - attempt < self._cooldown
            ):
                return token  # tried a moment ago, the token is used as it is

            # refreshing rotates the refresh token, it is not left half done when the caller is
            # cancelled or stops waiting - the file would not get the new one - so it runs on its
            # own task, which the scope waits for when it exits
            task = loop.create_task(
                self._refreshed(login, key=key, binary=binary),
                context=self._context.copy(),
            )
            self._refreshes[key] = task
            task.add_done_callback(partial(self._finished, key))

        if (remaining := _remaining(token)) > 0:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=min(self._refresh_timeout, remaining),
                )

            except TimeoutError:
                if _remaining(token) > 0:
                    ctx.log_warning(
                        "Refreshing the Claude Code login takes too long, its current token is used"
                    )
                    return token

            except ClaudeCodeUnavailable as exc:
                if _remaining(token) <= 0:
                    raise

                ctx.log_warning(
                    f"Refreshing the Claude Code login failed, its current token is used: {exc}"
                )
                return token

        # nothing to go on with, the refresh decides
        return await asyncio.shield(task)

    def _finished(
        self,
        key: str,
        task: asyncio.Task[ClaudeCodeAccessToken],
        /,
    ) -> None:
        if self._refreshes.get(key) is task:
            del self._refreshes[key]

        if not task.cancelled():
            task.exception()  # retrieved, a caller which went away does not report it

    async def _refreshed(
        self,
        login: ClaudeCodeLocalLogin,
        /,
        *,
        key: str,
        binary: str,
    ) -> ClaudeCodeAccessToken:
        directory: str = os.path.expanduser(login.config_directory)
        path: str = _credentials_path(login)
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            async with _FileLock(os.path.join(directory, _LOCK_FILE), self._lock_timeout):
                # another process might have refreshed it while this was waiting
                token: ClaudeCodeAccessToken = await asyncio.to_thread(_read_login, path)
                if _remaining(token) >= login.refresh_margin:
                    self._attempts.pop(key, None)
                    return token

                await self._refresh(login, binary=binary)
                token = await asyncio.to_thread(_read_login, path)

        except ClaudeCodeUnavailable as exc:
            self._attempts[key] = loop.time()
            ctx.log_warning(f"Refreshing the Claude Code login failed: {exc}")
            raise

        if _remaining(token) < login.refresh_margin:
            self._attempts[key] = loop.time()
            ctx.log_warning("Refreshing the Claude Code login did not extend its token")

        else:
            self._attempts.pop(key, None)

        return token

    async def _refresh(
        self,
        login: ClaudeCodeLocalLogin,
        /,
        *,
        binary: str,
    ) -> None:
        environment: dict[str, str] = await asyncio.to_thread(_refresh_environment, login)
        try:
            process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
                binary,
                "auth",
                "status",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
            )

        except OSError as exc:
            # the token is used while it is valid, the expiry is what decides
            ctx.log_warning(f"Claude Code can not be started to refresh the login: {exc.strerror}")
            return

        try:
            # the exit code is not a verdict, the file is - it is read again
            await asyncio.wait_for(process.wait(), timeout=self._refresh_limit)

        except TimeoutError:
            ctx.log_warning("Refreshing the Claude Code login hangs, it is stopped")

        finally:
            # only a hung refresh, or one torn down with its event loop, is still running here
            if process.returncode is None:
                process.kill()
                await process.wait()


def _credentials_path(
    login: ClaudeCodeLocalLogin,
    /,
) -> str:
    return os.path.join(os.path.expanduser(login.config_directory), _CREDENTIALS_FILE)


def _remaining(
    token: ClaudeCodeAccessToken,
    /,
) -> float:
    return (token.expires_at - datetime.now(UTC)).total_seconds()


def _refresh_environment(
    login: ClaudeCodeLocalLogin,
    /,
) -> dict[str, str]:
    # as the user of the login with its real home - the refresh needs the login, not an isolated
    # one, and the token it writes never reaches anything but the credentials file
    environment: dict[str, str] = dict(inherited_environment())
    environment["HOME"] = os.path.expanduser("~")
    directory: str = os.path.expanduser(login.config_directory)
    if os.path.realpath(directory) != os.path.realpath(os.path.expanduser(DEFAULT_LOGIN_DIRECTORY)):
        environment["CLAUDE_CONFIG_DIR"] = directory

    return environment


async def _in_thread[Result](
    call: Callable[[str], Result],
    argument: str,
    /,
    *,
    cleanup: Callable[[Result], None],
) -> Result:
    # a thread can not be cancelled - when the caller is, its result is still released
    future: asyncio.Future[Result] = asyncio.ensure_future(asyncio.to_thread(call, argument))
    try:
        return await asyncio.shield(future)

    except asyncio.CancelledError:
        future.add_done_callback(
            lambda done: (
                cleanup(done.result())
                if not done.cancelled() and done.exception() is None
                else None
            )
        )
        raise


def _release(
    descriptor: int | None,
    /,
) -> None:
    if descriptor is not None:
        os.close(descriptor)  # closing releases the lock


def _lock_attempt(
    path: str,
    /,
) -> int | None:
    descriptor: int = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    except BlockingIOError:
        os.close(descriptor)
        return None

    except BaseException:
        os.close(descriptor)
        raise

    return descriptor


class _FileLock:
    """Exclusive ``flock`` of a file within a bounded wait, which never blocks the loop and
    never holds a descriptor it is not going to release, also when cancelled."""

    __slots__ = ("_descriptor", "_path", "_timeout")

    def __init__(
        self,
        path: str,
        timeout: float,
        /,
    ) -> None:
        self._path: str = path
        self._timeout: float = timeout
        self._descriptor: int | None = None

    async def __aenter__(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        deadline: float = loop.time() + self._timeout
        while True:
            try:
                descriptor: int | None = await _in_thread(
                    _lock_attempt,
                    self._path,
                    cleanup=_release,
                )

            except OSError as exc:
                raise ClaudeCodeUnavailable(
                    f"the lock of the login refresh can not be used: {exc.strerror}"
                ) from exc

            if descriptor is not None:
                self._descriptor = descriptor
                return

            if loop.time() >= deadline:
                raise ClaudeCodeUnavailable("another process is refreshing the login for too long")

            await asyncio.sleep(_LOCK_POLL)

    async def __aexit__(
        self,
        *_: object,
    ) -> None:
        _release(self._descriptor)
        self._descriptor = None


def _read_login(
    path: str,
    /,
) -> ClaudeCodeAccessToken:
    outcome: ClaudeCodeAccessToken | str = _load(path)
    if isinstance(outcome, str):
        # raised outside of everything which holds the content of the file - neither the
        # exception it was loaded with nor the frames parsing it are a part of this one
        raise ClaudeCodeUnavailable(outcome)

    return outcome


def _load(
    path: str,
    /,
) -> ClaudeCodeAccessToken | str:
    try:
        with open(path, "rb") as file:
            data: bytes = file.read(_MAX_CREDENTIALS_BYTES + 1)

        if len(data) > _MAX_CREDENTIALS_BYTES:
            return _malformed(path)

        content: Any = json.loads(data.decode("utf-8"))

    except FileNotFoundError:
        return (
            f"the login has no credentials file ({path}), log in with `claude /login` as the"
            " user of the login - logins kept in the keychain (macOS) are not supported"
        )

    except OSError as exc:
        return f"the credentials file of the login ({path}) can not be read: {exc.strerror}"

    # malformed JSON or encoding, or nesting deeper than the parser goes - the exceptions and the
    # frames of the parser carry the content
    except ValueError, RecursionError:
        return _malformed(path)

    return _access_token(content, path=path)


def _malformed(
    path: str,
    /,
) -> str:
    return f"the credentials file of the login ({path}) is malformed"


def _access_token(
    content: Any,
    /,
    *,
    path: str,
) -> ClaudeCodeAccessToken | str:
    if not isinstance(content, Mapping):
        return _malformed(path)

    login: Any = cast(Mapping[str, Any], content).get("claudeAiOauth")
    if not isinstance(login, Mapping):
        return _malformed(path)

    fields: Mapping[str, Any] = cast(Mapping[str, Any], login)
    value: Any = fields.get("accessToken")
    scopes: Any = fields.get("scopes")
    expires: Any = fields.get("expiresAt")
    if (
        not isinstance(value, str)
        or not isinstance(scopes, list)
        or not all(isinstance(scope, str) for scope in cast(list[Any], scopes))
        or isinstance(expires, bool)
        or not isinstance(expires, int | float)
    ):
        return _malformed(path)

    try:
        return ClaudeCodeAccessToken(
            value=value,
            scopes=tuple(cast(list[str], scopes)),
            expires_at=datetime.fromtimestamp(expires / 1000, UTC),
        )

    except ValidationError, ValueError, OverflowError, OSError:
        return _malformed(path)
