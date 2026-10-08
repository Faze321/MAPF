"""A local-machine request budget shared by processes using the same endpoint."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit
import uuid


_DEFAULT_DIRECTORY = Path(__file__).resolve().parent / ".cache" / "llm_request_limits"
_POLL_INTERVAL = 0.05


def _endpoint_key(base_url: str) -> str:
    """Normalize transport/address/API path without persisting credentials."""
    try:
        parsed = urlsplit(str(base_url).strip())
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("The shared request limiter requires a valid HTTP(S) base_url") from None
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("The shared request limiter requires a valid HTTP(S) base_url")
    hostname = hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    if port == {"http": 80, "https": 443}[scheme]:
        port = None
    address = hostname if port is None else f"{hostname}:{port}"
    # Credentials and query parameters must not divide a server's request budget.
    endpoint = f"{scheme}://{address}{parsed.path.rstrip('/')}"
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()


def _try_file_lock(path: Path) -> BinaryIO | None:
    # Opening failures (permissions, disk errors) are never treated as contention.
    handle = path.open("a+b", buffering=0)
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                # The Windows CRT reports a busy byte-range lock as EACCES.
                winerror = getattr(exc, "winerror", None)
                if winerror == 33 or (winerror is None and exc.errno in {errno.EACCES, errno.EAGAIN}):
                    handle.close()
                    return None
                raise
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                    handle.close()
                    return None
                raise
        return handle
    except BaseException:
        handle.close()
        raise


def _release_file_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class SharedRequestLimiter:
    """Bound active requests across processes on this machine.

    Every client for an endpoint must configure the same positive limit and use
    the same directory. File locks are released by the OS if a worker exits.
    Waiting is cancellable and does not hold a thread or block the event loop.
    """

    def __init__(self, base_url: str, limit: int, *, directory: Path | None = None):
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_concurrent_requests_total must be a positive integer")
        self.limit = limit
        self.directory = Path(directory) if directory is not None else _DEFAULT_DIRECTORY
        self._endpoint_directory = self.directory / _endpoint_key(base_url)

    @asynccontextmanager
    async def slot(self):
        handle = None
        while handle is None:
            handle = self._try_acquire()
            if handle is None:
                await asyncio.sleep(_POLL_INTERVAL)
        try:
            yield
        finally:
            _release_file_lock(handle)

    def _read_limit(self) -> int | None:
        path = self._endpoint_directory / "limit.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Invalid shared request limit metadata; restore limit.json while requests are idle") from None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or type(value.get("limit")) is not int
            or value["limit"] <= 0
        ):
            raise ValueError("Invalid shared request limit metadata; restore limit.json while requests are idle")
        return value["limit"]

    def _write_limit(self) -> None:
        path = self._endpoint_directory / "limit.json"
        temporary = path.with_name(f"limit.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"schema_version": 1, "limit": self.limit}), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _has_active_slots(self) -> bool:
        # Called while holding the guard: nobody can acquire another slot until
        # this check and a potential metadata update finish.
        for path in self._endpoint_directory.glob("slot-*.lock"):
            handle = _try_file_lock(path)
            if handle is None:
                return True
            _release_file_lock(handle)
        return False

    def _try_acquire(self) -> BinaryIO | None:
        self._endpoint_directory.mkdir(parents=True, exist_ok=True)
        guard = _try_file_lock(self._endpoint_directory / "guard.lock")
        if guard is None:
            return None
        slot = None
        try:
            previous_limit = self._read_limit()
            if previous_limit != self.limit:
                if self._has_active_slots():
                    raise ValueError(
                        "Conflicting max_concurrent_requests_total for the same endpoint: "
                        f"active requests use {previous_limit!r}, requested {self.limit}. "
                        "Use the same limit in all clients sharing the endpoint."
                    )
                self._write_limit()
            for index in range(self.limit):
                slot = _try_file_lock(self._endpoint_directory / f"slot-{index}.lock")
                if slot is not None:
                    break
        finally:
            try:
                _release_file_lock(guard)
            except BaseException:
                if slot is not None:
                    _release_file_lock(slot)
                raise
        return slot
