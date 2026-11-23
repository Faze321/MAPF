from __future__ import annotations

import errno
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


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


def atomic_write_dataframe(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp{suffix}"
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        replace_cache_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.json"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        replace_cache_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def replace_cache_file(temporary: Path, path: Path) -> None:
    # Windows readers can briefly deny replacement; retain the atomic write.
    for attempt in range(11):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 10:
                raise
            time.sleep(0.05 * (attempt + 1))


@contextmanager
def process_file_lock(path: Path, *, blocking: bool, conflict_message: str | None = None):
    """Hold an OS-released lock; the caller selects waiting or fail-fast behavior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Both lock APIs permit ranges beyond EOF; no initialization write is needed.
    with path.open("a+b", buffering=0) as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if conflict_message is not None:
                raise RuntimeError(conflict_message) from exc
            raise
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
