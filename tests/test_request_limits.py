"""Shared request budgets, without running a model or contacting an API."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from request_limits import SharedRequestLimiter


_PROJECT = Path(__file__).resolve().parents[1]
_WORKER = r'''
import asyncio
import json
from pathlib import Path
import sys
import time
from request_limits import SharedRequestLimiter

async def main():
    directory, ready, start, duration = sys.argv[1:]
    limiter = SharedRequestLimiter("http://localhost:8000/v1", 2, directory=Path(directory))
    Path(ready).write_text("ready")
    while not Path(start).exists():
        await asyncio.sleep(0.01)
    async with limiter.slot():
        # Python 3.10's Windows monotonic clock can give equal timestamps to
        # successive workers. perf_counter is cross-process on Python 3.10+
        # and resolves their actual order without inventing overlapping slots.
        print(json.dumps(["start", time.perf_counter_ns()]), flush=True)
        Path(ready + ".acquired").write_text("active")
        await asyncio.sleep(float(duration))
        print(json.dumps(["end", time.perf_counter_ns()]), flush=True)
asyncio.run(main())
'''


class SharedRequestLimiterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        parent = _PROJECT / ".tmp_tests"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def limiter(self, limit=1, endpoint="http://localhost:8000/v1"):
        return SharedRequestLimiter(endpoint, limit, directory=self.root)

    async def assert_available(self, limiter):
        async def acquire():
            async with limiter.slot():
                return True
        self.assertTrue(await asyncio.wait_for(acquire(), timeout=1))

    async def test_equivalent_endpoints_share_slots_without_persisting_credentials(self):
        first = self.limiter(endpoint="HTTP://SecretUser:SecretPassword@LOCALHOST:80/v1/?key=SecretToken")
        second = self.limiter(endpoint="http://localhost/v1")
        async with first.slot():
            pending = asyncio.create_task(self.assert_available(second))
            await asyncio.sleep(0.08)
            self.assertFalse(pending.done())
        await pending
        persisted = "".join(str(path) + (path.read_text() if path.is_file() else "")
                            for path in self.root.rglob("*"))
        for secret in ("SecretUser", "SecretPassword", "SecretToken", "localhost"):
            self.assertNotIn(secret, persisted)

    async def test_distinct_endpoints_and_api_paths_have_independent_budgets(self):
        async with self.limiter().slot():
            for endpoint in ("http://localhost:8001/v1", "http://localhost:8000/v2", "https://localhost:8000/v1"):
                with self.subTest(endpoint=endpoint):
                    await self.assert_available(self.limiter(endpoint=endpoint))

    async def test_cancelling_waiter_does_not_release_an_active_request(self):
        limiter = self.limiter()
        async with limiter.slot():
            waiter = asyncio.create_task(self.assert_available(limiter))
            await asyncio.sleep(0.08)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            another = asyncio.create_task(self.assert_available(limiter))
            await asyncio.sleep(0.08)
            self.assertFalse(another.done())
        await another

    async def test_cancelling_active_request_releases_slot(self):
        limiter = self.limiter()
        started = asyncio.Event()

        async def request():
            async with limiter.slot():
                started.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(request())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.assert_available(limiter)

    async def test_request_exception_releases_slot(self):
        limiter = self.limiter()
        with self.assertRaisesRegex(RuntimeError, "failed request"):
            async with limiter.slot():
                raise RuntimeError("failed request")
        await self.assert_available(limiter)

    async def test_different_limit_is_rejected_while_requests_are_active(self):
        async with self.limiter(1).slot():
            with self.assertRaisesRegex(ValueError, "Conflicting max_concurrent_requests_total"):
                async with self.limiter(2).slot():
                    self.fail("Conflicting limits must not bypass the active budget")

    async def test_idle_limit_can_increase_and_decrease(self):
        for count in (1, 3, 1):
            await self.assert_available(self.limiter(count))
            metadata = list(self.root.rglob("limit.json"))
            self.assertEqual(len(metadata), 1)
            self.assertEqual(json.loads(metadata[0].read_text())["limit"], count)

    async def test_waiter_rechecks_limit_after_idle_reconfiguration(self):
        limiter = self.limiter(1)
        async with limiter.slot():
            waiter = asyncio.create_task(self.assert_available(limiter))
            await asyncio.sleep(0.08)
            self.assertFalse(waiter.done())
        # No await separates release and reconfiguration: the old waiter is
        # still sleeping and must read the new budget on its next attempt.
        async with self.limiter(2).slot():
            with self.assertRaisesRegex(ValueError, "Conflicting max_concurrent_requests_total"):
                await waiter

    async def test_invalid_metadata_is_reported_and_preserved(self):
        limiter = self.limiter()
        await self.assert_available(limiter)
        metadata = next(self.root.rglob("limit.json"))
        for invalid in ("broken", '{"schema_version": 1, "limit": 0}'):
            metadata.write_text(invalid)
            with self.assertRaisesRegex(ValueError, "metadata"):
                await self.assert_available(limiter)
            self.assertEqual(metadata.read_text(), invalid)

    async def test_open_permission_error_propagates_instead_of_retrying(self):
        with mock.patch.object(Path, "open", side_effect=PermissionError("permission denied")):
            with self.assertRaises(PermissionError):
                await self.assert_available(self.limiter())

    def test_invalid_limits_and_endpoints_are_rejected(self):
        for value in (0, -1, True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.limiter(value)
        for endpoint in ("", "localhost:8000", "file:///tmp", "http://localhost:bad"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.limiter(endpoint=endpoint)


class SharedRequestLimiterProcessTests(unittest.TestCase):
    def setUp(self):
        parent = _PROJECT / ".tmp_tests"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processes = []
        self.addCleanup(self.stop_workers)

    def stop_workers(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

    def worker(self, index, *, duration="0.2"):
        ready = self.root / f"ready-{index}"
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", _WORKER, str(self.root / "locks"),
             str(ready), str(self.root / "start"), duration],
            cwd=_PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append(process)
        return process, ready

    def wait_for(self, condition):
        deadline = time.monotonic() + 10
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("Request limiter subprocess did not become ready")
            for process in self.processes:
                if process.poll() is not None and process.returncode:
                    _, error = process.communicate()
                    self.fail(error)
            time.sleep(0.01)

    def test_processes_obey_the_same_total_budget(self):
        workers = [self.worker(index) for index in range(5)]
        self.wait_for(lambda: all(path.exists() for _, path in workers))
        (self.root / "start").touch()
        events = []
        for process, _ in workers:
            output, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, error)
            events.extend(json.loads(line) for line in output.splitlines())
        active = peak = 0
        for kind, _ in sorted(events, key=lambda event: event[1]):
            active += 1 if kind == "start" else -1
            peak = max(active, peak)
            self.assertLessEqual(active, 2, events)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 2)

    def test_process_exit_releases_its_slot(self):
        workers = [self.worker(index, duration="60") for index in range(2)]
        self.wait_for(lambda: all(path.exists() for _, path in workers))
        (self.root / "start").touch()
        self.wait_for(lambda: all(Path(str(path) + ".acquired").exists() for _, path in workers))
        replacement, ready = self.worker(2, duration="0.01")
        self.wait_for(ready.exists)
        time.sleep(0.08)
        self.assertIsNone(replacement.poll())
        workers[0][0].terminate()
        workers[0][0].communicate(timeout=10)
        output, error = replacement.communicate(timeout=10)
        self.assertEqual(replacement.returncode, 0, error)
        self.assertEqual([json.loads(line)[0] for line in output.splitlines()], ["start", "end"])


if __name__ == "__main__":
    unittest.main()
