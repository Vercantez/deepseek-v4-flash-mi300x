import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path


class _Logger:
    def warning(self, *args, **kwargs):
        return None


vllm = types.ModuleType("vllm")
logger_module = types.ModuleType("vllm.logger")
logger_module.init_logger = lambda _: _Logger()
base_module = types.ModuleType("vllm.v1.kv_offload.base")
base_module.OffloadKey = bytes


class ReqContext:
    def __init__(self, req_id: str):
        self.req_id = req_id


base_module.ReqContext = ReqContext
for name in (
    "vllm",
    "vllm.logger",
    "vllm.v1",
    "vllm.v1.kv_offload",
    "vllm.v1.kv_offload.base",
):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["vllm"] = vllm
sys.modules["vllm.logger"] = logger_module
sys.modules["vllm.v1.kv_offload.base"] = base_module

MODULE_PATH = Path(__file__).parents[1] / "patches" / "async_lookup.bounded.py"
SPEC = importlib.util.spec_from_file_location("async_lookup_bounded", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class ControlledLookupManager(module.AsyncLookupManager):
    def __init__(self, **kwargs):
        self.started = threading.Event()
        self.release = threading.Event()
        super().__init__(tier_type="test", **kwargs)

    def batch_lookup(self, keys, req_context):
        self.started.set()
        self.release.wait(5)
        return [True] * len(keys)


class ConcurrentLookupManager(module.AsyncLookupManager):
    def __init__(self, **kwargs):
        self.started_ids: set[str] = set()
        self.started_lock = threading.Lock()
        self.both_started = threading.Event()
        self.release = threading.Event()
        super().__init__(tier_type="test", **kwargs)

    def batch_lookup(self, keys, req_context):
        with self.started_lock:
            self.started_ids.add(req_context.req_id)
            if len(self.started_ids) == 2:
                self.both_started.set()
        self.release.wait(5)
        return [True] * len(keys)


class RecordingLookupManager(module.AsyncLookupManager):
    def __init__(self, **kwargs):
        self.batches: list[tuple[str, list[bytes]]] = []
        self.condition = threading.Condition()
        super().__init__(tier_type="test", **kwargs)

    def batch_lookup(self, keys, req_context):
        with self.condition:
            self.batches.append((req_context.req_id, list(keys)))
            self.condition.notify_all()
        return [True] * len(keys)

    def wait_for_batches(self, count: int, timeout: float = 2) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.batches) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
        return True


class CancelAwareLookupManager(module.AsyncLookupManager):
    def __init__(self, **kwargs):
        self.calls: list[str] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        super().__init__(tier_type="test", **kwargs)

    def batch_lookup(self, keys, req_context):
        self.calls.append(req_context.req_id)
        if req_context.req_id == "active":
            self.first_started.set()
            self.release_first.wait(5)
        return [True] * len(keys)


class AsyncLookupBoundTests(unittest.TestCase):
    def test_different_requests_probe_concurrently(self) -> None:
        manager = ConcurrentLookupManager(
            max_pending_batches=2,
            max_keys_per_step=8,
            num_workers=2,
        )
        try:
            manager.lookup(b"first", ReqContext("request-a"))
            manager.lookup(b"second", ReqContext("request-b"))
            manager.flush()
            self.assertTrue(
                manager.both_started.wait(2),
                "one request-local probe blocked the other",
            )
        finally:
            manager.release.set()
            manager.shutdown()

    def test_per_step_key_limit_fails_open_as_miss(self) -> None:
        manager = ControlledLookupManager(max_pending_batches=1, max_keys_per_step=1)
        context = ReqContext("request")
        try:
            self.assertIsNone(manager.lookup(b"first", context))
            self.assertFalse(manager.lookup(b"overflow", context))
            self.assertEqual(manager.take_dropped_keys(), 1)
        finally:
            manager.release.set()
            manager.shutdown()

    def test_full_batch_queue_drops_work_without_blocking_scheduler(self) -> None:
        manager = ControlledLookupManager(max_pending_batches=1, max_keys_per_step=8)
        try:
            manager.lookup(b"active", ReqContext("active"))
            manager.flush()
            self.assertTrue(manager.started.wait(2))

            manager.lookup(b"queued", ReqContext("queued"))
            manager.flush()
            manager.lookup(b"dropped", ReqContext("dropped"))
            manager.flush()

            self.assertFalse(manager.lookup(b"dropped", ReqContext("dropped")))
            self.assertEqual(manager.take_dropped_keys(), 1)
        finally:
            manager.release.set()
            manager.shutdown()

    def test_failed_load_can_invalidate_cached_hit(self) -> None:
        manager = ControlledLookupManager(max_pending_batches=1, max_keys_per_step=8)
        context = ReqContext("request")
        try:
            manager.lookup(b"key", context)
            manager.flush()
            self.assertTrue(manager.started.wait(2))
            manager.release.set()
            deadline = time.monotonic() + 2
            result = None
            while result is None and time.monotonic() < deadline:
                manager.drain_results()
                result = manager.lookup(b"key", context)
                time.sleep(0.01)
            self.assertTrue(result)
            manager.invalidate([b"key"])
            self.assertFalse(manager.lookup(b"key", context))
        finally:
            manager.release.set()
            manager.shutdown()

    def test_large_request_is_processed_as_independent_small_chunks(self) -> None:
        manager = RecordingLookupManager(
            max_pending_batches=8,
            max_keys_per_step=8,
            max_keys_per_batch=2,
            num_workers=1,
        )
        context = ReqContext("request")
        keys = [f"key-{index}".encode() for index in range(5)]
        try:
            for key in keys:
                self.assertIsNone(manager.lookup(key, context))
            manager.flush()
            self.assertTrue(manager.wait_for_batches(3))
            self.assertEqual([len(keys) for _, keys in manager.batches], [2, 2, 1])

            deadline = time.monotonic() + 2
            results = [None] * len(keys)
            while any(result is None for result in results) and time.monotonic() < deadline:
                manager.drain_results()
                results = [manager.lookup(key, context) for key in keys]
                time.sleep(0.01)
            self.assertEqual(results, [True] * len(keys))
        finally:
            manager.shutdown()

    def test_request_chunks_are_admitted_round_robin(self) -> None:
        manager = RecordingLookupManager(
            max_pending_batches=8,
            max_keys_per_step=8,
            max_keys_per_batch=1,
            num_workers=1,
        )
        try:
            for key in (b"a-1", b"a-2", b"a-3"):
                manager.lookup(key, ReqContext("request-a"))
            manager.lookup(b"b-1", ReqContext("request-b"))
            manager.flush()
            self.assertTrue(manager.wait_for_batches(4))
            self.assertEqual(
                [req_id for req_id, _ in manager.batches],
                ["request-a", "request-b", "request-a", "request-a"],
            )
        finally:
            manager.shutdown()

    def test_cancelled_request_does_not_consume_a_worker(self) -> None:
        manager = CancelAwareLookupManager(
            max_pending_batches=4,
            max_keys_per_step=4,
            max_keys_per_batch=1,
            num_workers=1,
        )
        try:
            manager.lookup(b"active", ReqContext("active"))
            manager.flush()
            self.assertTrue(manager.first_started.wait(2))

            manager.lookup(b"stale", ReqContext("stale"))
            manager.flush()
            manager.cleanup("stale")
            self.assertEqual(manager.take_cancelled_keys(), 1)

            manager.release_first.set()
            deadline = time.monotonic() + 2
            while manager.pending_batches and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.calls, ["active"])
        finally:
            manager.release_first.set()
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
