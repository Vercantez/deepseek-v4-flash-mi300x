import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "patches" / "tiering-fs-bounded-lru.py"
)
SPEC = importlib.util.spec_from_file_location("bounded_lru", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bounded_lru = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounded_lru)


class ShardLeaseIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_block(self, shard: str, name: str = "block.bin") -> str:
        directory = self.root / shard / "00_g0"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_bytes(b"kv")
        return str(path)

    def shard_path(self, shard: str) -> str:
        return os.path.realpath(self.root / shard)

    def test_positive_lookup_is_pinned_until_request_finishes(self) -> None:
        old = self.make_block("000")
        newer = self.make_block("001")
        os.utime(self.root / "000", (1, 1))
        os.utime(self.root / "001", (2, 2))
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")

        hits = index.lookup_and_pin(
            "req", [old], lambda paths: [os.path.exists(path) for path in paths]
        )
        self.assertEqual(hits, [True])
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("001"))

        index.finish_eviction(self.shard_path("001"), True)
        index.release_request("req")
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("000"))
        self.assertTrue(os.path.exists(newer))

    def test_lookup_and_eviction_decision_are_atomic(self) -> None:
        path_a = self.make_block("000")
        self.make_block("001")
        os.utime(self.root / "000", (1, 1))
        os.utime(self.root / "001", (2, 2))
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        lookup_result: list[bool] = []
        eviction_result: list[str | None] = []

        def slow_exists(paths: list[str]) -> list[bool]:
            lookup_started.set()
            self.assertTrue(release_lookup.wait(5))
            return [True for _ in paths]

        lookup_thread = threading.Thread(
            target=lambda: lookup_result.extend(
                index.lookup_and_pin("req", [path_a], slow_exists)
            )
        )
        lookup_thread.start()
        self.assertTrue(lookup_started.wait(5))
        eviction_thread = threading.Thread(
            target=lambda: eviction_result.append(index.begin_oldest_eviction())
        )
        eviction_thread.start()
        eviction_thread.join(5)
        self.assertEqual(eviction_result, [self.shard_path("001")])
        release_lookup.set()
        lookup_thread.join(5)

        self.assertEqual(lookup_result, [True])

    def test_jobs_and_cancelled_requests_fence_eviction(self) -> None:
        path = self.make_block("abc")
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        self.assertTrue(index.pin_job(7, [path]))
        self.assertIsNone(index.begin_oldest_eviction())
        index.release_job(7)
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("abc"))
        index.finish_eviction(self.shard_path("abc"), False)

        index.open_request("cancelled")
        index.release_request("cancelled")
        self.assertEqual(
            index.lookup_and_pin("cancelled", [path], lambda paths: [True]),
            [False],
        )

    def test_paths_outside_cache_root_are_rejected_without_resolution(self) -> None:
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        outside = str(self.root.parent / "000" / "00_g0" / "block.bin")
        with self.assertRaises(ValueError):
            index.shard_for_path(outside)

    def test_queued_lookup_is_protected_until_miss_resolves(self) -> None:
        path_a = self.make_block("aaa")
        self.make_block("bbb")
        os.utime(self.root / "aaa", (1, 1))
        os.utime(self.root / "bbb", (2, 2))
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")
        index.begin_lookup("req", path_a)
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("bbb"))
        index.finish_eviction(self.shard_path("bbb"), True)
        self.assertFalse(index.resolve_cached_lookup("req", path_a, False))
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("aaa"))

    def test_atomic_eviction_removes_only_selected_shard(self) -> None:
        selected = self.make_block("123")
        retained = self.make_block("456")
        bounded_lru.evict_shard_atomically(str(self.root / "123"))
        self.assertFalse(os.path.exists(selected))
        self.assertTrue(os.path.exists(retained))
        self.assertFalse(
            any(path.name.startswith(".evicting-") for path in self.root.iterdir())
        )

    def test_worker_reports_success_and_cleans_orphan_tombstone(self) -> None:
        self.make_block("def")
        worker = bounded_lru.ShardEvictionWorker()
        worker.submit(str(self.root / "def"))
        deadline = time.monotonic() + 5
        results = []
        while not results and time.monotonic() < deadline:
            results = worker.get_finished()
            time.sleep(0.01)
        worker.shutdown()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1], results)
        self.assertEqual(worker.outstanding, 0)


if __name__ == "__main__":
    unittest.main()
