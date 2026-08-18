import importlib.util
import os
import subprocess
import sys
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

    def test_cancelled_lookup_cannot_repin_after_disk_io_finishes(self) -> None:
        path = self.make_block("abc")
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        lookup_result: list[bool] = []

        def slow_exists(paths: list[str]) -> list[bool]:
            lookup_started.set()
            self.assertTrue(release_lookup.wait(5))
            return [True for _ in paths]

        lookup_thread = threading.Thread(
            target=lambda: lookup_result.extend(
                index.lookup_and_pin("req", [path], slow_exists)
            )
        )
        lookup_thread.start()
        self.assertTrue(lookup_started.wait(5))
        index.cancel_lookup("req")
        release_lookup.set()
        lookup_thread.join(5)

        self.assertEqual(lookup_result, [False])
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("abc"))

    def test_pending_lookup_metadata_is_compact_per_shard(self) -> None:
        paths = [self.make_block("abc", f"block-{index}.bin") for index in range(50)]
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")
        lookup_started = threading.Event()
        release_lookup = threading.Event()

        def slow_exists(candidate_paths: list[str]) -> list[bool]:
            lookup_started.set()
            self.assertTrue(release_lookup.wait(5))
            return [False for _ in candidate_paths]

        lookup_thread = threading.Thread(
            target=lambda: index.lookup_and_pin("req", paths, slow_exists)
        )
        lookup_thread.start()
        self.assertTrue(lookup_started.wait(5))
        pending = index._pending_lookup_shards["req"]
        self.assertEqual(pending, {self.shard_path("abc"): 50})
        release_lookup.set()
        lookup_thread.join(5)
        self.assertNotIn("req", index._pending_lookup_shards)

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

    def test_sibling_process_index_cannot_evict_a_leased_shard(self) -> None:
        path = self.make_block("abc")
        reader = bounded_lru.ShardLeaseIndex(str(self.root))
        evictor = bounded_lru.ShardLeaseIndex(str(self.root))
        reader.open_request("req")

        self.assertEqual(
            reader.lookup_and_pin(
                "req", [path], lambda paths: [os.path.exists(p) for p in paths]
            ),
            [True],
        )
        self.assertIsNone(evictor.begin_oldest_eviction())

        reader.release_request("req")
        self.assertEqual(evictor.begin_oldest_eviction(), self.shard_path("abc"))
        evictor.finish_eviction(self.shard_path("abc"), False)

    def test_sibling_process_index_cannot_evict_during_store_job(self) -> None:
        path = self.make_block("abc")
        writer = bounded_lru.ShardLeaseIndex(str(self.root))
        evictor = bounded_lru.ShardLeaseIndex(str(self.root))

        self.assertTrue(writer.pin_job(7, [path]))
        self.assertIsNone(evictor.begin_oldest_eviction())
        writer.release_job(7)
        self.assertEqual(evictor.begin_oldest_eviction(), self.shard_path("abc"))
        evictor.finish_eviction(self.shard_path("abc"), False)

    def test_promotion_job_replaces_request_lookup_lease(self) -> None:
        path = self.make_block("abc")
        index = bounded_lru.ShardLeaseIndex(str(self.root))
        index.open_request("req")
        self.assertEqual(
            index.lookup_and_pin("req", [path], lambda paths: [True]),
            [True],
        )
        self.assertTrue(index.pin_job(7, [path]))
        index.release_request_pins("req")
        self.assertIsNone(index.begin_oldest_eviction())

        index.release_job(7)
        self.assertEqual(index.begin_oldest_eviction(), self.shard_path("abc"))
        index.finish_eviction(self.shard_path("abc"), False)
        index.release_request("req")

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


class SharedEvictionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_sibling_owner_blocks_stores_until_eviction_finishes(self) -> None:
        owner = bounded_lru.SharedEvictionGate(str(self.root))
        sibling = bounded_lru.SharedEvictionGate(str(self.root))
        self.assertTrue(owner.try_begin_eviction())
        self.assertTrue(owner.try_fence_writes())
        self.assertTrue(sibling.eviction_active)
        self.assertFalse(sibling.try_pin_store(1))

        owner.finish_eviction()
        self.assertFalse(sibling.eviction_active)
        self.assertTrue(sibling.try_pin_store(1))
        sibling.release_store(1)
        owner.close()
        sibling.close()

    def test_eviction_waits_for_sibling_store_then_fences_new_stores(self) -> None:
        writer = bounded_lru.SharedEvictionGate(str(self.root))
        owner = bounded_lru.SharedEvictionGate(str(self.root))
        sibling = bounded_lru.SharedEvictionGate(str(self.root))
        self.assertTrue(writer.try_pin_store(7))
        self.assertTrue(owner.try_begin_eviction())
        self.assertFalse(owner.try_fence_writes())
        self.assertFalse(sibling.try_pin_store(8))

        writer.release_store(7)
        self.assertTrue(owner.try_fence_writes())
        self.assertFalse(sibling.try_pin_store(9))
        owner.finish_eviction()
        self.assertTrue(sibling.try_pin_store(10))
        sibling.release_store(10)
        writer.close()
        owner.close()
        sibling.close()

    def test_only_one_process_can_own_eviction(self) -> None:
        first = bounded_lru.SharedEvictionGate(str(self.root))
        second = bounded_lru.SharedEvictionGate(str(self.root))
        self.assertTrue(first.try_begin_eviction())
        self.assertFalse(second.try_begin_eviction())
        first.finish_eviction()
        self.assertTrue(second.try_begin_eviction())
        first.close()
        second.close()

    def test_crashed_owner_releases_shared_gate(self) -> None:
        script = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("bounded_lru_child", {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gate = module.SharedEvictionGate(sys.argv[1])
assert gate.try_begin_eviction()
assert gate.try_fence_writes()
print("ready", flush=True)
sys.stdin.buffer.read(1)
"""
        with subprocess.Popen(
            [sys.executable, "-c", script, str(self.root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as child:
            assert child.stdout is not None
            self.assertEqual(child.stdout.readline().strip(), "ready")
            sibling = bounded_lru.SharedEvictionGate(str(self.root))
            self.assertTrue(sibling.eviction_active)
            self.assertFalse(sibling.try_pin_store(1))

            child.kill()
            child.wait(timeout=5)
            self.assertFalse(sibling.eviction_active)
            self.assertTrue(sibling.try_begin_eviction())
            sibling.close()


if __name__ == "__main__":
    unittest.main()
