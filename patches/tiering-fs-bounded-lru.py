# SPDX-License-Identifier: Apache-2.0
"""Concurrency-safe shard leases for the filesystem KV-cache tier.

The filesystem layout places blocks below 4,096 three-hex-digit shard
directories.  Tracking shards instead of millions of block files keeps startup
bounded while still allowing the tier to own eviction safely.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable

_SHARD_NAME = re.compile(r"^[0-9a-f]{3}$")
_TOMBSTONE_PREFIX = ".evicting-"


class ShardLeaseIndex:
    """LRU metadata and leases shared by lookup, transfer, and eviction paths."""

    def __init__(self, run_root: str) -> None:
        self._input_run_root = os.path.abspath(run_root)
        self.run_root = os.path.realpath(run_root)
        self._input_root_prefix = self._input_run_root.rstrip(os.sep) + os.sep
        self._root_prefix = self.run_root.rstrip(os.sep) + os.sep
        self._lock = threading.RLock()
        self._last_used: dict[str, float] = {}
        self._request_shards: dict[str, set[str]] = {}
        # Eviction operates on 4,096 hash shards, so tracking millions of
        # individual candidate paths only wastes memory and lock time. Counts
        # preserve correctness when concurrent batches touch the same shard.
        self._pending_lookup_shards: dict[str, dict[str, int]] = {}
        self._job_shards: dict[int, set[str]] = {}
        self._evicting: set[str] = set()
        self._dirty_shards: set[str] = set()
        self._last_persisted: dict[str, float] = {}
        # A bounded cancellation fence prevents a late async lookup result
        # from re-leasing a request after cleanup.
        self._closed_requests: OrderedDict[str, None] = OrderedDict()
        self._lookup_cancelled_requests: set[str] = set()
        self._scan_shards()

    def _scan_shards(self) -> None:
        try:
            entries = os.scandir(self.run_root)
        except FileNotFoundError:
            return
        with entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not _SHARD_NAME.fullmatch(entry.name):
                    continue
                try:
                    self._last_used[entry.path] = entry.stat().st_mtime
                except FileNotFoundError:
                    pass

    def _normalize_path(self, path: str) -> str:
        # Paths come exclusively from FileMapper's content hashes. Resolving
        # every block with realpath() performs filesystem I/O twice per key and
        # made a 16K-key metadata batch take seconds on the production cache.
        # Lexical normalization retains the containment check without touching
        # the filesystem for every lookup.
        if path.startswith(self._root_prefix):
            return path
        if path.startswith(self._input_root_prefix):
            return self._root_prefix + path[len(self._input_root_prefix) :]
        normalized = os.path.abspath(path)
        if normalized.startswith(self._root_prefix):
            return normalized
        if normalized.startswith(self._input_root_prefix):
            return self._root_prefix + normalized[len(self._input_root_prefix) :]
        else:
            raise ValueError(f"KV path is outside the cache root: {path!r}")

    def _shard_for_normalized_path(self, path: str) -> str:
        if not path.startswith(self._root_prefix):
            raise ValueError(f"KV path is outside the cache root: {path!r}")
        relative = path[len(self._root_prefix) :]
        shard_name = relative.split(os.sep, 1)[0]
        if not _SHARD_NAME.fullmatch(shard_name):
            raise ValueError(f"KV path is outside a hash shard: {path!r}")
        shard = os.path.join(self.run_root, shard_name)
        return shard

    def shard_for_path(self, path: str) -> str:
        return self._shard_for_normalized_path(self._normalize_path(path))

    def lookup_and_pin(
        self,
        req_id: str,
        paths: list[str],
        exists: Callable[[list[str]], Iterable[bool]],
    ) -> list[bool]:
        """Check paths and lease positive results before eviction can begin."""
        with self._lock:
            if (
                req_id in self._closed_requests
                or req_id in self._lookup_cancelled_requests
            ):
                return [False] * len(paths)
            normalized_paths = [self._normalize_path(path) for path in paths]
            shards = [
                self._shard_for_normalized_path(path) for path in normalized_paths
            ]
            eligible = [shard not in self._evicting for shard in shards]
            pending = self._pending_lookup_shards.setdefault(req_id, {})
            for shard, ok in zip(shards, eligible):
                if ok:
                    pending[shard] = pending.get(shard, 0) + 1
        candidate_paths = [path for path, ok in zip(paths, eligible) if ok]
        candidate_shards = [shard for shard, ok in zip(shards, eligible) if ok]
        try:
            candidate_hits = iter(exists(candidate_paths))
            hits = [next(candidate_hits) if ok else False for ok in eligible]
        except Exception:
            with self._lock:
                self._release_pending_shards_locked(req_id, candidate_shards)
            raise
        with self._lock:
            self._release_pending_shards_locked(req_id, candidate_shards)
            if (
                req_id in self._closed_requests
                or req_id in self._lookup_cancelled_requests
            ):
                return [False] * len(paths)
            now = time.time()
            pinned = self._request_shards.setdefault(req_id, set())
            for shard, hit in zip(shards, hits):
                if not hit:
                    continue
                pinned.add(shard)
                self._last_used[shard] = now
                self._dirty_shards.add(shard)
            return hits

    def _release_pending_shards_locked(
        self, req_id: str, shards: Iterable[str]
    ) -> None:
        pending = self._pending_lookup_shards.get(req_id)
        if pending is None:
            return
        for shard in shards:
            remaining = pending.get(shard, 0) - 1
            if remaining > 0:
                pending[shard] = remaining
            else:
                pending.pop(shard, None)
        if not pending:
            self._pending_lookup_shards.pop(req_id, None)

    def begin_lookup(self, req_id: str, path: str) -> None:
        """Protect a path's shard while an existence check is queued."""
        with self._lock:
            if (
                req_id in self._closed_requests
                or req_id in self._lookup_cancelled_requests
            ):
                return
            shard = self.shard_for_path(path)
            pending = self._pending_lookup_shards.setdefault(req_id, {})
            pending[shard] = pending.get(shard, 0) + 1

    def resolve_cached_lookup(self, req_id: str, path: str, hit: bool) -> bool:
        """Resolve and optionally lease a result cached for another request."""
        shard = self.shard_for_path(path)
        with self._lock:
            self._release_pending_shards_locked(req_id, (shard,))
            if not hit:
                return False
            if (
                req_id in self._closed_requests
                or req_id in self._lookup_cancelled_requests
                or shard in self._evicting
            ):
                return False
            self._request_shards.setdefault(req_id, set()).add(shard)
            self._last_used[shard] = time.time()
            self._dirty_shards.add(shard)
            return True

    def touch(self, req_id: str, paths: Iterable[str]) -> None:
        now = time.time()
        with self._lock:
            if req_id in self._closed_requests:
                return
            # The scheduler broadcasts touch() to all tiers, including for
            # misses and primary-tier hits. Only shards positively found by
            # this filesystem tier are legitimate filesystem touches.
            pinned = self._request_shards.get(req_id)
            if not pinned:
                return
            for path in paths:
                shard = self.shard_for_path(path)
                if shard not in pinned or shard in self._evicting:
                    continue
                self._last_used[shard] = now
                self._dirty_shards.add(shard)

    def open_request(self, req_id: str) -> None:
        with self._lock:
            self._closed_requests.pop(req_id, None)
            self._lookup_cancelled_requests.discard(req_id)
            self._request_shards.setdefault(req_id, set())
            self._pending_lookup_shards.pop(req_id, None)

    def cancel_lookup(self, req_id: str, release_pins: bool = True) -> None:
        """Fence late metadata results without closing the request for stores."""
        with self._lock:
            self._lookup_cancelled_requests.add(req_id)
            self._pending_lookup_shards.pop(req_id, None)
            if release_pins:
                self._request_shards.pop(req_id, None)

    def release_request(self, req_id: str) -> None:
        with self._lock:
            self._request_shards.pop(req_id, None)
            self._pending_lookup_shards.pop(req_id, None)
            self._lookup_cancelled_requests.discard(req_id)
            self._closed_requests[req_id] = None
            self._closed_requests.move_to_end(req_id)
            if len(self._closed_requests) > 100_000:
                self._closed_requests.popitem(last=False)

    def pin_job(self, job_id: int, paths: Iterable[str]) -> bool:
        shards = {self.shard_for_path(path) for path in paths}
        with self._lock:
            if shards & self._evicting:
                return False
            self._job_shards[job_id] = shards
            return True

    def release_job(self, job_id: int) -> None:
        with self._lock:
            self._job_shards.pop(job_id, None)

    def record_store(self, paths: Iterable[str]) -> None:
        now = time.time()
        with self._lock:
            for path in paths:
                shard = self.shard_for_path(path)
                self._last_used[shard] = now
                self._dirty_shards.add(shard)

    def record_access(self, paths: Iterable[str]) -> None:
        now = time.time()
        with self._lock:
            for path in paths:
                shard = self.shard_for_path(path)
                if shard not in self._last_used or shard in self._evicting:
                    continue
                self._last_used[shard] = now
                self._dirty_shards.add(shard)

    def begin_oldest_eviction(self) -> str | None:
        """Lease the oldest idle shard for eviction, or return None."""
        with self._lock:
            pinned: set[str] = set()
            for shards in self._request_shards.values():
                pinned.update(shards)
            for shards in self._job_shards.values():
                pinned.update(shards)
            for shards in self._pending_lookup_shards.values():
                pinned.update(shards)
            candidates = (
                (last_used, shard)
                for shard, last_used in self._last_used.items()
                if shard not in pinned and shard not in self._evicting
            )
            try:
                _, shard = min(candidates)
            except ValueError:
                return None
            self._evicting.add(shard)
            return shard

    def finish_eviction(self, shard: str, success: bool) -> None:
        with self._lock:
            self._evicting.discard(shard)
            if success:
                self._last_used.pop(shard, None)
                self._dirty_shards.discard(shard)
                self._last_persisted.pop(shard, None)
            elif os.path.isdir(shard):
                try:
                    self._last_used[shard] = os.stat(shard).st_mtime
                except FileNotFoundError:
                    self._last_used.pop(shard, None)

    def flush_touches(
        self, minimum_age_seconds: float = 30, max_updates: int = 64
    ) -> None:
        """Persist a bounded batch of shard use without per-block atime I/O."""
        now = time.monotonic()
        with self._lock:
            ready = [
                shard
                for shard in self._dirty_shards
                if now - self._last_persisted.get(shard, 0)
                >= minimum_age_seconds
                and shard not in self._evicting
            ][:max_updates]
            self._dirty_shards.difference_update(ready)
            for shard in ready:
                self._last_persisted[shard] = now
        for shard in ready:
            try:
                os.utime(shard, None)
            except OSError:
                pass

    @property
    def evictions_in_flight(self) -> int:
        with self._lock:
            return len(self._evicting)


def evict_shard_atomically(shard: str) -> None:
    """Rename a shard out of the lookup namespace, then recursively delete it."""
    parent = os.path.dirname(shard)
    name = os.path.basename(shard)
    if not _SHARD_NAME.fullmatch(name):
        raise ValueError(f"Refusing to evict unexpected shard path: {shard!r}")
    tombstone = os.path.join(parent, f"{_TOMBSTONE_PREFIX}{name}-{uuid.uuid4().hex}")
    try:
        os.replace(shard, tombstone)
    except FileNotFoundError:
        return
    shutil.rmtree(tombstone)


class ShardEvictionWorker:
    """One background worker for potentially slow recursive shard deletion."""

    def __init__(self) -> None:
        self._pending: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._finished: queue.SimpleQueue[tuple[str, bool, str | None]] = (
            queue.SimpleQueue()
        )
        self._state_lock = threading.Lock()
        self._outstanding = 0
        self._thread = threading.Thread(
            target=self._run, name="vllm_kv_fs_evict", daemon=True
        )
        self._thread.start()

    def submit(self, shard: str) -> None:
        with self._state_lock:
            self._outstanding += 1
        self._pending.put(shard)

    def get_finished(self) -> list[tuple[str, bool, str | None]]:
        results: list[tuple[str, bool, str | None]] = []
        while True:
            try:
                results.append(self._finished.get_nowait())
            except queue.Empty:
                if results:
                    with self._state_lock:
                        self._outstanding -= len(results)
                return results

    @property
    def outstanding(self) -> int:
        with self._state_lock:
            return self._outstanding

    def shutdown(self) -> None:
        self._pending.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            shard = self._pending.get()
            if shard is None:
                return
            try:
                if os.path.basename(shard).startswith(_TOMBSTONE_PREFIX):
                    try:
                        shutil.rmtree(shard)
                    except FileNotFoundError:
                        pass
                else:
                    evict_shard_atomically(shard)
            except Exception as exc:  # surfaced to the scheduler logger
                self._finished.put((shard, False, repr(exc)))
            else:
                self._finished.put((shard, True, None))
