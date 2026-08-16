# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state and uses bounded background workers to execute batch_lookup() calls.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned exclusively by the scheduler
  thread.  lookup(), flush(), and cleanup() read and write them directly.

* _lookup_queue is written by the scheduler (flush → put_nowait, one item
  per request) and read by the background workers (get).  queue.Queue is
  thread-safe.

* _pending_results is written by the background thread (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

lookup() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), splitting
request work into small chunks and admitting those chunks round-robin. Multiple
workers can therefore cooperate on a large cache hit, while a miss, timeout, or
cancelled request leaves at most one small in-flight operation per worker.
drain_results() is called before any lookup() calls in the same step, so
lookup() is a pure OrderedDict operation.
"""

import queue
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


def _active_lookup_token() -> threading.Event:
    token = threading.Event()
    token.set()
    return token


@dataclass(slots=True)
class LookupState:
    result: bool | None = None  # True (found), False (not found), None
    request_ids: set[str] = field(default_factory=set)  # requests asking for the lookup
    # Workers retain only this cancellation token, never scheduler-owned maps.
    active: threading.Event = field(default_factory=_active_lookup_token)


class AsyncLookupManager(ABC):
    """
    Per-tier async lookup manager for secondary tier existence checks.

    Each secondary tier that wants non-blocking lookups composes its own
    AsyncLookupManager instance internally. The manager maintains lookup
    state (cache, queue) and uses a background thread to execute the actual
    batch_lookup() calls.

    Subclasses implement only batch_lookup() — all queue management,
    state tracking, and result delivery is provided by this base class.

    The owning tier delegates its lookup(), on_schedule_end(), and
    on_request_finished() to this manager:
      - lookup() → drain_results() + lookup state check
      - on_schedule_end() → flush()
      - on_request_finished() → cleanup()
    """

    def __init__(
        self,
        tier_type: str,
        max_pending_batches: int = 8,
        max_keys_per_step: int = 16_384,
        max_keys_per_batch: int = 256,
        num_workers: int = 4,
    ) -> None:
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        if max_keys_per_step <= 0:
            raise ValueError("max_keys_per_step must be positive")
        if max_keys_per_batch <= 0:
            raise ValueError("max_keys_per_batch must be positive")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self._tier_type = tier_type
        self._max_keys_per_step = max_keys_per_step
        self._max_keys_per_batch = max_keys_per_batch
        self._dropped_keys_delta = 0
        self._cancelled_keys_delta = 0

        # key → LookupState; scheduler-owned, no lock needed.
        self._lookup_state: dict[OffloadKey, LookupState] = {}
        # req_id → keys looked up by that request (reverse index for cleanup).
        self._req_keys: dict[str, set[OffloadKey]] = {}

        # Accumulates (key, req_context) pairs during lookup() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[
            tuple[OffloadKey, ReqContext, threading.Event]
        ] = []

        # Scheduler → workers: one request-local batch per item.
        # None is used as a shutdown sentinel.
        self._lookup_queue: queue.Queue[
            list[tuple[OffloadKey, ReqContext, threading.Event]] | None
        ] = queue.Queue(maxsize=max_pending_batches)

        # Worker → scheduler: completed result batches.
        # Each item is a list of (key, found) pairs.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[list[tuple[OffloadKey, bool]]] = (
            queue.SimpleQueue()
        )
        self._need_to_drain: bool = False
        self._in_flight_batches = 0
        self._in_flight_lock = threading.Lock()

        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"vllm_offloading_lookup_{tier_type}_{worker_idx}",
                daemon=True,
            )
            for worker_idx in range(num_workers)
        ]
        for thread in self._threads:
            thread.start()

    @abstractmethod
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        """
        Check whether a batch of blocks exist in this tier.

        Called from the worker thread — must be synchronous and must not
        touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not.
        """
        ...

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Non-blocking lookup called from the scheduler thread.

        Returns:
            True  — block is present in this tier.
            False — block is not present in this tier.
            None  — result not yet available; retry next step.
        """
        if self._need_to_drain:
            self.drain_results()
            self._need_to_drain = False
        req_id = req_context.req_id
        state = self._lookup_state.get(key)
        if state is None:
            state = LookupState()
            self._lookup_state[key] = state
            if len(self._lookup_batch) < self._max_keys_per_step:
                self._lookup_batch.append((key, req_context, state.active))
            else:
                # The filesystem cache is optional.  A saturated metadata
                # path is a cache miss, never scheduler backpressure.
                state.result = False
                self._dropped_keys_delta += 1
        state.request_ids.add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return state.result

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker thread.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. Request work is split into small chunks and admitted in
        round-robin order. This lets all workers cooperate on a long cached
        prefix without allowing one request to monopolize the metadata queue.
        Safe to call with an empty batch (no-op).
        """
        self._need_to_drain = True
        if self._lookup_batch:
            pending = self._lookup_batch
            self._lookup_batch = []
            batches: dict[
                str, list[tuple[OffloadKey, ReqContext, threading.Event]]
            ] = {}
            for key, req_context, active in pending:
                batches.setdefault(req_context.req_id, []).append(
                    (key, req_context, active)
                )
            request_chunks: deque[
                Iterator[tuple[OffloadKey, ReqContext, threading.Event]]
            ] = deque(iter(batch) for batch in batches.values())
            while request_chunks:
                batch_iter = request_chunks.popleft()
                batch = list(islice(batch_iter, self._max_keys_per_batch))
                if not batch:
                    continue
                try:
                    self._lookup_queue.put_nowait(batch)
                except queue.Full:
                    for key, _, active in batch:
                        state = self._lookup_state.get(key)
                        if state is not None and active.is_set():
                            state.result = False
                    self._dropped_keys_delta += len(batch)
                request_chunks.append(batch_iter)

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup() before checking state.
        """
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            for key, result in batch:
                state = self._lookup_state.get(key)
                if state is not None:
                    state.result = result

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state.get(key)
            if state is None:
                continue
            state.request_ids.discard(req_id)
            if not state.request_ids:
                if state.result is None:
                    self._cancelled_keys_delta += 1
                state.active.clear()
                del self._lookup_state[key]

    def invalidate(self, keys: Iterable[OffloadKey]) -> None:
        """Turn stale positive results into misses after a failed load."""
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None:
                state.result = False

    def take_dropped_keys(self) -> int:
        dropped = self._dropped_keys_delta
        self._dropped_keys_delta = 0
        return dropped

    def take_cancelled_keys(self) -> int:
        cancelled = self._cancelled_keys_delta
        self._cancelled_keys_delta = 0
        return cancelled

    @property
    def pending_batches(self) -> int:
        with self._in_flight_lock:
            return self._lookup_queue.qsize() + self._in_flight_batches

    @property
    def pending_keys(self) -> int:
        return len(self._lookup_batch)

    def shutdown(self) -> None:
        """Stop the lookup workers."""
        # Shutdown is allowed to wait for bounded queued metadata work. Runtime
        # scheduler calls never block on this queue.
        for _ in self._threads:
            self._lookup_queue.put(None)
        for thread in self._threads:
            thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            pending = self._lookup_queue.get()
            if pending is None:
                break

            if not pending:
                continue
            with self._in_flight_lock:
                self._in_flight_batches += 1
            try:
                active_pending = [item for item in pending if item[2].is_set()]
                if not active_pending:
                    continue
                req_context = active_pending[0][1]
                keys = [key for key, _, _ in active_pending]
                results: list[tuple[OffloadKey, bool]] = []
                try:
                    hits = self.batch_lookup(keys, req_context)
                except Exception as exc:
                    logger.warning(
                        "batch_lookup failed on tier %s for %d keys: %s",
                        self._tier_type,
                        len(keys),
                        exc,
                    )
                    hits = (False for _ in keys)

                for (key, _, active), hit in zip(active_pending, hits):
                    if active.is_set():
                        results.append((key, hit))

                # Publish each small chunk independently. A large request no
                # longer withholds early hits until every path has completed.
                if results:
                    self._pending_results.put(results)
            finally:
                with self._in_flight_lock:
                    self._in_flight_batches -= 1
