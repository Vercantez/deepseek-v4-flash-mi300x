# SPDX-License-Identifier: Apache-2.0
"""Bounded, fail-open policy for optional external KV-cache lookups.

The scheduler owns this object and calls it with monotonic timestamps.  It is
kept free of vLLM dependencies so its timeout and circuit-breaker behavior can
be fault-tested without a model or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LookupFailOpenSnapshot:
    deferred_requests: int
    oldest_deferred_seconds: float
    circuit_open: bool
    timeout_total: int
    circuit_bypass_total: int


class LookupFailOpenPolicy:
    """Turn an optional lookup into a miss before it can stall inference."""

    def __init__(
        self,
        timeout_seconds: float = 0.1,
        circuit_breaker_seconds: float = 30.0,
        timeout_threshold: int = 3,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if circuit_breaker_seconds <= 0:
            raise ValueError("circuit_breaker_seconds must be positive")
        if timeout_threshold <= 0:
            raise ValueError("timeout_threshold must be positive")
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker_seconds = circuit_breaker_seconds
        self.timeout_threshold = timeout_threshold
        self._deferred: dict[str, float] = {}
        self._consecutive_timeouts = 0
        self._circuit_open_until = 0.0
        self._timeout_delta = 0
        self._circuit_bypass_delta = 0

    def bypass_reason(self, request_id: str, now: float) -> str | None:
        """Return why this request should bypass external cache, if at all."""
        if now < self._circuit_open_until:
            self._deferred.pop(request_id, None)
            self._circuit_bypass_delta += 1
            return "circuit_open"
        started_at = self._deferred.get(request_id)
        if started_at is None or now < started_at + self.timeout_seconds:
            return None
        self._deferred.pop(request_id, None)
        self._consecutive_timeouts += 1
        self._timeout_delta += 1
        if self._consecutive_timeouts >= self.timeout_threshold:
            self._circuit_open_until = now + self.circuit_breaker_seconds
        return "deadline"

    def defer(self, request_id: str, started_at: float) -> None:
        """Start a request deadline without moving it on later scheduler ticks."""
        self._deferred.setdefault(request_id, started_at)

    def resolve(self, request_id: str) -> None:
        """Record a healthy lookup and close a half-open circuit."""
        was_deferred = self._deferred.pop(request_id, None) is not None
        if was_deferred or self._circuit_open_until:
            self._consecutive_timeouts = 0
            self._circuit_open_until = 0.0

    def finish(self, request_id: str) -> None:
        self._deferred.pop(request_id, None)

    def snapshot(self, now: float, *, reset_counters: bool = False) -> LookupFailOpenSnapshot:
        oldest = max((now - start for start in self._deferred.values()), default=0.0)
        snapshot = LookupFailOpenSnapshot(
            deferred_requests=len(self._deferred),
            oldest_deferred_seconds=max(0.0, oldest),
            circuit_open=now < self._circuit_open_until,
            timeout_total=self._timeout_delta,
            circuit_bypass_total=self._circuit_bypass_delta,
        )
        if reset_counters:
            self._timeout_delta = 0
            self._circuit_bypass_delta = 0
        return snapshot
