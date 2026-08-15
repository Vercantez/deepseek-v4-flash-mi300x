import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "patches" / "kv_lookup_fail_open.py"
SPEC = importlib.util.spec_from_file_location("kv_lookup_fail_open", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class LookupFailOpenPolicyTests(unittest.TestCase):
    def test_hung_lookup_fails_open_at_deadline(self) -> None:
        policy = module.LookupFailOpenPolicy(timeout_seconds=0.1)
        policy.defer("request", 10.0)
        self.assertIsNone(policy.bypass_reason("request", 10.099))
        self.assertEqual(policy.bypass_reason("request", 10.1), "deadline")
        snapshot = policy.snapshot(10.1)
        self.assertEqual(snapshot.timeout_total, 1)
        self.assertEqual(snapshot.deferred_requests, 0)

    def test_three_timeouts_open_and_healthy_probe_closes_circuit(self) -> None:
        policy = module.LookupFailOpenPolicy(
            timeout_seconds=0.1,
            circuit_breaker_seconds=30,
            timeout_threshold=3,
        )
        for index in range(3):
            request_id = f"timeout-{index}"
            policy.defer(request_id, float(index))
            self.assertEqual(
                policy.bypass_reason(request_id, float(index) + 0.1), "deadline"
            )
        self.assertEqual(policy.bypass_reason("bypassed", 3.0), "circuit_open")
        self.assertTrue(policy.snapshot(3.0).circuit_open)

        self.assertIsNone(policy.bypass_reason("probe", 32.1))
        policy.defer("probe", 32.1)
        policy.resolve("probe")
        self.assertFalse(policy.snapshot(32.1).circuit_open)

    def test_finish_cancels_deadline_and_counters_are_delta_based(self) -> None:
        policy = module.LookupFailOpenPolicy(timeout_seconds=0.1)
        policy.defer("finished", 1.0)
        policy.finish("finished")
        self.assertIsNone(policy.bypass_reason("finished", 2.0))

        policy.defer("timeout", 2.0)
        self.assertEqual(policy.bypass_reason("timeout", 2.1), "deadline")
        self.assertEqual(policy.snapshot(2.1, reset_counters=True).timeout_total, 1)
        self.assertEqual(policy.snapshot(2.1).timeout_total, 0)


if __name__ == "__main__":
    unittest.main()
