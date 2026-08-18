import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ComposeSharedMemoryIsolationTests(unittest.TestCase):
    def test_each_replica_uses_a_large_private_shm(self) -> None:
        base = (ROOT / "compose.yaml").read_text()
        replicas = (ROOT / "compose.tp1x2.yaml").read_text()

        self.assertNotRegex(base, r"(?m)^\s*ipc:\s*host\s*$")
        match = re.search(r'(?m)^\s*shm_size:\s*["\']?(\d+)gb["\']?\s*$', base)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 110)
        self.assertNotIn("VLLM_CLEAN_STALE_CPU_KV", replicas)

    def test_private_shm_is_cleaned_before_vllm_starts(self) -> None:
        entrypoint = (ROOT / "vllm-entrypoint.sh").read_text()

        cleanup = entrypoint.index("find /dev/shm")
        server_start = entrypoint.index('exec vllm serve "$@"')
        self.assertLess(cleanup, server_start)


if __name__ == "__main__":
    unittest.main()
