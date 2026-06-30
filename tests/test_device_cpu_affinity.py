import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COSMOS_ROOT = ROOT / "external" / "cosmos" / "packages" / "cosmos3"
if str(COSMOS_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS_ROOT))

from cosmos_framework.utils.device import resolve_cpu_affinity


class ResolveCpuAffinityTest(unittest.TestCase):
    def test_keeps_requested_affinity_when_allowed_contains_it(self) -> None:
        self.assertEqual(resolve_cpu_affinity([8, 9, 10], {0, 8, 9, 10, 11}), [8, 9, 10])

    def test_intersects_requested_affinity_with_allowed_cpuset(self) -> None:
        self.assertEqual(resolve_cpu_affinity([24, 25, 26, 27], {25, 27, 40}), [25, 27])

    def test_falls_back_to_allowed_cpuset_when_requested_is_disjoint(self) -> None:
        self.assertEqual(resolve_cpu_affinity([48, 49], {0, 1, 2, 3}), [0, 1, 2, 3])

    def test_returns_requested_affinity_when_allowed_cpuset_is_unknown(self) -> None:
        self.assertEqual(resolve_cpu_affinity([4, 5], None), [4, 5])


if __name__ == "__main__":
    unittest.main()
