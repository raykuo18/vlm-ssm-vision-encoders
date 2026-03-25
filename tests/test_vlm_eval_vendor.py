from __future__ import annotations

import sys
import unittest
from pathlib import Path


class VendoredVlmEvalImportTest(unittest.TestCase):
    def test_prismatic_loader_is_available(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        vendor_root = repo_root / "third_party" / "vlm-evaluation"
        sys.path.insert(0, str(vendor_root))
        try:
            from vlm_eval.models import _get_initializer

            initializer = _get_initializer("prismatic")
        finally:
            sys.path.pop(0)

        self.assertEqual(initializer.__name__, "PrismaticVLM")


if __name__ == "__main__":
    unittest.main()

