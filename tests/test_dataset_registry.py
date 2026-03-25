from __future__ import annotations

import unittest
from pathlib import Path

from prismatic.conf import DatasetConfig, DatasetRegistry


class DatasetRegistryTest(unittest.TestCase):
    def test_debug_dataset_is_registered(self) -> None:
        dataset_cfg = DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15_DEBUG_TINY.dataset_id)()
        self.assertEqual(dataset_cfg.dataset_id, "llava-v15-debug-320")
        self.assertEqual(
            dataset_cfg.finetune_stage_components[0],
            Path("download/llava-v1.5-instruct/llava_v1_5_mix665k_debug320.json"),
        )


if __name__ == "__main__":
    unittest.main()

