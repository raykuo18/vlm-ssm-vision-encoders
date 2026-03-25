from __future__ import annotations

import unittest

import torch

from prismatic.util.data_utils import PaddedCollatorForLanguageModeling


class PaddedCollatorTest(unittest.TestCase):
    def test_mixed_multimodal_and_unimodal_batch(self) -> None:
        collator = PaddedCollatorForLanguageModeling(
            model_max_length=8,
            pad_token_id=0,
            default_image_resolution=(3, 2, 2),
        )
        batch = collator(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "labels": torch.tensor([1, 2, 3], dtype=torch.long),
                    "pixel_values": torch.ones(3, 2, 2),
                },
                {
                    "input_ids": torch.tensor([4, 5], dtype=torch.long),
                    "labels": torch.tensor([4, 5], dtype=torch.long),
                    "pixel_values": None,
                },
            ]
        )

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["pixel_values"].shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(batch["multimodal_indices"], torch.tensor([0], dtype=torch.long)))
        self.assertTrue(torch.equal(batch["pixel_values"][1], torch.zeros(3, 2, 2)))


if __name__ == "__main__":
    unittest.main()

