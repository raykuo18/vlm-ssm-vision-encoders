from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from vlm_backbones.models.backbones.vision import vit_adapter


class ViTAdapterCheckpointResolutionTest(unittest.TestCase):
    def test_skip_pretrain_allows_missing_checkpoint(self) -> None:
        variant = vit_adapter.ViTAdapterVariant(
            config_relpath="configs/dummy.py",
            checkpoints=("missing_checkpoint.pth",),
            preprocess={},
        )

        with TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"VIT_ADAPTER_SKIP_PRETRAIN": "1"}, clear=False):
                with mock.patch.object(vit_adapter, "VIT_ADAPTER_CKPT_ROOT", Path(tmpdir)):
                    ckpt_path = vit_adapter._resolve_checkpoint(variant)

        self.assertEqual(ckpt_path, Path(tmpdir) / "missing_checkpoint.pth")

    def test_pretrained_still_requires_checkpoint(self) -> None:
        variant = vit_adapter.ViTAdapterVariant(
            config_relpath="configs/dummy.py",
            checkpoints=("missing_checkpoint.pth",),
            preprocess={},
        )

        with TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"VIT_ADAPTER_SKIP_PRETRAIN": "0"}, clear=False):
                with mock.patch.object(vit_adapter, "VIT_ADAPTER_CKPT_ROOT", Path(tmpdir)):
                    with self.assertRaises(FileNotFoundError):
                        vit_adapter._resolve_checkpoint(variant)


if __name__ == "__main__":
    unittest.main()
