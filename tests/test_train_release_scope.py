from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from prismatic.conf import ModelConfig


def _load_train_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("vlm_backbones_train_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainReleaseScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_module = _load_train_module()

    def test_default_model_id_matches_supported_release(self) -> None:
        self.assertEqual(self.train_module.DEFAULT_MODEL_ID, "in1k-224px-maxvit-t-letterbox-s3+7b-vicuna")

    def test_supported_backbone_passes_scope_validation(self) -> None:
        model_cfg = ModelConfig.get_choice_class("in1k-224px-maxvit-t-letterbox-s3+7b-vicuna")()
        self.train_module._validate_supported_release_scope(model_cfg)

    def test_unsupported_backbone_fails_scope_validation(self) -> None:
        model_cfg = ModelConfig.get_choice_class("prism-clip+7b")()
        with self.assertRaises(ValueError):
            self.train_module._validate_supported_release_scope(model_cfg)


if __name__ == "__main__":
    unittest.main()

