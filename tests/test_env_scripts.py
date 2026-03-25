from __future__ import annotations

import unittest
from pathlib import Path


class EnvScriptsTest(unittest.TestCase):
    def test_train_and_eval_scripts_have_expected_defaults(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_expectations = {
            "scripts/env/build_vit_train.sh": ('parse_common_args "vlm-backbones-train-vit"', "install_base_train_stack"),
            "scripts/env/build_vit_eval.sh": ('parse_common_args "vlm-backbones-eval-vit"', "install_base_eval_stack"),
            "scripts/env/build_maxvit_train.sh": ('parse_common_args "vlm-backbones-train-maxvit"', "install_base_train_stack"),
            "scripts/env/build_maxvit_eval.sh": ('parse_common_args "vlm-backbones-eval-maxvit"', "install_base_eval_stack"),
            "scripts/env/build_vmamba_train.sh": ('parse_common_args "vlm-backbones-train-vmamba"', "install_base_train_stack"),
            "scripts/env/build_vmamba_eval.sh": ('parse_common_args "vlm-backbones-eval-vmamba"', "install_base_eval_stack"),
            "scripts/env/build_mambavision_train.sh": ('parse_common_args "vlm-backbones-train-mambavision"', "install_base_train_stack"),
            "scripts/env/build_mambavision_eval.sh": ('parse_common_args "vlm-backbones-eval-mambavision"', "install_base_eval_stack"),
            "scripts/env/build_vitdet_train.sh": ('parse_common_args "vlm-backbones-train-vitdet"', "install_base_train_stack"),
            "scripts/env/build_vitdet_eval.sh": ('parse_common_args "vlm-backbones-eval-vitdet"', "install_base_eval_stack"),
            "scripts/env/build_vit_adapter_train.sh": ('parse_common_args "vlm-backbones-train-vit-adapter"', "install_base_train_stack"),
            "scripts/env/build_vit_adapter_eval.sh": ('parse_common_args "vlm-backbones-eval-vit-adapter"', "install_base_eval_stack"),
        }

        for relpath, required_strings in script_expectations.items():
            content = (repo_root / relpath).read_text(encoding="utf-8")
            for required in required_strings:
                self.assertIn(required, content, msg=f"{relpath} missing {required!r}")


if __name__ == "__main__":
    unittest.main()
