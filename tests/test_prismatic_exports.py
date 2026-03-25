from __future__ import annotations

import unittest

from prismatic.models.backbones.vision.mambavision import MambaVisionBackbone
from prismatic.models.backbones.vision.vit_adapter import ViTAdapterBackbone
from prismatic.models.backbones.vision.vitdet import ViTDetBackbone
from prismatic.models.backbones.vision.vmamba import VMambaBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.vlms import PrismaticVLM
from vlm_backbones.models.backbones.llm.prompting import PromptBuilder as ReleasePromptBuilder
from vlm_backbones.models.backbones.vision import (
    MambaVisionBackbone as ReleaseMambaVisionBackbone,
)
from vlm_backbones.models.backbones.vision import ViTAdapterBackbone as ReleaseViTAdapterBackbone
from vlm_backbones.models.backbones.vision import ViTDetBackbone as ReleaseViTDetBackbone
from vlm_backbones.models.backbones.vision import VMambaBackbone as ReleaseVMambaBackbone
from vlm_backbones.models.vlms.frozen import FrozenVLM


class PrismaticExportsTest(unittest.TestCase):
    def test_prismatic_vlm_is_compatible_wrapper(self) -> None:
        self.assertTrue(issubclass(PrismaticVLM, FrozenVLM))

    def test_prompt_builder_shim_matches_release_export(self) -> None:
        self.assertIs(PromptBuilder, ReleasePromptBuilder)

    def test_supported_vision_module_shims_match_release_exports(self) -> None:
        self.assertIs(MambaVisionBackbone, ReleaseMambaVisionBackbone)
        self.assertIs(ViTAdapterBackbone, ReleaseViTAdapterBackbone)
        self.assertIs(ViTDetBackbone, ReleaseViTDetBackbone)
        self.assertIs(VMambaBackbone, ReleaseVMambaBackbone)


if __name__ == "__main__":
    unittest.main()
