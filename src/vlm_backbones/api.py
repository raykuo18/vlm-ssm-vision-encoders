from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from vlm_backbones.download import download_and_extract
from vlm_backbones.manifest import ModelSpec, available_models as available_manifest_models, resolve_model_spec
from vlm_backbones.models.load import load_from_directory

FrozenDType = Literal["auto", "bfloat16", "float16", "float32"]


def available_models() -> list[ModelSpec]:
    return available_manifest_models()


def download_model(model_id: str, force: bool = False) -> Path:
    return download_and_extract(resolve_model_spec(model_id), force=force)


def load_model(model: str | Path, device: str = "cuda", dtype: FrozenDType = "bfloat16"):
    spec = None
    if isinstance(model, Path) or Path(str(model)).expanduser().exists():
        model_dir = Path(model).expanduser().resolve()
    else:
        spec = resolve_model_spec(str(model))
        model_dir = download_model(spec.id, force=False)

    override_cfg = None
    public_model_id = None
    if spec is not None:
        public_model_id = spec.id
        override_cfg = {
            "model_id": spec.id,
            "vision_backbone_id": spec.vision_backbone_id,
            "llm_backbone_id": spec.llm_backbone_id,
            "image_resize_strategy": spec.image_resize_strategy,
            "arch_specifier": spec.arch_specifier,
            "vmamba_feature_stage": spec.vmamba_feature_stage,
            "vmamba_feature_layer": spec.vmamba_feature_layer,
        }

    vlm = load_from_directory(model_dir, public_model_id=public_model_id, override_cfg=override_cfg)
    target_device = torch.device(device)
    vlm.to(target_device)

    if dtype == "auto":
        target_dtype = vlm.llm_backbone.half_precision_dtype
    else:
        target_dtype = getattr(torch, dtype)
    vlm.inference_dtype = target_dtype
    vlm.enable_mixed_precision_training = target_device.type == "cuda" and target_dtype != torch.float32
    return vlm
