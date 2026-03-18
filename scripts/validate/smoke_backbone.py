#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import os

from PIL import Image
import torch

from vlm_backbones.manifest import resolve_model_spec
from vlm_backbones.models.materialize import get_vision_backbone_and_transform


def _default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _move_to_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _prepare_inputs(transform, width: int, height: int):
    image = Image.new("RGB", (width, height), color=(127, 127, 127))
    pixel_values = transform(image)
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values
    if isinstance(pixel_values, dict):
        prepared = {}
        for key, value in pixel_values.items():
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                prepared[key] = value.unsqueeze(0)
            else:
                prepared[key] = value
        return prepared
    raise TypeError(f"Unexpected transform output type: {type(pixel_values)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the released vision backbone and run one dummy forward pass.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    spec = resolve_model_spec(args.model)
    if spec.family == "vmamba":
        os.environ.setdefault("VMAMBA_SKIP_PRETRAIN", "1")
    if spec.family == "mambavision":
        os.environ.setdefault("MAMBAVISION_SKIP_PRETRAIN", "1")
    if spec.family == "vitdet":
        os.environ.setdefault("VITDET_SKIP_PRETRAIN", "1")
    if spec.family == "vit_adapter":
        os.environ.setdefault("VIT_ADAPTER_SKIP_PRETRAIN", "1")

    backbone, _ = get_vision_backbone_and_transform(
        spec.vision_backbone_id,
        spec.image_resize_strategy,
        vmamba_feature_stage=spec.vmamba_feature_stage,
        vmamba_feature_layer=spec.vmamba_feature_layer,
    )
    device = _default_device(args.device)
    backbone = backbone.to(device)
    _, height, width = backbone.default_image_resolution
    pixel_values = _prepare_inputs(backbone.get_image_transform(), width, height)
    pixel_values = _move_to_device(pixel_values, device)

    with torch.inference_mode():
        features = backbone(pixel_values)

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Backbone forward returned unexpected type: {type(features)!r}")
    print(
        f"backbone ok: {spec.id} -> {backbone.identifier} "
        f"device={device} shape={tuple(features.shape)} dtype={features.dtype}"
    )


if __name__ == "__main__":
    main()
