"""
Compatibility loader for the vendored training/evaluation stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from vlm_backbones.api import download_model
from vlm_backbones.manifest import resolve_model_spec
from vlm_backbones.models.load import validate_artifact_layout
from vlm_backbones.overwatch import initialize_overwatch

from .materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from .registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from .vlms import PrismaticVLM

overwatch = initialize_overwatch(__name__)


def available_model_ids() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_ids_and_names() -> List[List[str]]:
    return [value["names"] for _, value in MODEL_REGISTRY.items()]


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }`; check `prismatic.available_model_ids()`")

    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))
    return description


def _resolve_public_model_path(model_id_or_name: str) -> tuple[Path, str, dict[str, object]]:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }`; check `prismatic.available_model_ids()`")

    canonical_model_id = str(GLOBAL_REGISTRY[model_id_or_name]["model_id"])
    spec = resolve_model_spec(canonical_model_id)
    run_dir = download_model(spec.id, force=False)
    override_cfg = {
        "model_id": spec.id,
        "vision_backbone_id": spec.vision_backbone_id,
        "llm_backbone_id": spec.llm_backbone_id,
        "image_resize_strategy": spec.image_resize_strategy,
        "arch_specifier": spec.arch_specifier,
        "vmamba_feature_stage": spec.vmamba_feature_stage,
        "vmamba_feature_layer": spec.vmamba_feature_layer,
    }
    return run_dir, spec.id, override_cfg


def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> PrismaticVLM:
    del cache_dir

    if os.path.isdir(model_id_or_path):
        run_dir = Path(model_id_or_path)
        public_model_id = None
        override_cfg = None
        overwatch.info(f"Loading from local path `{run_dir}`")
    else:
        run_dir, public_model_id, override_cfg = _resolve_public_model_path(str(model_id_or_path))
        overwatch.info(f"Resolved public model `{public_model_id}` to `{run_dir}`")

    run_dir = Path(run_dir).expanduser().resolve()
    config_json, checkpoint_pt = validate_artifact_layout(run_dir)

    with config_json.open("r", encoding="utf-8") as handle:
        full_cfg = json.load(handle)

    model_cfg = dict(full_cfg.get("model", {}))
    if override_cfg:
        for key, value in override_cfg.items():
            if value is not None:
                model_cfg[key] = value
    if public_model_id is not None:
        model_cfg["model_id"] = public_model_id

    vmamba_feature_stage = (
        model_cfg.get("vmamba_feature_stage")
        if "vmamba_feature_stage" in model_cfg
        else full_cfg.get("vmamba_feature_stage")
    )
    vmamba_feature_layer = (
        model_cfg.get("vmamba_feature_layer")
        if "vmamba_feature_layer" in model_cfg
        else full_cfg.get("vmamba_feature_layer")
    )

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        vmamba_feature_stage=vmamba_feature_stage,
        vmamba_feature_layer=vmamba_feature_layer,
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
    )

    return PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )
