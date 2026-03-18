"""
Low-level checkpoint loading utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from vlm_backbones.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from vlm_backbones.models.vlms import FrozenVLM
from vlm_backbones.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def validate_artifact_layout(run_dir: Path) -> tuple[Path, Path]:
    config_json = run_dir / "config.json"
    checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
    if not config_json.exists():
        raise FileNotFoundError(f"Missing config file: {config_json}")
    if not checkpoint_pt.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_pt}")
    return config_json, checkpoint_pt


def read_model_config(run_dir: Path) -> dict[str, Any]:
    config_json, _ = validate_artifact_layout(run_dir)
    with config_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_model_cfg(full_cfg: dict[str, Any], override_cfg: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = dict(full_cfg.get("model", {}))
    if override_cfg:
        for key, value in override_cfg.items():
            if value is not None:
                merged[key] = value
    return merged


def load_from_directory(
    run_dir: Path,
    *,
    hf_token: Optional[str] = None,
    public_model_id: Optional[str] = None,
    override_cfg: Optional[Mapping[str, Any]] = None,
) -> FrozenVLM:
    run_dir = Path(run_dir).expanduser().resolve()
    config_json, checkpoint_pt = validate_artifact_layout(run_dir)

    with config_json.open("r", encoding="utf-8") as handle:
        full_cfg = json.load(handle)

    model_cfg = _merge_model_cfg(full_cfg, override_cfg)
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

    extra_vmamba = ""
    if vmamba_feature_stage is not None or vmamba_feature_layer is not None:
        extra_vmamba = (
            f"\n             VMamba Tap      =>> stage {vmamba_feature_stage}, layer {vmamba_feature_layer}"
        )

    overwatch.info(
        f"Found Config =>> Loading [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
        f"{extra_vmamba}"
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

    return FrozenVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )
