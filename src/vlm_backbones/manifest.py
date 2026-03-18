from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from vlm_backbones.runtime_paths import get_repo_root


@dataclass(frozen=True)
class ModelMetrics:
    weighted_vqa: float
    weighted_loc: float
    weighted_overall: float


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str
    task: str
    download_url: str
    sha256: str
    artifact_filename: str
    internal_run_id: str
    vision_backbone_id: str
    image_resize_strategy: str
    arch_specifier: str
    llm_backbone_id: str
    metrics: ModelMetrics
    vmamba_feature_stage: Optional[int] = None
    vmamba_feature_layer: Optional[int] = None
    resolution: Optional[str] = None
    vision_tokens: Optional[int] = None
    params: Optional[str] = None
    pretraining: Optional[str] = None

    @property
    def has_download(self) -> bool:
        return bool(self.download_url) and "REPLACE_ME" not in self.download_url

    @property
    def has_sha256(self) -> bool:
        return bool(self.sha256) and "REPLACE_ME" not in self.sha256


def get_manifest_path() -> Path:
    return get_repo_root() / "model_zoo" / "models.yaml"


def _require_fields(entry: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in entry]
    if missing:
        raise ValueError(f"Manifest entry `{entry.get('id', '<unknown>')}` missing required fields: {missing}")


def _parse_entry(entry: dict[str, Any]) -> ModelSpec:
    _require_fields(
        entry,
        (
            "id",
            "display_name",
            "family",
            "task",
            "download_url",
            "sha256",
            "artifact_filename",
            "internal_run_id",
            "vision_backbone_id",
            "image_resize_strategy",
            "arch_specifier",
            "llm_backbone_id",
            "metrics",
        ),
    )
    metrics = entry["metrics"]
    _require_fields(metrics, ("weighted_vqa", "weighted_loc", "weighted_overall"))
    return ModelSpec(
        id=entry["id"],
        display_name=entry["display_name"],
        family=entry["family"],
        task=entry["task"],
        download_url=entry["download_url"],
        sha256=entry["sha256"],
        artifact_filename=entry["artifact_filename"],
        internal_run_id=entry["internal_run_id"],
        vision_backbone_id=entry["vision_backbone_id"],
        image_resize_strategy=entry["image_resize_strategy"],
        arch_specifier=entry["arch_specifier"],
        llm_backbone_id=entry["llm_backbone_id"],
        vmamba_feature_stage=entry.get("vmamba_feature_stage"),
        vmamba_feature_layer=entry.get("vmamba_feature_layer"),
        resolution=entry.get("resolution"),
        vision_tokens=entry.get("vision_tokens"),
        params=entry.get("params"),
        pretraining=entry.get("pretraining"),
        metrics=ModelMetrics(
            weighted_vqa=float(metrics["weighted_vqa"]),
            weighted_loc=float(metrics["weighted_loc"]),
            weighted_overall=float(metrics["weighted_overall"]),
        ),
    )


@lru_cache(maxsize=1)
def load_manifest() -> tuple[ModelSpec, ...]:
    manifest_path = get_manifest_path()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("models", [])
    specs = tuple(_parse_entry(entry) for entry in entries)
    if len({spec.id for spec in specs}) != len(specs):
        raise ValueError("Manifest contains duplicate public model ids.")
    return specs


def available_models() -> list[ModelSpec]:
    return list(load_manifest())


def resolve_model_spec(model_id: str) -> ModelSpec:
    for spec in load_manifest():
        if spec.id == model_id:
            return spec
    raise KeyError(f"Unknown public model id: {model_id}")
