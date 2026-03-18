from __future__ import annotations

from typing import Any

from vlm_backbones.manifest import ModelMetrics, ModelSpec

__all__ = [
    "FrozenVLM",
    "ModelMetrics",
    "ModelSpec",
    "available_models",
    "download_model",
    "load_model",
]


def available_models():
    from vlm_backbones.api import available_models as _available_models

    return _available_models()


def download_model(model_id: str, force: bool = False):
    from vlm_backbones.api import download_model as _download_model

    return _download_model(model_id, force=force)


def load_model(model: str, device: str = "cuda", dtype: str = "bfloat16"):
    from vlm_backbones.api import load_model as _load_model

    return _load_model(model, device=device, dtype=dtype)


def __getattr__(name: str) -> Any:
    if name == "FrozenVLM":
        from vlm_backbones.models.vlms import FrozenVLM

        return FrozenVLM
    raise AttributeError(name)
