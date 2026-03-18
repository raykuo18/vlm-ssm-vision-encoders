from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV_VAR = "VLM_BACKBONES_CACHE_DIR"
DEFAULT_CACHE_DIR = "~/.cache/vlm-ssm-vision-encoders"


def get_cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV_VAR, DEFAULT_CACHE_DIR)).expanduser().resolve()


def ensure_cache_dir(*parts: str) -> Path:
    path = get_cache_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_cache_dir(model_id: str) -> Path:
    return get_cache_root() / "models" / model_id


def get_artifact_cache_dir() -> Path:
    return ensure_cache_dir("artifacts")


def get_backbone_cache_dir(family: str) -> Path:
    return ensure_cache_dir("backbones", family)
