from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = []

    repo_root_override = os.environ.get("VLM_BACKBONES_REPO_ROOT")
    if repo_root_override:
        candidates.append(Path(repo_root_override).expanduser().resolve())

    candidates.extend((module_path.parents[1], module_path.parents[2]))

    for candidate in candidates:
        if (candidate / "model_zoo" / "models.yaml").exists() or (candidate / "third_party").exists():
            return candidate

    return module_path.parents[2]


def get_third_party_root() -> Path:
    third_party_override = os.environ.get("VLM_BACKBONES_THIRD_PARTY_ROOT")
    if third_party_override:
        return Path(third_party_override).expanduser().resolve()
    return get_repo_root() / "third_party"
