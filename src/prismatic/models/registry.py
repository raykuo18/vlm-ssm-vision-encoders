from __future__ import annotations

from vlm_backbones.manifest import available_models


def _build_model_registry() -> dict[str, dict[str, object]]:
    registry: dict[str, dict[str, object]] = {}
    for spec in available_models():
        description = {
            "name": spec.display_name,
            "family": spec.family,
            "task": spec.task,
            "visual_representation": spec.vision_backbone_id,
            "image_processing": spec.image_resize_strategy,
            "language_model": spec.llm_backbone_id,
            "source_run": spec.internal_run_id,
        }
        registry[spec.id] = {"model_id": spec.id, "names": [spec.display_name, spec.internal_run_id], "description": description}
        registry[spec.internal_run_id] = {
            "model_id": spec.id,
            "names": [spec.display_name, spec.internal_run_id],
            "description": description,
        }
    return registry


GLOBAL_REGISTRY = _build_model_registry()
MODEL_REGISTRY = {key: value for key, value in GLOBAL_REGISTRY.items() if key == value["model_id"]}

