"""
Factories for the inference-only public release.
"""

from __future__ import annotations

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from vlm_backbones.models.backbones.llm import LLaMa2LLMBackbone, LLMBackbone
from vlm_backbones.models.backbones.vision import (
    ImageTransform,
    IN1KViTBackbone,
    MambaVisionBackbone,
    MaxViTBackbone,
    ViTAdapterBackbone,
    ViTDetBackbone,
    VMambaBackbone,
    VisionBackbone,
)
from vlm_backbones.models.backbones.vision.mambavision import get_registered_mambavision_variants
from vlm_backbones.models.backbones.vision.maxvit import get_registered_maxvit_variants
from vlm_backbones.models.backbones.vision.vit_adapter import get_registered_vit_adapter_variants
from vlm_backbones.models.backbones.vision.vitdet import get_registered_vitdet_variants
from vlm_backbones.models.backbones.vision.vmamba import get_registered_vmamba_variants
from vlm_backbones.models.vlms import FrozenVLM

VISION_BACKBONES = {
    "in1k-vit-s": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-b": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in21k-vit-t": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in21k-vit-s": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in21k-vit-b": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in21k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1kft-vit-t": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1kft-vit-s": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1kft-vit-b": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1kft-vit-b2": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1kft-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
}

for variant_id in get_registered_maxvit_variants():
    VISION_BACKBONES[variant_id] = {"cls": MaxViTBackbone, "kwargs": {}}

for variant_id in get_registered_mambavision_variants():
    VISION_BACKBONES[variant_id] = {"cls": MambaVisionBackbone, "kwargs": {}}

for variant_id in get_registered_vmamba_variants():
    VISION_BACKBONES[variant_id] = {"cls": VMambaBackbone, "kwargs": {}}

for variant_id in get_registered_vitdet_variants():
    VISION_BACKBONES[variant_id] = {"cls": ViTDetBackbone, "kwargs": {}}

for variant_id in get_registered_vit_adapter_variants():
    VISION_BACKBONES[variant_id] = {"cls": ViTAdapterBackbone, "kwargs": {}}

LLM_BACKBONES = {
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
}


def get_vision_backbone_and_transform(
    vision_backbone_id: str,
    image_resize_strategy: str,
    vmamba_feature_stage: Optional[int] = None,
    vmamba_feature_layer: Optional[int] = None,
) -> Tuple[VisionBackbone, ImageTransform]:
    if vision_backbone_id not in VISION_BACKBONES:
        raise ValueError(f"Vision backbone `{vision_backbone_id}` is not supported by the public release.")

    vision_cfg = VISION_BACKBONES[vision_backbone_id]
    runtime_kwargs = dict(vision_cfg["kwargs"])

    if vmamba_feature_stage is not None or vmamba_feature_layer is not None:
        if not vision_backbone_id.startswith("vmamba"):
            raise ValueError("VMamba feature taps can only be specified for VMamba backbones.")
        if vmamba_feature_stage is not None:
            runtime_kwargs["feature_stage"] = vmamba_feature_stage
        if vmamba_feature_layer is not None:
            runtime_kwargs["feature_layer"] = vmamba_feature_layer

    vision_backbone: VisionBackbone = vision_cfg["cls"](
        vision_backbone_id,
        image_resize_strategy,
        **runtime_kwargs,
    )
    return vision_backbone, vision_backbone.get_image_transform()


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id not in LLM_BACKBONES:
        raise ValueError(f"LLM backbone `{llm_backbone_id}` is not supported by the public release.")

    llm_cfg = LLM_BACKBONES[llm_backbone_id]
    llm_backbone: LLMBackbone = llm_cfg["cls"](
        llm_backbone_id,
        llm_max_length=llm_max_length,
        hf_token=hf_token,
        inference_mode=inference_mode,
        **llm_cfg["kwargs"],
    )
    return llm_backbone, llm_backbone.get_tokenizer()


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
    vision_finetune_train_projector: bool = True,
) -> FrozenVLM:
    return FrozenVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        vision_finetune_train_projector=vision_finetune_train_projector,
    )
