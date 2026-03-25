from vlm_backbones.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
)

from prismatic.models.vlms import PrismaticVLM

__all__ = ["get_llm_backbone_and_tokenizer", "get_vision_backbone_and_transform", "get_vlm"]


def get_vlm(
    model_id,
    arch_specifier,
    vision_backbone,
    llm_backbone,
    enable_mixed_precision_training=True,
    vision_finetune_train_projector=True,
):
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        vision_finetune_train_projector=vision_finetune_train_projector,
    )
