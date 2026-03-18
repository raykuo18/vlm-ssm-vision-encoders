"""
in1k_vit.py

Vision Transformers trained / finetuned on ImageNet (ImageNet-21K =>> ImageNet-1K)
"""

from vlm_backbones.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported Vision Backbones (from TIMM)
# Three types:
# 1. in21k-vit-* : trained on ImageNet-21K only (_in21k)
# 2. in1kft-vit-*: pretrained on ImageNet-21K, fine-tuned on ImageNet-1K (_in21k_ft_in1k)
# 3. in1k-vit-*   : trained on ImageNet-1K only (_in1k)
IN1K_VISION_BACKBONES = {
    # Type 1: ImageNet-21K Only (pretrained on ImageNet-21K) - _in21k
    "in21k-vit-t": "vit_tiny_patch16_224.augreg_in21k",
    "in21k-vit-s": "vit_small_patch16_224.augreg_in21k",
    "in21k-vit-b": "vit_base_patch16_224.augreg_in21k",
    "in21k-vit-l": "vit_large_patch16_224.augreg_in21k",

    # Type 2: ImageNet-21K → ImageNet-1K (pretrained on ImageNet-21K, fine-tuned on ImageNet-1K) - _in21k_ft_in1k
    "in1kft-vit-t": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "in1kft-vit-s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "in1kft-vit-b": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "in1kft-vit-b2": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "in1kft-vit-l": "vit_large_patch16_224.augreg_in21k_ft_in1k",

    # Type 3: ImageNet-1K Only (trained directly on ImageNet-1K) - _in1k
    "in1k-vit-s": "vit_small_patch16_224.augreg_in1k",
    "in1k-vit-b": "vit_base_patch16_224.augreg_in1k",
    # "in1k-vit-l": "vit_large_patch16_224.augreg_in1k",
}


class IN1KViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            IN1K_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            pretrained=True,
        )
