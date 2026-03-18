"""
maxvit.py

MaxViT vision backbone wrapper using TIMM's PyTorch implementation + pretrained checkpoints.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
try:
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, resolve_model_data_config
except ImportError:  # timm<1.0
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, resolve_data_config as _resolve_data_config

    def resolve_model_data_config(model: nn.Module) -> Dict[str, object]:
        return _resolve_data_config({}, model=model)
from timm.models.maxxvit import MaxxVit, MaxxVitBlock
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from vlm_backbones.models.backbones.vision.base_vision import LetterboxPad, VisionBackbone
from vlm_backbones.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

def _to_2tuple(value: object) -> Tuple[object, object]:
    if isinstance(value, (tuple, list, torch.Size)):
        if len(value) == 2:
            return (value[0], value[1])
        if len(value) == 1:
            return (value[0], value[0])
    return (value, value)


def _patch_timm_same_padding() -> None:
    """Normalize int stride/dilation in timm's same-padding helpers for older builds."""
    try:
        from timm.layers import conv2d_same as timm_conv2d_same
        from timm.layers import padding as timm_padding
    except Exception:
        return
    if getattr(timm_padding, "_vlm_backbones_pad_same_patch", False):
        return
    orig_pad_same = timm_padding.pad_same

    def pad_same(x, kernel_size, stride, dilation):
        return orig_pad_same(
            x,
            _to_2tuple(kernel_size),
            _to_2tuple(stride),
            _to_2tuple(dilation),
        )

    timm_padding.pad_same = pad_same
    timm_conv2d_same.pad_same = pad_same
    timm_padding._vlm_backbones_pad_same_patch = True


_patch_timm_same_padding()

@dataclass(frozen=True)
class MaxViTVariant:
    timm_id: str
    image_size: int
    legacy_preprocess: bool = False
    feature_stage: Optional[int] = None


MAXVIT_VARIANTS: Dict[str, MaxViTVariant] = {
    # ImageNet-1K (tiny/small/base/large @ 224/384/512)
    "in1k-224px-maxvit-t": MaxViTVariant("maxvit_tiny_tf_224.in1k", 224, legacy_preprocess=True),
    "in1k-384px-maxvit-t": MaxViTVariant("maxvit_tiny_tf_384.in1k", 384, legacy_preprocess=True),
    "in1k-512px-maxvit-t": MaxViTVariant("maxvit_tiny_tf_512.in1k", 512, legacy_preprocess=True),

    "in1k-224px-maxvit-s": MaxViTVariant("maxvit_small_tf_224.in1k", 224, legacy_preprocess=True),
    "in1k-384px-maxvit-s": MaxViTVariant("maxvit_small_tf_384.in1k", 384, legacy_preprocess=True),
    "in1k-512px-maxvit-s": MaxViTVariant("maxvit_small_tf_512.in1k", 512, legacy_preprocess=True),

    "in1k-224px-maxvit-b": MaxViTVariant("maxvit_base_tf_224.in1k", 224, legacy_preprocess=True),
    "in1k-384px-maxvit-b": MaxViTVariant("maxvit_base_tf_384.in1k", 384, legacy_preprocess=True),
    "in1k-512px-maxvit-b": MaxViTVariant("maxvit_base_tf_512.in1k", 512, legacy_preprocess=True),

    "in1k-224px-maxvit-l": MaxViTVariant("maxvit_large_tf_224.in1k", 224, legacy_preprocess=True),
    "in1k-384px-maxvit-l": MaxViTVariant("maxvit_large_tf_384.in1k", 384, legacy_preprocess=True),
    "in1k-512px-maxvit-l": MaxViTVariant("maxvit_large_tf_512.in1k", 512, legacy_preprocess=True),

    # ImageNet-21K (base/large/xlarge @ 224)
    "in21k-224px-maxvit-b": MaxViTVariant("maxvit_base_tf_224.in21k", 224),
    "in21k-224px-maxvit-l": MaxViTVariant("maxvit_large_tf_224.in21k", 224),
    "in21k-224px-maxvit-xl": MaxViTVariant("maxvit_xlarge_tf_224.in21k", 224),

    # ImageNet-21K -> ImageNet-1K (base/large/xlarge @ 384/512)
    "in21kft-384px-maxvit-b": MaxViTVariant("maxvit_base_tf_384.in21k_ft_in1k", 384),
    "in21kft-512px-maxvit-b": MaxViTVariant("maxvit_base_tf_512.in21k_ft_in1k", 512),

    "in21kft-384px-maxvit-l": MaxViTVariant("maxvit_large_tf_384.in21k_ft_in1k", 384),
    "in21kft-512px-maxvit-l": MaxViTVariant("maxvit_large_tf_512.in21k_ft_in1k", 512),

    "in21kft-384px-maxvit-xl": MaxViTVariant("maxvit_xlarge_tf_384.in21k_ft_in1k", 384),
    "in21kft-512px-maxvit-xl": MaxViTVariant("maxvit_xlarge_tf_512.in21k_ft_in1k", 512),
}

for variant_id, variant in list(MAXVIT_VARIANTS.items()):
    MAXVIT_VARIANTS[f"{variant_id}-s3"] = MaxViTVariant(
        variant.timm_id,
        variant.image_size,
        legacy_preprocess=variant.legacy_preprocess,
        feature_stage=3,
    )

MAXVIT_CROP_PADDING = 32
MAXVIT_TF_MEAN = (0.5, 0.5, 0.5)
MAXVIT_TF_STD = (0.5, 0.5, 0.5)


def get_registered_maxvit_variants() -> Tuple[str, ...]:
    return tuple(sorted(MAXVIT_VARIANTS.keys()))


def _use_pretrained_weights() -> bool:
    value = os.environ.get("MAXVIT_SKIP_PRETRAIN", "")
    return value.strip().lower() not in {"1", "true", "yes"}

class MaxViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: Optional[int] = None,
        feature_stage: Optional[int] = None,
    ) -> None:
        if vision_backbone_id not in MAXVIT_VARIANTS:
            raise ValueError(f"Unknown MaxViT variant `{vision_backbone_id}`")
        variant = MAXVIT_VARIANTS[vision_backbone_id]
        if default_image_size is None:
            default_image_size = variant.image_size
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.variant = variant
        # MaxViT uses BatchNormAct2d layers; keep backbone params in FP32 to avoid BN dtype mismatches.
        self.dtype = torch.float32
        self._feature_stage = feature_stage or variant.feature_stage

        pretrained = _use_pretrained_weights()
        if not pretrained:
            overwatch.warning("MAXVIT_SKIP_PRETRAIN=1 -> loading MaxViT with random weights.")

        self.featurizer: MaxxVit = timm.create_model(
            variant.timm_id,
            pretrained=pretrained,
            num_classes=0,
            img_size=self.default_image_size,
        )
        self.featurizer.eval()
        # Route FSDP calls through the patched forward to avoid bypassing unflattening.
        self.featurizer.forward = self.featurizer.forward_features

        # Configure image transforms (aligned with the original MaxViT preprocessing).
        self.data_cfg = resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)
        self.image_transform = self._build_image_transform()

        self._feature_stage_index = self._resolve_feature_stage()
        self._embed_dim = self._infer_embed_dim()
        self._grid_size = self._infer_grid_size()

        if self._feature_stage is not None:
            overwatch.info(
                f"MaxViT feature stage => {self._feature_stage} (0-index {self._feature_stage_index})",
                ctx_level=1,
            )

    def _build_image_transform(self) -> Compose:
        if self.variant.legacy_preprocess:
            mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        else:
            mean, std = MAXVIT_TF_MEAN, MAXVIT_TF_STD

        interpolation = InterpolationMode.BICUBIC
        crop_pct = self.default_image_size / (self.default_image_size + MAXVIT_CROP_PADDING)
        resize_target = int(round(self.default_image_size / crop_pct))

        if self.image_resize_strategy == "resize-crop":
            return Compose(
                [
                    Resize(resize_target, interpolation=interpolation),
                    CenterCrop(self.default_image_size),
                    ToTensor(),
                    Normalize(mean, std),
                ]
            )
        if self.image_resize_strategy == "resize-naive":
            return Compose(
                [
                    Resize((self.default_image_size, self.default_image_size), interpolation=interpolation),
                    ToTensor(),
                    Normalize(mean, std),
                ]
            )
        if self.image_resize_strategy == "letterbox":
            fill = tuple(int(x * 255) for x in mean)
            return Compose(
                [
                    LetterboxPad(fill),
                    Resize((self.default_image_size, self.default_image_size), interpolation=interpolation),
                    ToTensor(),
                    Normalize(mean, std),
                ]
            )
        raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def _infer_grid_size(self) -> int:
        reduction = self._infer_stage_reduction()
        if getattr(self.featurizer, "feature_info", None):
            try:
                stage_info = self.featurizer.feature_info[self._feature_stage_index + 1]
                reduction = int(stage_info.get("reduction", reduction))
            except Exception:
                reduction = reduction
        return int(math.ceil(self.default_image_size / reduction))

    def _infer_stage_reduction(self) -> int:
        return 2 ** (self._feature_stage_index + 2)

    def _infer_embed_dim(self) -> int:
        if getattr(self.featurizer, "feature_info", None):
            try:
                stage_info = self.featurizer.feature_info[self._feature_stage_index + 1]
                num_chs = stage_info.get("num_chs")
                if num_chs is not None:
                    return int(num_chs)
            except Exception:
                pass
        return int(getattr(self.featurizer, "num_features", getattr(self.featurizer, "embed_dim", 0)))

    def _resolve_feature_stage(self) -> int:
        num_stages = len(getattr(self.featurizer, "stages", []))
        if num_stages == 0:
            return 0
        stage = num_stages if self._feature_stage is None else self._feature_stage
        if stage < 1 or stage > num_stages:
            raise ValueError(f"MaxViT stage selection {stage} is out of range for {num_stages} stages (1-indexed)")
        return stage - 1

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._feature_stage_index == len(self.featurizer.stages) - 1:
            return self.featurizer.forward_features(x)
        x = self.featurizer.stem(x)
        for idx, stage in enumerate(self.featurizer.stages):
            x = stage(x)
            if idx == self._feature_stage_index:
                break
        return x

    def get_fsdp_wrapping_policy(self) -> Callable:
        # Avoid wrapping the root MaxxVit module; stage-3 forward calls stem/stages
        # directly and needs unflattened params (root FSDP would flatten them).
        block_policy = partial(_module_wrap_policy, module_classes={MaxxVitBlock})
        return partial(_or_policy, policies=[block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(pixel_values, dict):
            raise ValueError("MaxViT expects a single tensor for `pixel_values`, not a dict.")
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 4 and pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        elif pixel_values.ndim != 4:
            raise ValueError(f"MaxViT expects pixel_values with 4 dims, got shape={tuple(pixel_values.shape)}")
        features = self._forward_features(pixel_values.contiguous())
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1).reshape(features.size(0), -1, features.size(1))
        return features

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._grid_size * self._grid_size

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
