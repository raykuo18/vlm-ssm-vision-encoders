"""
vitdet.py

ViTDet vision backbone wrapper using Detectron2 Mask R-CNN COCO checkpoints.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from vlm_backbones.cache import get_backbone_cache_dir
from vlm_backbones.models.backbones.vision.base_vision import ImageTransform, VisionBackbone
from vlm_backbones.overwatch import initialize_overwatch
from vlm_backbones.runtime_paths import get_third_party_root

overwatch = initialize_overwatch(__name__)

PIL_RESAMPLING = getattr(Image, "Resampling", Image)

VITDET_REPO_ROOT = get_third_party_root() / "detectron2"
VITDET_CKPT_ROOT = Path(os.environ.get("VITDET_CKPT_ROOT", str(get_backbone_cache_dir("vitdet"))))

VITDET_IMAGE_SIZE = 1024
VITDET_PATCH_SIZE = 16
VITDET_WINDOW_SIZE = 14
VITDET_PREPROCESS_STRATEGY = "vitdet-detection"

IMAGENET_RGB_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
IMAGENET_RGB_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)


def _use_pretrained_weights() -> bool:
    value = os.environ.get("VITDET_SKIP_PRETRAIN", "")
    return value.strip().lower() not in {"1", "true", "yes"}


@dataclass(frozen=True)
class ViTDetVariant:
    embed_dim: int
    depth: int
    num_heads: int
    drop_path_rate: float
    window_block_indexes: Tuple[int, ...]
    checkpoint: str
    url: str


VITDET_VARIANTS: Dict[str, ViTDetVariant] = {
    "vitdet-b-maskrcnn": ViTDetVariant(
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        checkpoint="vitdet_b_model_final_61ccd1.pkl",
        url="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl",
    ),
    "vitdet-l-maskrcnn": ViTDetVariant(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        window_block_indexes=tuple(
            list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
        ),
        checkpoint="vitdet_l_model_final_6146ed.pkl",
        url="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl",
    ),
    "vitdet-h-maskrcnn": ViTDetVariant(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        drop_path_rate=0.5,
        window_block_indexes=tuple(
            list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
        ),
        checkpoint="vitdet_h_model_final_7224f1.pkl",
        url="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl",
    ),
}

VITDET_ALIASES = {
    "vitdet-b": "vitdet-b-maskrcnn",
    "vitdet-l": "vitdet-l-maskrcnn",
    "vitdet-h": "vitdet-h-maskrcnn",
}


def get_registered_vitdet_variants() -> Tuple[str, ...]:
    return tuple(sorted(VITDET_VARIANTS.keys()))


def _resolve_variant_id(vision_backbone_id: str) -> str:
    return VITDET_ALIASES.get(vision_backbone_id, vision_backbone_id)


def _maybe_import_detectron2() -> None:
    if not VITDET_REPO_ROOT.exists():
        raise ImportError(
            f"detectron2 repo not found at {VITDET_REPO_ROOT}. Run `git submodule update --init --recursive`."
        )
    if str(VITDET_REPO_ROOT) not in sys.path:
        sys.path.append(str(VITDET_REPO_ROOT))
    try:
        import detectron2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime env issue
        raise ImportError(
            "Unable to import detectron2. Install its dependencies (e.g., cloudpickle, fvcore, yaml) "
            "and ensure the detectron2 submodule is on PYTHONPATH."
        ) from exc


def _load_vitdet_checkpoint(variant: ViTDetVariant) -> Dict[str, torch.Tensor]:
    ckpt_path = VITDET_CKPT_ROOT / variant.checkpoint
    if not ckpt_path.exists():
        VITDET_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        overwatch.info(f"Downloading ViTDet checkpoint to {ckpt_path}")
        from torch.hub import download_url_to_file

        download_url_to_file(variant.url, ckpt_path)

    with ckpt_path.open("rb") as handle:
        state = pickle.load(handle)
    weights = state.get("model", state)
    cleaned = {}
    for key, value in weights.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone.net."):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            cleaned[key[len("backbone.net.") :]] = value
    return cleaned


class ResizeKeepRatioMax:
    """Resize images while preserving aspect ratio under (max_long_edge, max_short_edge)."""

    def __init__(self, max_long_edge: int, max_short_edge: int, interpolation: InterpolationMode) -> None:
        self.max_long_edge = max_long_edge
        self.max_short_edge = max_short_edge
        if isinstance(interpolation, InterpolationMode):
            self.interpolation = getattr(PIL_RESAMPLING, interpolation.name, PIL_RESAMPLING.BILINEAR)
        elif isinstance(interpolation, str):
            self.interpolation = getattr(PIL_RESAMPLING, interpolation.upper(), PIL_RESAMPLING.BILINEAR)
        else:
            self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        long_edge = max(width, height)
        short_edge = min(width, height)
        if long_edge == 0 or short_edge == 0:
            return image
        scale = min(self.max_long_edge / long_edge, self.max_short_edge / short_edge)
        if scale >= 1.0 and long_edge <= self.max_long_edge and short_edge <= self.max_short_edge:
            return image
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), self.interpolation)


class PadToSquare:
    """Pad PIL images to a fixed square size, padding on the right/bottom."""

    def __init__(self, size: int, fill: Tuple[int, int, int]) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        pad_w = max(0, self.size - width)
        pad_h = max(0, self.size - height)
        if pad_w == 0 and pad_h == 0:
            return image
        padding = (0, 0, pad_w, pad_h)
        return ImageOps.expand(image, border=padding, fill=self.fill)


class ViTDetBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: Optional[int] = None,
    ) -> None:
        canonical_id = _resolve_variant_id(vision_backbone_id)
        if canonical_id not in VITDET_VARIANTS:
            raise ValueError(f"Unknown ViTDet variant `{vision_backbone_id}`")
        variant = VITDET_VARIANTS[canonical_id]

        if image_resize_strategy not in {VITDET_PREPROCESS_STRATEGY, "letterbox"}:
            raise ValueError(
                f"ViTDet backbone '{canonical_id}' requires image_resize_strategy='{VITDET_PREPROCESS_STRATEGY}' "
                f"or 'letterbox' (got '{image_resize_strategy}')."
            )

        if default_image_size is None:
            default_image_size = VITDET_IMAGE_SIZE
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        _maybe_import_detectron2()
        from detectron2.modeling.backbone.vit import Block, ViT

        self._block_cls = Block
        self._vit_cls = ViT
        self._variant = variant
        self._out_feature = "last_feat"
        self.dtype = torch.bfloat16

        self.featurizer: ViT = ViT(
            img_size=self.default_image_size,
            patch_size=VITDET_PATCH_SIZE,
            embed_dim=variant.embed_dim,
            depth=variant.depth,
            num_heads=variant.num_heads,
            drop_path_rate=variant.drop_path_rate,
            window_size=VITDET_WINDOW_SIZE,
            window_block_indexes=variant.window_block_indexes,
            residual_block_indexes=(),
            use_rel_pos=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            out_feature=self._out_feature,
        )
        self.featurizer.eval()
        self._load_checkpoint()

        self.image_transform = self._build_image_transform()
        self._embed_dim = variant.embed_dim
        grid_size = self.default_image_size // VITDET_PATCH_SIZE
        self._num_patches = grid_size * grid_size

    def _load_checkpoint(self) -> None:
        if not _use_pretrained_weights():
            overwatch.warning("VITDET_SKIP_PRETRAIN=1 -> loading ViTDet with random weights.")
            return
        weights = _load_vitdet_checkpoint(self._variant)
        incompatible = self.featurizer.load_state_dict(weights, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            overwatch.warning(
                "ViTDet checkpoint loaded with mismatched keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def _build_image_transform(self) -> ImageTransform:
        interpolation = InterpolationMode.BILINEAR
        fill = tuple(int(x * 255) for x in IMAGENET_RGB_MEAN)
        return Compose(
            [
                ResizeKeepRatioMax(self.default_image_size, self.default_image_size, interpolation),
                PadToSquare(self.default_image_size, fill),
                ToTensor(),
                Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_STD),
            ]
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

        vit_wrap_policy = partial(_module_wrap_policy, module_classes={self._vit_cls})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={self._block_cls})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(pixel_values, dict):
            raise ValueError("ViTDet expects a single tensor for `pixel_values`, not a dict.")
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 4 and pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        elif pixel_values.ndim != 4:
            raise ValueError(f"ViTDet expects pixel_values with 4 dims, got shape={tuple(pixel_values.shape)}")

        outputs = self.featurizer(pixel_values.contiguous())
        if isinstance(outputs, dict):
            features = outputs[self._out_feature]
        else:
            features = outputs

        if features.ndim == 4:
            if features.shape[1] != self._embed_dim and features.shape[-1] == self._embed_dim:
                features = features.permute(0, 3, 1, 2)
            features = features.flatten(2).transpose(1, 2)
        return features

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
