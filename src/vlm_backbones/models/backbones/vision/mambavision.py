"""MambaVision backbone integration for Prismatic VLMs."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.hub import download_url_to_file
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from vlm_backbones.cache import get_backbone_cache_dir
from vlm_backbones.models.backbones.vision.base_vision import LetterboxPad, VisionBackbone
from vlm_backbones.overwatch import initialize_overwatch
from vlm_backbones.runtime_paths import get_third_party_root

try:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except ImportError as exc:  # pragma: no cover - runtime env issue
    raise ImportError("MambaVision backbones require timm; install it in the mambavision env.") from exc

overwatch = initialize_overwatch(__name__)

MAMBAVISION_REPO_ROOT = get_third_party_root() / "MambaVision"
MAMBAVISION_CKPT_ROOT = Path(
    os.environ.get("MAMBAVISION_CKPT_ROOT", str(get_backbone_cache_dir("mambavision")))
)


@dataclass(frozen=True)
class MambaVisionVariant:
    model_name: str
    cfg_key: str
    feature_stage: Optional[int] = None


MAMBAVISION_VARIANTS: Dict[str, MambaVisionVariant] = {
    "mambavision-t": MambaVisionVariant("mamba_vision_T", "mamba_vision_T"),
    "mambavision-t2": MambaVisionVariant("mamba_vision_T2", "mamba_vision_T2"),
    "mambavision-s": MambaVisionVariant("mamba_vision_S", "mamba_vision_S"),
    "mambavision-b": MambaVisionVariant("mamba_vision_B", "mamba_vision_B"),
    "mambavision-b-21k": MambaVisionVariant("mamba_vision_B_21k", "mamba_vision_B_21k"),
    "mambavision-l": MambaVisionVariant("mamba_vision_L", "mamba_vision_L"),
    "mambavision-l-21k": MambaVisionVariant("mamba_vision_L_21k", "mamba_vision_L_21k"),
    "mambavision-l2": MambaVisionVariant("mamba_vision_L2", "mamba_vision_L2"),
    "mambavision-l2-512-21k": MambaVisionVariant("mamba_vision_L2_512_21k", "mamba_vision_L2_512_21k"),
    "mambavision-l3-256-21k": MambaVisionVariant("mamba_vision_L3_256_21k", "mamba_vision_L3_256_21k"),
    "mambavision-l3-512-21k": MambaVisionVariant("mamba_vision_L3_512_21k", "mamba_vision_L3_512_21k"),
}

for variant_id, variant in list(MAMBAVISION_VARIANTS.items()):
    MAMBAVISION_VARIANTS[f"{variant_id}-s3"] = MambaVisionVariant(
        variant.model_name,
        variant.cfg_key,
        feature_stage=3,
    )

MAMBAVISION_ALIASES = {
    "mambavision-tiny": "mambavision-t",
    "mambavision-small": "mambavision-s",
    "mambavision-base": "mambavision-b",
    "mambavision-large": "mambavision-l",
}


def get_registered_mambavision_variants() -> Tuple[str, ...]:
    return tuple(sorted(MAMBAVISION_VARIANTS.keys()))


def _resolve_variant_id(vision_backbone_id: str) -> str:
    canonical = MAMBAVISION_ALIASES.get(vision_backbone_id, vision_backbone_id)
    if canonical not in MAMBAVISION_VARIANTS:
        raise ValueError(
            f"Unknown MambaVision variant '{vision_backbone_id}'. Supported variants: {sorted(MAMBAVISION_VARIANTS)}"
        )
    return canonical


def _use_pretrained_weights() -> bool:
    value = os.environ.get("MAMBAVISION_SKIP_PRETRAIN", "")
    return value.strip().lower() not in {"1", "true", "yes"}


def _maybe_import_mambavision() -> Tuple[object, object]:
    if not MAMBAVISION_REPO_ROOT.exists():
        raise ImportError(
            f"MambaVision repo not found at {MAMBAVISION_REPO_ROOT}. Run `git submodule update --init --recursive`."
        )
    if str(MAMBAVISION_REPO_ROOT) not in sys.path:
        sys.path.append(str(MAMBAVISION_REPO_ROOT))
    try:
        from mambavision import create_model  # type: ignore
        from mambavision.models import mamba_vision as mv_module  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime env issue
        raise ImportError(
            "Unable to import MambaVision. Install its dependencies and ensure the MambaVision repo is on "
            "PYTHONPATH. See `scripts/env/build_mambavision.sh`."
        ) from exc
    return create_model, mv_module


def _resolve_default_cfg(variant: MambaVisionVariant) -> Dict[str, object]:
    _, mv_module = _maybe_import_mambavision()
    cfgs = getattr(mv_module, "default_cfgs", {})
    if variant.cfg_key not in cfgs:
        raise KeyError(f"MambaVision default config '{variant.cfg_key}' not found.")
    return dict(cfgs[variant.cfg_key])


def _resolve_interpolation(value: object) -> InterpolationMode:
    if isinstance(value, InterpolationMode):
        return value
    if isinstance(value, str):
        interp = value.lower()
        if interp == "bicubic":
            return InterpolationMode.BICUBIC
        if interp == "bilinear":
            return InterpolationMode.BILINEAR
        if interp == "nearest":
            return InterpolationMode.NEAREST
    return InterpolationMode.BICUBIC


def _ensure_checkpoint(cfg: Dict[str, object]) -> Path:
    url = cfg.get("url", "")
    if not url:
        raise ValueError("MambaVision default config is missing checkpoint URL.")
    filename = Path(url).name
    MAMBAVISION_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt_path = MAMBAVISION_CKPT_ROOT / filename
    if ckpt_path.exists():
        return ckpt_path
    if not _use_pretrained_weights():
        return ckpt_path

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    if is_distributed and rank != 0:
        timeout = int(os.environ.get("MAMBAVISION_CKPT_WAIT_SECS", "900"))
        poll = float(os.environ.get("MAMBAVISION_CKPT_POLL_SECS", "2.0"))
        start = time.time()
        while time.time() - start < timeout:
            if ckpt_path.exists():
                return ckpt_path
            time.sleep(poll)
        raise FileNotFoundError(
            f"Timed out waiting for MambaVision checkpoint at {ckpt_path}. "
            "Ensure rank0 can download from HF or pre-populate MAMBAVISION_CKPT_ROOT."
        )

    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    overwatch.info(f"Downloading MambaVision checkpoint -> {ckpt_path}")
    try:
        download_url_to_file(url=url, dst=tmp_path, progress=True)
        tmp_path.replace(ckpt_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return ckpt_path


class MambaVisionBackbone(VisionBackbone):
    """MambaVision backbone wrapper."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: Optional[int] = None,
        feature_stage: Optional[int] = None,
    ) -> None:
        canonical_id = _resolve_variant_id(vision_backbone_id)
        self.variant = MAMBAVISION_VARIANTS[canonical_id]
        self._feature_stage = feature_stage or self.variant.feature_stage
        self.cfg = _resolve_default_cfg(self.variant)

        input_size = self.cfg.get("input_size", (3, 224, 224))
        nominal_image_size = int(input_size[1]) if isinstance(input_size, (tuple, list)) else int(input_size)
        if default_image_size is None:
            default_image_size = nominal_image_size
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.dtype = torch.float32
        create_model, mv_module = _maybe_import_mambavision()
        self._mv_module = mv_module

        self.featurizer: nn.Module = create_model(self.variant.model_name, pretrained=False, num_classes=0)
        self.featurizer.eval()

        self._num_stages = len(getattr(self.featurizer, "levels", [])) + 1  # include patch_embed as stage 1
        self._feature_stage = self._resolve_feature_stage(self._feature_stage)
        self._embed_dim = self._infer_stage_embed_dim()

        self._grid_size = self._infer_grid_size(self.default_image_size)
        self._num_patches = self._grid_size * self._grid_size

        self._load_checkpoint()
        self.image_transform = self._build_image_transform()

    def _infer_grid_size(self, image_size: int) -> int:
        stride = self._infer_stage_stride()
        return max(1, int(math.ceil(image_size / stride)))

    def _build_image_transform(self) -> Compose:
        mean = tuple(self.cfg.get("mean", IMAGENET_DEFAULT_MEAN))
        std = tuple(self.cfg.get("std", IMAGENET_DEFAULT_STD))
        crop_pct_raw = self.cfg.get("crop_pct", 1.0)
        crop_pct = float(crop_pct_raw) if crop_pct_raw is not None else 1.0
        crop_mode = str(self.cfg.get("crop_mode", "center")).lower()
        interpolation = _resolve_interpolation(self.cfg.get("interpolation", "bicubic"))

        if self.image_resize_strategy in {"resize-crop", "mambavision-classification"}:
            if crop_mode == "squash":
                return Compose(
                    [
                        Resize((self.default_image_size, self.default_image_size), interpolation=interpolation),
                        ToTensor(),
                        Normalize(mean, std),
                    ]
                )
            resize_target = int(round(self.default_image_size / crop_pct))
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

    def _load_checkpoint(self) -> None:
        if not _use_pretrained_weights():
            overwatch.warning("MAMBAVISION_SKIP_PRETRAIN=1 -> loading MambaVision with random weights.")
            return
        ckpt_path = _ensure_checkpoint(self.cfg)
        if not ckpt_path.exists():
            overwatch.warning(
                f"MambaVision checkpoint not found at {ckpt_path}; proceeding without pretrained weights."
            )
            return
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("encoder."):
                key = key[len("encoder.") :]
            if key.startswith("head."):
                continue
            cleaned[key] = value

        incompatible = self.featurizer.load_state_dict(cleaned, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            overwatch.warning(
                "MambaVision checkpoint loaded with mismatched keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def get_fsdp_wrapping_policy(self) -> Callable:
        from torch.distributed.fsdp.wrap import _module_wrap_policy
        from functools import partial

        # Avoid wrapping the full MambaVision module because we call submodules directly
        # to keep spatial features; wrapping the root would leave params sharded.
        layer_cls = getattr(self._mv_module, "MambaVisionLayer", None)
        if layer_cls is not None:
            return partial(_module_wrap_policy, module_classes={layer_cls})
        block_cls = self._mv_module.Block
        conv_cls = self._mv_module.ConvBlock
        return partial(_module_wrap_policy, module_classes={block_cls, conv_cls})

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.featurizer.patch_embed(x)
        if self._feature_stage == 1:
            return x
        for idx, level in enumerate(self.featurizer.levels):
            x = level(x)
            if (idx + 2) == self._feature_stage:
                break
        if self._feature_stage == self._num_stages and hasattr(self.featurizer, "norm"):
            x = self.featurizer.norm(x)
        return x

    def _infer_stage_stride(self) -> int:
        stride = 4
        if self._feature_stage <= 1:
            return stride
        for idx, level in enumerate(self.featurizer.levels):
            if idx >= (self._feature_stage - 1):
                break
            if getattr(level, "downsample", None) is not None:
                stride *= 2
        return stride

    def _infer_stage_embed_dim(self) -> int:
        if self._feature_stage == 1:
            conv = getattr(self.featurizer.patch_embed, "conv_down", None)
            if conv is not None and len(conv) >= 4 and hasattr(conv[3], "out_channels"):
                return int(conv[3].out_channels)
        stage_idx = max(0, self._feature_stage - 2)
        if stage_idx < len(self.featurizer.levels):
            level = self.featurizer.levels[stage_idx]
            downsample = getattr(level, "downsample", None)
            reduction = getattr(downsample, "reduction", None) if downsample is not None else None
            if isinstance(reduction, nn.Sequential) and len(reduction) > 0:
                conv = reduction[0]
                if hasattr(conv, "out_channels"):
                    return int(conv.out_channels)
            if getattr(level, "blocks", None):
                block = level.blocks[0]
                if hasattr(block, "conv1"):
                    return int(block.conv1.out_channels)
                if hasattr(block, "norm1"):
                    shape = block.norm1.normalized_shape
                    if isinstance(shape, (tuple, list)):
                        return int(shape[0])
                    return int(shape)
        embed_dim = int(getattr(getattr(self.featurizer, "norm", None), "num_features", 0))
        if embed_dim == 0:
            embed_dim = int(getattr(self.featurizer, "num_features", 0))
        return embed_dim

    def _resolve_feature_stage(self, requested_stage: Optional[int]) -> int:
        stage = self._num_stages if requested_stage is None else requested_stage
        if stage < 1 or stage > self._num_stages:
            raise ValueError(
                f"MambaVision stage selection {stage} is out of range for {self._num_stages} stages (1-indexed)"
            )
        return stage

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(pixel_values, dict):
            raise ValueError("MambaVision expects a single tensor for `pixel_values`, not a dict.")
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 4 and pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        elif pixel_values.ndim != 4:
            raise ValueError(f"MambaVision expects pixel_values with 4 dims, got shape={tuple(pixel_values.shape)}")
        if pixel_values.dtype != self.dtype:
            pixel_values = pixel_values.to(dtype=self.dtype)

        features = self._forward_features(pixel_values.contiguous())
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
        return self._num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
