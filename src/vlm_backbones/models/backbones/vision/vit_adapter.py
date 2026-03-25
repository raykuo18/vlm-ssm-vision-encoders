"""
vit_adapter.py

ViT-Adapter ADE20K segmentation backbone wrapper.
"""

from __future__ import annotations

import os
import importlib
import re
import runpy
import sys
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.hub import download_url_to_file
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from vlm_backbones.cache import get_backbone_cache_dir
from vlm_backbones.models.backbones.vision.base_vision import ImageTransform, VisionBackbone
from vlm_backbones.overwatch import initialize_overwatch
from vlm_backbones.runtime_paths import get_third_party_root

PIL_RESAMPLING = getattr(Image, "Resampling", Image)

overwatch = initialize_overwatch(__name__)

VIT_ADAPTER_REPO_ROOT = get_third_party_root() / "ViT-Adapter"
VIT_ADAPTER_SEG_ROOT = VIT_ADAPTER_REPO_ROOT / "segmentation"
VIT_ADAPTER_DET_ROOT = VIT_ADAPTER_REPO_ROOT / "detection"

VIT_ADAPTER_CKPT_ROOT = Path(
    os.environ.get("VIT_ADAPTER_CKPT_ROOT", str(get_backbone_cache_dir("vit_adapter")))
)

VIT_ADAPTER_PREPROCESS_STRATEGY = "vit-adapter-segmentation"

MMSEG_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
MMSEG_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

VIT_ADAPTER_LEVEL_STRIDES = (4, 8, 16, 32)


def _use_pretrained_weights() -> bool:
    value = os.environ.get("VIT_ADAPTER_SKIP_PRETRAIN", "")
    return value.strip().lower() not in {"1", "true", "yes"}


@dataclass(frozen=True)
class ViTAdapterVariant:
    config_relpath: str
    checkpoints: Tuple[str, ...]
    preprocess: Dict[str, object]
    download_url: Optional[str] = None


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


class ResizeKeepRatioToFit:
    """Resize images while preserving aspect ratio to fit within (target_height, target_width)."""

    def __init__(self, target_height: int, target_width: int, interpolation: InterpolationMode) -> None:
        self.target_height = target_height
        self.target_width = target_width
        if isinstance(interpolation, InterpolationMode):
            self.interpolation = getattr(PIL_RESAMPLING, interpolation.name, PIL_RESAMPLING.BILINEAR)
        elif isinstance(interpolation, str):
            self.interpolation = getattr(PIL_RESAMPLING, interpolation.upper(), PIL_RESAMPLING.BILINEAR)
        else:
            self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == 0 or height == 0:
            return image
        scale = min(self.target_width / width, self.target_height / height)
        if scale >= 1.0 and width <= self.target_width and height <= self.target_height:
            return image
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), self.interpolation)


class PadToDivisible:
    """Pad PIL images so width/height are multiples of `divisor`."""

    def __init__(self, divisor: int, fill: Tuple[int, int, int]) -> None:
        self.divisor = divisor
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        pad_w = (self.divisor - (width % self.divisor)) % self.divisor
        pad_h = (self.divisor - (height % self.divisor)) % self.divisor
        if pad_w == 0 and pad_h == 0:
            return image
        padding = (0, 0, pad_w, pad_h)
        return ImageOps.expand(image, border=padding, fill=self.fill)


class PadToSize:
    """Pad PIL images to a fixed size, padding on the right/bottom."""

    def __init__(self, target_height: int, target_width: int, fill: Tuple[int, int, int]) -> None:
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        pad_w = max(0, self.target_width - width)
        pad_h = max(0, self.target_height - height)
        if pad_w == 0 and pad_h == 0:
            return image
        padding = (0, 0, pad_w, pad_h)
        return ImageOps.expand(image, border=padding, fill=self.fill)


def _seg_preprocess(long_edge: int, short_edge: int) -> Dict[str, object]:
    return {
        "type": "segmentation",
        "resize_scale": (long_edge, short_edge),
        "pad_divisor": 32,
        "pad_fill": tuple(int(x * 255) for x in MMSEG_MEAN),
        "mean": MMSEG_MEAN,
        "std": MMSEG_STD,
        "interpolation": InterpolationMode.BILINEAR,
        "nominal_resolution": (short_edge, long_edge),
    }


VIT_ADAPTER_VARIANTS: Dict[str, ViTAdapterVariant] = {
    # NOTE: ViT-Adapter-T ADE20K checkpoint link is broken upstream (404); entry is disabled for now.
    "vit-adapter-upernet-deit-s-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_deit_adapter_small_512_160k_ade20k.py",
        checkpoints=(
            "upernet_deit_adapter_small_512_160k_ade20k.pth",
            "upernet_deit_adapter_small_512_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
    ),
    "vit-adapter-upernet-deit-b-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_deit_adapter_base_512_160k_ade20k.py",
        checkpoints=(
            "upernet_deit_adapter_base_512_160k_ade20k.pth",
            "upernet_deit_adapter_base_512_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
        download_url="https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_base_512_160k_ade20k.pth.tar",
    ),
    "vit-adapter-upernet-augreg-t-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_augreg_adapter_tiny_512_160k_ade20k.py",
        checkpoints=(
            "upernet_augreg_adapter_tiny_512_160_ade20k.pth",
            "upernet_augreg_adapter_tiny_512_160_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
    ),
    "vit-adapter-upernet-augreg-b-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py",
        checkpoints=(
            "upernet_augreg_adapter_base_512_160k_ade20k.pth",
            "upernet_augreg_adapter_base_512_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
    ),
    "vit-adapter-upernet-augreg-l-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_augreg_adapter_large_512_160k_ade20k.py",
        checkpoints=(
            "upernet_augreg_adapter_large_512_160k_ade20k.pth",
            "upernet_augreg_adapter_large_512_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
    ),
    "vit-adapter-upernet-uniperceiver-l-ade20k-512": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k.py",
        checkpoints=(
            "upernet_uniperceiver_adapter_large_512_160k_ade20k.pth",
            "upernet_uniperceiver_adapter_large_512_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 512),
    ),
    "vit-adapter-upernet-beit-l-ade20k-640": ViTAdapterVariant(
        config_relpath="configs/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py",
        checkpoints=(
            "upernet_beit_adapter_large_640_160k_ade20k.pth",
            "upernet_beit_adapter_large_640_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 640),
    ),
    "vit-adapter-mask2former-beit-l-ade20k-640": ViTAdapterVariant(
        config_relpath="configs/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py",
        checkpoints=(
            "mask2former_beit_adapter_large_640_160k_ade20k.zip",
            "mask2former_beit_adapter_large_640_160k_ade20k.pth",
            "mask2former_beit_adapter_large_640_160k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(2048, 640),
    ),
    "vit-adapter-mask2former-beit-l-coco-ade20k-896": ViTAdapterVariant(
        config_relpath="configs/ade20k/mask2former_beit_adapter_large_896_80k_ade20k_ss.py",
        checkpoints=(
            "mask2former_beit_adapter_large_896_80k_ade20k.zip",
            "mask2former_beit_adapter_large_896_80k_ade20k.pth",
            "mask2former_beit_adapter_large_896_80k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(3584, 896),
    ),
    "vit-adapter-mask2former-beitv2-l-coco-ade20k-896": ViTAdapterVariant(
        config_relpath="configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py",
        checkpoints=(
            "mask2former_beitv2_adapter_large_896_80k_ade20k.zip",
            "mask2former_beitv2_adapter_large_896_80k_ade20k.pth",
            "mask2former_beitv2_adapter_large_896_80k_ade20k.pth.tar",
        ),
        preprocess=_seg_preprocess(3584, 896),
    ),
}


def get_registered_vit_adapter_variants() -> Tuple[str, ...]:
    return tuple(sorted(VIT_ADAPTER_VARIANTS.keys()))


def _ensure_vit_adapter_paths() -> None:
    if not VIT_ADAPTER_REPO_ROOT.exists():
        raise ImportError(
            f"ViT-Adapter repo not found at {VIT_ADAPTER_REPO_ROOT}. Run `git submodule update --init --recursive`."
        )
    for path in (VIT_ADAPTER_SEG_ROOT, VIT_ADAPTER_DET_ROOT):
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))

    try:
        import ops.modules  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime env issue
        raise ImportError(
            "ViT-Adapter deformable attention ops are missing. Build them via "
            "`scripts/env/build_vit_adapter.sh` (the `ops` symlink + make.sh step)."
        ) from exc


_BACKBONE_TYPE_MODULES: Dict[str, str] = {
    "ViTAdapter": "mmseg_custom.models.backbones.vit_adapter",
    "BEiTAdapter": "mmseg_custom.models.backbones.beit_adapter",
    "UniPerceiverAdapter": "mmseg_custom.models.backbones.uniperceiver_adapter",
    "ViTBaseline": "mmseg_custom.models.backbones.vit_baseline",
    "BEiTBaseline": "mmseg_custom.models.backbones.beit_baseline",
}


def _resolve_backbone_class(backbone_type: str):
    _ensure_vit_adapter_paths()
    module_path = _BACKBONE_TYPE_MODULES.get(backbone_type)
    if module_path is None:
        raise ValueError(
            f"Unsupported ViT-Adapter backbone type '{backbone_type}'. "
            f"Known types: {', '.join(sorted(_BACKBONE_TYPE_MODULES))}"
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - runtime env issue
        raise ImportError(
            "Unable to import ViT-Adapter backbone modules. "
            "Ensure third_party/ViT-Adapter/segmentation is available and that the "
            "deformable attention ops were built."
        ) from exc
    return getattr(module, backbone_type)


def _to_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _load_backbone_cfg(config_path: Path) -> Tuple[str, Dict[str, object]]:
    cfg = runpy.run_path(str(config_path))
    model_cfg = _to_dict(cfg.get("model"))
    if model_cfg is None:
        raise ValueError(f"Config {config_path} does not define a `model` dict")
    backbone_cfg = _to_dict(model_cfg.get("backbone"))
    if backbone_cfg is None:
        raise ValueError(f"Config {config_path} does not define `model.backbone`")
    backbone_cfg = dict(backbone_cfg)
    backbone_type = backbone_cfg.pop("type", None)
    if backbone_type is None:
        raise ValueError(f"Config {config_path} does not define `model.backbone.type`")
    backbone_cfg.pop("_delete_", None)
    backbone_cfg.pop("init_cfg", None)
    backbone_cfg.pop("pretrained", None)
    return backbone_type, backbone_cfg


def _checkpoint_prefix(name: str) -> str:
    for suffix in (".pth.tar", ".pth", ".pt", ".ckpt", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _maybe_extract_zip(path: Path) -> Optional[Path]:
    try:
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.endswith((".pth", ".pth.tar", ".pt", ".ckpt"))]
            if not members:
                return None
            zf.extractall(path.parent)
    except zipfile.BadZipFile:
        return None
    for member in members:
        candidate = path.parent / member
        if candidate.exists():
            return candidate
    return None


def _download_checkpoint(variant: ViTAdapterVariant) -> Optional[Path]:
    if not variant.download_url:
        return None

    target = VIT_ADAPTER_CKPT_ROOT / variant.checkpoints[0]
    if target.exists():
        return target
    if not _use_pretrained_weights():
        return target

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    if is_distributed and rank != 0:
        timeout = int(os.environ.get("VIT_ADAPTER_CKPT_WAIT_SECS", "900"))
        poll = float(os.environ.get("VIT_ADAPTER_CKPT_POLL_SECS", "2.0"))
        start = time.time()
        while time.time() - start < timeout:
            if target.exists():
                return target
            time.sleep(poll)
        raise FileNotFoundError(
            f"Timed out waiting for ViT-Adapter checkpoint at {target}. "
            "Ensure rank0 can download it or pre-populate VIT_ADAPTER_CKPT_ROOT."
        )

    tmp_path = target.with_name(f"{target.name}.partial")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    overwatch.info(f"Downloading ViT-Adapter checkpoint -> {target}")
    try:
        download_url_to_file(url=variant.download_url, dst=tmp_path, progress=True)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def _resolve_checkpoint(variant: ViTAdapterVariant) -> Path:
    VIT_ADAPTER_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = list(variant.checkpoints)
    for name in candidates:
        path = VIT_ADAPTER_CKPT_ROOT / name
        if path.exists():
            if path.suffix == ".zip":
                extracted = _maybe_extract_zip(path)
                if extracted is not None:
                    return extracted
            return path

    prefixes = {_checkpoint_prefix(name) for name in candidates}
    for prefix in prefixes:
        for ext in (".pth.tar", ".pth", ".pt", ".ckpt"):
            candidate = VIT_ADAPTER_CKPT_ROOT / f"{prefix}{ext}"
            if candidate.exists():
                return candidate

    if not _use_pretrained_weights():
        return VIT_ADAPTER_CKPT_ROOT / candidates[0]

    downloaded = _download_checkpoint(variant)
    if downloaded is not None and downloaded.exists():
        if downloaded.suffix == ".zip":
            extracted = _maybe_extract_zip(downloaded)
            if extracted is not None:
                return extracted
        return downloaded

    available = ", ".join(sorted(p.name for p in VIT_ADAPTER_CKPT_ROOT.glob("*") if p.is_file()))
    raise FileNotFoundError(
        "ViT-Adapter checkpoint not found. Looked for: "
        f"{', '.join(candidates)} under {VIT_ADAPTER_CKPT_ROOT}. "
        f"Available files: {available or 'none'}."
    )


@lru_cache(maxsize=None)
def _get_variant_spec(variant_id: str) -> Dict[str, object]:
    if variant_id not in VIT_ADAPTER_VARIANTS:
        raise ValueError(f"Unknown ViT-Adapter variant `{variant_id}`")
    meta = VIT_ADAPTER_VARIANTS[variant_id]
    config_path = VIT_ADAPTER_SEG_ROOT / meta.config_relpath
    if not config_path.exists():
        raise FileNotFoundError(f"ViT-Adapter config not found: {config_path}")
    backbone_type, backbone_cfg = _load_backbone_cfg(config_path)
    preprocess = dict(meta.preprocess)
    ckpt_path = _resolve_checkpoint(meta)
    return {
        "config_path": config_path,
        "checkpoint_path": ckpt_path,
        "backbone_type": backbone_type,
        "backbone_cfg": backbone_cfg,
        "preprocess": preprocess,
    }


class ViTAdapterBackbone(VisionBackbone):
    """Wrapper around ViT-Adapter segmentation backbones (ADE20K)."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: Optional[int] = None,
        feature_level: Optional[int] = None,
    ) -> None:
        spec = _get_variant_spec(vision_backbone_id)
        preprocess = dict(spec["preprocess"])
        nominal_height, nominal_width = preprocess.get("nominal_resolution", (512, 512))

        if default_image_size is not None:
            if image_resize_strategy == "letterbox":
                nominal_height, nominal_width = default_image_size, default_image_size
                preprocess["resize_scale"] = (default_image_size, default_image_size)
                preprocess["nominal_resolution"] = (default_image_size, default_image_size)
            else:
                short_side = min(nominal_height, nominal_width)
                long_side = max(nominal_height, nominal_width)
                scale = float(default_image_size) / float(short_side)
                scaled_short = default_image_size
                scaled_long = max(1, int(round(long_side * scale)))
                if nominal_height <= nominal_width:
                    nominal_height, nominal_width = scaled_short, scaled_long
                else:
                    nominal_height, nominal_width = scaled_long, scaled_short
                preprocess["resize_scale"] = (max(nominal_height, nominal_width), min(nominal_height, nominal_width))
                preprocess["nominal_resolution"] = (nominal_height, nominal_width)

        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=nominal_height)

        if image_resize_strategy not in {VIT_ADAPTER_PREPROCESS_STRATEGY, "letterbox"}:
            raise ValueError(
                f"ViT-Adapter variant '{vision_backbone_id}' requires image_resize_strategy="
                f"'{VIT_ADAPTER_PREPROCESS_STRATEGY}' or 'letterbox' (got '{image_resize_strategy}')."
            )

        backbone_cls = _resolve_backbone_class(spec["backbone_type"])
        backbone_cfg = dict(spec["backbone_cfg"])
        if default_image_size is not None:
            if nominal_height == nominal_width:
                backbone_cfg["img_size"] = nominal_height
                # ViT-Adapter uses `pretrain_size` to reshape positional embeddings at runtime.
                # Keep it aligned with the scaled square grid when profiling arbitrary resolutions.
                backbone_cfg["pretrain_size"] = nominal_height
            else:
                backbone_cfg["img_size"] = (nominal_height, nominal_width)
        self.featurizer = backbone_cls(**backbone_cfg)
        self._load_checkpoint(spec["checkpoint_path"])
        self.featurizer.eval()

        self.preprocess = preprocess
        self._nominal_input_size = (int(nominal_height), int(nominal_width))
        if self.image_resize_strategy == "letterbox":
            self._nominal_input_size = (int(nominal_height), int(nominal_height))
        self._feature_level = self._resolve_feature_level(feature_level)
        self._embed_dim = getattr(self.featurizer, "embed_dim", None)
        if self._embed_dim is None:
            raise ValueError("ViT-Adapter backbone could not determine embed_dim from the underlying model")
        self._num_patches = self._infer_num_patches()
        self.dtype = torch.float32

        self.image_transform = self._build_segmentation_transform()

        overwatch.info(
            f"ViT-Adapter feature level => index {self._feature_level} (stride {VIT_ADAPTER_LEVEL_STRIDES[self._feature_level]})",
            ctx_level=1,
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

        vit_cls = type(self.featurizer)
        policies = [partial(_module_wrap_policy, module_classes={vit_cls})]
        blocks = getattr(self.featurizer, "blocks", None)
        if blocks:
            block_cls = type(blocks[0])
            policies.append(partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls}))
        return partial(_or_policy, policies=policies)

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(pixel_values, dict):
            raise ValueError("ViT-Adapter expects a single tensor for `pixel_values`, not a dict.")
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 4 and pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        elif pixel_values.ndim != 4:
            raise ValueError(f"ViT-Adapter expects pixel_values with 4 dims, got shape={tuple(pixel_values.shape)}")

        param_dtype = next(self.featurizer.parameters()).dtype
        if pixel_values.dtype != param_dtype:
            pixel_values = pixel_values.to(dtype=param_dtype)
        outputs = self.featurizer(pixel_values.contiguous())
        if isinstance(outputs, (list, tuple)):
            features = outputs[self._feature_level]
        else:
            features = outputs

        if features.ndim == 4:
            if features.shape[1] != self._embed_dim and features.shape[-1] == self._embed_dim:
                features = features.permute(0, 3, 1, 2)
            b, c, h, w = features.shape
            features = features.reshape(b, c, h * w).transpose(1, 2)
        return features

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        height, width = self._nominal_input_size
        return (3, height, width)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_feature_level(self, requested_level: Optional[int]) -> int:
        level = len(VIT_ADAPTER_LEVEL_STRIDES) - 1 if requested_level is None else requested_level
        if level < 0:
            level = len(VIT_ADAPTER_LEVEL_STRIDES) + level
        if level < 0 or level >= len(VIT_ADAPTER_LEVEL_STRIDES):
            raise ValueError(
                f"Feature level {requested_level} is out of range for {len(VIT_ADAPTER_LEVEL_STRIDES)} outputs"
            )
        return level

    def _infer_num_patches(self) -> int:
        height, width = self._nominal_input_size
        stride = VIT_ADAPTER_LEVEL_STRIDES[self._feature_level]
        return (height // stride) * (width // stride)

    def _build_segmentation_transform(self) -> ImageTransform:
        scale = self.preprocess.get("resize_scale", (2048, 512))
        max_long, max_short = int(scale[0]), int(scale[1])
        pad_divisor = int(self.preprocess.get("pad_divisor", 32))
        pad_fill = tuple(self.preprocess.get("pad_fill", (0, 0, 0)))
        mean = tuple(self.preprocess.get("mean", MMSEG_MEAN))
        std = tuple(self.preprocess.get("std", MMSEG_STD))
        interpolation = self.preprocess.get("interpolation", InterpolationMode.BILINEAR)
        if self.image_resize_strategy == "letterbox":
            target_size = max_short
            return Compose([
                ResizeKeepRatioToFit(target_size, target_size, interpolation),
                PadToSize(target_size, target_size, pad_fill),
                ToTensor(),
                Normalize(mean, std),
            ])
        return Compose([
            ResizeKeepRatioMax(max_long, max_short, interpolation),
            PadToDivisible(pad_divisor, pad_fill),
            ToTensor(),
            Normalize(mean, std),
        ])

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        if not _use_pretrained_weights():
            overwatch.warning("VIT_ADAPTER_SKIP_PRETRAIN=1 -> loading ViT-Adapter with random weights.")
            return
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                weights = state["model"]
            elif "state_dict" in state:
                weights = state["state_dict"]
            else:
                weights = state
        else:
            weights = state

        cleaned = {}
        for key, value in weights.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                cleaned[key[len("backbone."):]] = value
        if cleaned:
            weights = cleaned

        incompatible = self.featurizer.load_state_dict(weights, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            gamma_only = False
            if missing and not unexpected:
                gamma_only = all(re.match(r"^blocks\\.(\\d+)\\.gamma[12]$", key) for key in missing)
            if gamma_only:
                overwatch.info(
                    "ViT-Adapter checkpoint missing LayerScale gamma parameters; using defaults.",
                    ctx_level=1,
                )
            else:
                overwatch.warning(
                    "ViT-Adapter checkpoint loaded with mismatched keys: "
                    f"missing={missing}, unexpected={unexpected}"
                )
