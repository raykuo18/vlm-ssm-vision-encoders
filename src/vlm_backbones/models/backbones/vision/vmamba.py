"""VMamba backbone integration for Prismatic VLMs."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageOps

PIL_RESAMPLING = getattr(Image, "Resampling", Image)
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from torch.hub import download_url_to_file

from vlm_backbones.cache import get_backbone_cache_dir
from vlm_backbones.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone
from vlm_backbones.overwatch import initialize_overwatch
from vlm_backbones.runtime_paths import get_third_party_root

try:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("VMamba backbones require timm; install it in the vmamba env.") from exc

# Initialize logger
overwatch = initialize_overwatch(__name__)

# Default locations
VMAMBA_REPO_ROOT = get_third_party_root() / "VMamba"
VMAMBA_CONF_ROOT = Path(__file__).resolve().parent / "vmamba_conf"
VMAMBA_CONF_ROOT.mkdir(parents=True, exist_ok=True)
VMAMBA_CKPT_ROOT = Path(os.environ.get("VMAMBA_CKPT_ROOT", str(get_backbone_cache_dir("vmamba"))))

# Preprocessing defaults
IMAGENET_MEAN = tuple(IMAGENET_DEFAULT_MEAN)
IMAGENET_STD = tuple(IMAGENET_DEFAULT_STD)
MMDET_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
MMDET_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)

CLASSIFICATION_PREPROCESS = {
    "type": "classification",
    "img_size": 224,
    "interpolation": InterpolationMode.BICUBIC,
    "crop_pct": 224 / 256,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
    "nominal_resolution": (224, 224),
}
DETECTION_PREPROCESS = {
    "type": "detection",
    "resize_scale": (1333, 800),  # (max_long_edge, max_short_edge)
    "interpolation": InterpolationMode.BILINEAR,
    "pad_divisor": 32,
    "pad_fill": tuple(int(x * 255) for x in MMDET_MEAN),
    "mean": MMDET_MEAN,
    "std": MMDET_STD,
    "nominal_resolution": (800, 1333),
}
SEGMENTATION_PREPROCESS = {
    "type": "segmentation",
    "resize_scale": (2048, 512),
    "interpolation": InterpolationMode.BILINEAR,
    "pad_divisor": 32,
    "pad_fill": tuple(int(x * 255) for x in MMDET_MEAN),
    "mean": MMDET_MEAN,
    "std": MMDET_STD,
    "nominal_resolution": (512, 2048),
}

VMAMBA_PREPROCESS_STRATEGIES = {
    "classification": "vmamba-classification",
    "detection": "vmamba-detection",
    "segmentation": "vmamba-segmentation",
}
VMAMBA_DETECTION_LETTERBOX_SQUARE_STRATEGIES = {
    "letterbox-square-224": 224,
    "letterbox-square-256": 256,
    "letterbox-square-512": 512,
    "letterbox-square-1024": 1024,
}
# Classification variants (9 checkpoints)
_CLASSIFICATION_VARIANTS = [
    (
        "vmamba-tiny-vanilla",
        "classification/configs/vssm/vmambav0_tiny_224.yaml",
        "vssmtiny_dp01_ckpt_epoch_292.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth",
    ),
    (
        "vmamba-small-vanilla",
        "classification/configs/vssm/vmambav0_small_224.yaml",
        "vssmsmall_dp03_ckpt_epoch_238.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmsmall_dp03_ckpt_epoch_238.pth",
    ),
    (
        "vmamba-base-vanilla",
        "classification/configs/vssm/vmambav0_base_224.yaml",
        "vssmbase_dp06_ckpt_epoch_241.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmbase_dp06_ckpt_epoch_241.pth",
    ),
    (
        "vmamba-tiny-s2l5",
        "classification/configs/vssm/vmambav2_tiny_224.yaml",
        "vssm_tiny_0230_ckpt_epoch_262.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_tiny_0230_ckpt_epoch_262.pth",
    ),
    (
        "vmamba-small-s2l15",
        "classification/configs/vssm/vmambav2_small_224.yaml",
        "vssm_small_0229_ckpt_epoch_222.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth",
    ),
    (
        "vmamba-base-s2l15",
        "classification/configs/vssm/vmambav2_base_224.yaml",
        "vssm_base_0229_ckpt_epoch_237.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth",
    ),
    (
        "vmamba-tiny-s1l8",
        "classification/configs/vssm/vmambav2v_tiny_224.yaml",
        "vssm1_tiny_0230s_ckpt_epoch_264.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    ),
    (
        "vmamba-small-s1l20",
        "classification/configs/vssm/vmambav2v_small_224.yaml",
        "vssm1_small_0229s_ckpt_epoch_240.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_small_0229s_ckpt_epoch_240.pth",
    ),
    (
        "vmamba-base-s1l20",
        "classification/configs/vssm/vmambav2v_base_224.yaml",
        "vssm1_base_0229s_ckpt_epoch_225.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_base_0229s_ckpt_epoch_225.pth",
    ),
]

# Detection variants (13 checkpoints)
_DETECTION_VARIANTS = [
    (
        "vmamba-tiny-vanilla-det-maskrcnn-1x",
        "detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py",
        "vssmtiny_mask_rcnn_swin_fpn_coco_epoch_12.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0det/vssmtiny_mask_rcnn_swin_fpn_coco_epoch_12.pth",
    ),
    (
        "vmamba-small-vanilla-det-maskrcnn-1x",
        "detection/configs/vssm/mask_rcnn_vssm_fpn_coco_small.py",
        "vssmsmall_mask_rcnn_swin_fpn_coco_epoch_12.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0det/vssmsmall_mask_rcnn_swin_fpn_coco_epoch_12.pth",
    ),
    (
        "vmamba-base-vanilla-det-maskrcnn-1x",
        "detection/configs/vssm/mask_rcnn_vssm_fpn_coco_base.py",
        "vssmbase_mask_rcnn_swin_fpn_coco_epoch_12.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0det/vssmbase_mask_rcnn_swin_fpn_coco_epoch_12.pth",
    ),
    (
        "vmamba-tiny-s2l5-det-maskrcnn-1x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny1.py",
        "mask_rcnn_vssm_fpn_coco_tiny_epoch_12.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_tiny_epoch_12.pth",
    ),
    (
        "vmamba-small-s2l15-det-maskrcnn-1x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small.py",
        "mask_rcnn_vssm_fpn_coco_small_epoch_11.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_small_epoch_11.pth",
    ),
    (
        "vmamba-base-s2l15-det-maskrcnn-1x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py",
        "mask_rcnn_vssm_fpn_coco_base_epoch_11.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_base_epoch_11.pth",
    ),
    (
        "vmamba-base-s2l15-det-maskrcnn-1x-bs8",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py",
        "mask_rcnn_vssm_fpn_coco_base_epoch_12_bs8.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_base_epoch_12_bs8.pth",
    ),
    (
        "vmamba-tiny-s1l8-det-maskrcnn-1x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py",
        "mask_rcnn_vssm_fpn_coco_tiny_s_epoch_12.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_tiny_s_epoch_12.pth",
    ),
    (
        "vmamba-tiny-vanilla-det-maskrcnn-3x",
        "detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py",
        "vssmtiny_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0det/vssmtiny_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth",
    ),
    (
        "vmamba-small-vanilla-det-maskrcnn-3x",
        "detection/configs/vssm/mask_rcnn_vssm_fpn_coco_small_ms_3x.py",
        "vssmsmall_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0det/vssmsmall_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth",
    ),
    (
        "vmamba-tiny-s2l5-det-maskrcnn-3x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny1_ms_3x.py",
        "mask_rcnn_vssm_fpn_coco_tiny_ms_3x_epoch_36.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_epoch_36.pth",
    ),
    (
        "vmamba-small-s2l15-det-maskrcnn-3x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small_ms_3x.py",
        "mask_rcnn_vssm_fpn_coco_small_ms_3x_epoch_32.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_small_ms_3x_epoch_32.pth",
    ),
    (
        "vmamba-tiny-s1l8-det-maskrcnn-3x",
        "detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py",
        "mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s_epoch_31.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2det/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s_epoch_31.pth",
    ),
]

# Segmentation variants (7 checkpoints)
_SEGMENTATION_VARIANTS = [
    (
        "vmamba-tiny-vanilla-seg-ade20k",
        "segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py",
        "vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0seg/vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
    ),
    (
        "vmamba-small-vanilla-seg-ade20k",
        "segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py",
        "vssmsmall_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0seg/vssmsmall_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
    ),
    (
        "vmamba-base-vanilla-seg-ade20k",
        "segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py",
        "vssmbase_upernet_4xb4-160k_ade20k-512x512_iter_128000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v0seg/vssmbase_upernet_4xb4-160k_ade20k-512x512_iter_128000.pth",
    ),
    (
        "vmamba-tiny-s2l5-seg-ade20k",
        "segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny1.py",
        "upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2seg/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth",
    ),
    (
        "vmamba-small-s2l15-seg-ade20k",
        "segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py",
        "upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2seg/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth",
    ),
    (
        "vmamba-base-s2l15-seg-ade20k",
        "segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py",
        "upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2seg/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
    ),
    (
        "vmamba-tiny-s1l8-seg-ade20k",
        "segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py",
        "upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth",
        "https://github.com/MzeroMiko/VMamba/releases/download/%23v2seg/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth",
    ),
]

ARCH_BASE_MAP = {
    # Detection (Mask R-CNN)
    "vmamba-tiny-vanilla-det-maskrcnn-1x": "vmamba-tiny-vanilla",
    "vmamba-small-vanilla-det-maskrcnn-1x": "vmamba-small-vanilla",
    "vmamba-base-vanilla-det-maskrcnn-1x": "vmamba-base-vanilla",
    "vmamba-tiny-vanilla-det-maskrcnn-3x": "vmamba-tiny-vanilla",
    "vmamba-small-vanilla-det-maskrcnn-3x": "vmamba-small-vanilla",
    "vmamba-tiny-s2l5-det-maskrcnn-1x": "vmamba-tiny-s2l5",
    "vmamba-tiny-s2l5-det-maskrcnn-3x": "vmamba-tiny-s2l5",
    "vmamba-tiny-s1l8-det-maskrcnn-1x": "vmamba-tiny-s1l8",
    "vmamba-tiny-s1l8-det-maskrcnn-3x": "vmamba-tiny-s1l8",
    "vmamba-small-s2l15-det-maskrcnn-1x": "vmamba-small-s2l15",
    "vmamba-small-s2l15-det-maskrcnn-3x": "vmamba-small-s2l15",
    "vmamba-base-s2l15-det-maskrcnn-1x": "vmamba-base-s2l15",
    "vmamba-base-s2l15-det-maskrcnn-1x-bs8": "vmamba-base-s2l15",
    # Segmentation (ADE20K)
    "vmamba-tiny-vanilla-seg-ade20k": "vmamba-tiny-vanilla",
    "vmamba-small-vanilla-seg-ade20k": "vmamba-small-vanilla",
    "vmamba-base-vanilla-seg-ade20k": "vmamba-base-vanilla",
    "vmamba-tiny-s2l5-seg-ade20k": "vmamba-tiny-s2l5",
    "vmamba-tiny-s1l8-seg-ade20k": "vmamba-tiny-s1l8",
    "vmamba-small-s2l15-seg-ade20k": "vmamba-small-s2l15",
    "vmamba-base-s2l15-seg-ade20k": "vmamba-base-s2l15",
}

def _build_metadata() -> Dict[str, Dict[str, object]]:
    meta: Dict[str, Dict[str, object]] = {}
    for variant, config_relpath, filename, url in _CLASSIFICATION_VARIANTS:
        meta[variant] = {
            "family": "classification",
            "config_relpath": config_relpath,
            "checkpoint_filename": filename,
            "checkpoint_url": url,
            "preprocess": dict(CLASSIFICATION_PREPROCESS),
        }
    for variant, config_relpath, filename, url in _DETECTION_VARIANTS:
        meta[variant] = {
            "family": "detection",
            "config_relpath": config_relpath,
            "checkpoint_filename": filename,
            "checkpoint_url": url,
            "preprocess": {
                **DETECTION_PREPROCESS,
                "nominal_resolution": (800, 1333),
            },
        }
    for variant, config_relpath, filename, url in _SEGMENTATION_VARIANTS:
        meta[variant] = {
            "family": "segmentation",
            "config_relpath": config_relpath,
            "checkpoint_filename": filename,
            "checkpoint_url": url,
            "preprocess": dict(SEGMENTATION_PREPROCESS),
        }
    for variant, base in ARCH_BASE_MAP.items():
        if variant in meta:
            meta[variant]["arch_base"] = base
    return meta


VMAMBA_VARIANT_METADATA = _build_metadata()
REGISTERED_VMAMBA_VARIANTS = tuple(VMAMBA_VARIANT_METADATA.keys())
VMAMBA_VARIANTS = VMAMBA_VARIANT_METADATA  # Backwards compatibility alias


VMAMBA_ALIASES = {
    "vmamba-tiny": "vmamba-tiny-s1l8",
    "vmamba-small": "vmamba-small-s2l15",
    "vmamba-base": "vmamba-base-s2l15",
}


def get_registered_vmamba_variants() -> Tuple[str, ...]:
    """Return all canonical VMamba backbone identifiers."""
    return REGISTERED_VMAMBA_VARIANTS


def get_vmamba_variant_spec(vision_backbone_id: str) -> Dict[str, object]:
    """Return the resolved variant spec (including model kwargs) for a VMamba backbone."""
    canonical = _resolve_variant_id(vision_backbone_id)
    return _get_variant_spec(canonical).copy()

def _resolve_variant_id(vision_backbone_id: str) -> str:
    canonical = VMAMBA_ALIASES.get(vision_backbone_id, vision_backbone_id)
    if canonical not in VMAMBA_VARIANT_METADATA:
        raise ValueError(
            f"Unknown VMamba variant '{vision_backbone_id}'. Supported variants: {sorted(VMAMBA_VARIANT_METADATA)}"
        )
    return canonical


def _copy_vmamba_config(rel_path: str) -> Path:
    source_path = VMAMBA_REPO_ROOT / rel_path
    if not source_path.exists():
        raise FileNotFoundError(f"VMamba config missing: {source_path}")
    dest_path = VMAMBA_CONF_ROOT / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    needs_copy = not dest_path.exists() or source_path.stat().st_mtime > dest_path.stat().st_mtime
    if needs_copy:
        shutil.copy2(source_path, dest_path)
    return dest_path


def _ensure_checkpoint(meta: Dict[str, object]) -> Path:
    filename = str(meta["checkpoint_filename"])
    family = str(meta["family"])
    target = VMAMBA_CKPT_ROOT / family / filename
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        overwatch.info(f"Downloading VMamba checkpoint {filename} to {target}")
        download_url_to_file(str(meta["checkpoint_url"]), str(target), progress=True)
    return target


@lru_cache(maxsize=None)
def _load_model_kwargs_from_config(config_path: Path) -> Dict[str, object]:
    if config_path.suffix in {".yml", ".yaml"}:
        return _parse_classification_config(config_path)
    if config_path.suffix == ".py":
        return _parse_python_backbone_config(config_path)
    raise ValueError(f"Unsupported VMamba config format: {config_path}")


def _parse_classification_config(config_path: Path) -> Dict[str, object]:
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg.get("MODEL", {})
    vssm_cfg = model_cfg.get("VSSM", {})
    defaults = {
        "PATCH_SIZE": 4,
        "IN_CHANS": 3,
        "DEPTHS": [2, 2, 9, 2],
        "EMBED_DIM": 96,
        "SSM_D_STATE": 16,
        "SSM_RATIO": 2.0,
        "SSM_RANK_RATIO": 2.0,
        "SSM_DT_RANK": "auto",
        "SSM_ACT_LAYER": "silu",
        "SSM_CONV": 3,
        "SSM_CONV_BIAS": True,
        "SSM_DROP_RATE": 0.0,
        "SSM_INIT": "v0",
        "SSM_FORWARDTYPE": "v2",
        "MLP_RATIO": 4.0,
        "MLP_ACT_LAYER": "gelu",
        "MLP_DROP_RATE": 0.0,
        "PATCH_NORM": True,
        "NORM_LAYER": "ln",
        "DOWNSAMPLE": "v2",
        "PATCHEMBED": "v2",
        "POSEMBED": False,
        "GMLP": False,
        "USE_CHECKPOINT": False,
    }
    params = {**defaults, **{k.upper(): v for k, v in vssm_cfg.items()}}
    kwargs = {
        "patch_size": int(params["PATCH_SIZE"]),
        "in_chans": int(params["IN_CHANS"]),
        "depths": list(params["DEPTHS"]),
        "dims": params["EMBED_DIM"],
        "ssm_d_state": params["SSM_D_STATE"],
        "ssm_ratio": params.get("SSM_RATIO", params.get("SSM_RANK_RATIO", 2.0)),
        "ssm_dt_rank": params["SSM_DT_RANK"],
        "ssm_act_layer": params.get("SSM_ACT_LAYER", "silu"),
        "ssm_conv": params.get("SSM_CONV", 3),
        "ssm_conv_bias": params.get("SSM_CONV_BIAS", False),
        "ssm_drop_rate": params.get("SSM_DROP_RATE", 0.0),
        "ssm_init": params.get("SSM_INIT", "v0"),
        "forward_type": params.get("SSM_FORWARDTYPE", "v05_noz"),
        "mlp_ratio": params.get("MLP_RATIO", 4.0),
        "mlp_act_layer": params.get("MLP_ACT_LAYER", "gelu"),
        "mlp_drop_rate": params.get("MLP_DROP_RATE", 0.0),
        "patch_norm": params.get("PATCH_NORM", True),
        "norm_layer": params.get("NORM_LAYER", "ln"),
        "downsample_version": params.get("DOWNSAMPLE", "v2"),
        "patchembed_version": params.get("PATCHEMBED", "v2"),
        "posembed": params.get("POSEMBED", False),
        "gmlp": params.get("GMLP", False),
        "use_checkpoint": params.get("USE_CHECKPOINT", False),
    }
    return kwargs


def _parse_python_backbone_config(config_path: Path) -> Dict[str, object]:
    namespace = runpy.run_path(str(config_path))
    model_cfg = namespace.get("model")
    if not model_cfg or "backbone" not in model_cfg:
        raise ValueError(f"No backbone definition found in {config_path}")
    backbone = model_cfg["backbone"]
    return _normalize_backbone_kwargs(backbone)


def _normalize_backbone_kwargs(backbone: Dict[str, object]) -> Dict[str, object]:
    kwargs = backbone.copy()
    mapping = {
        "dims": "dims",
        "embed_dim": "dims",
        "depths": "depths",
        "ssm_d_state": "ssm_d_state",
        "ssm_dt_rank": "ssm_dt_rank",
        "ssm_ratio": "ssm_ratio",
        "ssm_conv": "ssm_conv",
        "ssm_conv_bias": "ssm_conv_bias",
        "ssm_drop_rate": "ssm_drop_rate",
        "ssm_init": "ssm_init",
        "forward_type": "forward_type",
        "mlp_ratio": "mlp_ratio",
        "downsample_version": "downsample_version",
        "patchembed_version": "patchembed_version",
        "drop_path_rate": "drop_path_rate",
        "norm_layer": "norm_layer",
        "patch_norm": "patch_norm",
        "posembed": "posembed",
        "gmlp": "gmlp",
        "use_checkpoint": "use_checkpoint",
        "mlp_act_layer": "mlp_act_layer",
        "ssm_act_layer": "ssm_act_layer",
    }
    normalized: Dict[str, object] = {}
    for src, dst in mapping.items():
        if src in kwargs:
            normalized[dst] = kwargs[src]
    normalized.setdefault("ssm_act_layer", "silu")
    normalized.setdefault("mlp_act_layer", "gelu")
    normalized.setdefault("patch_norm", True)
    normalized.setdefault("posembed", False)
    normalized.setdefault("gmlp", False)
    normalized.setdefault("use_checkpoint", False)
    normalized.setdefault("ssm_conv_bias", False)
    normalized.setdefault("ssm_ratio", 2.0)
    normalized.setdefault("ssm_dt_rank", "auto")
    normalized.setdefault("forward_type", "v05_noz")
    normalized.setdefault("downsample_version", "v2")
    normalized.setdefault("patchembed_version", "v2")
    normalized.setdefault("norm_layer", "ln2d")
    if "depths" in normalized:
        normalized["depths"] = list(normalized["depths"])
    return normalized

def _maybe_import_vmamba() -> Tuple[object, Path]:
    if not VMAMBA_REPO_ROOT.exists():
        raise ImportError(
            f"VMamba repo not found at {VMAMBA_REPO_ROOT}. Run `git submodule update --init --recursive third_party/VMamba`."
        )
    if str(VMAMBA_REPO_ROOT) not in sys.path:
        sys.path.append(str(VMAMBA_REPO_ROOT))
    try:
        import vmamba as vmamba_pkg  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime env issue
        raise ImportError(
            "Unable to import VMamba. Activate the VMamba environment or rerun `scripts/env/build_vmamba.sh`."
        ) from exc
    return vmamba_pkg, VMAMBA_REPO_ROOT

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

    def __call__(self, image):
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

    def __call__(self, image):
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

    def __call__(self, image):
        width, height = image.size
        pad_w = (self.divisor - (width % self.divisor)) % self.divisor
        pad_h = (self.divisor - (height % self.divisor)) % self.divisor
        if pad_w == 0 and pad_h == 0:
            return image
        padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
        return ImageOps.expand(image, border=padding, fill=self.fill)


class PadToSize:
    """Pad PIL images to a fixed size, padding on the right/bottom."""

    def __init__(self, target_height: int, target_width: int, fill: Tuple[int, int, int]) -> None:
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        pad_w = max(0, self.target_width - width)
        pad_h = max(0, self.target_height - height)
        if pad_w == 0 and pad_h == 0:
            return image
        padding = (0, 0, pad_w, pad_h)
        return ImageOps.expand(image, border=padding, fill=self.fill)

@lru_cache(maxsize=None)
def _get_variant_spec(canonical_id: str) -> Dict[str, object]:
    meta = VMAMBA_VARIANT_METADATA[canonical_id].copy()
    config_relpath = Path(meta["config_relpath"])
    source_config_path = VMAMBA_REPO_ROOT / config_relpath
    if not source_config_path.exists():
        raise FileNotFoundError(f"VMamba config not found: {source_config_path}")
    local_config_path = _copy_vmamba_config(str(config_relpath))
    checkpoint_path = _ensure_checkpoint(meta)
    arch_base = meta.get("arch_base")
    if arch_base:
        base_spec = _get_variant_spec(arch_base)
        model_kwargs = dict(base_spec["model_kwargs"])
    else:
        model_kwargs = _load_model_kwargs_from_config(source_config_path)
    preprocess = meta["preprocess"].copy()
    if "nominal_resolution" not in preprocess:
        preprocess["nominal_resolution"] = preprocess.get("nominal_resolution", (preprocess.get("img_size", 224),) * 2)
    spec = {
        "family": meta["family"],
        "config_path": local_config_path,
        "source_config_path": source_config_path,
        "checkpoint_path": checkpoint_path,
        "model_kwargs": model_kwargs,
        "preprocess": preprocess,
    }
    return spec

class VMambaBackbone(VisionBackbone):
    """Wrapper around VMamba feature extractor for downstream VLM training."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        feature_stage: Optional[int] = None,
        feature_layer: Optional[int] = None,
    ) -> None:
        canonical_id = _resolve_variant_id(vision_backbone_id)
        spec = _get_variant_spec(canonical_id)
        preprocess = spec["preprocess"]
        nominal_height, nominal_width = preprocess.get("nominal_resolution", (default_image_size, default_image_size))
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=nominal_height)
        self.variant_spec = spec
        self.preprocess = preprocess
        preprocess_type = preprocess.get("type")
        expected_strategy = VMAMBA_PREPROCESS_STRATEGIES.get(preprocess_type)
        allowed_strategies = {expected_strategy} if expected_strategy is not None else set()
        if preprocess_type in {"classification", "detection", "segmentation"}:
            allowed_strategies.add("letterbox")
        if preprocess_type in {"classification", "detection"}:
            allowed_strategies.update(VMAMBA_DETECTION_LETTERBOX_SQUARE_STRATEGIES.keys())
        if allowed_strategies and self.image_resize_strategy not in allowed_strategies:
            raise ValueError(
                f"VMamba variant '{canonical_id}' expects image_resize_strategy in {sorted(allowed_strategies)} "
                f"to preserve its upstream preprocessing (got '{self.image_resize_strategy}'). "
                "Update the model dataclass or CLI flags to use a supported VMamba strategy."
            )
        self._nominal_input_size = (int(nominal_height), int(nominal_width))
        if self.image_resize_strategy == "letterbox" and preprocess_type == "segmentation":
            self._nominal_input_size = (int(nominal_height), int(nominal_height))
        if preprocess_type in {"classification", "detection"}:
            square_size = VMAMBA_DETECTION_LETTERBOX_SQUARE_STRATEGIES.get(self.image_resize_strategy)
            if square_size is not None:
                self._nominal_input_size = (square_size, square_size)

        vmamba_pkg, _ = _maybe_import_vmamba()
        model_kwargs = dict(spec["model_kwargs"])
        model_kwargs.setdefault("imgsize", preprocess.get("img_size", nominal_height))

        self.stage_depths = list(model_kwargs.get("depths", []))
        num_stages = len(self.stage_depths)
        if num_stages == 0:
            raise ValueError("VMamba configuration must specify `depths` per stage")

        self._feature_stage_index = self._resolve_stage_index(feature_stage, num_stages)
        self._feature_layer_index = self._resolve_layer_index(self._feature_stage_index, feature_layer)
        stage_depth = self.stage_depths[self._feature_stage_index]
        self._direct_stage_capture = self._feature_layer_index == (stage_depth - 1)

        out_indices = (
            (self._feature_stage_index,)
            if self._direct_stage_capture
            else tuple(range(num_stages))
        )
        self.featurizer: nn.Module = vmamba_pkg.Backbone_VSSM(
            out_indices=out_indices,
            pretrained=None,
            **model_kwargs,
        )
        self._load_checkpoint()
        self.featurizer.eval()

        self.image_transform = self._build_image_transform()
        self._embed_dim = self._infer_embed_dim()
        self._num_patches = self._infer_num_patches(model_kwargs.get("patch_size", 4))
        self.dtype = torch.float32  # selective-scan kernels are most stable in fp32

        overwatch.info(
            f"VMamba feature tap => stage {self._feature_stage_index + 1}, layer {self._feature_layer_index}",
            ctx_level=1,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._forward_features(pixel_values)

    def get_fsdp_wrapping_policy(self) -> Callable:
        vmamba_pkg, _ = _maybe_import_vmamba()
        vss_block = vmamba_pkg.VSSBlock
        from torch.distributed.fsdp.wrap import _module_wrap_policy

        def policy(module, recurse, nonwrapped_numel):
            return _module_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                module_classes={vss_block},
            )

        return policy

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
        return torch.float32

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._direct_stage_capture:
            outputs = self.featurizer(pixel_values)
            feat = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        else:
            feat = self._select_feature_map(pixel_values)
        if feat.ndim != 4:
            raise ValueError("VMamba backbone expected channel-first feature maps")
        b, c, h, w = feat.shape
        feat = feat.reshape(b, c, h * w).transpose(1, 2)
        return feat

    def _select_feature_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        module = self.featurizer
        if self._direct_stage_capture:
            raise RuntimeError("_select_feature_map should not be called when direct stage capture is enabled")
        target_stage = self._feature_stage_index
        target_layer = self._feature_layer_index
        x = module.patch_embed(pixel_values)
        selected = None

        for stage_idx, layer in enumerate(module.layers):
            captured = False
            for block_idx, block in enumerate(layer.blocks):
                x = block(x)
                if stage_idx == target_stage and block_idx == target_layer:
                    selected = x
                    captured = True
                    break

            if captured and stage_idx == target_stage:
                break

            x = layer.downsample(x)

        if selected is None:
            raise RuntimeError(
                f"Unable to select VMamba feature map (stage={target_stage + 1}, layer={target_layer});"
                " check the requested indices."
            )

        norm_layer = getattr(module, f"outnorm{target_stage}")
        selected = norm_layer(selected)
        if not module.channel_first:
            selected = selected.permute(0, 3, 1, 2)
        return selected.contiguous()

    def _build_image_transform(self) -> ImageTransform:
        ptype = self.preprocess["type"]
        if ptype == "classification":
            return self._build_classification_transform()
        if ptype == "detection":
            return self._build_detection_transform()
        if ptype == "segmentation":
            return self._build_segmentation_transform()
        raise ValueError(f"Unsupported VMamba preprocess type: {ptype}")

    def _build_classification_transform(self) -> ImageTransform:
        size = int(self.preprocess.get("img_size", 224))
        interpolation = self.preprocess.get("interpolation", InterpolationMode.BICUBIC)
        crop_pct = float(self.preprocess.get("crop_pct", 224 / 256))
        resize_target = int(round(size / crop_pct))
        square_size = VMAMBA_DETECTION_LETTERBOX_SQUARE_STRATEGIES.get(self.image_resize_strategy)
        if square_size is not None:
            pad_fill = tuple(int(x * 255) for x in IMAGENET_MEAN)
            return Compose([
                ResizeKeepRatioToFit(square_size, square_size, interpolation),
                PadToSize(square_size, square_size, pad_fill),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        if self.image_resize_strategy == VMAMBA_PREPROCESS_STRATEGIES["classification"]:
            return Compose([
                Resize(resize_target, interpolation=interpolation),
                CenterCrop(size),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        if self.image_resize_strategy == "letterbox":
            return Compose([
                LetterboxPad(tuple(int(x * 255) for x in IMAGENET_MEAN)),
                Resize((size, size), interpolation=interpolation),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        if self.image_resize_strategy == "resize-naive":
            return Compose([
                Resize((size, size), interpolation=interpolation),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return Compose([
            Resize(resize_target, interpolation=interpolation),
            CenterCrop(size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _build_detection_transform(self) -> ImageTransform:
        scale = self.preprocess.get("resize_scale", (1333, 800))
        max_long, max_short = int(scale[0]), int(scale[1])
        pad_divisor = int(self.preprocess.get("pad_divisor", 32))
        pad_fill = tuple(self.preprocess.get("pad_fill", (0, 0, 0)))
        mean = tuple(self.preprocess.get("mean", MMDET_MEAN))
        std = tuple(self.preprocess.get("std", MMDET_STD))
        interpolation = self.preprocess.get("interpolation", InterpolationMode.BILINEAR)
        square_size = VMAMBA_DETECTION_LETTERBOX_SQUARE_STRATEGIES.get(self.image_resize_strategy)
        if square_size is not None:
            return Compose([
                ResizeKeepRatioToFit(square_size, square_size, interpolation),
                PadToSize(square_size, square_size, pad_fill),
                ToTensor(),
                Normalize(mean, std),
            ])
        if self.image_resize_strategy == "letterbox":
            target_height = max_short
            target_width = max_long
            return Compose([
                ResizeKeepRatioToFit(target_height, target_width, interpolation),
                PadToSize(target_height, target_width, pad_fill),
                ToTensor(),
                Normalize(mean, std),
            ])
        return Compose([
            ResizeKeepRatioMax(max_long, max_short, interpolation),
            PadToDivisible(pad_divisor, pad_fill),
            ToTensor(),
            Normalize(mean, std),
        ])

    def _build_segmentation_transform(self) -> ImageTransform:
        scale = self.preprocess.get("resize_scale", (2048, 512))
        max_long, max_short = int(scale[0]), int(scale[1])
        pad_divisor = int(self.preprocess.get("pad_divisor", 32))
        pad_fill = tuple(self.preprocess.get("pad_fill", (0, 0, 0)))
        mean = tuple(self.preprocess.get("mean", MMDET_MEAN))
        std = tuple(self.preprocess.get("std", MMDET_STD))
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

    def _load_checkpoint(self) -> None:
        ckpt_path = self.variant_spec["checkpoint_path"]
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
        if any(k.startswith("backbone.") for k in weights.keys()):
            weights = {k.split("backbone.", 1)[1]: v for k, v in weights.items() if k.startswith("backbone.")}
        weights = dict(weights)

        # Classification checkpoints store a classifier norm/head that is not part of the
        # feature-extractor interface used by this repo. Reuse the classifier norm for the
        # tapped stage outnorm when shapes match, then drop the classifier-only head keys.
        stage_outnorm_prefix = f"outnorm{self._feature_stage_index}"
        featurizer_state = self.featurizer.state_dict()
        for suffix in ("weight", "bias"):
            classifier_norm_key = f"classifier.norm.{suffix}"
            stage_outnorm_key = f"{stage_outnorm_prefix}.{suffix}"
            if (
                classifier_norm_key in weights
                and stage_outnorm_key in featurizer_state
                and stage_outnorm_key not in weights
                and weights[classifier_norm_key].shape == featurizer_state[stage_outnorm_key].shape
            ):
                weights[stage_outnorm_key] = weights[classifier_norm_key]

        for key in list(weights.keys()):
            if key.startswith("classifier.norm.") or key.startswith("classifier.head."):
                weights.pop(key)

        incompatible = self.featurizer.load_state_dict(weights, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        tapped_outnorm_keys = [
            f"{stage_outnorm_prefix}.weight",
            f"{stage_outnorm_prefix}.bias",
        ]
        if not unexpected_keys and set(missing_keys) == set(tapped_outnorm_keys):
            overwatch.info(
                "VMamba classification checkpoint does not include the tapped stage outnorm; using module defaults.",
                ctx_level=1,
            )
        elif missing_keys or unexpected_keys:
            overwatch.warning(
                f"VMamba checkpoint loaded with mismatched keys: missing={missing_keys}, "
                f"unexpected={unexpected_keys}"
            )

    def _infer_embed_dim(self) -> int:
        dims = self.featurizer.dims if hasattr(self.featurizer, "dims") else None
        if dims is None:
            raise ValueError("VMamba Backbone could not determine stage dimensions from the underlying model")
        return dims[self._feature_stage_index]

    def _infer_num_patches(self, patch_size: int) -> int:
        if patch_size <= 0:
            raise ValueError("VMamba patch size must be positive")
        height, width = self._nominal_input_size
        base_h = max(1, (height // patch_size) // (2 ** self._feature_stage_index))
        base_w = max(1, (width // patch_size) // (2 ** self._feature_stage_index))
        return base_h * base_w

    def _resolve_stage_index(self, requested_stage: Optional[int], num_stages: int) -> int:
        stage = num_stages if requested_stage is None else requested_stage
        if stage < 1 or stage > num_stages:
            raise ValueError(
                f"VMamba stage selection {stage} is out of range for {num_stages} stages (1-indexed)"
            )
        return stage - 1

    def _resolve_layer_index(self, stage_index: int, requested_layer: Optional[int]) -> int:
        depth = self.stage_depths[stage_index]
        if depth <= 0:
            raise ValueError(f"Stage {stage_index + 1} has no blocks to select")

        if requested_layer is None:
            resolved = depth - 1
        else:
            resolved = requested_layer
            if resolved < 0:
                resolved = depth + resolved

        if resolved < 0 or resolved >= depth:
            raise ValueError(
                f"Layer index {requested_layer} (resolved to {resolved}) is out of bounds for stage {stage_index + 1}"
            )
        return resolved
