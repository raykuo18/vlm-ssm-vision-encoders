"""Utility helpers for projecting bounding boxes into the model's preprocessing frame."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))


def _as_resolution(transform_info: Dict[str, Sequence[int]], image_wh: Tuple[int, int]) -> Tuple[float, float]:
    target = transform_info.get("target_resolution")
    if isinstance(target, Sequence) and len(target) == 2:
        h, w = target
        if h and w:
            return float(h), float(w)
    # Fallback to original resolution if metadata is missing
    return float(image_wh[1]), float(image_wh[0])


def _normalize_letterbox(bbox_xyxy: Sequence[float], image_wh: Tuple[int, int]) -> List[float]:
    width, height = map(float, image_wh)
    max_wh = max(width, height)
    pad_x = (max_wh - width) / 2 if width < max_wh else 0.0
    pad_y = (max_wh - height) / 2 if height < max_wh else 0.0

    transformed = [
        (bbox_xyxy[0] + pad_x) / max_wh,
        (bbox_xyxy[1] + pad_y) / max_wh,
        (bbox_xyxy[2] + pad_x) / max_wh,
        (bbox_xyxy[3] + pad_y) / max_wh,
    ]
    return [_clamp01(coord) for coord in transformed]


def _normalize_resize_naive(bbox_xyxy: Sequence[float], image_wh: Tuple[int, int]) -> List[float]:
    width, height = map(float, image_wh)
    transformed = [
        bbox_xyxy[0] / width,
        bbox_xyxy[1] / height,
        bbox_xyxy[2] / width,
        bbox_xyxy[3] / height,
    ]
    return [_clamp01(coord) for coord in transformed]


def _normalize_resize_crop(
    bbox_xyxy: Sequence[float], image_wh: Tuple[int, int], transform_info: Dict[str, Sequence[int]]
) -> List[float]:
    width, height = map(float, image_wh)
    target_h, target_w = _as_resolution(transform_info, image_wh)

    # Uniformly scale until both dimensions exceed the model's target resolution
    scale = max(target_w / width, target_h / height)
    scaled_w, scaled_h = width * scale, height * scale

    scaled_bbox = [coord * scale for coord in bbox_xyxy]
    offset_x = max(0.0, (scaled_w - target_w) / 2.0)
    offset_y = max(0.0, (scaled_h - target_h) / 2.0)

    transformed = [
        (scaled_bbox[0] - offset_x) / target_w,
        (scaled_bbox[1] - offset_y) / target_h,
        (scaled_bbox[2] - offset_x) / target_w,
        (scaled_bbox[3] - offset_y) / target_h,
    ]
    return [_clamp01(coord) for coord in transformed]


def normalize_bbox_for_transform(
    bbox_xyxy: Sequence[float], image_wh: Tuple[int, int], transform_info: Dict[str, Sequence[int]] | None
) -> List[float]:
    """Project an absolute COCO/OCID box into the VLM's normalized coordinate frame."""
    if transform_info is None:
        transform_info = {}

    policy = transform_info.get("policy", "resize-naive").lower()

    if policy == "letterbox":
        return _normalize_letterbox(bbox_xyxy, image_wh)

    if policy == "resize-crop":
        return _normalize_resize_crop(bbox_xyxy, image_wh, transform_info)

    # Default to naive resize semantics
    return _normalize_resize_naive(bbox_xyxy, image_wh)
