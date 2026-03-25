#!/usr/bin/env python3
"""Visualize the localization bbox bug before/after the geometry fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont

import importlib.util

GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "vlm_eval" / "tasks" / "harnesses" / "geometry.py"
spec = importlib.util.spec_from_file_location("vlm_eval_geometry", GEOMETRY_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load geometry helpers from {GEOMETRY_PATH}")
geometry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_module)  # type: ignore[arg-type]
normalize_bbox_for_transform = geometry_module.normalize_bbox_for_transform


def _load_metadata(dataset_root: Path, dataset: str) -> Dict[str, Dict]:
    meta_path = dataset_root / "datasets" / dataset / "metadata-full-v2.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_transform(transform_path: Path) -> Dict:
    with transform_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "policy" not in data:
        raise ValueError(f"image_transform.json missing `policy`: {transform_path}")
    return data


def _target_hw(transform_info: Dict, fallback_wh: Sequence[int]) -> List[int]:
    target = transform_info.get("target_resolution")
    if isinstance(target, Sequence) and len(target) == 2:
        return [int(target[0]), int(target[1])]
    width, height = fallback_wh
    return [int(height), int(width)]


def _apply_transform(image: Image.Image, transform_info: Dict) -> Image.Image:
    policy = transform_info.get("policy", "resize-naive").lower()
    target_h, target_w = _target_hw(transform_info, image.size)

    if policy == "letterbox":
        max_wh = max(image.width, image.height)
        padded = Image.new("RGB", (max_wh, max_wh), color=(114, 114, 114))
        pad_x = (max_wh - image.width) // 2
        pad_y = (max_wh - image.height) // 2
        padded.paste(image, (pad_x, pad_y))
        return padded.resize((target_w, target_h), Image.BICUBIC)

    if policy == "resize-crop":
        scale = max(target_w / image.width, target_h / image.height)
        scaled_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        scaled = image.resize(scaled_size, Image.BICUBIC)
        left = max(0, (scaled.width - target_w) // 2)
        top = max(0, (scaled.height - target_h) // 2)
        return scaled.crop((left, top, left + target_w, top + target_h))

    return image.resize((target_w, target_h), Image.BICUBIC)


def _norm_to_pixels(norm_bbox: Sequence[float], target_hw: Sequence[int]) -> List[float]:
    target_h, target_w = target_hw
    return [
        norm_bbox[0] * target_w,
        norm_bbox[1] * target_h,
        norm_bbox[2] * target_w,
        norm_bbox[3] * target_h,
    ]


def _draw_rect(image: Image.Image, bbox_xyxy: Sequence[float], color: str, width: int = 3) -> None:
    draw_ctx = ImageDraw.Draw(image)
    draw_ctx.rectangle(bbox_xyxy, outline=color, width=width)


def _annotate(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), text, fill="white", font=font, stroke_width=2, stroke_fill="black")


def visualize_example(
    entry: Dict,
    dataset_root: Path,
    dataset: str,
    transform_info: Dict,
    output_dir: Path,
) -> Path:
    img_rel_path = entry["img_path"]
    img_path = dataset_root / img_rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    original = Image.open(img_path).convert("RGB")
    original_with_box = original.copy()
    _draw_rect(original_with_box, entry["bbox_xyxy"], color="yellow", width=4)
    _annotate(original_with_box, "Original (absolute GT)")

    processed = _apply_transform(original, transform_info)
    target_hw = _target_hw(transform_info, original.size)

    bbox_xyxy = entry["bbox_xyxy"]
    image_wh = (entry["image_width"], entry["image_height"])

    naive_norm = normalize_bbox_for_transform(bbox_xyxy, image_wh, {"policy": "resize-naive"})
    fixed_norm = normalize_bbox_for_transform(bbox_xyxy, image_wh, transform_info)

    naive_px = _norm_to_pixels(naive_norm, target_hw)
    fixed_px = _norm_to_pixels(fixed_norm, target_hw)

    processed_with_boxes = processed.copy()
    _draw_rect(processed_with_boxes, naive_px, color="red", width=3)
    _draw_rect(processed_with_boxes, fixed_px, color="lime", width=3)
    _annotate(processed_with_boxes, "Preprocessed Frame (red=old, green=new)")

    spacing = 24
    canvas = Image.new(
        "RGB",
        (original_with_box.width + processed_with_boxes.width + spacing, max(original_with_box.height, processed_with_boxes.height)),
        color=(20, 20, 20),
    )
    canvas.paste(original_with_box, (0, 0))
    canvas.paste(processed_with_boxes, (original_with_box.width + spacing, 0))

    example_id = entry["example_id"]
    out_path = output_dir / f"{dataset}-example-{example_id}.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize localization GT re-projection.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing the prepared datasets/")
    parser.add_argument("--dataset", type=str, default="refcoco", help="Dataset key under datasets/<name>/metadata-full-v2.json")
    parser.add_argument("--image-transform-json", type=Path, required=True, help="Path to image_transform.json for the evaluated model.")
    parser.add_argument("--example-ids", type=int, nargs="+", required=True, help="Example IDs to visualize (match metadata keys).")
    parser.add_argument("--output-dir", type=Path, default=Path("bug_visualization"), help="Directory to store rendered images.")
    args = parser.parse_args()

    metadata = _load_metadata(args.dataset_root, args.dataset)
    transform_info = _load_transform(args.image_transform_json)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for example_id in args.example_ids:
        key = str(example_id)
        if key not in metadata:
            raise KeyError(f"Example ID {example_id} not present in metadata for {args.dataset}")
        entry = metadata[key]
        out_path = visualize_example(entry, args.dataset_root, args.dataset, transform_info, output_dir)
        print(f"Wrote visualization to {out_path}")


if __name__ == "__main__":
    main()
