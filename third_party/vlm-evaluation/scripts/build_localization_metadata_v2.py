#!/usr/bin/env python
"""Create side-by-side RefCOCO*/OCID-Ref metadata files with absolute boxes + image size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"),
        help="Dataset root that already holds the downloaded RefCOCO/OCID assets",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["refcoco", "ocid-ref"],
        help="Which datasets to rebuild (choices: refcoco, ocid-ref)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="v2",
        help="Suffix for the new metadata file (default: v2 => metadata-full-v2.json)",
    )
    return parser.parse_args()


def ensure_float_list(values: Iterable) -> List[float]:
    return [float(v) for v in values]


def build_refcoco_metadata(root_dir: Path, suffix: str) -> None:
    dataset_dir = root_dir / "datasets/refcoco"
    src = dataset_dir / "metadata-full.json"
    dst = dataset_dir / f"metadata-full-{suffix}.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing RefCOCO metadata at {src}")

    with open(src, "r") as f:
        metadata = json.load(f)

    new_entries: Dict[str, Dict] = {}
    for key, entry in metadata.items():
        img_path = root_dir / entry["img_path"]
        with Image.open(img_path) as img:
            width, height = img.size
        bbox_norm = ensure_float_list(entry["bbox"])
        bbox_xyxy = [
            round(bbox_norm[0] * width, 2),
            round(bbox_norm[1] * height, 2),
            round(bbox_norm[2] * width, 2),
            round(bbox_norm[3] * height, 2),
        ]
        new_entry = dict(entry)
        new_entry["bbox"] = bbox_norm
        new_entry["bbox_xyxy"] = bbox_xyxy
        new_entry["image_width"] = width
        new_entry["image_height"] = height
        new_entries[str(key)] = new_entry

    with open(dst, "w") as f:
        json.dump(new_entries, f, indent=2)
    print(f"[INFO] Wrote RefCOCO metadata with absolute boxes to {dst}")


def build_ocid_metadata(root_dir: Path, suffix: str) -> None:
    dataset_dir = root_dir / "datasets/ocid-ref"
    src = dataset_dir / "metadata-full.json"
    dst = dataset_dir / f"metadata-full-{suffix}.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing OCID-Ref metadata at {src}")

    with open(src, "r") as f:
        metadata = json.load(f)

    new_entries: Dict[str, Dict] = {}
    for key, entry in metadata.items():
        img_path = root_dir / entry["img_path"]
        with Image.open(img_path) as img:
            width, height = img.size
        bbox_norm = ensure_float_list(entry["bbox"])
        bbox_xyxy = [
            round(bbox_norm[0] * width, 2),
            round(bbox_norm[1] * height, 2),
            round(bbox_norm[2] * width, 2),
            round(bbox_norm[3] * height, 2),
        ]
        new_entry = dict(entry)
        new_entry["bbox"] = bbox_norm
        new_entry["bbox_xyxy"] = bbox_xyxy
        new_entry["image_width"] = width
        new_entry["image_height"] = height
        new_entries[str(key)] = new_entry

    with open(dst, "w") as f:
        json.dump(new_entries, f, indent=2)
    print(f"[INFO] Wrote OCID-Ref metadata with absolute boxes to {dst}")


def main() -> None:
    args = parse_args()
    datasets = set(args.datasets)
    if "refcoco" in datasets:
        build_refcoco_metadata(args.root_dir, args.suffix)
    if "ocid-ref" in datasets:
        build_ocid_metadata(args.root_dir, args.suffix)


if __name__ == "__main__":
    main()
