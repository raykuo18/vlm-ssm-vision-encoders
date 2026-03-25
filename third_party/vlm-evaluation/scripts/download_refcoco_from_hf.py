#!/usr/bin/env python
"""Utility script for mirroring RefCOCO/RefCOCO+/RefCOCOg assets from HuggingFace."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

DATA_ROOT_DEFAULT = "/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"

HF_VARIANTS: Dict[str, Dict[str, str]] = {
    "refcoco": {
        "repo": "jxu124/refcoco",
        "pickle_name": "refs(unc).p",
    },
    "refcoco+": {
        "repo": "jxu124/refcocoplus",
        "pickle_name": "refs(unc).p",
    },
    "refcocog": {
        "repo": "jxu124/refcocog",
        "pickle_name": "refs(umd).p",
    },
}

COCO_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 5, "name": "airplane", "supercategory": "vehicle"},
    {"id": 6, "name": "bus", "supercategory": "vehicle"},
    {"id": 7, "name": "train", "supercategory": "vehicle"},
    {"id": 8, "name": "truck", "supercategory": "vehicle"},
    {"id": 9, "name": "boat", "supercategory": "vehicle"},
    {"id": 10, "name": "traffic light", "supercategory": "outdoor"},
    {"id": 11, "name": "fire hydrant", "supercategory": "outdoor"},
    {"id": 12, "name": "stop sign", "supercategory": "outdoor"},
    {"id": 13, "name": "parking meter", "supercategory": "outdoor"},
    {"id": 14, "name": "bench", "supercategory": "outdoor"},
    {"id": 15, "name": "bird", "supercategory": "animal"},
    {"id": 16, "name": "cat", "supercategory": "animal"},
    {"id": 17, "name": "dog", "supercategory": "animal"},
    {"id": 18, "name": "horse", "supercategory": "animal"},
    {"id": 19, "name": "sheep", "supercategory": "animal"},
    {"id": 20, "name": "cow", "supercategory": "animal"},
    {"id": 21, "name": "elephant", "supercategory": "animal"},
    {"id": 22, "name": "bear", "supercategory": "animal"},
    {"id": 23, "name": "zebra", "supercategory": "animal"},
    {"id": 24, "name": "giraffe", "supercategory": "animal"},
    {"id": 25, "name": "backpack", "supercategory": "accessory"},
    {"id": 26, "name": "umbrella", "supercategory": "accessory"},
    {"id": 27, "name": "handbag", "supercategory": "accessory"},
    {"id": 28, "name": "tie", "supercategory": "accessory"},
    {"id": 29, "name": "suitcase", "supercategory": "accessory"},
    {"id": 30, "name": "frisbee", "supercategory": "sports"},
    {"id": 31, "name": "skis", "supercategory": "sports"},
    {"id": 32, "name": "snowboard", "supercategory": "sports"},
    {"id": 33, "name": "sports ball", "supercategory": "sports"},
    {"id": 34, "name": "kite", "supercategory": "sports"},
    {"id": 35, "name": "baseball bat", "supercategory": "sports"},
    {"id": 36, "name": "baseball glove", "supercategory": "sports"},
    {"id": 37, "name": "skateboard", "supercategory": "sports"},
    {"id": 38, "name": "surfboard", "supercategory": "sports"},
    {"id": 39, "name": "tennis racket", "supercategory": "sports"},
    {"id": 40, "name": "bottle", "supercategory": "kitchen"},
    {"id": 41, "name": "wine glass", "supercategory": "kitchen"},
    {"id": 42, "name": "cup", "supercategory": "kitchen"},
    {"id": 43, "name": "fork", "supercategory": "kitchen"},
    {"id": 44, "name": "knife", "supercategory": "kitchen"},
    {"id": 45, "name": "spoon", "supercategory": "kitchen"},
    {"id": 46, "name": "bowl", "supercategory": "kitchen"},
    {"id": 47, "name": "banana", "supercategory": "food"},
    {"id": 48, "name": "apple", "supercategory": "food"},
    {"id": 49, "name": "sandwich", "supercategory": "food"},
    {"id": 50, "name": "orange", "supercategory": "food"},
    {"id": 51, "name": "broccoli", "supercategory": "food"},
    {"id": 52, "name": "carrot", "supercategory": "food"},
    {"id": 53, "name": "hot dog", "supercategory": "food"},
    {"id": 54, "name": "pizza", "supercategory": "food"},
    {"id": 55, "name": "donut", "supercategory": "food"},
    {"id": 56, "name": "cake", "supercategory": "food"},
    {"id": 57, "name": "chair", "supercategory": "furniture"},
    {"id": 58, "name": "couch", "supercategory": "furniture"},
    {"id": 59, "name": "potted plant", "supercategory": "furniture"},
    {"id": 60, "name": "bed", "supercategory": "furniture"},
    {"id": 61, "name": "dining table", "supercategory": "furniture"},
    {"id": 62, "name": "toilet", "supercategory": "furniture"},
    {"id": 63, "name": "tv", "supercategory": "electronic"},
    {"id": 64, "name": "laptop", "supercategory": "electronic"},
    {"id": 65, "name": "mouse", "supercategory": "electronic"},
    {"id": 66, "name": "remote", "supercategory": "electronic"},
    {"id": 67, "name": "keyboard", "supercategory": "electronic"},
    {"id": 68, "name": "cell phone", "supercategory": "electronic"},
    {"id": 69, "name": "microwave", "supercategory": "appliance"},
    {"id": 70, "name": "oven", "supercategory": "appliance"},
    {"id": 71, "name": "toaster", "supercategory": "appliance"},
    {"id": 72, "name": "sink", "supercategory": "appliance"},
    {"id": 73, "name": "refrigerator", "supercategory": "appliance"},
    {"id": 74, "name": "book", "supercategory": "indoor"},
    {"id": 75, "name": "clock", "supercategory": "indoor"},
    {"id": 76, "name": "vase", "supercategory": "indoor"},
    {"id": 77, "name": "scissors", "supercategory": "indoor"},
    {"id": 78, "name": "teddy bear", "supercategory": "indoor"},
    {"id": 79, "name": "hair drier", "supercategory": "indoor"},
    {"id": 80, "name": "toothbrush", "supercategory": "indoor"},
]


def _normalize_tokens(tokens) -> List[str]:
    if tokens is None:
        return []
    if isinstance(tokens, list):
        return [str(tok) for tok in tokens]
    try:
        return [str(tok) for tok in tokens.tolist()]
    except AttributeError:
        return [str(tokens)]


def _parse_field(blob):
    if blob is None:
        return None
    if isinstance(blob, str):
        return json.loads(blob)
    return blob


def build_refs_from_parquet(parquet_paths: Iterable[str]) -> Tuple[List[Dict], Dict[int, Dict], Dict[int, Dict]]:
    refs: List[Dict] = []
    images: Dict[int, Dict] = {}
    annotations: Dict[int, Dict] = {}

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        for row in df.itertuples(index=False):
            sentences = []
            raw_sentences_data = getattr(row, "sentences", None)
            if raw_sentences_data is None:
                raw_sentences = []
            elif isinstance(raw_sentences_data, (list, tuple)):
                raw_sentences = raw_sentences_data
            else:
                try:
                    raw_sentences = list(raw_sentences_data)
                except TypeError:
                    raw_sentences = [raw_sentences_data]
            for sent in raw_sentences:
                sentence_entry = {
                    "sent_id": int(sent.get("sent_id", 0)),
                    "sent": str(sent.get("sent", "")),
                }
                if "raw" in sent:
                    sentence_entry["raw"] = str(sent["raw"])
                tokens = sent.get("tokens")
                if tokens is not None:
                    sentence_entry["tokens"] = _normalize_tokens(tokens)
                sentences.append(sentence_entry)

            refs.append(
                {
                    "image_id": int(getattr(row, "image_id")),
                    "ann_id": int(getattr(row, "ann_id")),
                    "ref_id": int(getattr(row, "ref_id")),
                    "sentences": sentences,
                    "category_id": int(getattr(row, "category_id")),
                    "file_name": str(getattr(row, "file_name")),
                    "split": str(getattr(row, "split")),
                }
            )

            image_info = _parse_field(getattr(row, "raw_image_info", None))
            if image_info is not None:
                image_id = int(image_info["id"])
                image_info["id"] = image_id
                images[image_id] = image_info

            ann_info = _parse_field(getattr(row, "raw_anns", None))
            if ann_info is not None:
                ann_id = int(ann_info["id"])
                ann_info["id"] = ann_id
                ann_info["image_id"] = int(ann_info["image_id"])
                ann_info["category_id"] = int(ann_info["category_id"])
                annotations[ann_id] = ann_info

    return refs, images, annotations


def write_instances(instances_path: Path, images: Dict[int, Dict], annotations: Dict[int, Dict]) -> None:
    payload = {
        "images": list(images.values()),
        "annotations": list(annotations.values()),
        "categories": COCO_CATEGORIES,
    }
    instances_path.write_text(json.dumps(payload))


def download_variant(variant: str, cfg: Dict[str, str], download_root: Path, force: bool) -> None:
    variant_dir = download_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = variant_dir / cfg["pickle_name"]
    instances_path = variant_dir / "instances.json"

    if pickle_path.exists() and instances_path.exists() and not force:
        print(f"[skip] {variant} artifacts already present")
        return

    print(f"[info] Downloading {variant} metadata from {cfg['repo']}")
    repo_files = list_repo_files(cfg["repo"], repo_type="dataset")
    parquet_files = sorted([f for f in repo_files if f.endswith(".parquet")])
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {cfg['repo']}")

    local_paths = [
        hf_hub_download(cfg["repo"], filename=filename, repo_type="dataset")
        for filename in parquet_files
    ]

    refs, images, annotations = build_refs_from_parquet(local_paths)
    print(f"[info] Writing {len(refs)} references -> {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(refs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[info] Writing {len(images)} images / {len(annotations)} annotations -> {instances_path}"
    )
    write_instances(instances_path, images, annotations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RefCOCO/RefCOCO+/RefCOCOg metadata from HuggingFace and stage it for vlm-eval."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path(DATA_ROOT_DEFAULT),
        help="Dataset root where vlm-eval data is stored (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing refs/instances files if they are already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_root = args.root_dir / "download" / "refcoco"
    download_root.mkdir(parents=True, exist_ok=True)

    for variant, cfg in HF_VARIANTS.items():
        try:
            download_variant(variant, cfg, download_root, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] Failed to prepare {variant}: {exc}", file=sys.stderr)
            raise

    print("[done] RefCOCO assets prepared; run download_all.sh to build metadata files.")


if __name__ == "__main__":
    main()
