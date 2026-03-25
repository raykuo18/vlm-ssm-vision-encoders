#!/usr/bin/env python
"""Backfill `image_transform.json` for all localization eval runs under a runs root."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

TARGET_DATASETS = ("refcoco-full", "ocid-ref-full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("../vlm_runs").resolve(),
        help="Directory containing per-run folders (defaults to ../vlm_runs)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(TARGET_DATASETS),
        help="Dataset groups to scan inside each eval_results/<tag>/ (default: %(default)s)",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Optional[Dict]:
    for candidate in [run_dir / "config.yaml", run_dir / "config.json"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                if candidate.suffix == ".json":
                    return json.load(f)
                if yaml is not None:
                    return yaml.safe_load(f)
    return None


def extract_model_cfg(cfg: Dict) -> Dict:
    if isinstance(cfg, dict) and "model" in cfg and isinstance(cfg["model"], dict):
        return cfg["model"]
    return cfg


def infer_resolution(model_cfg: Dict, model_id: str) -> List[int]:
    hints: List[Optional[int]] = [
        model_cfg.get("default_image_size"),
        model_cfg.get("vision_default_image_size"),
    ]
    vb_id = model_cfg.get("vision_backbone_id")
    for token in filter(None, [vb_id, model_id]):
        match = re.search(r"(\d{2,4})\s*px", token, re.IGNORECASE)
        if match:
            hints.append(int(match.group(1)))
    for hint in hints:
        if isinstance(hint, int) and hint > 0:
            return [hint, hint]
    return [336, 336]


def build_metadata(model_cfg: Dict, model_id: str) -> Dict:
    policy = model_cfg.get("image_resize_strategy", "letterbox")
    resolution = infer_resolution(model_cfg, model_id)
    return {
        "policy": policy,
        "target_resolution": resolution,
        "vision_backbone": model_cfg.get("vision_backbone_id"),
        "model_id": model_id,
        "model_family": "prismatic",
    }


def iter_model_dirs(run_dir: Path, dataset_groups: Iterable[str]) -> Iterator[Tuple[Path, Path]]:
    eval_root = run_dir / "eval_results"
    if not eval_root.exists():
        return
    for tag_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        for dataset_group in dataset_groups:
            ds_group_dir = tag_dir / dataset_group
            if not ds_group_dir.exists():
                continue
            for family_dir in sorted(p for p in ds_group_dir.iterdir() if p.is_dir()):
                for dataset_id_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                    for model_dir in sorted(p for p in dataset_id_dir.iterdir() if p.is_dir()):
                        yield dataset_id_dir, model_dir


def main() -> None:
    args = parse_args()
    runs_root: Path = args.runs_root
    dataset_groups: List[str] = list(dict.fromkeys(args.datasets))

    created = 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        model_cfg_raw = load_run_config(run_dir)
        if model_cfg_raw is None:
            print(f"[WARN] Missing config for {run_dir.name}; skipping")
            continue
        model_cfg = extract_model_cfg(model_cfg_raw)

        for dataset_id_dir, model_dir in iter_model_dirs(run_dir, dataset_groups):
            transform_path = model_dir / "image_transform.json"
            if transform_path.exists():
                continue
            model_id = model_dir.name
            metadata = build_metadata(model_cfg, model_id)
            transform_path.write_text(json.dumps(metadata, indent=2))
            created += 1
            print(f"[INFO] Wrote {transform_path.relative_to(run_dir)}")

    print(f"[DONE] Created {created} image_transform.json files under {runs_root}")


if __name__ == "__main__":
    main()
