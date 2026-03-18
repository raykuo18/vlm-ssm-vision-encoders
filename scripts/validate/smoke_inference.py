#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from vlm_backbones.api import load_model
from vlm_backbones.images import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one inference step on a released checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    vlm = load_model(args.model, device=args.device, dtype=args.dtype)
    image = load_image(args.image)
    print(vlm.generate(image, args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
