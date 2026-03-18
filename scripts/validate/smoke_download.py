#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from vlm_backbones.api import download_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract a released model artifact.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = download_model(args.model, force=args.force)
    print(path)


if __name__ == "__main__":
    main()
