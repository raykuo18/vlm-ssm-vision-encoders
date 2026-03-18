#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_backbones.manifest import available_models


def main() -> None:
    models = available_models()
    print(f"manifest ok: {len(models)} models")


if __name__ == "__main__":
    main()
