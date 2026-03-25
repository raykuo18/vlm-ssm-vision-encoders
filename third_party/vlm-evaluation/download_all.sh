#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # or set to your repo root
cd "$ROOT_DIR"

DATA_ROOT="${VLM_EVAL_DATA_ROOT:-/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data}"
DATASETS=(vqa-v2 gqa vizwiz text-vqa refcoco ocid-ref tally-qa pope vsr)

dataset_is_prepared() {
  local dataset_family="$1"
  python - "$dataset_family" "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

from vlm_eval.tasks.registry import DATASET_REGISTRY

dataset_family = sys.argv[1]
root_dir = Path(sys.argv[2])

if dataset_family not in DATASET_REGISTRY:
    print(f"Unknown dataset family: {dataset_family}", file=sys.stderr)
    sys.exit(2)

paths = DATASET_REGISTRY[dataset_family]["paths"]
dataset_dir = root_dir / paths["dataset_dir"]
index_files = [root_dir / idx for idx in paths.get("index_files", [])]

if dataset_dir.exists():
    if not index_files:
        sys.exit(0)
    for idx in index_files:
        if idx.exists():
            sys.exit(0)

sys.exit(1)
PY
}

for ds in "${DATASETS[@]}"; do
  if dataset_is_prepared "$ds"; then
    echo "[skip] ${ds} already prepared under ${DATA_ROOT}"
    continue
  fi

  echo "==> Downloading: $ds"
  EXTRA_ARGS=()
  if [[ "$ds" == "vsr" ]]; then
    EXTRA_ARGS+=(--create_slim_dataset false)
  fi

  if python scripts/datasets/prepare.py --dataset_family "$ds" --root_dir "$DATA_ROOT" "${EXTRA_ARGS[@]}"; then
    echo "✓ Done: $ds"
  else
    echo "✗ Failed: $ds" >&2
  fi
done
