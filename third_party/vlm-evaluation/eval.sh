#!/usr/bin/env bash
set -euo pipefail

# Ensure conda function is available (non-login shell)
CONDA_PROFILE="/lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_PROFILE" ]]; then
  echo "Missing shared conda profile: $CONDA_PROFILE" >&2
  exit 1
fi
source "$CONDA_PROFILE"
conda activate "${HOME}/.conda/envs/vlm-eval"

python scripts/evaluate.py \
  --model_family prismatic \
  --model_id in1kft-224px+7b_finetune \
  --model_dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs/in1kft-224px+7b_finetune+stage-finetune+x7 \
  --dataset.type text-vqa-slim \
  --dataset.root_dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data \
  --results_dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs/evaluations \
  --load_precision fp32

python scripts/score.py \
  --model_id in1kft-224px+7b_finetune \
  --dataset.type text-vqa-slim \
  --dataset.root_dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data \
  --results_dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs/evaluations
