# Train And Eval Guide

This guide covers the v2 release workflow for users who want to reproduce training and benchmark evaluation with the same codepaths used in `Mamba-MLLM`, but from inside `vlm-ssm-vision-encoders`.

## Environment Matrix

| Family | Inference | Training | Evaluation | Notes |
| --- | --- | --- | --- | --- |
| ViT | `scripts/env/build_vit.sh` | `scripts/env/build_vit_train.sh` | `scripts/env/build_vit_eval.sh` | No CUDA extension build unless FlashAttention is requested |
| MaxViT | `scripts/env/build_maxvit.sh` | `scripts/env/build_maxvit_train.sh` | `scripts/env/build_maxvit_eval.sh` | No CUDA extension build unless FlashAttention is requested |
| VMamba | `scripts/env/build_vmamba.sh` | `scripts/env/build_vmamba_train.sh` | `scripts/env/build_vmamba_eval.sh` | Requires `nvcc` for `mamba-ssm` and selective scan kernels |
| MambaVision | `scripts/env/build_mambavision.sh` | `scripts/env/build_mambavision_train.sh` | `scripts/env/build_mambavision_eval.sh` | Requires `nvcc` |
| ViTDet | `scripts/env/build_vitdet.sh` | `scripts/env/build_vitdet_train.sh` | `scripts/env/build_vitdet_eval.sh` | Detectron2-style Python deps are installed in all three envs |
| ViT-Adapter | `scripts/env/build_vit_adapter.sh` | `scripts/env/build_vit_adapter_train.sh` | `scripts/env/build_vit_adapter_eval.sh` | Requires `nvcc` for deformable attention ops |

Default env names follow:

- Inference: `vlm-backbones-<family>`
- Training: `vlm-backbones-train-<family>`
- Evaluation: `vlm-backbones-eval-<family>`

## Default Paths

The repo uses these environment variables:

- `RUN_ROOT_DIR`: root for training runs and checkpoints
- `DATASET_ROOT`: root for train datasets such as `llava-v15`
- `VLM_EVAL_DATA_ROOT`: root for the vendored evaluation datasets
- `HF_TOKEN`: optional environment variable name if you do not want to store the token in `.hf_token`

On NVWULF, the recommended setup is:

```bash
source scripts/setup_paths.sh
```

That exports:

- `RUN_ROOT_DIR=/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs`
- `DATASET_ROOT=/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_dataset/cobra_dataset`
- `VLM_EVAL_DATA_ROOT=/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data`

Outside NVWULF, set these variables explicitly before training or evaluation.

## HF Token

The train and eval entrypoints accept either:

- `.hf_token` in the repo root
- `--hf_token SOME_ENV_VAR_NAME` where `SOME_ENV_VAR_NAME` is already exported

Example:

```bash
export HF_TOKEN=hf_xxx
printf '%s\n' "$HF_TOKEN" > .hf_token
```

## Debug Dry Run

This is the parity check used for the public train release:

```bash
source scripts/setup_paths.sh
./scripts/env/build_maxvit_train.sh
conda activate vlm-backbones-train-maxvit
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --stage finetune \
  --model.type in1k-224px-maxvit-t-letterbox-s3+7b-vicuna \
  --dataset.type llava-v15-debug-320 \
  --run_root_dir "$RUN_ROOT_DIR" \
  --dry_run true
```

Expected behavior class:

- FSDP initializes successfully on one GPU
- world-size-1 setup downgrades to `NO_SHARD`
- the debug dataset runs to `Max Steps = 3`

## Released-Checkpoint Evaluation

To run a slim benchmark against a released public checkpoint:

```bash
source scripts/setup_paths.sh
./scripts/env/build_maxvit_eval.sh
conda activate vlm-backbones-eval-maxvit
python scripts/evaluate.py \
  --model_family prismatic \
  --model_id maxvit-t-in1k-224-s3 \
  --dataset.type text-vqa-slim \
  --results_dir results
python scripts/score.py \
  --model_id maxvit-t-in1k-224-s3 \
  --dataset.type text-vqa-slim \
  --results_dir results
```

Scored metrics are written under:

- `results/text-vqa/text-vqa-slim/maxvit-t-in1k-224-s3/metrics.json`

## Evaluating A Local Training Run

If you want to evaluate a run produced by `scripts/train.py`, point `--model_dir` at the run directory:

```bash
python scripts/evaluate.py \
  --model_family prismatic \
  --model_id in1k-224px-maxvit-t-letterbox-s3+7b-vicuna+ep1+ft+x7 \
  --model_dir "$RUN_ROOT_DIR/in1k-224px-maxvit-t-letterbox-s3+7b-vicuna+ep1+ft+x7" \
  --dataset.type text-vqa-slim \
  --results_dir results
python scripts/score.py \
  --model_id in1k-224px-maxvit-t-letterbox-s3+7b-vicuna+ep1+ft+x7 \
  --dataset.type text-vqa-slim \
  --results_dir results
```

Local runs keep the original layout:

- `<run_root>/<run_id>/config.yaml`
- `<run_root>/<run_id>/config.json`
- `<run_root>/<run_id>/checkpoints/latest-checkpoint.pt`

## Paper-Facing Artifacts

There are two identifier spaces in this repo:

- Public checkpoint ids in `model_zoo/models.yaml`, such as `maxvit-t-in1k-224-s3`
- Internal paper run ids in `model_zoo/models.yaml`, such as `in1k-224px-maxvit-t-letterbox-s3+7b-vicuna+ep1+ft+x7`

Use:

- Public ids for released-checkpoint download and evaluation
- Internal run ids when referring to training outputs under `$RUN_ROOT_DIR`
