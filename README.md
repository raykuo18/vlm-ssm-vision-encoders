# VLM SSM Vision Encoders

Public release for the paper _Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders_.

This repository now ships three workflows:

- `src/vlm_backbones/` for released-checkpoint download and inference
- `src/prismatic/` for parity-oriented training inside this repo
- `third_party/vlm-evaluation/` plus repo-local wrappers for benchmark evaluation and scoring

## Scope

This release supports Vicuna v1.5 7B checkpoints with these vision backbone families:

- ViT
- MaxViT
- VMamba
- MambaVision
- ViTDet
- ViT-Adapter

The public package is manifest-driven. Each released checkpoint has a stable public id in `model_zoo/models.yaml`, while paper-facing run ids remain metadata only.

Training and evaluation stay close to the original `Mamba-MLLM` stack to minimize behavioral drift. The inference surface remains the stable `vlm_backbones` API from v1.

## Credit

This codebase builds on `TRI-ML/prismatic-vlms` and the original `Mamba-MLLM` train/eval stack. The train/eval path is intentionally vendored with a narrow compatibility layer instead of being reimplemented from scratch.

## Clone

Clone the repository before building an environment:

```bash
git clone --recurse-submodules https://github.com/raykuo18/vlm-ssm-vision-encoders.git
cd vlm-ssm-vision-encoders
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

The environment builders can clone the required `third_party/` repos automatically if they are missing, but a recursive clone is still the cleaner default because several released backbone families load code directly from those source trees at runtime.

## Environment Builders

Environment setup is split by purpose:

- `scripts/env/build_<family>.sh` for inference-only environments
- `scripts/env/build_<family>_train.sh` for training environments
- `scripts/env/build_<family>_eval.sh` for evaluation environments

See [docs/TRAIN_EVAL_GUIDE.md](/lustre/nvwulf/home/skuo/vlm-project/vlm-ssm-vision-encoders/docs/TRAIN_EVAL_GUIDE.md) for the full matrix and exact commands.

## Inference Quickstart

1. Load a CUDA toolkit that provides `nvcc` if you are building an environment that compiles CUDA extensions.

```bash
module load cuda12.8/toolkit/12.8.1
```

If your cluster uses a different module name, load the equivalent CUDA toolkit first. In this repo, `build_vmamba.sh`, `build_mambavision.sh`, and `build_vit_adapter.sh` require `nvcc` during setup. `build_vit.sh`, `build_maxvit.sh`, and `build_vitdet.sh` do not require `nvcc` unless you opt into FlashAttention.

If possible, run CUDA-extension builds from a session where an NVIDIA GPU is visible. This is the recommended path for general users. The build scripts use the visible GPU compute capability to infer `TORCH_CUDA_ARCH_LIST` for CUDA extensions such as VMamba kernels and ViT-Adapter ops.

If no GPU is visible during the build, set `TORCH_CUDA_ARCH_LIST` manually before running the builder. For example, H200 should use:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
```

Manual `TORCH_CUDA_ARCH_LIST` is a fallback, not the primary path. CUDA extension builds can still be more brittle without a visible GPU. If the build continues to fail in a CPU-only session, rerun it from a GPU-visible session.

Backbone environments skip FlashAttention by default because it is not required for backbone inference in this repo. If you want FlashAttention anyway, pass `--with-flash-attn`.

2. Build the inference environment for the family you want to use.

```bash
./scripts/env/build_vmamba.sh --env-name vlm-vmamba
```

The environment build may take a long time because it installs large PyTorch wheels and, for some families, compiles CUDA-dependent packages such as VMamba kernels or ViT-Adapter ops. By default, the build uses up to 8 CPU cores for compile-heavy steps.

If you want to limit compile parallelism, set `ENV_BUILD_JOBS` before running the builder. For example:

```bash
export ENV_BUILD_JOBS=8
./scripts/env/build_vmamba.sh --env-name vlm-vmamba
```

If you opt into FlashAttention and still hit source-build failures, set a smaller `FLASH_ATTN_BUILD_JOBS` value explicitly. For example:

```bash
export FLASH_ATTN_BUILD_JOBS=2
./scripts/env/build_vmamba.sh --env-name vlm-vmamba --with-flash-attn
```

If a previous build failed after partially creating the environment, fix the missing prerequisite and rerun the same command. The builder will reuse the existing environment by default.

```bash
./scripts/env/build_vmamba.sh --env-name vlm-vmamba
```

For a clean rebuild from scratch, use:

```bash
./scripts/env/build_vmamba.sh --env-name vlm-vmamba --recreate
```

3. Activate the environment.

```bash
conda activate vlm-vmamba
```

4. Export a Hugging Face token before loading gated base weights such as Vicuna.

```bash
export HF_TOKEN=[hf_token]
```

5. Download a released checkpoint.

```bash
vlm-backbones-download --model vmamba-s-in1k-224-s3
```

Available released public ids:

- `vit-s-in1k-224`
- `maxvit-t-in1k-224-s3`
- `mambavision-b-in1k-224-s3`
- `vmamba-s-in1k-224-s3`
- `vitdet-b-coco-1024`
- `vmamba-s-coco-1333x800`
- `vit-adapter-deit-b-ade20k-512`
- `vmamba-s-ade20k-512`

See `Released Models` below for the task, resolution, and metric summary for each checkpoint.

6. Run inference.

```bash
vlm-backbones-chat \
  --model vmamba-s-in1k-224-s3 \
  --image /path/to/example.jpg \
  --prompt "Describe the image."
```

`vlm-backbones-chat` treats `--prompt` as a raw user message and applies the model's chat prompt template automatically.

By default, `vlm-backbones-chat` uses deterministic generation.

If you want sampled generation, set a temperature. The CLI will enable sampling automatically:

```bash
vlm-backbones-chat \
  --model vmamba-s-in1k-224-s3 \
  --image /path/to/example.jpg \
  --prompt "Describe the image." \
  --temperature 0.2
```

If `--temperature` is omitted, generation stays deterministic.

For gated Vicuna weights, authenticate with Hugging Face first or export `HF_TOKEN`.

## Training Quickstart

Use the train builders for reproducible paper-style runs. On NVWULF, `scripts/setup_paths.sh` exports the known training and evaluation dataset roots.

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

The train/eval release only supports the published ViT, MaxViT, VMamba, MambaVision, ViTDet, and ViT-Adapter families with Vicuña v1.5 7B.

## Evaluation Quickstart

Evaluation uses the vendored `vlm_eval` package through repo-local wrappers:

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

To evaluate a local training run instead of a released public checkpoint, pass `--model_dir "$RUN_ROOT_DIR/<run_id>"` and set `--model_id` to the run name you want written under `results/`.

## Train/Eval Guide

Detailed train/eval instructions live in [docs/TRAIN_EVAL_GUIDE.md](/lustre/nvwulf/home/skuo/vlm-project/vlm-ssm-vision-encoders/docs/TRAIN_EVAL_GUIDE.md).

## Released Models

| Public ID | Family | Task | Resolution | Tokens | Params | Weighted VQA | Weighted Loc | Weighted Overall |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `vit-s-in1k-224` | ViT | Classification | 224x224 | 196 | 22M | 57.25 | 17.82 | 51.95 |
| `maxvit-t-in1k-224-s3` | MaxViT | Classification | 224x224 | 196 | 31M | 58.75 | 15.79 | 52.98 |
| `mambavision-b-in1k-224-s3` | MambaVision | Classification | 224x224 | 196 | 98M | 56.53 | 31.17 | 53.12 |
| `vmamba-s-in1k-224-s3` | VMamba | Classification | 224x224 | 196 | 50M | 62.39 | 39.17 | 59.27 |
| `vitdet-b-coco-1024` | ViTDet | Detection-adapted | 1024x1024 | 4096 | 111M | 63.00 | 43.74 | 60.42 |
| `vmamba-s-coco-1333x800` | VMamba | Detection-adapted | 1333x800 | 4150 | 50M | 62.78 | 47.94 | 60.78 |
| `vit-adapter-deit-b-ade20k-512` | ViT-Adapter | Segmentation-adapted | 512x512 | 1024 | 134M | 60.69 | 33.77 | 57.07 |
| `vmamba-s-ade20k-512` | VMamba | Segmentation-adapted | 512x512 | 1024 | 50M | 63.21 | 44.98 | 60.76 |

Full metadata is in `MODEL_ZOO.md`.

## Main Results

Matched 224 classification setting:

| Model | Weighted VQA | Weighted Loc | Weighted Overall |
| --- | ---: | ---: | ---: |
| `vit-s-in1k-224` | 57.25 | 17.82 | 51.95 |
| `maxvit-t-in1k-224-s3` | 58.75 | 15.79 | 52.98 |
| `mambavision-b-in1k-224-s3` | 56.53 | 31.17 | 53.12 |
| `vmamba-s-in1k-224-s3` | 62.39 | 39.17 | 59.27 |

Dense-objective adapted setting:

| Model | Weighted VQA | Weighted Loc | Weighted Overall |
| --- | ---: | ---: | ---: |
| `vitdet-b-coco-1024` | 63.00 | 43.74 | 60.42 |
| `vmamba-s-coco-1333x800` | 62.78 | 47.94 | 60.78 |
| `vit-adapter-deit-b-ade20k-512` | 60.69 | 33.77 | 57.07 |
| `vmamba-s-ade20k-512` | 63.21 | 44.98 | 60.76 |

Machine-readable copies live in `results/`. See `RESULTS.md` for the full table files.

## Layout

```text
src/vlm_backbones/     Python package
src/prismatic/         Vendored train/eval compatibility package
model_zoo/             Released model manifest
results/               Static release-facing tables
scripts/env/           Family-specific environment builders
scripts/train.py       Training entrypoint
scripts/evaluate.py    Evaluation wrapper
scripts/score.py       Scoring wrapper
scripts/validate/      Smoke validation scripts
third_party/           Pinned source dependencies
```
