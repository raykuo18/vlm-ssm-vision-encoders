# VLM SSM Vision Encoders

[arXiv](https://arxiv.org/abs/2603.19209) | [Project Page](https://lab-spell.github.io/vlm-ssm-vision-encoders/) | [Hugging Face Checkpoints](https://huggingface.co/raykuo188/vlm-ssm-vision-encoders-checkpoints)

Inference-only release for the paper _Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders_.

This repository packages a clean public inference surface around a representative set of released checkpoints. It focuses on:

- loading a released checkpoint
- running single-image generation
- reproducing the main public result tables included in `results/`
- building validated environments for the released backbone families

Training and evaluation code, along with additional checkpoints, will be released soon.

## Scope

This release supports Vicuna v1.5 7B checkpoints with these vision backbone families:

- ViT
- MaxViT
- VMamba
- MambaVision
- ViTDet
- ViT-Adapter

The public package is manifest-driven. Each released model has a stable public id in `model_zoo/models.yaml`, while raw internal run ids remain metadata only.

## Credit

This codebase builds on `TRI-ML/prismatic-vlms`, with the public release reduced to an inference-only package and backbone-specific runtime wrappers.

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

## Quickstart

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

2. Build the environment for the family you want to use.

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

Released checkpoints are hosted at [raykuo188/vlm-ssm-vision-encoders-checkpoints](https://huggingface.co/raykuo188/vlm-ssm-vision-encoders-checkpoints).

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
model_zoo/             Released model manifest
results/               Static release-facing tables
scripts/env/           Family-specific environment builders
scripts/validate/      Smoke validation scripts
third_party/           Pinned source dependencies
```
