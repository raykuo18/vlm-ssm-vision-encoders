"""
models.py

Draccus Dataclass Definition for a ModelConfig object, with various registered subclasses for each model family and
variant thereof. A given model variant configures the following attributes:
    - Pretrained Visual Representation (e.g., OpenAI CLIP ViT-L/14) + Pretrained LLM Backbone (e.g., LLaMa-2 7B)
    - VLM Configuration + Parameters (e.g., MLP Projector, Image Preprocessing, etc.)
    - [Optional] Stage 1 (`align`) Optimization Hyperparameters
    - Stage 2 (`finetune`) Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    model_id: str                                           # Unique Model ID that fully specifies a given variant
    arch_specifier: str                                     # Architecture specifier string (e.g., "gelu-mlp")

    # Pretrained Backbones
    vision_backbone_id: str                                 # Pretrained Visual Featurizer (from TIMM) to load
    llm_backbone_id: str                                    # Pretrained LLM (from HF Transformers) to load

    # Backbone Parameters
    image_resize_strategy: str                              # Resizing strategy in < crop | letterbox | corner-pad >
    llm_max_length: int                                     # Maximum context length for LLM (can be < than max!)

    # === Multi-Stage Optimization Hyperparameters ===
    # By default, we assume an AdamW optimizer with FSDP (Gradient Sharding or Full Sharding depending on stage)

    # Align Stage Optimization Parameters
    align_epochs: int                                       # Epochs to Run (in case `max_steps` is not specified)
    align_max_steps: Optional[int]                          # [Optional] Max Gradient Steps (overrides epochs)
    align_global_batch_size: int                            # Global Batch Size (divided across processes)
    align_per_device_batch_size: int                        # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    align_learning_rate: float                              # Peak Learning Rate (lr_scheduler sets warmup/decay)
    align_weight_decay: float                               # Weight Decay for AdamW Optimizer
    align_max_grad_norm: float                              # Max Grad Norm (for global gradient clipping)
    align_lr_scheduler_type: str                            # LR Scheduler (default: "linear-warmup+cosine-decay")
    align_warmup_ratio: float                               # Fraction of total steps to warmup

    align_train_strategy: str                               # Align Train Strategy (default: "fsdp-shard-grad-op")

    # Finetune Stage Optimization Parameters
    finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    finetune_max_steps: Optional[int]                       # [Optional] Max Gradient Steps (overrides epochs)
    finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    finetune_warmup_ratio: float                            # Fraction of total steps to warmup

    finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")

    # Vision Finetune Stage Optimization Parameters
    vision_finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    vision_finetune_max_steps: Optional[int]                       # [Optional] Max Gradient Steps (overrides epochs)
    vision_finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    vision_finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    vision_finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    vision_finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    vision_finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    vision_finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    vision_finetune_warmup_ratio: float                            # Fraction of total steps to warmup

    vision_finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")
    vision_finetune_train_projector: bool = True                      # Whether projector updates during vision finetune

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True

    # Enable Traditional Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True            # Whether to enable mixed precision training
    reduce_in_full_precision: bool = False                  # Whether to run gradient reduction in FP32

    # fmt: on


# === LLaVa v1.5 Reproduction - Fully Specified Configurations ===
@dataclass
class LLaVa_v15_Reproduction_7B(ModelConfig):
    model_id: str = "reproduction-llava-v15+7b"
    arch_specifier: str = "gelu-mlp"

    vision_backbone_id: str = "clip-vit-l-336px"
    llm_backbone_id: str = "vicuna-v15-7b"

    image_resize_strategy: str = "letterbox"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_max_steps: Optional[int] = None
    align_global_batch_size: int = 256
    align_per_device_batch_size: int = 16

    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03

    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 1
    finetune_max_steps: Optional[int] = None
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 16

    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03

    finetune_train_strategy: str = "fsdp-full-shard"

    # Vision Finetune Stage Optimization Parameters
    vision_finetune_epochs: int = 1
    vision_finetune_max_steps: Optional[int] = None
    vision_finetune_global_batch_size: int = 128
    vision_finetune_per_device_batch_size: int = 16

    vision_finetune_learning_rate: float = 2e-5
    vision_finetune_weight_decay: float = 0.1
    vision_finetune_max_grad_norm: float = 1.0
    vision_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    vision_finetune_warmup_ratio: float = 0.03

    vision_finetune_train_strategy: str = "fsdp-full-shard"


@dataclass
class LLaVa_v15_Reproduction_13B(LLaVa_v15_Reproduction_7B):
    model_id: str = "reproduction-llava-v15+13b"
    llm_backbone_id: str = "vicuna-v15-13b"


# === Mamba-MLLM Backbone Sweep Templates ===
#
# These dataclasses back every sweep we run in the Mamba-MLLM project. Each one
# keeps the baseline optimizer above, overrides identifying fields (model_id,
# backbones, resize policy), and leaves finetune duration to the launcher via
# --finetune_epochs_override so we can sweep epochs without editing configs.
#
# Layout
#   Section 1 — LLaMA-2-7B-pure head (primary target)
#       1A. ImageNet-1K ViTs (trained from scratch on IN1K)
#       1B. ImageNet-1K Fine-Tuned ViTs (IN21K -> IN1K)
#       1C. ImageNet-21K-only ViTs
#       1D. VMamba (Tiny/Small/Base; throughput vs. accuracy variants)
#   Section 2 — Vicuña-v1.5-7B head (baseline mirror of Section 1)
#


# ------------------------------------------------------------------------------
# Section 2 :: Vicuña-v1.5-7B Backbones
# ------------------------------------------------------------------------------
#
# 2A. ImageNet-1K ViTs (scratch on IN1K)
@dataclass
class Exp_7B_IN1K_ViT_S_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K ViT-S/16 @ 224px"""
    model_id: str = "in1k-224px-vit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-vit-s"

@dataclass
class Exp_7B_IN1K_ViT_S_p16_224px_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K ViT-S/16 @ 224px (fused projector)"""
    model_id: str = "in1k-224px-vit-s-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "in1k-vit-s"


@dataclass
class Exp_7B_IN1K_ViT_B_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K ViT-B/16 @ 224px"""
    model_id: str = "in1k-224px-vit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-vit-b"


# 2B. ImageNet-1K Fine-Tuned ViTs (IN21K -> IN1K)
@dataclass
class Exp_7B_IN1KFT_ViT_T_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1KFT ViT-T/16 @ 224px"""
    model_id: str = "in1kft-224px-vit-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1kft-vit-t"


@dataclass
class Exp_7B_IN1KFT_ViT_S_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1KFT ViT-S/16 @ 224px"""
    model_id: str = "in1kft-224px-vit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1kft-vit-s"


@dataclass
class Exp_7B_IN1KFT_ViT_B_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1KFT ViT-B/16 @ 224px"""
    model_id: str = "in1kft-224px-vit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1kft-vit-b"


@dataclass
class Exp_7B_IN1KFT_ViT_B2_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1KFT ViT-B/16 @ 224px (augreg2 variant)"""
    model_id: str = "in1kft-224px-vit-b2+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1kft-vit-b2"


@dataclass
class Exp_7B_IN1KFT_ViT_L_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1KFT ViT-L/16 @ 224px"""
    model_id: str = "in1kft-224px-vit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1kft-vit-l"


# 2C. ImageNet-21K-only ViTs
@dataclass
class Exp_7B_IN21K_ViT_T_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K ViT-T/16 @ 224px"""
    model_id: str = "in21k-224px-vit-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-vit-t"


@dataclass
class Exp_7B_IN21K_ViT_S_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K ViT-S/16 @ 224px"""
    model_id: str = "in21k-224px-vit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-vit-s"


@dataclass
class Exp_7B_IN21K_ViT_B_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K ViT-B/16 @ 224px"""
    model_id: str = "in21k-224px-vit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-vit-b"

@dataclass
class Exp_7B_IN21K_ViT_L_p16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K ViT-L/16 @ 224px"""
    model_id: str = "in21k-224px-vit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-vit-l"


# 2C+. MaxViT (ImageNet-1K)
@dataclass
class Exp_7B_IN1K_MaxViT_T_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 224px"""
    model_id: str = "in1k-224px-maxvit-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-t"
    image_resize_strategy: str = "resize-crop"

@dataclass
class Exp_7B_IN1K_MaxViT_T_224px_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 224px (fused projector)"""
    model_id: str = "in1k-224px-maxvit-t-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-t"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_T_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 384px"""
    model_id: str = "in1k-384px-maxvit-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-t"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_T_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 512px"""
    model_id: str = "in1k-512px-maxvit-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-t"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_S_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 224px"""
    model_id: str = "in1k-224px-maxvit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-s"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_S_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 384px"""
    model_id: str = "in1k-384px-maxvit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-s"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_S_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 512px"""
    model_id: str = "in1k-512px-maxvit-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-s"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_B_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 224px"""
    model_id: str = "in1k-224px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_B_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 384px"""
    model_id: str = "in1k-384px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_B_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 512px"""
    model_id: str = "in1k-512px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_L_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 224px"""
    model_id: str = "in1k-224px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_L_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 384px"""
    model_id: str = "in1k-384px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN1K_MaxViT_L_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 512px"""
    model_id: str = "in1k-512px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


# 2C+. MaxViT (ImageNet-21K)
@dataclass
class Exp_7B_IN21K_MaxViT_B_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-B @ 224px"""
    model_id: str = "in21k-224px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21K_MaxViT_L_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-L @ 224px"""
    model_id: str = "in21k-224px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21K_MaxViT_XL_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-XL @ 224px"""
    model_id: str = "in21k-224px-maxvit-xl+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-xl"
    image_resize_strategy: str = "resize-crop"


# 2C+. MaxViT (IN21K -> IN1K)
@dataclass
class Exp_7B_IN21KFT_MaxViT_B_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 384px"""
    model_id: str = "in21kft-384px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21KFT_MaxViT_B_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 512px"""
    model_id: str = "in21kft-512px-maxvit-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 384px"""
    model_id: str = "in21kft-384px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 512px"""
    model_id: str = "in21kft-512px-maxvit-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-l"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 384px"""
    model_id: str = "in21kft-384px-maxvit-xl+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-xl"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_512px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 512px"""
    model_id: str = "in21kft-512px-maxvit-xl+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-xl"
    image_resize_strategy: str = "resize-crop"


# 2C+. MaxViT (Letterbox variants)
@dataclass
class Exp_7B_IN1K_MaxViT_T_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 224px (letterbox)"""
    model_id: str = "in1k-224px-maxvit-t-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-t"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_T_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 384px (letterbox)"""
    model_id: str = "in1k-384px-maxvit-t-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-t"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_T_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 512px (letterbox)"""
    model_id: str = "in1k-512px-maxvit-t-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-t"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 224px (letterbox)"""
    model_id: str = "in1k-224px-maxvit-s-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-s"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 384px (letterbox)"""
    model_id: str = "in1k-384px-maxvit-s-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-s"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 512px (letterbox)"""
    model_id: str = "in1k-512px-maxvit-s-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-s"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 224px (letterbox)"""
    model_id: str = "in1k-224px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 384px (letterbox)"""
    model_id: str = "in1k-384px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 512px (letterbox)"""
    model_id: str = "in1k-512px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 224px (letterbox)"""
    model_id: str = "in1k-224px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 384px (letterbox)"""
    model_id: str = "in1k-384px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 512px (letterbox)"""
    model_id: str = "in1k-512px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_B_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-B @ 224px (letterbox)"""
    model_id: str = "in21k-224px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_L_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-L @ 224px (letterbox)"""
    model_id: str = "in21k-224px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_XL_224px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-XL @ 224px (letterbox)"""
    model_id: str = "in21k-224px-maxvit-xl-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-xl"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_B_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 384px (letterbox)"""
    model_id: str = "in21kft-384px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_B_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 512px (letterbox)"""
    model_id: str = "in21kft-512px-maxvit-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 384px (letterbox)"""
    model_id: str = "in21kft-384px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 512px (letterbox)"""
    model_id: str = "in21kft-512px-maxvit-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_384px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 384px (letterbox)"""
    model_id: str = "in21kft-384px-maxvit-xl-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-xl"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_512px_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 512px (letterbox)"""
    model_id: str = "in21kft-512px-maxvit-xl-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-xl"
    image_resize_strategy: str = "letterbox"


# 2C+. MaxViT (Letterbox variants, stage 3)
@dataclass
class Exp_7B_IN1K_MaxViT_T_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 224px (letterbox, stage 3)"""
    model_id: str = "in1k-224px-maxvit-t-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-t-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_T_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 384px (letterbox, stage 3)"""
    model_id: str = "in1k-384px-maxvit-t-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-t-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_T_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-T @ 512px (letterbox, stage 3)"""
    model_id: str = "in1k-512px-maxvit-t-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-t-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 224px (letterbox, stage 3)"""
    model_id: str = "in1k-224px-maxvit-s-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-s-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 384px (letterbox, stage 3)"""
    model_id: str = "in1k-384px-maxvit-s-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-s-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_S_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-S @ 512px (letterbox, stage 3)"""
    model_id: str = "in1k-512px-maxvit-s-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-s-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 224px (letterbox, stage 3)"""
    model_id: str = "in1k-224px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 384px (letterbox, stage 3)"""
    model_id: str = "in1k-384px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_B_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-B @ 512px (letterbox, stage 3)"""
    model_id: str = "in1k-512px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 224px (letterbox, stage 3)"""
    model_id: str = "in1k-224px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-224px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 384px (letterbox, stage 3)"""
    model_id: str = "in1k-384px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-384px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN1K_MaxViT_L_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN1K MaxViT-L @ 512px (letterbox, stage 3)"""
    model_id: str = "in1k-512px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in1k-512px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_B_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-B @ 224px (letterbox, stage 3)"""
    model_id: str = "in21k-224px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_L_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-L @ 224px (letterbox, stage 3)"""
    model_id: str = "in21k-224px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21K_MaxViT_XL_224px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21K MaxViT-XL @ 224px (letterbox, stage 3)"""
    model_id: str = "in21k-224px-maxvit-xl-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21k-224px-maxvit-xl-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_B_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 384px (letterbox, stage 3)"""
    model_id: str = "in21kft-384px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_B_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-B @ 512px (letterbox, stage 3)"""
    model_id: str = "in21kft-512px-maxvit-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-b-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 384px (letterbox, stage 3)"""
    model_id: str = "in21kft-384px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_L_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-L @ 512px (letterbox, stage 3)"""
    model_id: str = "in21kft-512px-maxvit-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-l-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_384px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 384px (letterbox, stage 3)"""
    model_id: str = "in21kft-384px-maxvit-xl-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-384px-maxvit-xl-s3"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_IN21KFT_MaxViT_XL_512px_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + IN21KFT MaxViT-XL @ 512px (letterbox, stage 3)"""
    model_id: str = "in21kft-512px-maxvit-xl-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "in21kft-512px-maxvit-xl-s3"
    image_resize_strategy: str = "letterbox"

# 2D. Vim Variants (Tiny/Small/Base)
@dataclass
class Exp_7B_Vim_Tiny_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Tiny"""
    model_id: str = "vim-tiny+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-tiny"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_Vim_Tiny_FT_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Tiny (finetuned)"""
    model_id: str = "vim-tiny-ft+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-tiny-ft"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_Vim_Small_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Small"""
    model_id: str = "vim-small+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-small"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_Vim_Small_FT_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Small (finetuned)"""
    model_id: str = "vim-small-ft+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-small-ft"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_Vim_Base_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Base"""
    model_id: str = "vim-base+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-base"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "resize-crop"


# 2D. Vim Variants [Letterbox Legacy]
@dataclass
class Exp_7B_Vim_Tiny_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Tiny (letterbox)"""
    model_id: str = "vim-tiny-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-tiny"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_Vim_Tiny_FT_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Tiny (finetuned, letterbox)"""
    model_id: str = "vim-tiny-ft-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-tiny-ft"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_Vim_Small_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Small (letterbox)"""
    model_id: str = "vim-small-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-small"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_Vim_Small_FT_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Small (finetuned, letterbox)"""
    model_id: str = "vim-small-ft-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-small-ft"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_Vim_Base_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vim Base (letterbox)"""
    model_id: str = "vim-base-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vim-base"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


# 2E. MambaVision Variants (T/T2/S/B/L/L2 + 21K)
@dataclass
class Exp_7B_MambaVision_T_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T"""
    model_id: str = "mambavision-t+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_T2_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T2"""
    model_id: str = "mambavision-t2+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t2"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_S_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-S"""
    model_id: str = "mambavision-s+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-s"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_B_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B"""
    model_id: str = "mambavision-b+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_B_21K_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B (IN21K)"""
    model_id: str = "mambavision-b-21k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L"""
    model_id: str = "mambavision-l+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L_21K_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L (IN21K)"""
    model_id: str = "mambavision-l-21k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L2_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2"""
    model_id: str = "mambavision-l2+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L2_512_21K_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2 (IN21K, 512px)"""
    model_id: str = "mambavision-l2-512-21k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2-512-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L3_256_21K_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 256px)"""
    model_id: str = "mambavision-l3-256-21k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-256-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


@dataclass
class Exp_7B_MambaVision_L3_512_21K_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 512px)"""
    model_id: str = "mambavision-l3-512-21k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-512-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "mambavision-classification"


# 2E. MambaVision Variants [Letterbox Legacy]
@dataclass
class Exp_7B_MambaVision_T_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T (letterbox)"""
    model_id: str = "mambavision-t-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_T2_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T2 (letterbox)"""
    model_id: str = "mambavision-t2-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t2"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_S_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-S (letterbox)"""
    model_id: str = "mambavision-s-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-s"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_B_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B (letterbox)"""
    model_id: str = "mambavision-b-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_B_21K_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B (IN21K, letterbox)"""
    model_id: str = "mambavision-b-21k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L (letterbox)"""
    model_id: str = "mambavision-l-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L_21K_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L (IN21K, letterbox)"""
    model_id: str = "mambavision-l-21k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L2_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2 (letterbox)"""
    model_id: str = "mambavision-l2-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L2_512_21K_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2 (IN21K, 512px, letterbox)"""
    model_id: str = "mambavision-l2-512-21k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2-512-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L3_256_21K_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 256px, letterbox)"""
    model_id: str = "mambavision-l3-256-21k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-256-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L3_512_21K_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 512px, letterbox)"""
    model_id: str = "mambavision-l3-512-21k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-512-21k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


# 2E. MambaVision Variants [Letterbox Legacy, Stage 3]
@dataclass
class Exp_7B_MambaVision_T_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T (letterbox, stage 3)"""
    model_id: str = "mambavision-t-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_T2_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-T2 (letterbox, stage 3)"""
    model_id: str = "mambavision-t2-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-t2-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_S_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-S (letterbox, stage 3)"""
    model_id: str = "mambavision-s-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-s-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_B_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B (letterbox, stage 3)"""
    model_id: str = "mambavision-b-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_B_21K_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-B (IN21K, letterbox, stage 3)"""
    model_id: str = "mambavision-b-21k-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-b-21k-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L (letterbox, stage 3)"""
    model_id: str = "mambavision-l-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L_21K_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L (IN21K, letterbox, stage 3)"""
    model_id: str = "mambavision-l-21k-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l-21k-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L2_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2 (letterbox, stage 3)"""
    model_id: str = "mambavision-l2-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L2_512_21K_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L2 (IN21K, 512px, letterbox, stage 3)"""
    model_id: str = "mambavision-l2-512-21k-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l2-512-21k-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L3_256_21K_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 256px, letterbox, stage 3)"""
    model_id: str = "mambavision-l3-256-21k-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-256-21k-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_MambaVision_L3_512_21K_Letterbox_S3_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + MambaVision-L3 (IN21K, 512px, letterbox, stage 3)"""
    model_id: str = "mambavision-l3-512-21k-letterbox-s3+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "mambavision-l3-512-21k-s3"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"

# 2F. VMamba Variants (Tiny/Small/Base)
@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8]"""
    model_id: str = "vmamba-tiny-s1l8+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"

@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (fused projector)"""
    model_id: str = "vmamba-tiny-s1l8-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"

@dataclass
class Exp_7B_VMamba_Tiny_S2L5_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s2l5]"""
    model_id: str = "vmamba-tiny-s2l5+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s2l5"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Tiny_Vanilla_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny (vanilla)"""
    model_id: str = "vmamba-tiny-vanilla+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-vanilla"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Small_S2L15_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15]"""
    model_id: str = "vmamba-small-s2l15+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Small_S1L20_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s1l20]"""
    model_id: str = "vmamba-small-s1l20+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s1l20"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Small_Vanilla_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small (vanilla)"""
    model_id: str = "vmamba-small-vanilla+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-vanilla"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Base_S2L15_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15]"""
    model_id: str = "vmamba-base-s2l15+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Base_S1L20_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s1l20]"""
    model_id: str = "vmamba-base-s1l20+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s1l20"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


@dataclass
class Exp_7B_VMamba_Base_Vanilla_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base (vanilla)"""
    model_id: str = "vmamba-base-vanilla+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-vanilla"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-classification"


# 2F. VMamba Variants (Tiny/Small/Base) [Letterbox Legacy]
@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Tiny_S2L5_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s2l5] (letterbox)"""
    model_id: str = "vmamba-tiny-s2l5-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s2l5"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S2L15_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (letterbox)"""
    model_id: str = "vmamba-small-s2l15-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S2L15_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (letterbox, fused projector)"""
    model_id: str = "vmamba-small-s2l15-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S1L20_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s1l20] (letterbox)"""
    model_id: str = "vmamba-small-s1l20-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s1l20"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_Vanilla_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small (vanilla, letterbox)"""
    model_id: str = "vmamba-small-vanilla-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-vanilla"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S2L15_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (letterbox)"""
    model_id: str = "vmamba-base-s2l15-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (letterbox, fused projector)"""
    model_id: str = "vmamba-tiny-s1l8-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S2L15_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (letterbox, fused projector)"""
    model_id: str = "vmamba-base-s2l15-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S1L20_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s1l20] (letterbox)"""
    model_id: str = "vmamba-base-s1l20-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s1l20"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_Vanilla_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base (vanilla, letterbox)"""
    model_id: str = "vmamba-base-vanilla-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-vanilla"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


# 2F. VMamba Variants (Tiny/Small/Base) [Letterbox Square]
@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (256x256 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Small_S2L15_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (256x256 letterbox)"""
    model_id: str = "vmamba-small-s2l15-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Base_S2L15_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (256x256 letterbox)"""
    model_id: str = "vmamba-base-s2l15-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (512x512 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Small_S2L15_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (512x512 letterbox)"""
    model_id: str = "vmamba-small-s2l15-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Base_S2L15_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (512x512 letterbox)"""
    model_id: str = "vmamba-base-s2l15-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


# 2E. VMamba Downstream Finetunes (COCO detection + ADE20K segmentation)
@dataclass
class Exp_7B_VMamba_Tiny_Vanilla_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Tiny (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-vanilla-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-vanilla-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Small_Vanilla_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Small (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-small-vanilla-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-vanilla-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Base_Vanilla_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Base (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-base-vanilla-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-vanilla-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_S2L5_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s2l5] (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-s2l5-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s2l5-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_Vanilla_MaskRCNN_3x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Tiny (MaskRCNN@3x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-vanilla-maskrcnn-3x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-vanilla-det-maskrcnn-3x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Small_Vanilla_MaskRCNN_3x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Small (MaskRCNN@3x COCO checkpoint)"""
    model_id: str = "vmamba-small-vanilla-maskrcnn-3x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-vanilla-det-maskrcnn-3x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_S2L5_MaskRCNN_3x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s2l5] (MaskRCNN@3x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-s2l5-maskrcnn-3x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s2l5-det-maskrcnn-3x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_3x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny (MaskRCNN@3x COCO checkpoint)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-3x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-3x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_3x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small (MaskRCNN@3x COCO checkpoint)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-3x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-3x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base (MaskRCNN@1x COCO checkpoint)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_BS8_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base (MaskRCNN@1x COCO checkpoint, batch size 8 variant)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-bs8+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x-bs8"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-detection"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, letterbox)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, letterbox)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_512_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox, fused projector)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-512-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox, fused projector)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-512-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 512x512 letterbox, fused projector)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-512-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-512"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_224_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, 224x224 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-224+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-224"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_224_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, 224x224 letterbox)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-224+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-224"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_224_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 224x224 letterbox)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-224+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-224"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, 256x256 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, 256x256 letterbox)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_256_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 256x256 letterbox)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-256+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-256"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_1024_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (MaskRCNN@1x COCO checkpoint, 1024x1024 letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-maskrcnn-1x-letterbox-1024+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-1024"


@dataclass
class Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (MaskRCNN@1x COCO checkpoint, 1024x1024 letterbox)"""
    model_id: str = "vmamba-small-s2l15-maskrcnn-1x-letterbox-1024+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-1024"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 1024x1024 letterbox)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-1024+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-1024"


@dataclass
class Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (MaskRCNN@1x COCO checkpoint, 1024x1024 letterbox, fused projector)"""
    model_id: str = "vmamba-base-s2l15-maskrcnn-1x-letterbox-1024-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-det-maskrcnn-1x"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox-square-1024"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-tiny-s1l8-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Tiny_S2L5_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s2l5] (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-tiny-s2l5-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s2l5-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Tiny_Vanilla_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Tiny (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-tiny-vanilla-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-vanilla-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Small_S2L15_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-small-s2l15-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Small_Vanilla_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Small (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-small-vanilla-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-vanilla-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Base_S2L15_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-base-s2l15-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Base_Vanilla_ADE20K_UperNet_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + Vanilla VMamba Base (UperNet ADE20K checkpoint)"""
    model_id: str = "vmamba-base-vanilla-ade20k+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-vanilla-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vmamba-segmentation"


@dataclass
class Exp_7B_VMamba_Tiny_S1L8_ADE20K_UperNet_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Tiny [s1l8] (UperNet ADE20K checkpoint, letterbox)"""
    model_id: str = "vmamba-tiny-s1l8-ade20k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-tiny-s1l8-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Small_S2L15_ADE20K_UperNet_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Small [s2l15] (UperNet ADE20K checkpoint, letterbox)"""
    model_id: str = "vmamba-small-s2l15-ade20k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-small-s2l15-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_VMamba_Base_S2L15_ADE20K_UperNet_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + VMamba Base [s2l15] (UperNet ADE20K checkpoint, letterbox)"""
    model_id: str = "vmamba-base-s2l15-ade20k-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vmamba-base-s2l15-seg-ade20k"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"

# 2F. ViTDet (Mask R-CNN COCO)
@dataclass
class Exp_7B_ViTDet_B_MaskRCNN_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-B (Mask R-CNN COCO checkpoint)"""
    model_id: str = "vitdet-b-maskrcnn+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-b-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vitdet-detection"


@dataclass
class Exp_7B_ViTDet_L_MaskRCNN_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-L (Mask R-CNN COCO checkpoint)"""
    model_id: str = "vitdet-l-maskrcnn+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-l-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vitdet-detection"


@dataclass
class Exp_7B_ViTDet_H_MaskRCNN_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-H (Mask R-CNN COCO checkpoint)"""
    model_id: str = "vitdet-h-maskrcnn+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-h-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vitdet-detection"


@dataclass
class Exp_7B_ViTDet_B_MaskRCNN_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-B (Mask R-CNN COCO checkpoint, letterbox)"""
    model_id: str = "vitdet-b-maskrcnn-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-b-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTDet_B_MaskRCNN_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-B (Mask R-CNN COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vitdet-b-maskrcnn-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vitdet-b-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTDet_L_MaskRCNN_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-L (Mask R-CNN COCO checkpoint, letterbox)"""
    model_id: str = "vitdet-l-maskrcnn-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-l-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTDet_L_MaskRCNN_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-L (Mask R-CNN COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vitdet-l-maskrcnn-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vitdet-l-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTDet_H_MaskRCNN_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-H (Mask R-CNN COCO checkpoint, letterbox)"""
    model_id: str = "vitdet-h-maskrcnn-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vitdet-h-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTDet_H_MaskRCNN_Letterbox_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViTDet-H (Mask R-CNN COCO checkpoint, letterbox, fused projector)"""
    model_id: str = "vitdet-h-maskrcnn-letterbox-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "vitdet-h-maskrcnn"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


# 2G. ViT-Adapter (ADE20K segmentation)
# NOTE: ViT-Adapter-T ADE20K checkpoint link is broken upstream (404); disable until fixed.
# @dataclass
# class Exp_7B_ViTAdapter_UperNet_DeiT_T_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
#     """Vicuña-v1.5-7B + ViT-Adapter-T (UperNet ADE20K, 512px)"""
#     model_id: str = "vit-adapter-upernet-deit-t-ade20k-512+7b-vicuna"
#     arch_specifier: str = "no-align+gelu-mlp"
#     vision_backbone_id: str = "vit-adapter-upernet-deit-t-ade20k-512"
#     llm_backbone_id: str = "vicuna-v15-7b"
#     image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_DeiT_S_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-S (UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-deit-s-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-deit-s-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_DeiT_B_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-B (UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-deit-b-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-deit-b-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_T_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-T (AugReg, UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-augreg-t-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-t-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_B_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-B (AugReg, UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-augreg-b-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-b-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_L_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (AugReg, UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-augreg-l-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-l-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_UniPerceiver_L_ADE20K_512_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Uni-Perceiver, UperNet ADE20K, 512px)"""
    model_id: str = "vit-adapter-upernet-uniperceiver-l-ade20k-512+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-uniperceiver-l-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_BEiT_L_ADE20K_640_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (BEiT, UperNet ADE20K, 640px)"""
    model_id: str = "vit-adapter-upernet-beit-l-ade20k-640+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-beit-l-ade20k-640"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiT_L_ADE20K_640_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiT, ADE20K, 640px)"""
    model_id: str = "vit-adapter-mask2former-beit-l-ade20k-640+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beit-l-ade20k-640"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiT_L_COCO_ADE20K_896_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiT+COCO, ADE20K, 896px)"""
    model_id: str = "vit-adapter-mask2former-beit-l-coco-ade20k-896+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beit-l-coco-ade20k-896"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiTv2_L_COCO_ADE20K_896_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiTv2+COCO, ADE20K, 896px)"""
    model_id: str = "vit-adapter-mask2former-beitv2-l-coco-ade20k-896+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beitv2-l-coco-ade20k-896"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "vit-adapter-segmentation"


@dataclass
class Exp_7B_ViTAdapter_UperNet_DeiT_S_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-S (UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-deit-s-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-deit-s-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_DeiT_B_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-B (UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-deit-b-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-deit-b-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_T_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-T (AugReg, UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-augreg-t-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-t-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_B_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-B (AugReg, UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-augreg-b-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-b-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_AugReg_L_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (AugReg, UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-augreg-l-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-augreg-l-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_UniPerceiver_L_ADE20K_512_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Uni-Perceiver, UperNet ADE20K, 512px, letterbox)"""
    model_id: str = "vit-adapter-upernet-uniperceiver-l-ade20k-512-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-uniperceiver-l-ade20k-512"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_UperNet_BEiT_L_ADE20K_640_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (BEiT, UperNet ADE20K, 640px, letterbox)"""
    model_id: str = "vit-adapter-upernet-beit-l-ade20k-640-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-upernet-beit-l-ade20k-640"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiT_L_ADE20K_640_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiT, ADE20K, 640px, letterbox)"""
    model_id: str = "vit-adapter-mask2former-beit-l-ade20k-640-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beit-l-ade20k-640"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiT_L_COCO_ADE20K_896_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiT+COCO, ADE20K, 896px, letterbox)"""
    model_id: str = "vit-adapter-mask2former-beit-l-coco-ade20k-896-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beit-l-coco-ade20k-896"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_ViTAdapter_Mask2Former_BEiTv2_L_COCO_ADE20K_896_Letterbox_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + ViT-Adapter-L (Mask2Former BEiTv2+COCO, ADE20K, 896px, letterbox)"""
    model_id: str = "vit-adapter-mask2former-beitv2-l-coco-ade20k-896-letterbox+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "vit-adapter-mask2former-beitv2-l-coco-ade20k-896"
    llm_backbone_id: str = "vicuna-v15-7b"
    image_resize_strategy: str = "letterbox"

# ------------------------------------------------------------------------------
# Section 3 :: Original Vision Backbones (CLIP / SigLIP / DINO families)
# ------------------------------------------------------------------------------
#
# 3B. Vicuña-v1.5-7B Head
@dataclass
class Exp_7B_CLIP_ViT_L_336px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + CLIP ViT-L/14 @ 336px"""
    model_id: str = "clip-336px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_CLIP_ViT_B_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + CLIP ViT-B/16 @ 224px"""
    model_id: str = "clip-vit-b-224px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "clip-vit-b"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_CLIP_ViT_L_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + CLIP ViT-L/14 @ 224px"""
    model_id: str = "clip-vit-l-224px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "clip-vit-l"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_B16_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + SigLIP ViT-B/16 @ 224px"""
    model_id: str = "siglip-vit-b16-224px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "siglip-vit-b16-224px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_B16_256px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + SigLIP ViT-B/16 @ 256px"""
    model_id: str = "siglip-vit-b16-256px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "siglip-vit-b16-256px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_B16_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + SigLIP ViT-B/16 @ 384px"""
    model_id: str = "siglip-vit-b16-384px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "siglip-vit-b16-384px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_SO400M_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + SigLIP SO400M @ 224px"""
    model_id: str = "siglip-vit-so400m-224px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "siglip-vit-so400m"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_SO400M_384px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + SigLIP SO400M @ 384px"""
    model_id: str = "siglip-vit-so400m-384px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOv2_ViT_L_224px_Vicuna(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOv2 ViT-L/14 @ 224px"""
    model_id: str = "dinov2-224px+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "dinov2-vit-l"


@dataclass
class Exp_7B_DINOCLIP_ViT_L_336px_Vicuna_MLP(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOCLIP ViT-L/14 @ 336px (MLP projector)"""
    model_id: str = "dinoclip-336px-mlp+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "dinoclip-vit-l-336px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOCLIP_ViT_L_336px_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOCLIP ViT-L/14 @ 336px (fused projector)"""
    model_id: str = "dinoclip-336px-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "dinoclip-vit-l-336px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_384px_Vicuna_MLP(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOSigLIP ViT-SO/14 @ 384px (MLP projector)"""
    model_id: str = "dinosiglip-384px-mlp+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_384px_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOSigLIP ViT-SO/14 @ 384px (fused projector)"""
    model_id: str = "dinosiglip-384px-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_224px_Vicuna_MLP(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOSigLIP ViT-SO/14 @ 224px (MLP projector)"""
    model_id: str = "dinosiglip-224px-mlp+7b-vicuna"
    arch_specifier: str = "no-align+gelu-mlp"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_224px_Vicuna_Fused(LLaVa_v15_Reproduction_7B):
    """Vicuña-v1.5-7B + DINOSigLIP ViT-SO/14 @ 224px (fused projector)"""
    model_id: str = "dinosiglip-224px-fused+7b-vicuna"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "letterbox"




# === Section 4.1 :: Optimization Procedure ===


# Section 4.1A :: 🚀 --> Necessity of Multi-Stage Training
@dataclass
class Exp_7B_One_Stage(LLaVa_v15_Reproduction_7B):
    model_id: str = "one-stage+7b"
    arch_specifier: str = "no-align+gelu-mlp"


@dataclass
class Exp_13B_One_Stage(LLaVa_v15_Reproduction_13B):
    model_id: str = "one-stage+13b"
    arch_specifier: str = "no-align+gelu-mlp"


# Section 4.1B :: 🛠️ --> Full Finetuning through Visual Backbones
#   =>> Note :: Run with `--stage full-finetune`
@dataclass
class Exp_7B_Full_Finetune_Multi_Stage(LLaVa_v15_Reproduction_7B):
    model_id: str = "full-ft-multi-stage+7b"


@dataclass
class Exp_7B_Full_Finetune_One_Stage(Exp_7B_One_Stage):
    model_id: str = "full-ft-one-stage+7b"


# === Section 4.2 :: Image Processing and Visual Representations ===


# Section 4.2A :: 📸 --> Choosing a Pretrained Representation
@dataclass
class Exp_7B_IN1KFT_ViT_L_p16_224px_VisionFT(LLaVa_v15_Reproduction_7B):
    model_id: str = "in1kft-224px+7b-visionft"
    vision_backbone_id: str = "in1k-vit-l"


@dataclass
class Exp_7B_DINOv2_ViT_L_p14_224px(Exp_7B_One_Stage):
    model_id: str = "dinov2-224px+7b"
    vision_backbone_id: str = "dinov2-vit-l"


@dataclass
class Exp_7B_CLIP_ViT_L_p14_224px(Exp_7B_One_Stage):
    model_id: str = "clip-224px+7b"
    vision_backbone_id: str = "clip-vit-l"


@dataclass
class Exp_7B_SigLIP_ViT_SO_p14_224px(Exp_7B_One_Stage):
    model_id: str = "siglip-224px+7b"
    vision_backbone_id: str = "siglip-vit-so400m"


# Section 4.2B :: 📐 --> Choosing an Image Preprocessing Strategy
@dataclass
class Exp_7B_CLIP_ViT_L_p14_336px_Resize_Crop(Exp_7B_One_Stage):
    model_id: str = "clip-336px-resize-crop+7b"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_CLIP_ViT_L_p14_336px_Resize_Naive(Exp_7B_One_Stage):
    model_id: str = "clip-336px-resize-naive+7b"
    image_resize_strategy: str = "resize-naive"


@dataclass
class Exp_7B_SigLIP_ViT_SO_p14_384px_Letterbox(Exp_7B_One_Stage):
    model_id: str = "siglip-384px-letterbox+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "letterbox"


@dataclass
class Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Crop(Exp_7B_One_Stage):
    model_id: str = "siglip-384px-resize-crop+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-crop"


@dataclass
class Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Naive(Exp_7B_One_Stage):
    model_id: str = "siglip-384px-resize-naive+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"


# Section 4.2D :: 🥞 --> Stacking/Ensembling Visual Representations
@dataclass
class Exp_7B_DINOCLIP_ViT_L_p14_336px_Letterbox(Exp_7B_One_Stage):
    model_id: str = "dinoclip-336px-letterbox+7b"
    vision_backbone_id: str = "dinoclip-vit-l-336px"
    image_resize_strategy: str = "letterbox"
    arch_specifier: str = "no-align+fused-gelu-mlp"


@dataclass
class Exp_7B_DINOCLIP_ViT_L_p14_336px_Resize_Naive(Exp_7B_One_Stage):
    model_id: str = "dinoclip-336px-resize-naive+7b"
    vision_backbone_id: str = "dinoclip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    arch_specifier: str = "no-align+fused-gelu-mlp"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_p14_384px_Letterbox(Exp_7B_One_Stage):
    model_id: str = "dinosiglip-384px-letterbox+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "letterbox"
    arch_specifier: str = "no-align+fused-gelu-mlp"


@dataclass
class Exp_7B_DINOSigLIP_ViT_L_p14_384px_Resize_Naive(Exp_7B_One_Stage):
    model_id: str = "dinosiglip-384px-resize-naive+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    arch_specifier: str = "no-align+fused-gelu-mlp"


# === Section 4.3 :: Language Models ===


# Section 4.3A :: 📝 --> Base vs. Instruct-Tuned (Chat) LLMs
@dataclass
class Exp_7B_Llama2(Exp_7B_One_Stage):
    model_id: str = "llama2+7b"
    llm_backbone_id: str = "llama2-7b-pure"


@dataclass
class Exp_13B_Llama2(Exp_13B_One_Stage):
    model_id: str = "llama2+13b"
    llm_backbone_id: str = "llama2-13b-pure"


# ~ Additional LLM Backbones :: LLaMa-2 Chat, Mistral v0.1, Mistral v0.1 Instruct, Phi-2 ~
@dataclass
class Ext_Exp_7B_Llama2_Chat(Exp_7B_One_Stage):
    model_id: str = "llama2-chat+7b"
    llm_backbone_id: str = "llama2-7b-chat"


@dataclass
class Ext_Exp_13B_Llama2_Chat(Exp_13B_One_Stage):
    model_id: str = "llama2-chat+13b"
    llm_backbone_id: str = "llama2-13b-chat"


@dataclass
class Ext_Exp_7B_Mistral_V1(Exp_7B_One_Stage):
    model_id: str = "mistral-v0.1+7b"
    llm_backbone_id: str = "mistral-v0.1-7b-pure"


@dataclass
class Ext_Exp_7B_Mistral_Instruct_V1(Exp_7B_One_Stage):
    model_id: str = "mistral-instruct-v0.1+7b"
    llm_backbone_id: str = "mistral-v0.1-7b-instruct"


@dataclass
class Ext_Exp_3B_Phi_2(Exp_7B_One_Stage):
    model_id: str = "phi-2+3b"
    llm_backbone_id: str = "phi-2-3b"


# Section 4.3B :: ✌️ --> Co-training on Language-only Data
#   =>> Note :: Run with `--dataset.type "llava-multimodal" (multimodal data only / no co-training)
@dataclass
class Exp_7B_Vicuna_No_Cotraining(Exp_7B_One_Stage):
    model_id: str = "vicuna-no-cotraining+7b"


@dataclass
class Exp_7B_Llama2_No_Cotraining(Exp_7B_One_Stage):
    model_id: str = "llama2-no-cotraining+7b"
    llm_backbone_id: str = "llama2-7b-pure"


# === Section 4.4 :: Scaling Properties - Train Time & Data ===


# Section 4.4A :: ⏰ --> Scaling Train Time
@dataclass
class Exp_7B_1p25_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-1.25-epochs+7b"
    finetune_max_steps: int = 6500


@dataclass
class Exp_7B_1p5_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-1.5-epochs+7b"
    finetune_max_steps: int = 7800


@dataclass
class Exp_7B_2_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-2-epochs+7b"
    finetune_epochs: int = 2


@dataclass
class Exp_7B_3_Epochs(Exp_7B_One_Stage):
    model_id: str = "train-3-epochs+7b"
    finetune_epochs: int = 3


# Section 4.4B :: 📚 --> Scaling Data
#   =>> Note :: Run with `--dataset.type "llava-lvis4v"`
@dataclass
class Exp_7B_LLaVa_LVIS4V(Exp_7B_One_Stage):
    model_id: str = "llava-lvis4v+7b"


#   =>> Note :: Run with `--dataset.type "llava-lrv"`
@dataclass
class Exp_7B_LLaVa_LRV(Exp_7B_One_Stage):
    model_id: str = "llava-lrv+7b"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Exp_7B_LLaVa_LVIS4V_LRV(Exp_7B_One_Stage):
    model_id: str = "llava-lvis4v-lrv+7b"


# === Section 5 :: Prisms ===


# Prism-CLIP
@dataclass
class Prism_7B_CLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-clip-controlled+7b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"


@dataclass
class Prism_13B_CLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-clip-controlled+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_CLIP(Exp_7B_One_Stage):
    model_id: str = "prism-clip+7b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_CLIP(Exp_13B_One_Stage):
    model_id: str = "prism-clip+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    finetune_epochs: int = 2


# Prism-SigLIP
@dataclass
class Prism_7B_SigLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-siglip-controlled+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"


@dataclass
class Prism_13B_SigLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-siglip-controlled+13b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_SigLIP(Exp_7B_One_Stage):
    model_id: str = "prism-siglip+7b"
    vision_backbone_id: str = "siglip-vit-so400m-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_SigLIP(Exp_13B_One_Stage):
    model_id: str = "prism-siglip+13b"
    vision_backbone_id: str = "clip-vit-l-336px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    finetune_epochs: int = 2


# Prism-DINOSigLIP
@dataclass
class Prism_7B_DINOSigLIP_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-controlled+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


@dataclass
class Prism_13B_DINOSigLIP_Controlled(Exp_13B_One_Stage):
    model_id: str = "prism-dinosiglip-controlled+13b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_DINOSigLIP(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_13B_DINOSigLIP(Exp_13B_One_Stage):
    model_id: str = "prism-dinosiglip+13b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-13b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


# [Inference-Optimized] 224px Prism Models
@dataclass
class Prism_7B_DINOSigLIP_224px_Controlled(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-224px-controlled+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"


#   =>> Note :: Run with `--dataset.type "llava-lvis4v-lrv"`
@dataclass
class Prism_7B_DINOSigLIP_224px(Exp_7B_One_Stage):
    model_id: str = "prism-dinosiglip-224px+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "llama2-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2


# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # === LLaVa v1.5 Base Reproductions ===
    REPRODUCTION_7B = LLaVa_v15_Reproduction_7B
    REPRODUCTION_13B = LLaVa_v15_Reproduction_13B

    # === Section 4.1 :: Optimization Procedure ===
    EXP_ONE_STAGE_7B = Exp_7B_One_Stage
    EXP_ONE_STAGE_13B = Exp_13B_One_Stage

    EXP_FULL_FT_MULTI_STAGE = Exp_7B_Full_Finetune_Multi_Stage
    EXP_FULL_FT_ONE_STAGE = Exp_7B_Full_Finetune_One_Stage

      # === For Mamba-MLLM Project ===

    # GROUP 2: Vicuña-v1.5-7B + IN1K Vision Backbones (Baseline)
    EXP_IN1K_VIT_S_Vicuna = Exp_7B_IN1K_ViT_S_p16_224px_Vicuna
    EXP_IN1K_VIT_S_FUSED_Vicuna = Exp_7B_IN1K_ViT_S_p16_224px_Vicuna_Fused
    EXP_IN1K_VIT_B_Vicuna = Exp_7B_IN1K_ViT_B_p16_224px_Vicuna
    EXP_IN1KFT_VIT_S_Vicuna = Exp_7B_IN1KFT_ViT_S_p16_224px_Vicuna
    EXP_IN1KFT_VIT_B_Vicuna = Exp_7B_IN1KFT_ViT_B_p16_224px_Vicuna
    EXP_IN1KFT_VIT_B2_Vicuna = Exp_7B_IN1KFT_ViT_B2_p16_224px_Vicuna
    EXP_IN1KFT_VIT_L_Vicuna = Exp_7B_IN1KFT_ViT_L_p16_224px_Vicuna
    EXP_IN1KFT_VIT_T_Vicuna = Exp_7B_IN1KFT_ViT_T_p16_224px_Vicuna
    EXP_IN21K_VIT_T_Vicuna = Exp_7B_IN21K_ViT_T_p16_224px_Vicuna
    EXP_IN21K_VIT_S_Vicuna = Exp_7B_IN21K_ViT_S_p16_224px_Vicuna
    EXP_IN21K_VIT_B_Vicuna = Exp_7B_IN21K_ViT_B_p16_224px_Vicuna
    EXP_IN21K_VIT_L_Vicuna = Exp_7B_IN21K_ViT_L_p16_224px_Vicuna
    EXP_IN1K_MAXVIT_T_224PX_Vicuna = Exp_7B_IN1K_MaxViT_T_224px_Vicuna
    EXP_IN1K_MAXVIT_T_224PX_FUSED_Vicuna = Exp_7B_IN1K_MaxViT_T_224px_Vicuna_Fused
    EXP_IN1K_MAXVIT_T_384PX_Vicuna = Exp_7B_IN1K_MaxViT_T_384px_Vicuna
    EXP_IN1K_MAXVIT_T_512PX_Vicuna = Exp_7B_IN1K_MaxViT_T_512px_Vicuna
    EXP_IN1K_MAXVIT_S_224PX_Vicuna = Exp_7B_IN1K_MaxViT_S_224px_Vicuna
    EXP_IN1K_MAXVIT_S_384PX_Vicuna = Exp_7B_IN1K_MaxViT_S_384px_Vicuna
    EXP_IN1K_MAXVIT_S_512PX_Vicuna = Exp_7B_IN1K_MaxViT_S_512px_Vicuna
    EXP_IN1K_MAXVIT_B_224PX_Vicuna = Exp_7B_IN1K_MaxViT_B_224px_Vicuna
    EXP_IN1K_MAXVIT_B_384PX_Vicuna = Exp_7B_IN1K_MaxViT_B_384px_Vicuna
    EXP_IN1K_MAXVIT_B_512PX_Vicuna = Exp_7B_IN1K_MaxViT_B_512px_Vicuna
    EXP_IN1K_MAXVIT_L_224PX_Vicuna = Exp_7B_IN1K_MaxViT_L_224px_Vicuna
    EXP_IN1K_MAXVIT_L_384PX_Vicuna = Exp_7B_IN1K_MaxViT_L_384px_Vicuna
    EXP_IN1K_MAXVIT_L_512PX_Vicuna = Exp_7B_IN1K_MaxViT_L_512px_Vicuna
    EXP_IN21K_MAXVIT_B_224PX_Vicuna = Exp_7B_IN21K_MaxViT_B_224px_Vicuna
    EXP_IN21K_MAXVIT_L_224PX_Vicuna = Exp_7B_IN21K_MaxViT_L_224px_Vicuna
    EXP_IN21K_MAXVIT_XL_224PX_Vicuna = Exp_7B_IN21K_MaxViT_XL_224px_Vicuna
    EXP_IN21KFT_MAXVIT_B_384PX_Vicuna = Exp_7B_IN21KFT_MaxViT_B_384px_Vicuna
    EXP_IN21KFT_MAXVIT_B_512PX_Vicuna = Exp_7B_IN21KFT_MaxViT_B_512px_Vicuna
    EXP_IN21KFT_MAXVIT_L_384PX_Vicuna = Exp_7B_IN21KFT_MaxViT_L_384px_Vicuna
    EXP_IN21KFT_MAXVIT_L_512PX_Vicuna = Exp_7B_IN21KFT_MaxViT_L_512px_Vicuna
    EXP_IN21KFT_MAXVIT_XL_384PX_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_384px_Vicuna
    EXP_IN21KFT_MAXVIT_XL_512PX_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_512px_Vicuna
    EXP_IN1K_MAXVIT_T_224PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_T_224px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_T_384PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_T_384px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_T_512PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_T_512px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_S_224PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_S_224px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_S_384PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_S_384px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_S_512PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_S_512px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_B_224PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_B_224px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_B_384PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_B_384px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_B_512PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_B_512px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_L_224PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_L_224px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_L_384PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_L_384px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_L_512PX_LETTERBOX_Vicuna = Exp_7B_IN1K_MaxViT_L_512px_Letterbox_Vicuna
    EXP_IN21K_MAXVIT_B_224PX_LETTERBOX_Vicuna = Exp_7B_IN21K_MaxViT_B_224px_Letterbox_Vicuna
    EXP_IN21K_MAXVIT_L_224PX_LETTERBOX_Vicuna = Exp_7B_IN21K_MaxViT_L_224px_Letterbox_Vicuna
    EXP_IN21K_MAXVIT_XL_224PX_LETTERBOX_Vicuna = Exp_7B_IN21K_MaxViT_XL_224px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_B_384PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_B_384px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_B_512PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_B_512px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_L_384PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_L_384px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_L_512PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_L_512px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_XL_384PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_384px_Letterbox_Vicuna
    EXP_IN21KFT_MAXVIT_XL_512PX_LETTERBOX_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_512px_Letterbox_Vicuna
    EXP_IN1K_MAXVIT_T_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_T_224px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_T_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_T_384px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_T_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_T_512px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_S_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_S_224px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_S_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_S_384px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_S_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_S_512px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_B_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_B_224px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_B_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_B_384px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_B_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_B_512px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_L_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_L_224px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_L_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_L_384px_Letterbox_S3_Vicuna
    EXP_IN1K_MAXVIT_L_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN1K_MaxViT_L_512px_Letterbox_S3_Vicuna
    EXP_IN21K_MAXVIT_B_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21K_MaxViT_B_224px_Letterbox_S3_Vicuna
    EXP_IN21K_MAXVIT_L_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21K_MaxViT_L_224px_Letterbox_S3_Vicuna
    EXP_IN21K_MAXVIT_XL_224PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21K_MaxViT_XL_224px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_B_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_B_384px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_B_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_B_512px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_L_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_L_384px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_L_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_L_512px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_XL_384PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_384px_Letterbox_S3_Vicuna
    EXP_IN21KFT_MAXVIT_XL_512PX_LETTERBOX_S3_Vicuna = Exp_7B_IN21KFT_MaxViT_XL_512px_Letterbox_S3_Vicuna
    EXP_VIM_TINY_Vicuna = Exp_7B_Vim_Tiny_Vicuna
    EXP_VIM_TINY_FT_Vicuna = Exp_7B_Vim_Tiny_FT_Vicuna
    EXP_VIM_SMALL_Vicuna = Exp_7B_Vim_Small_Vicuna
    EXP_VIM_SMALL_FT_Vicuna = Exp_7B_Vim_Small_FT_Vicuna
    EXP_VIM_BASE_Vicuna = Exp_7B_Vim_Base_Vicuna
    EXP_VIM_TINY_LETTERBOX_Vicuna = Exp_7B_Vim_Tiny_Letterbox_Vicuna
    EXP_VIM_TINY_FT_LETTERBOX_Vicuna = Exp_7B_Vim_Tiny_FT_Letterbox_Vicuna
    EXP_VIM_SMALL_LETTERBOX_Vicuna = Exp_7B_Vim_Small_Letterbox_Vicuna
    EXP_VIM_SMALL_FT_LETTERBOX_Vicuna = Exp_7B_Vim_Small_FT_Letterbox_Vicuna
    EXP_VIM_BASE_LETTERBOX_Vicuna = Exp_7B_Vim_Base_Letterbox_Vicuna
    EXP_MAMBAVISION_T_Vicuna = Exp_7B_MambaVision_T_Vicuna
    EXP_MAMBAVISION_T2_Vicuna = Exp_7B_MambaVision_T2_Vicuna
    EXP_MAMBAVISION_S_Vicuna = Exp_7B_MambaVision_S_Vicuna
    EXP_MAMBAVISION_B_Vicuna = Exp_7B_MambaVision_B_Vicuna
    EXP_MAMBAVISION_B_21K_Vicuna = Exp_7B_MambaVision_B_21K_Vicuna
    EXP_MAMBAVISION_L_Vicuna = Exp_7B_MambaVision_L_Vicuna
    EXP_MAMBAVISION_L_21K_Vicuna = Exp_7B_MambaVision_L_21K_Vicuna
    EXP_MAMBAVISION_L2_Vicuna = Exp_7B_MambaVision_L2_Vicuna
    EXP_MAMBAVISION_L2_512_21K_Vicuna = Exp_7B_MambaVision_L2_512_21K_Vicuna
    EXP_MAMBAVISION_L3_256_21K_Vicuna = Exp_7B_MambaVision_L3_256_21K_Vicuna
    EXP_MAMBAVISION_L3_512_21K_Vicuna = Exp_7B_MambaVision_L3_512_21K_Vicuna
    EXP_MAMBAVISION_T_LETTERBOX_Vicuna = Exp_7B_MambaVision_T_Letterbox_Vicuna
    EXP_MAMBAVISION_T2_LETTERBOX_Vicuna = Exp_7B_MambaVision_T2_Letterbox_Vicuna
    EXP_MAMBAVISION_S_LETTERBOX_Vicuna = Exp_7B_MambaVision_S_Letterbox_Vicuna
    EXP_MAMBAVISION_B_LETTERBOX_Vicuna = Exp_7B_MambaVision_B_Letterbox_Vicuna
    EXP_MAMBAVISION_B_21K_LETTERBOX_Vicuna = Exp_7B_MambaVision_B_21K_Letterbox_Vicuna
    EXP_MAMBAVISION_L_LETTERBOX_Vicuna = Exp_7B_MambaVision_L_Letterbox_Vicuna
    EXP_MAMBAVISION_L_21K_LETTERBOX_Vicuna = Exp_7B_MambaVision_L_21K_Letterbox_Vicuna
    EXP_MAMBAVISION_L2_LETTERBOX_Vicuna = Exp_7B_MambaVision_L2_Letterbox_Vicuna
    EXP_MAMBAVISION_L2_512_21K_LETTERBOX_Vicuna = Exp_7B_MambaVision_L2_512_21K_Letterbox_Vicuna
    EXP_MAMBAVISION_L3_256_21K_LETTERBOX_Vicuna = Exp_7B_MambaVision_L3_256_21K_Letterbox_Vicuna
    EXP_MAMBAVISION_L3_512_21K_LETTERBOX_Vicuna = Exp_7B_MambaVision_L3_512_21K_Letterbox_Vicuna
    EXP_MAMBAVISION_T_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_T_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_T2_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_T2_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_S_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_S_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_B_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_B_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_B_21K_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_B_21K_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L_21K_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L_21K_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L2_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L2_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L2_512_21K_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L2_512_21K_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L3_256_21K_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L3_256_21K_Letterbox_S3_Vicuna
    EXP_MAMBAVISION_L3_512_21K_LETTERBOX_S3_Vicuna = Exp_7B_MambaVision_L3_512_21K_Letterbox_S3_Vicuna
    EXP_VMAMBA_TINY_S1L8_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Vicuna
    EXP_VMAMBA_TINY_S1L8_FUSED_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Vicuna_Fused
    EXP_VMAMBA_TINY_S2L5_Vicuna = Exp_7B_VMamba_Tiny_S2L5_Vicuna
    EXP_VMAMBA_TINY_VANILLA_Vicuna = Exp_7B_VMamba_Tiny_Vanilla_Vicuna
    EXP_VMAMBA_SMALL_S2L15_Vicuna = Exp_7B_VMamba_Small_S2L15_Vicuna
    EXP_VMAMBA_SMALL_S1L20_Vicuna = Exp_7B_VMamba_Small_S1L20_Vicuna
    EXP_VMAMBA_SMALL_VANILLA_Vicuna = Exp_7B_VMamba_Small_Vanilla_Vicuna
    EXP_VMAMBA_BASE_S2L15_Vicuna = Exp_7B_VMamba_Base_S2L15_Vicuna
    EXP_VMAMBA_BASE_S1L20_Vicuna = Exp_7B_VMamba_Base_S1L20_Vicuna
    EXP_VMAMBA_BASE_VANILLA_Vicuna = Exp_7B_VMamba_Base_Vanilla_Vicuna
    EXP_VMAMBA_TINY_S1L8_LETTERBOX_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Letterbox_Vicuna
    EXP_VMAMBA_TINY_S2L5_LETTERBOX_Vicuna = Exp_7B_VMamba_Tiny_S2L5_Letterbox_Vicuna
    EXP_VMAMBA_SMALL_S2L15_LETTERBOX_Vicuna = Exp_7B_VMamba_Small_S2L15_Letterbox_Vicuna
    EXP_VMAMBA_TINY_S1L8_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Letterbox_Vicuna_Fused
    EXP_VMAMBA_SMALL_S2L15_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Small_S2L15_Letterbox_Vicuna_Fused
    EXP_VMAMBA_SMALL_S1L20_LETTERBOX_Vicuna = Exp_7B_VMamba_Small_S1L20_Letterbox_Vicuna
    EXP_VMAMBA_SMALL_VANILLA_LETTERBOX_Vicuna = Exp_7B_VMamba_Small_Vanilla_Letterbox_Vicuna
    EXP_VMAMBA_BASE_S2L15_LETTERBOX_Vicuna = Exp_7B_VMamba_Base_S2L15_Letterbox_Vicuna
    EXP_VMAMBA_BASE_S2L15_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Base_S2L15_Letterbox_Vicuna_Fused
    EXP_VMAMBA_TINY_S1L8_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Letterbox_256_Vicuna
    EXP_VMAMBA_SMALL_S2L15_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Small_S2L15_Letterbox_256_Vicuna
    EXP_VMAMBA_BASE_S2L15_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Base_S2L15_Letterbox_256_Vicuna
    EXP_VMAMBA_TINY_S1L8_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Tiny_S1L8_Letterbox_512_Vicuna
    EXP_VMAMBA_SMALL_S2L15_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Small_S2L15_Letterbox_512_Vicuna
    EXP_VMAMBA_BASE_S2L15_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Base_S2L15_Letterbox_512_Vicuna
    EXP_VMAMBA_BASE_S1L20_LETTERBOX_Vicuna = Exp_7B_VMamba_Base_S1L20_Letterbox_Vicuna
    EXP_VMAMBA_BASE_VANILLA_LETTERBOX_Vicuna = Exp_7B_VMamba_Base_Vanilla_Letterbox_Vicuna
    EXP_VMAMBA_TINY_VANILLA_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Tiny_Vanilla_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_SMALL_VANILLA_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Small_Vanilla_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_BASE_VANILLA_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Base_Vanilla_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_TINY_S2L5_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Tiny_S2L5_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_TINY_VANILLA_MASKRCNN_3X_Vicuna = Exp_7B_VMamba_Tiny_Vanilla_MaskRCNN_3x_Vicuna
    EXP_VMAMBA_SMALL_VANILLA_MASKRCNN_3X_Vicuna = Exp_7B_VMamba_Small_Vanilla_MaskRCNN_3x_Vicuna
    EXP_VMAMBA_TINY_S2L5_MASKRCNN_3X_Vicuna = Exp_7B_VMamba_Tiny_S2L5_MaskRCNN_3x_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_3X_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_3x_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_3X_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_3x_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_BS8_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_BS8_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_Vicuna_Fused
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_Vicuna_Fused
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_FUSED_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_Vicuna_Fused
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_512_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_512_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_512_FUSED_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_512_Vicuna_Fused
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_512_FUSED_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna_Fused
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_512_FUSED_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_512_Vicuna_Fused
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_224_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_224_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_224_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_224_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_224_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_224_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_256_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_256_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_256_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_256_Vicuna
    EXP_VMAMBA_TINY_S1L8_MASKRCNN_1X_LETTERBOX_1024_Vicuna = Exp_7B_VMamba_Tiny_S1L8_MaskRCNN_1x_Letterbox_1024_Vicuna
    EXP_VMAMBA_SMALL_S2L15_MASKRCNN_1X_LETTERBOX_1024_Vicuna = Exp_7B_VMamba_Small_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_1024_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna
    EXP_VMAMBA_BASE_S2L15_MASKRCNN_1X_LETTERBOX_1024_FUSED_Vicuna = Exp_7B_VMamba_Base_S2L15_MaskRCNN_1x_Letterbox_1024_Vicuna_Fused
    EXP_VMAMBA_TINY_S1L8_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Tiny_S1L8_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_TINY_S2L5_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Tiny_S2L5_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_TINY_VANILLA_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Tiny_Vanilla_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_SMALL_S2L15_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Small_S2L15_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_SMALL_VANILLA_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Small_Vanilla_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_BASE_S2L15_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Base_S2L15_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_BASE_VANILLA_ADE20K_UPERNET_Vicuna = Exp_7B_VMamba_Base_Vanilla_ADE20K_UperNet_Vicuna
    EXP_VMAMBA_TINY_S1L8_ADE20K_UPERNET_LETTERBOX_Vicuna = Exp_7B_VMamba_Tiny_S1L8_ADE20K_UperNet_Letterbox_Vicuna
    EXP_VMAMBA_SMALL_S2L15_ADE20K_UPERNET_LETTERBOX_Vicuna = Exp_7B_VMamba_Small_S2L15_ADE20K_UperNet_Letterbox_Vicuna
    EXP_VMAMBA_BASE_S2L15_ADE20K_UPERNET_LETTERBOX_Vicuna = Exp_7B_VMamba_Base_S2L15_ADE20K_UperNet_Letterbox_Vicuna
    EXP_VITDET_B_MASKRCNN_Vicuna = Exp_7B_ViTDet_B_MaskRCNN_Vicuna
    EXP_VITDET_L_MASKRCNN_Vicuna = Exp_7B_ViTDet_L_MaskRCNN_Vicuna
    EXP_VITDET_H_MASKRCNN_Vicuna = Exp_7B_ViTDet_H_MaskRCNN_Vicuna
    EXP_VITDET_B_MASKRCNN_LETTERBOX_Vicuna = Exp_7B_ViTDet_B_MaskRCNN_Letterbox_Vicuna
    EXP_VITDET_B_MASKRCNN_LETTERBOX_FUSED_Vicuna = Exp_7B_ViTDet_B_MaskRCNN_Letterbox_Vicuna_Fused
    EXP_VITDET_L_MASKRCNN_LETTERBOX_Vicuna = Exp_7B_ViTDet_L_MaskRCNN_Letterbox_Vicuna
    EXP_VITDET_L_MASKRCNN_LETTERBOX_FUSED_Vicuna = Exp_7B_ViTDet_L_MaskRCNN_Letterbox_Vicuna_Fused
    EXP_VITDET_H_MASKRCNN_LETTERBOX_Vicuna = Exp_7B_ViTDet_H_MaskRCNN_Letterbox_Vicuna
    EXP_VITDET_H_MASKRCNN_LETTERBOX_FUSED_Vicuna = Exp_7B_ViTDet_H_MaskRCNN_Letterbox_Vicuna_Fused
    EXP_VITADAPTER_UPERNET_DEIT_S_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_DeiT_S_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_DEIT_B_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_DeiT_B_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_T_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_T_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_B_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_B_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_L_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_L_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_UNIPERCEIVER_L_ADE20K_512_Vicuna = Exp_7B_ViTAdapter_UperNet_UniPerceiver_L_ADE20K_512_Vicuna
    EXP_VITADAPTER_UPERNET_BEIT_L_ADE20K_640_Vicuna = Exp_7B_ViTAdapter_UperNet_BEiT_L_ADE20K_640_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEIT_L_ADE20K_640_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiT_L_ADE20K_640_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEIT_L_COCO_ADE20K_896_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiT_L_COCO_ADE20K_896_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEITV2_L_COCO_ADE20K_896_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiTv2_L_COCO_ADE20K_896_Vicuna
    EXP_VITADAPTER_UPERNET_DEIT_S_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_DeiT_S_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_DEIT_B_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_DeiT_B_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_T_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_T_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_B_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_B_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_AUGREG_L_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_AugReg_L_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_UNIPERCEIVER_L_ADE20K_512_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_UniPerceiver_L_ADE20K_512_Letterbox_Vicuna
    EXP_VITADAPTER_UPERNET_BEIT_L_ADE20K_640_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_UperNet_BEiT_L_ADE20K_640_Letterbox_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEIT_L_ADE20K_640_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiT_L_ADE20K_640_Letterbox_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEIT_L_COCO_ADE20K_896_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiT_L_COCO_ADE20K_896_Letterbox_Vicuna
    EXP_VITADAPTER_MASK2FORMER_BEITV2_L_COCO_ADE20K_896_LETTERBOX_Vicuna = Exp_7B_ViTAdapter_Mask2Former_BEiTv2_L_COCO_ADE20K_896_Letterbox_Vicuna


    EXP_CLIP_336PX_Vicuna = Exp_7B_CLIP_ViT_L_336px_Vicuna
    EXP_CLIP_VIT_B_224PX_Vicuna = Exp_7B_CLIP_ViT_B_224px_Vicuna
    EXP_CLIP_VIT_L_224PX_Vicuna = Exp_7B_CLIP_ViT_L_224px_Vicuna
    EXP_SIGLIP_VIT_B16_224PX_Vicuna = Exp_7B_SigLIP_ViT_B16_224px_Vicuna
    EXP_SIGLIP_VIT_B16_256PX_Vicuna = Exp_7B_SigLIP_ViT_B16_256px_Vicuna
    EXP_SIGLIP_VIT_B16_384PX_Vicuna = Exp_7B_SigLIP_ViT_B16_384px_Vicuna
    EXP_SIGLIP_SO400M_224PX_Vicuna = Exp_7B_SigLIP_ViT_SO400M_224px_Vicuna
    EXP_SIGLIP_SO400M_384PX_Vicuna = Exp_7B_SigLIP_ViT_SO400M_384px_Vicuna
    EXP_DINOV2_224PX_Vicuna = Exp_7B_DINOv2_ViT_L_224px_Vicuna
    EXP_DINOCLIP_336PX_MLP_Vicuna = Exp_7B_DINOCLIP_ViT_L_336px_Vicuna_MLP
    EXP_DINOCLIP_336PX_FUSED_Vicuna = Exp_7B_DINOCLIP_ViT_L_336px_Vicuna_Fused
    EXP_DINOSIGLIP_384PX_MLP_Vicuna = Exp_7B_DINOSigLIP_ViT_L_384px_Vicuna_MLP
    EXP_DINOSIGLIP_224PX_MLP_Vicuna = Exp_7B_DINOSigLIP_ViT_L_224px_Vicuna_MLP
    EXP_DINOSIGLIP_384PX_FUSED_Vicuna = Exp_7B_DINOSigLIP_ViT_L_384px_Vicuna_Fused
    EXP_DINOSIGLIP_224PX_FUSED_Vicuna = Exp_7B_DINOSigLIP_ViT_L_224px_Vicuna_Fused

    # === Section 4.2 :: Image Processing and Visual Representations ===
    # EXP_IN1KFT_224PX = Exp_7B_IN1KFT_ViT_L_p16_224px  # Commented out - use EXP_IN1KFT_VIT_L_Vicuna instead
    EXP_IN1KFT_224PX_VISION_FT = Exp_7B_IN1KFT_ViT_L_p16_224px_VisionFT
    EXP_DINOV2_224PX = Exp_7B_DINOv2_ViT_L_p14_224px
    EXP_CLIP_224PX = Exp_7B_CLIP_ViT_L_p14_224px
    EXP_SIGLIP_224PX = Exp_7B_SigLIP_ViT_SO_p14_224px

    EXP_CLIP_336PX_RESIZE_CROP = Exp_7B_CLIP_ViT_L_p14_336px_Resize_Crop
    EXP_CLIP_336PX_RESIZE_NAIVE = Exp_7B_CLIP_ViT_L_p14_336px_Resize_Naive
    EXP_SIGLIP_384PX_LETTERBOX = Exp_7B_SigLIP_ViT_SO_p14_384px_Letterbox
    EXP_SIGLIP_384PX_RESIZE_CROP = Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Crop
    EXP_SIGLIP_384PX_RESIZE_NAIVE = Exp_7B_SigLIP_ViT_SO_p14_384px_Resize_Naive

    EXP_DINOCLIP_336PX_LETTERBOX = Exp_7B_DINOCLIP_ViT_L_p14_336px_Letterbox
    EXP_DINOCLIP_336PX_RESIZE_NAIVE = Exp_7B_DINOCLIP_ViT_L_p14_336px_Resize_Naive
    EXP_DINOSIGLIP_384PX_LETTERBOX = Exp_7B_DINOSigLIP_ViT_L_p14_384px_Letterbox
    EXP_DINOSIGLIP_384PX_RESIZE_NAIVE = Exp_7B_DINOSigLIP_ViT_L_p14_384px_Resize_Naive

    # === Section 4.3 :: Language Models ===
    EXP_LLAMA2_7B = Exp_7B_Llama2
    EXP_LLAMA2_13B = Exp_13B_Llama2

    # ~ Additional LLM Backbone Experiments :: LLaMa-2 Chat, Mistral v0.1, Mistral v0.1 Instruct, Phi-2 ~
    EXT_EXP_LLAMA2_CHAT_7B = Ext_Exp_7B_Llama2_Chat
    EXT_EXP_LLAMA2_CHAT_13B = Ext_Exp_13B_Llama2_Chat
    EXT_EXP_MISTRAL_V1_7B = Ext_Exp_7B_Mistral_V1
    EXT_EXP_MISTRAL_INSTRUCT_V1_7B = Ext_Exp_7B_Mistral_Instruct_V1
    EXT_EXP_PHI_2_3B = Ext_Exp_3B_Phi_2

    # Cotraining w/ Unimodal Data

    EXP_VICUNA_NO_COTRAINING_7B = Exp_7B_Vicuna_No_Cotraining
    EXP_LLAMA2_NO_COTRAINING_7B = Exp_7B_Llama2_No_Cotraining

    # === Section 4.4 :: Scaling Properties - Train Time & Data ===
    EXP_1P25_EPOCHS = Exp_7B_1p25_Epochs
    EXP_1P5_EPOCHS = Exp_7B_1p5_Epochs
    EXP_2_EPOCHS = Exp_7B_2_Epochs
    EXP_3_EPOCHS = Exp_7B_3_Epochs

    EXP_LLAVA_LVIS4V = Exp_7B_LLaVa_LVIS4V
    EXP_LLAVA_LRV = Exp_7B_LLaVa_LRV
    EXP_LLAVA_LVIS4V_LRV = Exp_7B_LLaVa_LVIS4V_LRV

    # === Section 5 :: Prisms ===
    PRISM_CLIP_CONTROLLED_7B = Prism_7B_CLIP_Controlled
    PRISM_CLIP_CONTROLLED_13B = Prism_13B_CLIP_Controlled
    PRISM_CLIP_7B = Prism_7B_CLIP
    PRISM_CLIP_13B = Prism_13B_CLIP

    PRISM_SIGLIP_CONTROLLED_7B = Prism_7B_SigLIP_Controlled
    PRISM_SIGLIP_CONTROLLED_13B = Prism_13B_SigLIP_Controlled
    PRISM_SIGLIP_7B = Prism_7B_SigLIP
    PRISM_SIGLIP_13B = Prism_13B_SigLIP

    PRISM_DINOSIGLIP_CONTROLLED_7B = Prism_7B_DINOSigLIP_Controlled
    PRISM_DINOSIGLIP_CONTROLLED_13B = Prism_13B_DINOSigLIP_Controlled
    PRISM_DINOSIGLIP_7B = Prism_7B_DINOSigLIP
    PRISM_DINOSIGLIP_13B = Prism_13B_DINOSigLIP

    # === Inference Optimized :: 224px Prism Models ===
    PRISM_DINOSIGLIP_224PX_CONTROLLED_7B = Prism_7B_DINOSigLIP_224px_Controlled
    PRISM_DINOSIGLIP_224PX_7B = Prism_7B_DINOSigLIP_224px

    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)
