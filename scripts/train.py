"""
train.py

Training script for the train/eval release, kept close to the original Prismatic/Mamba-MLLM entrypoint and using
Fully-Sharded Data Parallel (FSDP) for distributed execution.


Notes & Prerequisites:
    - We're loading LLaMa-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For LLaMa-2, make sure to first get Meta approval, then fill out the form at the top of the HF LLaMa-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or environment variable name

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/train.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.vision.vmamba import (
    VMAMBA_ALIASES,
    get_vmamba_variant_spec,
)
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

DEFAULT_MODEL_ID = "in1k-224px-maxvit-t-letterbox-s3+7b-vicuna"
DEFAULT_RUN_ROOT_DIR = Path(os.getenv("RUN_ROOT_DIR", "runs"))
SUPPORTED_LLM_BACKBONES = {"vicuna-v15-7b"}
SUPPORTED_VISION_PREFIXES = (
    "in1k-vit-",
    "in1kft-vit-",
    "in21k-vit-",
    "vmamba",
    "mambavision",
    "vitdet",
    "vit-adapter",
)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(DEFAULT_MODEL_ID)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | vision_finetune (projector + vision) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = DEFAULT_RUN_ROOT_DIR                       # Path to directory to store logs & checkpoints
    run_suffix: Optional[str] = None                                # Optional suffix appended to auto-generated run_id
    seed: int = 7                                                   # Random seed (for reproducibility)
    use_shm_checkpointing: bool = False                             # Stage checkpoints under /dev/shm (async by default)

    # Epoch Override Support
    finetune_epochs_override: Optional[int] = None                 # Override finetune_epochs from model config
    projector_arch_override: Optional[str] = None                 # Override projector connector (e.g., fused MLP)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    # wandb_project: str = "prismatic"                                # Name of W&B project (default: `prismatic`)
    # wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)
    wandb_project: str = "prismatic-vlms"
    wandb_entity: str = "raykuo-sj"

    # VMamba feature tap controls
    vmamba_feature_stage: Optional[int] = None
    vmamba_feature_layer: Optional[int] = None

    # Utility flags
    dry_run: bool = False
    visualize_model_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        stage_key = self.stage.replace("-", "_")
        self.normalized_stage = stage_key

        if self.projector_arch_override:
            self.model.arch_specifier = self.projector_arch_override

        if stage_key == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif stage_key == "finetune":
            # Apply epoch override if specified
            self.epochs = self.finetune_epochs_override if self.finetune_epochs_override is not None else self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        elif stage_key == "vision_finetune":
            self.epochs = self.model.vision_finetune_epochs
            self.max_steps = self.model.vision_finetune_max_steps
            self.global_batch_size = self.model.vision_finetune_global_batch_size
            self.per_device_batch_size = self.model.vision_finetune_per_device_batch_size

            self.learning_rate = self.model.vision_finetune_learning_rate
            self.weight_decay = self.model.vision_finetune_weight_decay
            self.max_grad_norm = self.model.vision_finetune_max_grad_norm
            self.lr_scheduler_type = self.model.vision_finetune_lr_scheduler_type
            self.warmup_ratio = self.model.vision_finetune_warmup_ratio

            self.train_strategy = self.model.vision_finetune_train_strategy

        elif stage_key == "full_finetune":
            self.epochs = self.model.vision_finetune_epochs
            self.max_steps = self.model.vision_finetune_max_steps
            self.global_batch_size = self.model.vision_finetune_global_batch_size
            self.per_device_batch_size = self.model.vision_finetune_per_device_batch_size

            self.learning_rate = self.model.vision_finetune_learning_rate
            self.weight_decay = self.model.vision_finetune_weight_decay
            self.max_grad_norm = self.model.vision_finetune_max_grad_norm
            self.lr_scheduler_type = self.model.vision_finetune_lr_scheduler_type
            self.warmup_ratio = self.model.vision_finetune_warmup_ratio

            self.train_strategy = self.model.vision_finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

        _validate_supported_release_scope(self.model)

    # fmt: on


def _validate_supported_release_scope(model_cfg: ModelConfig) -> None:
    if model_cfg.llm_backbone_id not in SUPPORTED_LLM_BACKBONES:
        raise ValueError(
            "This release only supports Vicuña v1.5 7B for train/eval parity. "
            f"Got llm_backbone_id={model_cfg.llm_backbone_id!r}."
        )

    vision_backbone_id = model_cfg.vision_backbone_id
    if "maxvit" in vision_backbone_id:
        return
    if any(vision_backbone_id.startswith(prefix) for prefix in SUPPORTED_VISION_PREFIXES):
        return

    raise ValueError(
        "This release only supports the published ViT, MaxViT, VMamba, MambaVision, ViTDet, and ViT-Adapter "
        f"families. Got vision_backbone_id={vision_backbone_id!r}."
    )


def _resolve_hf_token(token_or_path: Union[str, Path]) -> str:
    if isinstance(token_or_path, Path):
        return token_or_path.read_text().strip()

    candidate_path = Path(token_or_path).expanduser()
    if candidate_path.exists():
        return candidate_path.read_text().strip()

    return os.environ[token_or_path]


def _visualize_vlm(vlm: torch.nn.Module, image_shape: Tuple[int, int, int], max_seq_len: int, output_path: Path) -> None:
    try:
        from torchview import draw_graph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("`torchview` is required for --visualize_model_path") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm = vlm.to(device).eval()
    c, h, w = image_shape
    dummy_images = torch.randn(1, c, h, w, device=device)
    text_seq_len = min(8, max_seq_len)
    dummy_input_ids = torch.zeros(1, text_seq_len, dtype=torch.long, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_path.suffix[1:] if output_path.suffix else "png"
    filename = output_path.stem if output_path.suffix else output_path.name

    draw_graph(
        model=vlm,
        input_data={"input_ids": dummy_input_ids, "pixel_values": dummy_images},
        input_size=None,
        graph_name=filename,
        depth=4,
        device=device,
        dtypes=None,
        mode=None,
        strict=True,
        expand_nested=True,
        graph_dir="TB",
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=True,
        show_shapes=True,
        save_graph=True,
        filename=filename,
        directory=str(output_path.parent),
    )

    final_path = (output_path.parent / filename).with_suffix(f".{fmt}")
    overwatch.info(f"Saved VLM architecture graph to {final_path}")


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Prismatic VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    try:
        device_id = overwatch.local_rank()
    except AttributeError:
        device_id = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
    else:
        device_id = -1

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    is_rank_zero = overwatch.is_rank_zero()
    is_distributed = dist.is_available() and dist.is_initialized()

    def _resolve_vmamba_tap_suffix(vision_backbone_id: str) -> str:
        canonical = VMAMBA_ALIASES.get(vision_backbone_id, vision_backbone_id)
        variant = get_vmamba_variant_spec(canonical)

        depths = variant["model_kwargs"].get("depths", [])
        if not depths:
            raise ValueError(f"VMamba variant '{vision_backbone_id}' missing `depths` info")

        stage = cfg.vmamba_feature_stage or len(depths)
        if stage < 1 or stage > len(depths):
            raise ValueError(
                f"Requested VMamba stage {stage} is out of bounds (1-{len(depths)}) for backbone '{vision_backbone_id}'"
            )
        stage_idx = stage - 1
        depth = depths[stage_idx]

        if depth <= 0:
            raise ValueError(f"VMamba stage {stage} has no blocks to select")

        layer_raw = cfg.vmamba_feature_layer if cfg.vmamba_feature_layer is not None else depth - 1
        resolved_layer = layer_raw
        if resolved_layer < 0:
            resolved_layer = depth + resolved_layer

        if resolved_layer < 0 or resolved_layer >= depth:
            raise ValueError(
                f"Resolved VMamba layer index {resolved_layer} is out of range for stage {stage} (depth={depth})"
            )

        return f"+s{stage}l{resolved_layer}"

    # Always add epoch information to model_id
    if cfg.stage == "finetune":
        epochs_to_use = cfg.finetune_epochs_override if cfg.finetune_epochs_override is not None else cfg.model.finetune_epochs
        model_id = f"{model_id}+ep{epochs_to_use}"

    vmamba_tap_suffix = ""
    vmamba_resize_strategy = getattr(cfg.model, "image_resize_strategy", None)
    uses_vmamba_preproc = isinstance(vmamba_resize_strategy, str) and vmamba_resize_strategy.startswith("vmamba-")
    is_vmamba_model = cfg.model.vision_backbone_id.startswith("vmamba")
    if is_vmamba_model and (cfg.vmamba_feature_stage is not None or cfg.vmamba_feature_layer is not None):
        vmamba_tap_suffix = _resolve_vmamba_tap_suffix(cfg.model.vision_backbone_id)
        model_id = f"{model_id}{vmamba_tap_suffix}"

    if cfg.run_id is None:
        # Extract core components for cleaner naming
        if cfg.stage == "vision-finetune":
            # For vision finetune, extract the vision backbone (e.g., "in1kft-224px")
            vision_backbone = model_id.split("+")[0]  # Take the first part before '+'
            cfg.run_id = f"{vision_backbone}+vision-ft+x{cfg.seed}"
        elif cfg.stage == "finetune":
            # For finetune, the model_id is already descriptive enough
            cfg.run_id = f"{model_id}+ft+x{cfg.seed}"
        elif cfg.stage == "align":
            # For align stage, simplify to just model + align
            cfg.run_id = f"{model_id}+align+x{cfg.seed}"
        else:
            # For other stages, keep simplified structure
            cfg.run_id = f"{model_id}+{cfg.stage}+x{cfg.seed}"
    else:
        # Keep user-provided run_id as-is
        cfg.run_id = cfg.run_id

    if uses_vmamba_preproc:
        resize_suffix = f"+img-{vmamba_resize_strategy}"
        if resize_suffix not in cfg.run_id:
            cfg.run_id = f"{cfg.run_id}{resize_suffix}"

    if cfg.run_suffix:
        cfg.run_id = f"{cfg.run_id}{cfg.run_suffix}"

    stage_key = getattr(cfg, "normalized_stage", cfg.stage.replace("-", "_"))
    if stage_key == "vision_finetune" and not cfg.model.vision_finetune_train_projector:
        cfg.run_id = f"{cfg.run_id}-projFrozen"

    force_train = os.getenv("PRISMATIC_FORCE_TRAIN", "false").lower() == "true"
    if is_rank_zero:
        run_dir = cfg.run_root_dir / cfg.run_id
        checkpoint_path = run_dir / "checkpoints" / "latest-checkpoint.pt"
        if run_dir.exists():
            if checkpoint_path.exists():
                if not force_train and not cfg.dry_run:
                    raise RuntimeError(
                        f"Run directory already exists at {run_dir} with latest-checkpoint.pt; "
                        "set PRISMATIC_FORCE_TRAIN=true to overwrite."
                    )
                if not cfg.dry_run:
                    shutil.rmtree(run_dir)
            else:
                if not cfg.dry_run:
                    shutil.rmtree(run_dir)

    if is_distributed:
        run_id_exchange = [cfg.run_id if is_rank_zero else None]
        dist.broadcast_object_list(run_id_exchange, src=0)
        cfg.run_id = run_id_exchange[0]

    enable_checkpointing = True
    checkpoint_stage_dir = None
    if cfg.use_shm_checkpointing:
        checkpoint_stage_dir = Path("/dev/shm") / f"prismatic-{cfg.run_id}"
        overwatch.info(f"Checkpoint staging enabled under {checkpoint_stage_dir}")

    if cfg.dry_run:
        overwatch.warning("Dry-run mode enabled: checkpoints, staged copies, and W&B logging are disabled.")
        cfg.use_shm_checkpointing = False
        enable_checkpointing = False
        checkpoint_stage_dir = None
        filtered_trackers = tuple(tr for tr in cfg.trackers if tr != "wandb")
        cfg.trackers = filtered_trackers or ("jsonl",)

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = _resolve_hf_token(cfg.hf_token)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    run_dir = cfg.run_root_dir / cfg.run_id
    checkpoints_dir = run_dir / "checkpoints"

    if is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

    if is_distributed:
        dist.barrier()

    if not is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id,
        image_resize_strategy=cfg.model.image_resize_strategy,
        vmamba_feature_stage=cfg.vmamba_feature_stage,
        vmamba_feature_layer=cfg.vmamba_feature_layer,
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        vision_finetune_train_projector=cfg.model.vision_finetune_train_projector,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")

    vlm.freeze_backbones(stage_key)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(stage_key, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    if cfg.visualize_model_path is not None:
        _visualize_vlm(
            vlm,
            vision_backbone.default_image_resolution,
            cfg.model.llm_max_length,
            cfg.visualize_model_path,
        )
        overwatch.info("Visualization complete; skipping training.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])
            dist.destroy_process_group()
        return

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        stage_key,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )
    if overwatch.is_rank_zero() and hasattr(train_dataset, "get_max_text_tokens"):
        try:
            max_text_tokens = train_dataset.get_max_text_tokens(multimodal_only=True)
            total_tokens = max_text_tokens + vision_backbone.num_patches
            if total_tokens > cfg.model.llm_max_length:
                overwatch.warning(
                    "Longest multimodal conversation + vision tokens exceeds LLM context: "
                    f"text={max_text_tokens}, vision={vision_backbone.num_patches}, "
                    f"total={total_tokens}, max={cfg.model.llm_max_length}."
                )
            else:
                overwatch.info(
                    "Longest multimodal conversation token budget: "
                    f"text={max_text_tokens}, vision={vision_backbone.num_patches}, "
                    f"total={total_tokens}, max={cfg.model.llm_max_length}."
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            overwatch.warning(f"Failed to compute longest conversation length: {exc}")

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        checkpoint_stage_dir=checkpoint_stage_dir,
        enable_checkpointing=enable_checkpointing,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")

    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    train_strategy.wait_for_pending_io()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        pretrain()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
