"""
evaluate.py

Entry point for all VLM-Evaluation evaluations; specify model and dataset, get results.

Run with `accelerate` from repository root (for naive parallelization):
    =>> [Single-GPU] CUDA_VISIBLE_DEVICES={0-7} accelerate launch --num_processes=1 scripts/evaluate.py < args >
    =>> [Multi-GPU]  accelerate launch --num_processes={>1} scripts/evaluate.py < args >
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

import draccus
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.utils.data import Subset

from vlm_eval.conf import DatasetConfig, DatasetRegistry
from vlm_eval.models import load_vlm
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import get_task_runner

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)
DEFAULT_MODEL_ID = "maxvit-t-in1k-224-s3"
DEFAULT_RESULTS_DIR = Path(os.getenv("VLM_EVAL_RESULTS_DIR", "results"))


def _resolve_hf_token(token_or_path: Union[str, Path]) -> str:
    if isinstance(token_or_path, Path):
        return token_or_path.read_text().strip()

    candidate_path = Path(token_or_path).expanduser()
    if candidate_path.exists():
        return candidate_path.read_text().strip()

    return os.environ[token_or_path]


@dataclass
class EvaluationConfig:
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # === Model Parameters =>> Prismatic ===
    model_family: str = "prismatic"                 # Model family to load from in < `prismatic` | `llava-v15` | ... >
    model_id: Optional[str] = (                     # Model ID to load and run (instance of `model_family`)
        DEFAULT_MODEL_ID
    )
    model_dir: Optional[Path] = None                # Path to model checkpoint to load --> should be self-contained

    # === Model Parameters =>> Official LLaVa ===
    # model_family: str = "llava-v15"
    # model_id: str = "llava-v1.5-7b"
    # model_dir: Path = "liuhaotian/llava-v1.5-7b"

    # === Model Parameters =>> Official InstructBLIP ===
    # model_family: str = "instruct-blip"
    # model_id: str = "instructblip-vicuna-7b"
    # model_dir: Path = "Salesforce/instructblip-vicuna-7b"

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)
    load_precision: str = "bf16"                   # Precision for model weights (set to fp32 on CPU-only runs)
    max_examples: Optional[int] = None             # Optional debug cap on dataset size (does not affect defaults)

    # Artifact Parameters
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        DEFAULT_RESULTS_DIR
    )

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir
        if self.model_id is None and self.model_dir is not None:
            self.model_id = self.model_dir.resolve().name

    # fmt: on

def evaluate_after_parse(cfg, vlm=None):
    overwatch.info(f"Starting Evaluation for Dataset `{cfg.dataset.dataset_id}` w/ Model `{cfg.model_id}`")
    set_seed(cfg.seed)

    # Short-Circuit (if results/metrics already exist)
    dataset_results_id = cfg.dataset.results_dataset_id or cfg.dataset.dataset_id
    task_results_dir = cfg.results_dir / cfg.dataset.dataset_family / dataset_results_id / cfg.model_id
    task_results_dir.mkdir(parents=True, exist_ok=True)
    if (task_results_dir / "metrics.json").exists():
        overwatch.info(f"Metrics for `{cfg.dataset.dataset_id}` w/ `{cfg.model_id}` exist =>> exiting!")
        return

    # Build the VLM --> Download/Load Pretrained Model from Checkpoint
    overwatch.info("Initializing VLM =>> Bundling Models, Image Processors, and Tokenizer")
    hf_token = _resolve_hf_token(cfg.hf_token)
    if vlm is None:
        vlm = load_vlm(
            cfg.model_family,
            cfg.model_id,
            cfg.run_dir,
            hf_token=hf_token,
            ocr=cfg.dataset.ocr,
            load_precision=cfg.load_precision,
        )

    # Create Task Runner
    overwatch.info(f"Building Evaluation Runner for Dataset `{cfg.dataset.dataset_id}`")
    task_runner = get_task_runner(
        cfg.dataset.dataset_family,
        cfg.dataset.root_dir,
        cfg.dataset.index_file,
        task_results_dir,
        cfg.model_id,
        prompt_fn=vlm.get_prompt_fn(cfg.dataset.dataset_family),
        image_processor=vlm.image_processor,
    )
    if cfg.max_examples is not None:
        if cfg.max_examples <= 0:
            raise ValueError(f"`max_examples` must be positive, got {cfg.max_examples}")
        if not hasattr(task_runner, "dataset"):
            raise ValueError("Task runner does not expose a dataset, so `max_examples` is unsupported here.")
        limited_examples = min(int(cfg.max_examples), len(task_runner.dataset))
        task_runner.dataset = Subset(task_runner.dataset, range(limited_examples))
        overwatch.info(f"Limiting evaluation to first {limited_examples} examples for debug validation.")

    _write_transform_metadata(task_results_dir, vlm, cfg)

    # Run Evaluation
    overwatch.info("Starting (Distributed) Evaluation Loop")
    task_runner.evaluate(vlm, cfg.device_batch_size, cfg.num_workers)


def _write_transform_metadata(task_results_dir: Path, vlm, cfg: EvaluationConfig) -> None:
    transform_getter = getattr(vlm, "get_image_transform_info", None)
    if not callable(transform_getter):
        return

    transform_info = transform_getter() or {}
    transform_info.setdefault("model_id", cfg.model_id)
    transform_info.setdefault("model_family", cfg.model_family)

    with open(task_results_dir / "image_transform.json", "w") as f:
        json.dump(transform_info, f, indent=2)


@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    try:
        evaluate_after_parse(cfg)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    evaluate()
