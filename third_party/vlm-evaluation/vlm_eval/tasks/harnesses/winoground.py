"""
winoground.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the Winoground benchmark.
"""
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Dataset Indexing / Building Utilities ===
def build_winoground_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse Winoground --> build & write index files w/ necessary keys + additional metadata."""
    paths = DATASET_REGISTRY["winoground"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    assert slim_dataset_sizes is None, "Winoground is tiny -- no slim dataset!"
    index_files = [dataset_dir / "metadata-full.json"]
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Load raw Winoground annotations
    examples = []
    with open(root_dir / paths["examples"], "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    images_dir = root_dir / paths["images"]
    image_lookup = {p.stem: p.name for p in images_dir.glob("*") if p.is_file()}

    def _resolve_image(name: str) -> str:
        name = str(name)
        if Path(name).suffix:
            return name
        if name in image_lookup:
            return image_lookup[name]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = images_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate.name
        raise FileNotFoundError(f"Image `{name}` not found under {images_dir}")

    # Build Full Metadata Structure
    index = {}
    for example in tqdm(examples, desc="=> Processing Winoground Dataset:", leave=False):
        example_id = int(example["id"])
        img0_name = _resolve_image(example["image_0"])
        img1_name = _resolve_image(example["image_1"])
        img0_path = paths["images"] / img0_name
        img1_path = paths["images"] / img1_name

        assert (root_dir / img0_path).exists(), f"Image `{img0_path}` for Example ID `{example_id}` does not exist!"
        assert (root_dir / img1_path).exists(), f"Image `{img1_path}` for Example ID `{example_id}` does not exist!"

        # fmt: off
        index[example_id] = {
            "example_id": example_id,
            "caption_0": example["caption_0"],
            "caption_1": example["caption_1"],
            "img_path_0": str(img0_path),
            "img_path_1": str(img1_path),
            "tag": example.get("tag"),
            "secondary_tag": example.get("secondary_tag"),
            "collapsed_tag": example.get("collapsed_tag"),
            "num_main_preds": example.get("num_main_preds"),
        }
        # fmt: on

    assert len(index) == 400, "Expected 400 Winoground examples!"

    # Write metadata
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump(index, f)
        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class WinogroundIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, str, Path, Path]:
        """Return (example_id, caption_0, caption_1, img_path_0, img_path_1)."""
        ex = self.examples[idx]
        return (
            ex["example_id"],
            ex["caption_0"],
            ex["caption_1"],
            Path(self.root_dir / ex["img_path_0"]),
            Path(self.root_dir / ex["img_path_1"]),
        )

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class WinogroundMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, image_processor: ImageProcessor
    ) -> None:
        """
        Constructs a Map-Style Dataset for Winoground. Returns captions + two processed images per example.
        """
        self.image_processor = image_processor
        self.root_dir, self.index_file = root_dir, index_file

        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def _process_image(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            return self.image_processor(img)
        return self.image_processor(img, return_tensors="pt")["pixel_values"][0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, str, str, torch.Tensor, torch.Tensor, Optional[str], Optional[str], Optional[str], Optional[int]]:
        """Return (example_id, caption_0, caption_1, pixel_values_0, pixel_values_1, tag info)."""
        ex = self.examples[idx]
        img0 = self._process_image(Path(self.root_dir / ex["img_path_0"]))
        img1 = self._process_image(Path(self.root_dir / ex["img_path_1"]))
        return (
            ex["example_id"],
            ex["caption_0"],
            ex["caption_1"],
            img0,
            img1,
            ex.get("tag"),
            ex.get("secondary_tag"),
            ex.get("collapsed_tag"),
            ex.get("num_main_preds"),
        )

    def __len__(self) -> int:
        return len(self.examples)


def _repeat_pixel_values(pixel_values: torch.Tensor, repeats: int) -> torch.Tensor:
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.repeat_interleave(repeats, dim=0)
    return {k: v.repeat_interleave(repeats, dim=0) for k, v in pixel_values.items()}


# === Winoground Task Runner ===
class WinogroundTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for Winoground; loads data then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        from accelerate import PartialState

        self.distributed_state = PartialState()

        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"Winoground Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        overwatch.info(f"Assembling Winoground Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = WinogroundMapDataset(self.root_dir, self.index_file, self.image_processor)

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        """Initialize Dataloader & partition data across ranks, writing metrics to disk on termination."""
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(self.dataset, batch_size=device_batch_size, sampler=sampler, num_workers=num_workers)

        result_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for (
                example_ids,
                caption_0,
                caption_1,
                pixel_values_0,
                pixel_values_1,
                tag,
                secondary_tag,
                collapsed_tag,
                num_main_preds,
            ) in tqdm(
                dataloader,
                desc="=>> Evaluating",
                disable=not self.distributed_state.is_main_process,
            ):
                prompts = []
                for cap0, cap1 in zip(caption_0, caption_1, strict=True):
                    prompts.append(self.prompt_fn(cap0))
                    prompts.append(self.prompt_fn(cap1))

                # Process image 0 (repeat each sample for caption_0 and caption_1)
                if isinstance(pixel_values_0, torch.Tensor):
                    pixel_values_0 = pixel_values_0.to(self.distributed_state.device)
                else:
                    pixel_values_0 = {k: v.to(self.distributed_state.device) for k, v in pixel_values_0.items()}
                repeated_img0 = _repeat_pixel_values(pixel_values_0, 2)
                probs_img0 = vlm.generate_answer(
                    repeated_img0, prompts, return_string_probabilities=["True", "False"]
                )

                # Process image 1 (repeat each sample for caption_0 and caption_1)
                if isinstance(pixel_values_1, torch.Tensor):
                    pixel_values_1 = pixel_values_1.to(self.distributed_state.device)
                else:
                    pixel_values_1 = {k: v.to(self.distributed_state.device) for k, v in pixel_values_1.items()}
                repeated_img1 = _repeat_pixel_values(pixel_values_1, 2)
                probs_img1 = vlm.generate_answer(
                    repeated_img1, prompts, return_string_probabilities=["True", "False"]
                )

                for idx, example_id in enumerate(example_ids):
                    base = 2 * idx
                    ex_id = int(example_id.item())
                    result_pairs[ex_id] = {
                        "example_id": ex_id,
                        "caption_0": caption_0[idx],
                        "caption_1": caption_1[idx],
                        "true_false_probabilities": {
                            "i0_c0": probs_img0[base],
                            "i0_c1": probs_img0[base + 1],
                            "i1_c0": probs_img1[base],
                            "i1_c1": probs_img1[base + 1],
                        },
                        "tags": {
                            "tag": tag[idx],
                            "secondary_tag": secondary_tag[idx],
                            "collapsed_tag": collapsed_tag[idx],
                            "num_main_preds": int(num_main_preds[idx]) if num_main_preds is not None else None,
                        },
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_pairs, f, indent=2)

        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function ===
class WinogroundScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "test",
        **_: str,
    ) -> None:
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_pairs = full_result_pairs

    def score(self, model_id: str) -> Dict[str, float]:
        n_image_correct = 0
        n_text_correct = 0
        n_group_correct = 0

        for example in self.full_result_pairs.values():
            probs = example["true_false_probabilities"]
            p00 = probs["i0_c0"][0]  # True prob for (img0, cap0)
            p01 = probs["i0_c1"][0]  # True prob for (img0, cap1)
            p10 = probs["i1_c0"][0]  # True prob for (img1, cap0)
            p11 = probs["i1_c1"][0]  # True prob for (img1, cap1)

            image_correct = (p00 > p01) and (p11 > p10)
            text_correct = (p00 > p10) and (p11 > p01)
            group_correct = image_correct and text_correct

            n_image_correct += int(image_correct)
            n_text_correct += int(text_correct)
            n_group_correct += int(group_correct)

        total = len(self.full_result_pairs)
        image_acc = float(n_image_correct / total)
        text_acc = float(n_text_correct / total)
        group_acc = float(n_group_correct / total)

        metrics = {
            "accuracy__Winoground-Image": image_acc,
            "accuracy__Winoground-Text": text_acc,
            "accuracy__Winoground-Group": group_acc,
        }

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Image Accuracy : {image_acc:.3f}\n"
            f"          => Text Accuracy  : {text_acc:.3f}\n"
            f"          => Group Accuracy : {group_acc:.3f}"
        )

        return metrics
