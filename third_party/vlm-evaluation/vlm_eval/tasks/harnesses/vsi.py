"""
vsi.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the VSI benchmark.
"""
import json
import math
import os
import re
from functools import lru_cache
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


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def _normalize_options(options: List[str]) -> List[str]:
    cleaned = []
    for opt in options:
        if opt is None:
            continue
        cleaned.append(re.sub(r"^[A-Z]\\s*\\.\\s*", "", str(opt)).strip())
    return cleaned


def _sample_indices(total: int, num_frames: int) -> List[int]:
    if total <= 0:
        return [0] * num_frames
    if num_frames == 1:
        return [int(total // 2)]
    return np.linspace(0, total - 1, num_frames).astype(int).tolist()


def _read_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    # Try decord
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(video_path)
        indices = _sample_indices(len(vr), num_frames)
        return [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
    except Exception:
        pass

    # Try imageio
    try:
        import imageio.v2 as imageio  # type: ignore

        reader = imageio.get_reader(video_path)
        try:
            total = reader.count_frames()
        except Exception:
            total = reader.get_length() or 0
        indices = _sample_indices(total, num_frames)
        frames = [Image.fromarray(reader.get_data(idx)) for idx in indices]
        reader.close()
        return frames
    except Exception:
        pass

    # Try OpenCV
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _sample_indices(total, num_frames)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if frames:
            return frames
    except Exception:
        pass

    raise RuntimeError(
        "VSI evaluation requires a video decoder. Install one of: `decord`, `imageio[ffmpeg]`, or `opencv-python`."
    )


def _make_grid(frames: List[Image.Image], cols: int) -> Image.Image:
    if not frames:
        raise ValueError("No frames available to build VSI grid image.")
    cols = max(1, cols)
    rows = int(math.ceil(len(frames) / cols))
    base_w, base_h = frames[0].size
    grid = Image.new("RGB", (cols * base_w, rows * base_h), color=(0, 0, 0))
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        if frame.size != (base_w, base_h):
            frame = frame.resize((base_w, base_h), Image.BICUBIC)
        grid.paste(frame, (col * base_w, row * base_h))
    return grid


@lru_cache(maxsize=64)
def _load_video_grid(video_path: str, num_frames: int, grid_cols: int) -> Image.Image:
    frames = _read_video_frames(video_path, num_frames)
    return _make_grid(frames, grid_cols)


# === Dataset Indexing / Building Utilities ===
def build_vsi_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse VSI --> build & write index files for full + debiased subsets."""
    paths = DATASET_REGISTRY["vsi"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    assert slim_dataset_sizes is None, "VSI uses full/debiased splits only."
    index_files = [
        dataset_dir / "metadata-full.json",
        dataset_dir / "metadata-debiased.json",
    ]
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Load pruned IDs (for debiased split)
    with open(root_dir / paths["pruned_ids"], "r") as f:
        pruned_ids = {line.strip() for line in f if line.strip()}

    # Load raw annotations
    examples = []
    with open(root_dir / paths["questions"], "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    full_index = {}
    debiased_index = {}
    for example in tqdm(examples, desc="=> Processing VSI Dataset:", leave=False):
        example_id = int(example["id"])
        dataset_name = example["dataset"]
        scene_name = example["scene_name"]
        video_path = paths["videos"] / dataset_name / f"{scene_name}.mp4"

        assert (root_dir / video_path).exists(), f"Video `{video_path}` for Example ID `{example_id}` does not exist!"

        record = {
            "example_id": example_id,
            "question_type": example["question_type"],
            "question": example["question"],
            "options": example.get("options"),
            "ground_truth": example["ground_truth"],
            "dataset": dataset_name,
            "scene_name": scene_name,
            "video_path": str(video_path),
            "pruned": str(example_id) in pruned_ids,
        }
        full_index[example_id] = record
        if not record["pruned"]:
            debiased_index[example_id] = record

    # Write metadata
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump(full_index, f)
        elif index_file.name == "metadata-debiased.json":
            with open(index_file, "w") as f:
                json.dump(debiased_index, f)
        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class VSIIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, str]:
        """Return (example_id, question, video_path, ground_truth)."""
        ex = self.examples[idx]
        return ex["example_id"], ex["question"], Path(self.root_dir / ex["video_path"]), ex["ground_truth"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class VSIMapDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        prompt_fn: Callable[[str, Optional[List[str]]], str],
        image_processor: ImageProcessor,
    ) -> None:
        """
        Constructs a Map-Style Dataset for VSI. Returns prompts, processed video grids, and QA metadata.
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

        self.num_frames = int(os.environ.get("VSI_NUM_FRAMES", "4"))
        self.grid_cols = int(os.environ.get("VSI_GRID_COLS", "2"))

    def _load_video_image(self, video_path: Path) -> Image.Image:
        return _load_video_grid(str(video_path), self.num_frames, self.grid_cols)

    def _process_image(self, video_path: Path) -> torch.Tensor:
        img = self._load_video_image(video_path)
        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            return self.image_processor(img)
        return self.image_processor(img, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, Optional[List[str]], str, str]:
        """
        Return (example_id, prompt, pixel_values, question, choices, ground_truth, question_type).
        """
        ex = self.examples[idx]
        options = ex.get("options")
        choices = _normalize_options(options) if options else None
        prompt = self.prompt_fn(ex["question"], choices)
        pixel_values = self._process_image(Path(self.root_dir / ex["video_path"]))

        return (
            ex["example_id"],
            prompt,
            pixel_values,
            ex["question"],
            choices,
            ex["ground_truth"],
            ex["question_type"],
        )

    def __len__(self) -> int:
        return len(self.examples)


def _vsi_collate(
    batch: List[Tuple[int, str, torch.Tensor, str, Optional[List[str]], str, str]]
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str], List[Optional[List[str]]], List[str], List[str]]:
    example_ids, prompts, pixel_values, questions, choices, ground_truths, question_types = zip(*batch)
    example_ids = torch.tensor(example_ids, dtype=torch.long)
    if isinstance(pixel_values[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values, dim=0)
    elif isinstance(pixel_values[0], dict):
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values], dim=0) for k in pixel_values[0]}
    else:
        raise TypeError(f"Unsupported VSI pixel_values type: {type(pixel_values[0])}")

    return (
        example_ids,
        list(prompts),
        pixel_values,
        list(questions),
        list(choices),
        list(ground_truths),
        list(question_types),
    )


def _slice_pixel_values(pixel_values: torch.Tensor, indices: List[int]) -> torch.Tensor:
    if not indices:
        return pixel_values
    index_tensor = torch.tensor(indices, device=next(iter(pixel_values.values())).device) if isinstance(pixel_values, dict) else torch.tensor(indices, device=pixel_values.device)
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.index_select(0, index_tensor)
    return {k: v.index_select(0, index_tensor) for k, v in pixel_values.items()}


# === VSI Task Runner ===
class VSITaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str, Optional[List[str]]], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for VSI; loads data then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        from accelerate import PartialState

        self.distributed_state = PartialState()

        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"VSI Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        overwatch.info(f"Assembling VSI Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = VSIMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        """Initialize Dataloader & partition data across ranks, writing metrics to disk on termination."""
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=device_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_vsi_collate,
        )

        result_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for (
                example_ids,
                prompts,
                pixel_values,
                questions,
                choices,
                ground_truths,
                question_types,
            ) in tqdm(
                dataloader,
                desc="=>> Evaluating",
                disable=not self.distributed_state.is_main_process,
            ):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(self.distributed_state.device)
                else:
                    pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}

                # Split MC vs numeric questions
                mc_indices = [i for i, opts in enumerate(choices) if opts]
                na_indices = [i for i, opts in enumerate(choices) if not opts]

                # MC groups by number of options (2/3/4)
                for n_choices in sorted({len(choices[i]) for i in mc_indices}):
                    group_indices = [i for i in mc_indices if len(choices[i]) == n_choices]
                    if not group_indices:
                        continue
                    group_prompts = [prompts[i] for i in group_indices]
                    group_pixels = _slice_pixel_values(pixel_values, group_indices)
                    return_strings = [chr(ord("A") + idx) for idx in range(n_choices)]
                    gen_probs = vlm.generate_answer(
                        group_pixels, group_prompts, return_string_probabilities=return_strings
                    )
                    for idx, probs in zip(group_indices, gen_probs, strict=True):
                        pred_label = return_strings[int(np.argmax(probs))]
                        ex_id = int(example_ids[idx].item())
                        result_pairs[ex_id] = {
                            "example_id": ex_id,
                            "question": questions[idx],
                            "question_type": question_types[idx],
                            "choices": choices[idx],
                            "ground_truth": ground_truths[idx],
                            "prediction": pred_label,
                            "mc_probabilities": probs,
                        }

                # Numeric answer questions
                if na_indices:
                    na_prompts = [prompts[i] for i in na_indices]
                    na_pixels = _slice_pixel_values(pixel_values, na_indices)
                    gen_answers = vlm.generate_answer(na_pixels, na_prompts)
                    for idx, answer in zip(na_indices, gen_answers, strict=True):
                        ex_id = int(example_ids[idx].item())
                        result_pairs[ex_id] = {
                            "example_id": ex_id,
                            "question": questions[idx],
                            "question_type": question_types[idx],
                            "choices": choices[idx],
                            "ground_truth": ground_truths[idx],
                            "prediction": answer,
                        }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_pairs, f, indent=2)

        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function ===
class VSIScorer:
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

    @staticmethod
    def _fuzzy_matching(pred: str) -> str:
        return pred.split(" ")[0].rstrip(".").strip() if isinstance(pred, str) else ""

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _mean_relative_accuracy(pred: Optional[float], target: Optional[float], start: float, end: float, interval: float) -> float:
        if pred is None or target in (None, 0):
            return 0.0
        num_pts = (end - start) / interval + 2
        conf_intervals = np.linspace(start, end, int(num_pts))
        accuracy = abs(pred - target) / target <= 1 - conf_intervals
        return float(accuracy.mean())

    def score(self, model_id: str) -> Dict[str, float]:
        def _score_subset(examples: Dict[str, Dict], prefix: str, accuracy_key: str) -> Dict[str, float]:
            per_type_scores: Dict[str, List[float]] = {}
            for example in examples.values():
                qtype = example["question_type"]
                pred = example.get("prediction", "")
                gt = example.get("ground_truth", "")

                if qtype in MCA_QUESTION_TYPES:
                    score = 1.0 if str(pred).strip().upper() == str(gt).strip().upper() else 0.0
                elif qtype in NA_QUESTION_TYPES:
                    pred_val = self._to_float(self._fuzzy_matching(str(pred)))
                    gt_val = self._to_float(self._fuzzy_matching(str(gt)))
                    score = self._mean_relative_accuracy(pred_val, gt_val, start=0.5, end=0.95, interval=0.05)
                else:
                    raise ValueError(f"Unknown VSI question type: {qtype}")

                per_type_scores.setdefault(qtype, []).append(score)

            metrics: Dict[str, float] = {}
            for qtype, scores in per_type_scores.items():
                metrics[f"{prefix}{qtype}"] = float(np.mean(scores))

            # Aggregate directional subtypes
            direction_keys = [
                f"{prefix}object_rel_direction_easy",
                f"{prefix}object_rel_direction_medium",
                f"{prefix}object_rel_direction_hard",
            ]
            if all(k in metrics for k in direction_keys):
                metrics[f"{prefix}object_rel_direction"] = float(np.mean([metrics.pop(k) for k in direction_keys]))

            overall = float(np.mean(list(metrics.values()))) if metrics else 0.0
            metrics[accuracy_key] = overall
            return metrics

        # Load pruned IDs from the metadata to compute debiased scores.
        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)
        pruned_ids = {str(k) for k, v in annotations.items() if v.get("pruned")}

        full_metrics = _score_subset(self.full_result_pairs, prefix="vsi_", accuracy_key="accuracy__VSI-overall")
        debiased_pairs = {k: v for k, v in self.full_result_pairs.items() if str(k) not in pruned_ids}
        debiased_metrics = _score_subset(
            debiased_pairs,
            prefix="vsi_debiased_",
            accuracy_key="accuracy__VSI-debiased",
        )

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Overall Score : {full_metrics['accuracy__VSI-overall']:.3f}\n"
            f"          => Debiased Score: {debiased_metrics['accuracy__VSI-debiased']:.3f}"
        )

        return {**full_metrics, **debiased_metrics}
