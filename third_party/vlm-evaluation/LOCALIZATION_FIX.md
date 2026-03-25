# Localization Benchmark Fix Plan

## Summary of the Bug
- RefCOCO/RefCOCO+/RefCOCOg and OCID-Ref metadata only stored bounding boxes normalized by the raw image size. During evaluation every model (Prismatic, when trained with the `letterbox` strategy) runs on square, letterboxed frames so the predictions are in a padded coordinate system. Comparing those predictions to unpadded boxes tanked IOU and accuracy.
- Older evaluation runs also lacked any record of the preprocessing policy/resolution, so the scorer could not project ground-truth boxes into the correct frame.

## Code Changes (already applied)
1. Metadata builders now retain per-example `bbox_xyxy`, `image_width`, and `image_height` alongside the older normalized box.
2. New helper `vlm_eval/tasks/harnesses/geometry.py` projects ground-truth boxes into whatever preprocessing frame a model reports (letterbox, resize-crop, resize-naive).
3. Scorers for RefCOCO* and OCID-Ref read `image_transform.json` from the evaluation results directory and normalize GT boxes before computing IOU.
4. `scripts/evaluate.py` now writes `image_transform.json` automatically for new runs (the Prismatic, LLaVA, and InstructBLIP wrappers report their resize strategy + resolution).
5. `scripts/backfill_transform_info.py` can populate `image_transform.json` for existing Prismatic runs using the saved `config.json` from `vlm_runs/`.

## Rebuilding Dataset Metadata
Create parallel `metadata-full-v2.json` files (the original metadata stays untouched):

```bash
cd external/vlm-evaluation
python scripts/build_localization_metadata_v2.py \
  --root-dir /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data
```

Run it once and it will emit `datasets/refcoco/metadata-full-v2.json` and `datasets/ocid-ref/metadata-full-v2.json`,
each augmented with `image_width`/`image_height` and absolute boxes.

## Backfilling Transform Metadata for Existing Runs
To avoid re-running inference, create the missing `image_transform.json` files under every old localization result (this reads each run’s config to grab the resize policy/resolution):

```bash
cd external/vlm-evaluation
python scripts/backfill_transform_info.py \
  --runs-root /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs
```

This walks every run under the provided root and drops an `image_transform.json` next to each localization result (if it
doesn’t already exist).

## Re-scoring Without Re-running Inference
Run everything with a single command (use `--dry-run` to preview, or `--try-one <run-path>` to sanity-check a single
checkpoint such as `/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs/clip-336px+7b-llama2+ep1+ft+x7`):

```bash
cd external/vlm-evaluation
python scripts/rescore_localization_all.py \
  --runs-root /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs \
  --dataset-root /lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data
```

The script automatically locates every `refcoco-full` and `ocid-ref-full` evaluation across all runs and re-invokes
`scripts/score.py` for the new `DatasetRegistry.REFCOCO_FULL_V2` / `DatasetRegistry.OCIDREF_FULL_V2` configs. Existing
`metrics.json` files are preserved and the fresh scores land in `metrics_v2.json`, letting you compare old vs. new
numbers while keeping the original checkpoints untouched.

Use `--dry-run` to print the planned score commands without executing, or `--try-one /lustre/.../vlm_runs/<run>` to
validate a single checkpoint before batch processing.

Because `scripts/evaluate.py` now writes `image_transform.json` automatically, any **new** evaluation (or re-run) will carry the right metadata. For archived runs, the backfill script plus the commands above allow you to regenerate metrics without touching GPUs.

## Notes
- All SLURM templates continue to pass `--model_family prismatic`, so there is no need to patch the LLaVA/InstructBLIP harnesses for this workflow.
- If you later evaluate non-Prismatic models, just re-run `backfill_transform_info.py` (or re-evaluate) so the scorer sees their preprocessing metadata as well.
