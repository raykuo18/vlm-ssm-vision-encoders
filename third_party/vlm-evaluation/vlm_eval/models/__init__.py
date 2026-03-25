from pathlib import Path
from typing import Optional

from vlm_eval.util.interfaces import VLM


def _get_initializer(model_family: str):
    """Resolve the loader for the requested model family lazily.

    Importing the LLaVa module unconditionally causes Transformers to register a
    duplicate "llava" config, which raises `ValueError` even when we only want to
    evaluate Prismatic checkpoints. Delaying the import until the family is
    requested sidesteps that issue while keeping existing behaviour for callers
    who do need LLaVa or InstructBLIP support.
    """

    if model_family == "prismatic":
        from .prismatic import PrismaticVLM  # local import to avoid side effects

        return PrismaticVLM

    if model_family == "llava-v15":
        from .llava import LLaVa  # type: ignore

        return LLaVa

    if model_family == "instruct-blip":
        from .instructblip import InstructBLIP  # type: ignore

        return InstructBLIP

    raise ValueError(f"Model family `{model_family}` not supported!")


def load_vlm(
    model_family: str,
    model_id: str,
    run_dir: Path,
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=1.0,
) -> VLM:
    initializer = _get_initializer(model_family)
    return initializer(
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )
