from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from vlm_backbones.api import download_model, load_model
from vlm_backbones.images import load_image


def _write_jsonl(records: Iterable[dict], output_path: str | None) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    if output_path:
        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        for line in lines:
            print(line)


def _generation_kwargs(max_new_tokens: int, temperature: float | None) -> dict:
    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    return kwargs


def _build_chat_prompt(model, prompt: str, system_prompt: str | None = None) -> str:
    prompt_builder = model.get_prompt_builder(system_prompt=system_prompt)
    prompt_builder.add_turn(role="human", message=prompt)
    return prompt_builder.get_prompt()


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download a released checkpoint artifact.")
    parser.add_argument("--model", required=True, help="Public model id from model_zoo/models.yaml")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract the artifact.")
    args = parser.parse_args()
    print(download_model(args.model, force=args.force))


def chat_main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image chat inference.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", help="Prompt text. If omitted, start a simple REPL.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature. If unset, generation stays deterministic.",
    )
    parser.add_argument("--system-prompt", help="Optional override for the default chat system prompt.")
    args = parser.parse_args()

    image = load_image(args.image)
    model = load_model(args.model, device=args.device, dtype=args.dtype)

    if args.prompt:
        prompt_text = _build_chat_prompt(model, args.prompt, system_prompt=args.system_prompt)
        print(
            model.generate(
                image,
                prompt_text,
                **_generation_kwargs(args.max_new_tokens, args.temperature),
            )
        )
        return

    prompt_builder = model.get_prompt_builder(system_prompt=args.system_prompt)
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = model.generate(
            image,
            prompt_text,
            **_generation_kwargs(args.max_new_tokens, args.temperature),
        )
        prompt_builder.add_turn(role="gpt", message=generated_text)
        print(generated_text)


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Run batched inference from JSON or JSONL input.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL file with image/prompt fields.")
    parser.add_argument("--output", help="Optional JSONL output path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature. If unset, generation stays deterministic.",
    )
    parser.add_argument("--system-prompt", help="Optional override for the default chat system prompt.")
    args = parser.parse_args()

    input_path = Path(args.input)
    content = input_path.read_text(encoding="utf-8").strip()
    if input_path.suffix.lower() == ".json":
        records = json.loads(content)
        if isinstance(records, dict):
            records = [records]
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    outputs = []
    for record in records:
        image = load_image(record["image"])
        prompt = record["prompt"]
        prompt_text = _build_chat_prompt(
            model,
            prompt,
            system_prompt=record.get("system_prompt", args.system_prompt),
        )
        prediction = model.generate(
            image,
            prompt_text,
            **_generation_kwargs(args.max_new_tokens, args.temperature),
        )
        enriched = dict(record)
        enriched["prediction"] = prediction
        outputs.append(enriched)

    _write_jsonl(outputs, args.output)
