# Notice

This repository is an inference-only release derived in part from the public `TRI-ML/prismatic-vlms` codebase.

Included and adapted components:

- The multimodal projector and generation path under `src/vlm_backbones/models/vlms/`
- Vicuna/LLaMA loading and prompt formatting under `src/vlm_backbones/models/backbones/llm/`
- Vision backbone wrappers under `src/vlm_backbones/models/backbones/vision/`

Third-party source trees are expected under `third_party/` and keep their original upstream licenses:

- `third_party/VMamba`
- `third_party/MambaVision`
- `third_party/ViT-Adapter`
- `third_party/detectron2`

Before publishing a public tag, update:

- `model_zoo/models.yaml` with final direct checkpoint URLs and SHA256 checksums
- `CITATION.cff` with the final repository URL and author metadata
- `.gitmodules` if any fork URLs or pinned commits change
