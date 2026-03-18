# Model Zoo

The canonical release manifest is `model_zoo/models.yaml`.

Each entry includes:

- stable public model id
- display name
- model family
- task type
- direct checkpoint artifact URL
- SHA256 checksum
- public artifact filename
- internal run id
- runtime backbone configuration
- weighted public metrics

## Released IDs

| Public ID | Display Name | Family | Task | Vision Backbone | Resize Strategy |
| --- | --- | --- | --- | --- | --- |
| `vit-s-in1k-224` | ViT-S / IN1K / 224 | vit | classification | `in1k-vit-s` | `letterbox` |
| `maxvit-t-in1k-224-s3` | MaxViT-T / IN1K / 224 / stage 3 | maxvit | classification | `in1k-224px-maxvit-t-s3` | `letterbox` |
| `mambavision-b-in1k-224-s3` | MambaVision-B / IN1K / 224 / stage 3 | mambavision | classification | `mambavision-b-s3` | `letterbox` |
| `vmamba-s-in1k-224-s3` | VMamba-S / IN1K / 224 / stage 3 | vmamba | classification | `vmamba-small-s2l15` | `letterbox` |
| `vitdet-b-coco-1024` | ViTDet-B / COCO / 1024 | vitdet | detection | `vitdet-b-maskrcnn` | `letterbox` |
| `vmamba-s-coco-1333x800` | VMamba-S / COCO / 1333x800 | vmamba | detection | `vmamba-small-s2l15-det-maskrcnn-1x` | `letterbox` |
| `vit-adapter-deit-b-ade20k-512` | ViT-Adapter DeiT-B / ADE20K / 512 | vit_adapter | segmentation | `vit-adapter-upernet-deit-b-ade20k-512` | `letterbox` |
| `vmamba-s-ade20k-512` | VMamba-S / ADE20K / 512 | vmamba | segmentation | `vmamba-small-s2l15-seg-ade20k` | `letterbox` |

## Checkpoint Packaging Contract

Each downloadable model artifact must extract to:

```text
<cache>/models/<public-id>/
  config.json
  checkpoints/latest-checkpoint.pt
```

The public downloader normalizes one nested root directory if present, but the final extracted layout must match the contract above.
