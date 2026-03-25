from __future__ import annotations

import json
import importlib
import hashlib
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from prismatic import load
from prismatic.models.vlms import PrismaticVLM
from vlm_backbones.download import download_and_extract
from vlm_backbones.manifest import ModelMetrics, ModelSpec


class PrismaticLoadTest(unittest.TestCase):
    def test_load_local_checkpoint_directory(self) -> None:
        load_module = importlib.import_module("prismatic.models.load")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "model_id": "in1k-224px-maxvit-t-letterbox-s3+7b-vicuna",
                            "vision_backbone_id": "in1k-224px-maxvit-t-s3",
                            "llm_backbone_id": "vicuna-v15-7b",
                            "image_resize_strategy": "letterbox",
                            "arch_specifier": "no-align+gelu-mlp",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "latest-checkpoint.pt").write_bytes(b"checkpoint")

            sentinel = object()
            with mock.patch.object(load_module, "get_vision_backbone_and_transform", return_value=(object(), object())):
                with mock.patch.object(load_module, "get_llm_backbone_and_tokenizer", return_value=(object(), object())):
                    with mock.patch.object(PrismaticVLM, "from_pretrained", return_value=sentinel) as from_pretrained:
                        loaded = load(run_dir)

        self.assertIs(loaded, sentinel)
        from_pretrained.assert_called_once()

    def test_invalid_cached_public_model_is_rebuilt(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models" / "vit-s-in1k-224"
            artifact_cache_dir = root / "artifacts"
            payload_dir = root / "payload"
            payload_nested = payload_dir / "release"
            archive_path = root / "vit-s-in1k-224.tar"

            model_dir.mkdir(parents=True)
            artifact_cache_dir.mkdir(parents=True)
            (model_dir / "stale.txt").write_text("stale", encoding="utf-8")

            (payload_nested / "checkpoints").mkdir(parents=True)
            (payload_nested / "config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "model_id": "in1k-224px-vit-s+7b-vicuna",
                            "vision_backbone_id": "in1k-vit-s",
                            "llm_backbone_id": "vicuna-v15-7b",
                            "image_resize_strategy": "letterbox",
                            "arch_specifier": "no-align+gelu-mlp",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (payload_nested / "checkpoints" / "latest-checkpoint.pt").write_bytes(b"checkpoint")

            with tarfile.open(archive_path, "w") as archive:
                archive.add(payload_nested, arcname="release")

            archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            cached_artifact = artifact_cache_dir / archive_path.name
            cached_artifact.write_bytes(b"corrupted")

            spec = ModelSpec(
                id="vit-s-in1k-224",
                display_name="ViT-S",
                family="vit",
                task="classification",
                download_url=str(archive_path),
                sha256=archive_sha256,
                artifact_filename=archive_path.name,
                internal_run_id="in1k-224px-vit-s+7b-vicuna+ep1+ft+x7",
                vision_backbone_id="in1k-vit-s",
                image_resize_strategy="letterbox",
                arch_specifier="no-align+gelu-mlp",
                llm_backbone_id="vicuna-v15-7b",
                metrics=ModelMetrics(weighted_vqa=0.0, weighted_loc=0.0, weighted_overall=0.0),
            )

            with mock.patch("vlm_backbones.download.get_model_cache_dir", return_value=model_dir):
                with mock.patch("vlm_backbones.download.get_artifact_cache_dir", return_value=artifact_cache_dir):
                    rebuilt_dir = download_and_extract(spec, force=False)

            self.assertEqual(rebuilt_dir, model_dir)
            self.assertTrue((rebuilt_dir / "config.json").exists())
            self.assertTrue((rebuilt_dir / "checkpoints" / "latest-checkpoint.pt").exists())
            self.assertFalse((rebuilt_dir / "stale.txt").exists())
            self.assertEqual(hashlib.sha256(cached_artifact.read_bytes()).hexdigest(), archive_sha256)


if __name__ == "__main__":
    unittest.main()
