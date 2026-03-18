from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from vlm_backbones.cache import get_artifact_cache_dir, get_model_cache_dir
from vlm_backbones.manifest import ModelSpec
from vlm_backbones.models.load import validate_artifact_layout

console = Console(stderr=True)


def _copy_local_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        _copy_local_file(Path(parsed.path), dst)
        return
    if parsed.scheme == "" and Path(url).exists():
        _copy_local_file(Path(url), dst)
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or "0")
        with dst.open("wb") as handle:
            if total > 0:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Downloading {dst.name}", total=total)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.advance(task, len(chunk))
            else:
                console.print(f"Downloading {dst.name} ...")
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination, filter="data")
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    raise ValueError(f"Unsupported checkpoint artifact format: {archive_path.name}")


def _normalize_extracted_tree(destination: Path) -> None:
    if (destination / "config.json").exists():
        return
    children = [path for path in destination.iterdir()]
    if len(children) != 1 or not children[0].is_dir():
        return
    nested_root = children[0]
    for child in nested_root.iterdir():
        shutil.move(str(child), destination / child.name)
    nested_root.rmdir()


def download_and_extract(spec: ModelSpec, force: bool = False) -> Path:
    if not spec.has_download:
        raise ValueError(
            f"Model `{spec.id}` does not have a real download URL yet. Update model_zoo/models.yaml first."
        )

    model_dir = get_model_cache_dir(spec.id)
    if model_dir.exists() and not force:
        validate_artifact_layout(model_dir)
        return model_dir

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = get_artifact_cache_dir() / spec.artifact_filename
    if force or not artifact_path.exists():
        _download_file(spec.download_url, artifact_path)

    if spec.has_sha256 and _sha256(artifact_path) != spec.sha256:
        raise ValueError(f"Checksum mismatch for `{artifact_path.name}`.")

    _extract_archive(artifact_path, model_dir)
    _normalize_extracted_tree(model_dir)
    validate_artifact_layout(model_dir)
    return model_dir
