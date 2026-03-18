from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def load_image(source: str | Path) -> Image.Image:
    if isinstance(source, Path):
        return Image.open(source).convert("RGB")

    parsed = urlparse(str(source))
    if parsed.scheme in {"http", "https"}:
        response = requests.get(str(source), timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(Path(source)).convert("RGB")
