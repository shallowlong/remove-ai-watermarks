"""Shared fixtures for remove-ai-watermarks test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

CORPUS_NEG_DIR = Path(__file__).resolve().parent.parent / "data" / "synthid_corpus" / "images" / "neg"


@pytest.fixture
def clean_photo() -> Path:
    """A verified-negative real photo from the corpus neg/ set.

    Used by the "non-AI image" assertions (no SynthID, verdict unknown). These
    are real photos with no AI provenance, the ground truth for "must not false-
    positive". Skips if the corpus is not checked out.
    """
    files = sorted(CORPUS_NEG_DIR.glob("*")) if CORPUS_NEG_DIR.exists() else []
    if not files:
        pytest.skip("no corpus neg/ images present")
    return files[0]


@pytest.fixture
def tmp_image_path(tmp_path: Path) -> Path:
    """Create a minimal 200x200 test PNG image and return its path."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_large_image_path(tmp_path: Path) -> Path:
    """Create a 1200x1200 test PNG image (triggers large watermark branch)."""
    img = np.random.randint(0, 255, (1200, 1200, 3), dtype=np.uint8)
    path = tmp_path / "test_large.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_jpeg_path(tmp_path: Path) -> Path:
    """Create a minimal JPEG test image."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_png_with_ai_metadata(tmp_path: Path) -> Path:
    """Create a PNG with AI-related metadata keys."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    pnginfo = PngInfo()
    pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, CFG scale: 7")
    pnginfo.add_text("prompt", "a beautiful landscape")
    pnginfo.add_text("Author", "Test Author")
    path = tmp_path / "ai_metadata.png"
    img.save(path, pnginfo=pnginfo)
    return path


@pytest.fixture
def tmp_clean_png(tmp_path: Path) -> Path:
    """Create a PNG with no AI metadata."""
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    pnginfo = PngInfo()
    pnginfo.add_text("Author", "Human Artist")
    pnginfo.add_text("Title", "Test Artwork")
    path = tmp_path / "clean.png"
    img.save(path, pnginfo=pnginfo)
    return path
