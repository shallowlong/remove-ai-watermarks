"""Tests for the optional Adobe TrustMark detector.

TrustMark is an optional dependency (extra ``trustmark``) that downloads model
weights on first use, so the decode path is only exercised when it is installed
(mirrors the imwatermark handling). The always-on test pins the graceful
absent/error behaviour: detect must return None, never raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from remove_ai_watermarks.trustmark_detector import detect_trustmark, is_available

if TYPE_CHECKING:
    from pathlib import Path


def test_detect_never_raises(tmp_clean_png: Path):
    # Whether or not trustmark is installed, a clean image must yield None
    # (no watermark) without raising. When absent, the import guard returns None.
    assert detect_trustmark(tmp_clean_png) is None


def test_unreadable_file_returns_none(tmp_path: Path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_bytes(b"not an image")
    assert detect_trustmark(bad) is None


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_clean_image_reports_no_watermark(tmp_clean_png: Path):
    # With the decoder present, an un-watermarked image must report absent.
    assert detect_trustmark(tmp_clean_png) is None
