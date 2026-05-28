"""Unit tests for the text-protection change-map helper (no model download).

``build_change_map`` is the pure cv2/numpy part of ``text_protector``: it turns
detected text polygons into a Differential-Diffusion change map. The polarity is
load-bearing and was verified empirically (white = preserve, black = change), so
a regression here would either freeze the whole image or fail to protect text.
The PP-OCRv3 detector itself needs a model download and is not exercised here.
"""

from __future__ import annotations

import numpy as np

from remove_ai_watermarks.text_protector import build_change_map


class TestBuildChangeMap:
    def test_no_boxes_is_all_change(self):
        m = build_change_map([], 32, 48)
        assert m.shape == (32, 48)
        assert m.dtype == np.float32
        assert float(m.max()) == 0.0

    def test_text_region_is_preserved_background_is_change(self):
        # A 20x20 box centered in a 64x64 map, no feather for a crisp check.
        box = np.array([[22, 22], [42, 22], [42, 42], [22, 42]])
        m = build_change_map([box], 64, 64, preserve=0.9, feather=0)
        # Inside the polygon: painted to preserve value.
        assert m[32, 32] == np.float32(0.9)
        # Far background: untouched -> full change (0.0).
        assert m[2, 2] == 0.0
        # Polarity: text preserved more than background.
        assert m[32, 32] > m[2, 2]

    def test_preserve_value_is_respected(self):
        box = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        m = build_change_map([box], 40, 40, preserve=0.5, feather=0)
        assert m[20, 20] == np.float32(0.5)

    def test_feather_creates_soft_edge_gradient(self):
        box = np.array([[20, 20], [44, 20], [44, 44], [20, 44]])
        m = build_change_map([box], 64, 64, preserve=1.0, feather=15)
        center = m[32, 32]
        # An edge pixel just outside the polygon should be partially blended:
        # strictly between full-change (0) and the preserved center.
        edge = m[32, 47]
        assert 0.0 < edge < center
        assert center <= 1.0

    def test_even_feather_does_not_crash(self):
        box = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        m = build_change_map([box], 40, 40, feather=14)
        assert m.shape == (40, 40)

    def test_values_stay_in_unit_range(self):
        box = np.array([[5, 5], [35, 5], [35, 35], [5, 35]])
        m = build_change_map([box], 40, 40, preserve=1.0, feather=9)
        assert float(m.min()) >= 0.0
        assert float(m.max()) <= 1.0
