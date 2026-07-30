#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/utils.py
# ----------------------------------------------------------------------------------
# Purpose:
# Computation utilities for azimuthal integration, such as automatic npt selection.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# scale pixel-distance to farthest corner; 2D needs more bins for azimuthal resolution
_NPT_FACTORS = {"1d": 1.5, "2d": 2.0}


def calculate_npt(poni_path: str | Path, frame_shape: tuple[int, int], mode: str) -> int:
    """Calculate radial integration points from beam center to farthest image corner."""
    ai = AzimuthalIntegrator()
    ai.load(str(poni_path))
    if ai.pixel1 <= 0 or ai.pixel2 <= 0:
        raise ValueError(f"Invalid pixel size in {poni_path}: pixel1={ai.pixel1}, pixel2={ai.pixel2}")
    center_y = ai.poni1 / ai.pixel1
    center_x = ai.poni2 / ai.pixel2
    h, w = frame_shape
    max_dist = max(np.sqrt((r - center_y) ** 2 + (c - center_x) ** 2) for r, c in ((0, 0), (0, w), (h, 0), (h, w)))
    return int(max_dist * _NPT_FACTORS[mode])

