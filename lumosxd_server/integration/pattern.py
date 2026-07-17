#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/pattern.py
# ----------------------------------------------------------------------------------
# Purpose:
# Data type for a 1D diffraction pattern produced by azimuthal integration.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Pattern:
    """1D azimuthally integrated diffraction pattern."""

    radial: np.ndarray
    intensity: np.ndarray
    unit: str
