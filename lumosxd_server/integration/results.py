#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/results.py
# ----------------------------------------------------------------------------------
# Purpose:
# Result types returned by azimuthal integration: Pattern (1D) and Cake (2D).
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


@dataclass(frozen=True, slots=True)
class Cake:
    """2D azimuthally integrated cake (azimuthal × radial)."""

    radial: np.ndarray
    azimuthal: np.ndarray
    intensity: np.ndarray
    unit: str
