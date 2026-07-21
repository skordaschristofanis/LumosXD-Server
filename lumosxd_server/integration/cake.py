#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/cake.py
# ----------------------------------------------------------------------------------
# Purpose:
# Data type for a 2D cake image produced by azimuthal integration.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Cake:
    """2D azimuthally integrated cake (azimuthal × radial)."""

    radial: np.ndarray
    azimuthal: np.ndarray
    intensity: np.ndarray
    unit: str
