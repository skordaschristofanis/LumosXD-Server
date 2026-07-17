#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_pattern.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for the Pattern result type.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

import numpy as np

from lumosxd_server.integration import Pattern


def test_pattern_construction() -> None:
    radial = np.linspace(0.0, 10.0, 5)
    intensity = np.ones(5)
    pattern = Pattern(radial=radial, intensity=intensity, unit="2th_deg")

    assert pattern.unit == "2th_deg"
    assert pattern.radial.shape == (5,)
    assert pattern.intensity.shape == (5,)
    np.testing.assert_array_equal(pattern.radial, radial)
    np.testing.assert_array_equal(pattern.intensity, intensity)
