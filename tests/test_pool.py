#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_pool.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for parallel FrameStack integration.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import FrameStack, Pattern, integrate_stack


@pytest.fixture
def poni_file(tmp_path: Path) -> Path:
    ai = AzimuthalIntegrator(
        dist=0.1,
        poni1=0.05,
        poni2=0.05,
        pixel1=1e-4,
        pixel2=1e-4,
        wavelength=1.0e-10,
    )
    path = tmp_path / "calibration.poni"
    ai.save(str(path))
    return path


@pytest.fixture
def stack() -> FrameStack:
    data = np.ones((4, 64, 64), dtype=np.float64)
    data[1] *= 2.0
    data[2] *= 3.0
    data[3] *= 4.0
    return FrameStack(data)


def test_integrate_stack_serial(poni_file: Path, stack: FrameStack) -> None:
    patterns = integrate_stack(poni_file, stack, npt=32, workers=1)

    assert len(patterns) == 4
    assert all(isinstance(pattern, Pattern) for pattern in patterns)
    assert all(pattern.radial.shape == (32,) for pattern in patterns)
    assert patterns[0].intensity.mean() < patterns[3].intensity.mean()


def test_integrate_stack_parallel_matches_serial(poni_file: Path, stack: FrameStack) -> None:
    serial = integrate_stack(poni_file, stack, npt=32, workers=1)
    parallel = integrate_stack(poni_file, stack, npt=32, workers=2)

    assert len(parallel) == len(serial)
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(left.radial, right.radial)
        np.testing.assert_allclose(left.intensity, right.intensity)


def test_integrate_stack_with_mask(poni_file: Path, stack: FrameStack) -> None:
    mask = np.zeros(stack.frame_shape, dtype=bool)
    mask[:, :32] = True
    patterns = integrate_stack(poni_file, stack, npt=32, mask=mask, workers=2)

    assert len(patterns) == 4
    assert all(np.all(np.isfinite(pattern.intensity)) for pattern in patterns)


def test_integrate_stack_rejects_mask_mismatch(poni_file: Path, stack: FrameStack) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="Mask shape"):
        integrate_stack(poni_file, stack, npt=16, mask=mask, workers=1)
