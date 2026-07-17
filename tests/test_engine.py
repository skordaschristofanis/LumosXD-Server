#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_engine.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for AzimuthalEngine calibration loading.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import AzimuthalEngine, Pattern


@pytest.fixture
def integrator() -> AzimuthalIntegrator:
    return AzimuthalIntegrator(
        dist=0.1,
        poni1=0.05,
        poni2=0.05,
        pixel1=1e-4,
        pixel2=1e-4,
        wavelength=1.0e-10,
    )


@pytest.fixture
def poni_file(tmp_path: Path, integrator: AzimuthalIntegrator) -> Path:
    path = tmp_path / "calibration.poni"
    integrator.save(str(path))
    return path


def test_from_poni_loads_integrator(poni_file: Path) -> None:
    engine = AzimuthalEngine.from_poni(poni_file, npt=512)

    assert isinstance(engine.integrator, AzimuthalIntegrator)
    assert engine.npt == 512
    assert engine.unit == "2th_deg"
    assert engine.integrator.dist == pytest.approx(0.1)


def test_from_poni_rejects_invalid_npt(poni_file: Path) -> None:
    with pytest.raises(ValueError, match="npt"):
        AzimuthalEngine.from_poni(poni_file, npt=0)


def test_from_poni_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.poni"
    with pytest.raises(Exception):
        AzimuthalEngine.from_poni(missing, npt=128)


def test_integrate_returns_pattern(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 64)
    image = np.ones((100, 100), dtype=np.float64)

    pattern = engine.integrate(image)

    assert isinstance(pattern, Pattern)
    assert pattern.unit == "2th_deg"
    assert pattern.radial.shape == (64,)
    assert pattern.intensity.shape == (64,)
    assert np.all(np.isfinite(pattern.radial))
    assert np.all(np.isfinite(pattern.intensity))


def test_integrate_rejects_non_2d(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 32)
    with pytest.raises(ValueError, match="2D"):
        engine.integrate(np.ones(10, dtype=np.float64))


def test_warmup_then_repeated_integrate(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 64)
    shape = (100, 100)
    image = np.ones(shape, dtype=np.float64)

    assert engine.warmed_shape is None
    engine.warmup(shape)
    assert engine.warmed_shape == shape

    patterns = [engine.integrate(image) for _ in range(5)]
    assert len(patterns) == 5
    assert all(pattern.radial.shape == (64,) for pattern in patterns)


def test_warmup_rejects_invalid_shape(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 32)
    with pytest.raises(ValueError, match="shape"):
        engine.warmup((100,))


def test_integrate_with_mask(integrator: AzimuthalIntegrator) -> None:
    shape = (100, 100)
    image = np.ones(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    mask[:, :50] = True

    unmasked = AzimuthalEngine(integrator, 64).integrate(image)
    masked = AzimuthalEngine(integrator, 64, mask=mask).integrate(image)

    assert masked.radial.shape == unmasked.radial.shape
    assert masked.intensity.shape == unmasked.intensity.shape
    assert not np.allclose(masked.intensity, unmasked.intensity)


def test_set_mask_clears_warmup(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 64)
    shape = (100, 100)
    engine.warmup(shape)
    assert engine.warmed_shape == shape

    mask = np.zeros(shape, dtype=bool)
    engine.set_mask(mask)
    assert engine.mask is mask
    assert engine.warmed_shape is None


def test_integrate_rejects_mask_shape_mismatch(integrator: AzimuthalIntegrator) -> None:
    mask = np.zeros((50, 50), dtype=bool)
    engine = AzimuthalEngine(integrator, 32, mask=mask)
    with pytest.raises(ValueError, match="Mask shape"):
        engine.integrate(np.ones((100, 100), dtype=np.float64))


def test_default_method_is_cython(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 32)
    assert "cython" in str(engine.method).lower()


def test_prefer_opencl_falls_back_or_uses_opencl(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, 32, prefer_opencl=True)
    method_name = str(engine.method).lower()
    assert "cython" in method_name or "opencl" in method_name

    pattern = engine.integrate(np.ones((64, 64), dtype=np.float64))
    assert pattern.radial.shape == (32,)
    assert np.all(np.isfinite(pattern.intensity))
