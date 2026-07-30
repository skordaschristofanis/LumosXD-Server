#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_cake.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for Cake integration — AzimuthalEngine.integrate_cake() and
# integrate_cake_stack().
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import AzimuthalEngine, Cake, integrate_cake_stack


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


@pytest.fixture
def stack() -> np.ndarray:
    data = np.ones((4, 64, 64), dtype=np.float64)
    data[1] *= 2.0
    data[2] *= 3.0
    data[3] *= 4.0
    return data


def test_cake_fields() -> None:
    radial = np.linspace(0, 10, 64)
    azimuthal = np.linspace(-180, 180, 36)
    intensity = np.ones((36, 64), dtype=np.float64)
    cake = Cake(radial=radial, azimuthal=azimuthal, intensity=intensity, unit="2th_deg")

    assert cake.unit == "2th_deg"
    assert cake.radial.shape == (64,)
    assert cake.azimuthal.shape == (36,)
    assert cake.intensity.shape == (36, 64)


def test_integrate_cake_returns_cake(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=64, npt_azim=36)
    image = np.ones((100, 100), dtype=np.float64)

    cake = engine.integrate_cake(image)

    assert isinstance(cake, Cake)
    assert cake.unit == "2th_deg"
    assert cake.radial.shape == (64,)
    assert cake.azimuthal.shape == (36,)
    assert cake.intensity.shape == (36, 64)
    assert np.all(np.isfinite(cake.radial))
    assert np.all(np.isfinite(cake.azimuthal))


def test_integrate_cake_default_npt_azim(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=64)
    cake = engine.integrate_cake(np.ones((100, 100), dtype=np.float64))

    assert cake.azimuthal.shape == (360,)
    assert cake.intensity.shape == (360, 64)


def test_integrate_cake_rejects_non_2d(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=32)
    with pytest.raises(ValueError, match="2D"):
        engine.integrate_cake(np.ones(10, dtype=np.float64))


def test_integrate_cake_rejects_mask_shape_mismatch(integrator: AzimuthalIntegrator) -> None:
    mask = np.zeros((50, 50), dtype=bool)
    engine = AzimuthalEngine(integrator, npt=32, mask=mask)
    with pytest.raises(ValueError, match="Mask shape"):
        engine.integrate_cake(np.ones((100, 100), dtype=np.float64))


def test_integrate_cake_with_mask(integrator: AzimuthalIntegrator) -> None:
    shape = (100, 100)
    image = np.ones(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    mask[:, :50] = True

    unmasked = AzimuthalEngine(integrator, npt=32).integrate_cake(image)
    masked = AzimuthalEngine(integrator, npt=32, mask=mask).integrate_cake(image)

    assert masked.intensity.shape == unmasked.intensity.shape
    assert not np.allclose(masked.intensity, unmasked.intensity)


def test_warmup_cake_then_repeated_integrate(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=64, npt_azim=36)
    shape = (100, 100)

    assert engine.warmed_cake_shape is None
    engine.warmup_cake(shape)
    assert engine.warmed_cake_shape == shape

    cakes = [engine.integrate_cake(np.ones(shape, dtype=np.float64)) for _ in range(3)]
    assert len(cakes) == 3
    assert all(c.intensity.shape == (36, 64) for c in cakes)


def test_warmup_cake_rejects_invalid_shape(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=32)
    with pytest.raises(ValueError, match="shape"):
        engine.warmup_cake((100,))


def test_set_mask_clears_cake_warmup(integrator: AzimuthalIntegrator) -> None:
    engine = AzimuthalEngine(integrator, npt=32)
    shape = (100, 100)
    engine.warmup_cake(shape)
    assert engine.warmed_cake_shape == shape

    engine.set_mask(np.zeros(shape, dtype=bool))
    assert engine.warmed_cake_shape is None


def test_from_poni_with_npt_azim(poni_file: Path) -> None:
    engine = AzimuthalEngine.from_poni(poni_file, npt=64, npt_azim=180)

    assert engine.npt == 64
    assert engine.npt_azim == 180

    cake = engine.integrate_cake(np.ones((100, 100), dtype=np.float64))
    assert cake.azimuthal.shape == (180,)
    assert cake.intensity.shape == (180, 64)


def test_invalid_npt_azim(integrator: AzimuthalIntegrator) -> None:
    with pytest.raises(ValueError, match="npt_azim"):
        AzimuthalEngine(integrator, npt=32, npt_azim=0)


def test_integrate_cake_stack_serial(poni_file: Path, stack: np.ndarray) -> None:
    cakes = integrate_cake_stack(poni_file, stack, npt=32, npt_azim=36, workers=1)

    assert len(cakes) == 4
    assert all(isinstance(c, Cake) for c in cakes)
    assert all(c.intensity.shape == (36, 32) for c in cakes)
    assert cakes[0].intensity.mean() < cakes[3].intensity.mean()


def test_integrate_cake_stack_parallel_matches_serial(poni_file: Path, stack: np.ndarray) -> None:
    serial = integrate_cake_stack(poni_file, stack, npt=32, npt_azim=36, workers=1)
    parallel = integrate_cake_stack(poni_file, stack, npt=32, npt_azim=36, workers=2)

    assert len(parallel) == len(serial)
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(left.radial, right.radial)
        np.testing.assert_allclose(left.azimuthal, right.azimuthal)
        np.testing.assert_allclose(left.intensity, right.intensity)


def test_integrate_cake_stack_with_mask(poni_file: Path, stack: np.ndarray) -> None:
    mask = np.zeros(stack.shape[1:], dtype=bool)
    mask[:, :32] = True
    cakes = integrate_cake_stack(poni_file, stack, npt=32, npt_azim=36, mask=mask, workers=2)

    assert len(cakes) == 4
    assert all(np.all(np.isfinite(c.intensity)) for c in cakes)


def test_integrate_cake_stack_rejects_mask_mismatch(poni_file: Path, stack: np.ndarray) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="Mask shape"):
        integrate_cake_stack(poni_file, stack, npt=16, mask=mask, workers=1)

