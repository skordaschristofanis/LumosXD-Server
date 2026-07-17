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

import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import AzimuthalEngine


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
