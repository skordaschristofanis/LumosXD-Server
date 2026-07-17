#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/engine.py
# ----------------------------------------------------------------------------------
# Purpose:
# Long-lived azimuthal integration engine that uses a pyFAI integrator.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from logging import getLogger
from pathlib import Path
from typing import Self

import numpy as np
from pyFAI import load as pyfai_load
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration.pattern import Pattern

logger = getLogger(__name__)

DEFAULT_UNIT = "2th_deg"
DEFAULT_POLARIZATION_FACTOR = 0.99
DEFAULT_METHOD = ("bbox", "csr", "cython")


class AzimuthalEngine:
    """Uses one pyFAI AzimuthalIntegrator for repeated 1D integration."""

    def __init__(self, integrator: AzimuthalIntegrator, npt: int, unit: str = DEFAULT_UNIT) -> None:
        if npt < 1:
            raise ValueError(f"npt must be >= 1, got {npt}")
        self._integrator = integrator
        self._npt = npt
        self._unit = unit
        self._warmed_shape: tuple[int, int] | None = None

    @classmethod
    def from_poni(cls, poni_path: str | Path, npt: int, unit: str = DEFAULT_UNIT) -> Self:
        """Load calibration from a .poni file and build an engine."""
        path = Path(poni_path)
        logger.info("Loading calibration from %s", path)
        loaded = pyfai_load(str(path))
        if not isinstance(loaded, AzimuthalIntegrator):
            raise TypeError(f"Expected AzimuthalIntegrator from {path}, got {type(loaded).__name__}")
        return cls(loaded, npt, unit)

    @property
    def integrator(self) -> AzimuthalIntegrator:
        return self._integrator

    @property
    def npt(self) -> int:
        return self._npt

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def warmed_shape(self) -> tuple[int, int] | None:
        return self._warmed_shape

    def warmup(self, shape: tuple[int, int]) -> None:
        """Build sparse integration tables for shape using a zero frame."""
        if len(shape) != 2:
            raise ValueError(f"Expected shape (height, width), got {shape}")

        height, width = int(shape[0]), int(shape[1])
        logger.info("Warming up integrator for shape=(%s, %s)", height, width)
        self.integrate(np.zeros((height, width), dtype=np.float64))
        self._warmed_shape = (height, width)

    def integrate(self, image: np.ndarray) -> Pattern:
        """Integrate a 2D detector frame to a 1D pattern."""
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {image.shape}")

        logger.debug("Integrating frame shape=%s npt=%s unit=%s", image.shape, self._npt, self._unit)
        result = self._integrator.integrate1d(
            image,
            self._npt,
            method=DEFAULT_METHOD,
            unit=self._unit,
            polarization_factor=DEFAULT_POLARIZATION_FACTOR,
            correctSolidAngle=True,
        )
        return Pattern(
            radial=np.asarray(result.radial, dtype=np.float64),
            intensity=np.asarray(result.intensity, dtype=np.float64),
            unit=self._unit,
        )
