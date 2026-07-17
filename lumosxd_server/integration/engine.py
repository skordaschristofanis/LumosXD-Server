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

from pyFAI import load as pyfai_load
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

logger = getLogger(__name__)

DEFAULT_UNIT = "2th_deg"


class AzimuthalEngine:
    """Uses one pyFAI AzimuthalIntegrator for repeated 1D integration."""

    def __init__(self, integrator: AzimuthalIntegrator, npt: int, unit: str = DEFAULT_UNIT) -> None:
        if npt < 1:
            raise ValueError(f"npt must be >= 1, got {npt}")
        self._integrator = integrator
        self._npt = npt
        self._unit = unit

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
