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
from typing import Any, Self

import numpy as np
from pyFAI import load as pyfai_load
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.method_registry import IntegrationMethod

from lumosxd_server.integration.results import Cake, Pattern

logger = getLogger(__name__)

DEFAULT_UNIT = "2th_deg"
DEFAULT_POLARIZATION_FACTOR = 0.99
DEFAULT_NPT_AZIM = 360
DEFAULT_METHOD = ("bbox", "csr", "cython")
OPENCL_METHOD = ("bbox", "csr", "opencl")


def _select_method(prefer_opencl: bool, dim: int = 1) -> Any:
    """Pick an integration method for the given dimension, preferring OpenCL CSR when requested and available."""
    if prefer_opencl:
        opencl_methods = IntegrationMethod.select_method(
            dim=dim,
            split=OPENCL_METHOD[0],
            algo=OPENCL_METHOD[1],
            impl=OPENCL_METHOD[2],
            degradable=False,
        )
        if opencl_methods:
            logger.info("Using OpenCL CSR %dD integration method", dim)
            return opencl_methods[0]
        logger.warning("OpenCL CSR %dD unavailable; falling back to Cython CSR", dim)

    cython_methods = IntegrationMethod.select_method(
        dim=dim,
        split=DEFAULT_METHOD[0],
        algo=DEFAULT_METHOD[1],
        impl=DEFAULT_METHOD[2],
    )
    if not cython_methods:
        raise RuntimeError(f"No Cython CSR {dim}D integration method available — check pyFAI installation")
    return cython_methods[0]


class AzimuthalEngine:
    """Uses one pyFAI AzimuthalIntegrator for repeated 1D and 2D (cake) integration."""

    def __init__(
        self,
        integrator: AzimuthalIntegrator,
        npt: int,
        unit: str = DEFAULT_UNIT,
        mask: np.ndarray | None = None,
        prefer_opencl: bool = False,
        npt_azim: int = DEFAULT_NPT_AZIM,
    ) -> None:
        if npt < 1:
            raise ValueError(f"npt must be >= 1, got {npt}")
        if npt_azim < 1:
            raise ValueError(f"npt_azim must be >= 1, got {npt_azim}")
        self._integrator = integrator
        self._npt = npt
        self._npt_azim = npt_azim
        self._unit = unit
        self._mask = mask
        self._method = _select_method(prefer_opencl, dim=1)
        self._cake_method = _select_method(prefer_opencl, dim=2)

    @classmethod
    def from_poni(
        cls,
        poni_path: str | Path,
        npt: int,
        unit: str = DEFAULT_UNIT,
        prefer_opencl: bool = False,
        npt_azim: int = DEFAULT_NPT_AZIM,
    ) -> Self:
        """Load calibration from a .poni file and build an engine."""
        path = Path(poni_path)
        logger.info("Loading calibration from %s", path)
        loaded = pyfai_load(str(path))
        if not isinstance(loaded, AzimuthalIntegrator):
            raise TypeError(f"Expected AzimuthalIntegrator from {path}, got {type(loaded).__name__}")
        return cls(loaded, npt, unit, prefer_opencl=prefer_opencl, npt_azim=npt_azim)

    @property
    def npt(self) -> int:
        return self._npt

    @property
    def npt_azim(self) -> int:
        return self._npt_azim

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Set or clear the pixel mask. Call before warmup — changing the mask after warmup does not rebuild lookup tables."""
        self._mask = mask

    def warmup(self, shape: tuple[int, ...], dim: str = "1d") -> None:
        """Pre-build integration lookup tables for the given frame shape and dimension."""
        if len(shape) != 2:
            raise ValueError(f"Expected shape (height, width), got {shape}")
        if self._mask is not None and self._mask.shape != shape:
            raise ValueError(f"Mask shape {self._mask.shape} does not match warmup shape {shape}")
        height, width = int(shape[0]), int(shape[1])
        logger.info("Warming up integrator for shape=(%s, %s) dim=%s", height, width, dim)
        if dim == "1d":
            self.integrate(np.zeros((height, width), dtype=np.float64))
        else:
            self.integrate_cake(np.zeros((height, width), dtype=np.float64))

    def integrate(self, image: np.ndarray) -> Pattern:
        """Integrate a 2D detector frame to a 1D pattern."""
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {image.shape}")
        if self._mask is not None and self._mask.shape != image.shape:
            raise ValueError(f"Mask shape {self._mask.shape} does not match image shape {image.shape}")

        logger.debug("Integrating frame shape=%s npt=%s unit=%s", image.shape, self._npt, self._unit)
        result = self._integrator.integrate1d(
            image,
            self._npt,
            method=self._method,
            unit=self._unit,
            mask=self._mask,
            polarization_factor=DEFAULT_POLARIZATION_FACTOR,
            correctSolidAngle=True,
        )
        return Pattern(
            radial=np.asarray(result.radial, dtype=np.float64),
            intensity=np.asarray(result.intensity, dtype=np.float64),
            unit=self._unit,
        )

    def integrate_cake(self, image: np.ndarray) -> Cake:
        """Integrate a 2D detector frame to a cake (azimuthal × radial)."""
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {image.shape}")
        if self._mask is not None and self._mask.shape != image.shape:
            raise ValueError(f"Mask shape {self._mask.shape} does not match image shape {image.shape}")

        logger.debug(
            "Cake integrating frame shape=%s npt=%s npt_azim=%s unit=%s",
            image.shape,
            self._npt,
            self._npt_azim,
            self._unit,
        )
        result = self._integrator.integrate2d(
            image,
            self._npt,
            self._npt_azim,
            method=self._cake_method,
            unit=self._unit,
            mask=self._mask,
            polarization_factor=DEFAULT_POLARIZATION_FACTOR,
            correctSolidAngle=True,
        )
        return Cake(
            radial=np.asarray(result.radial, dtype=np.float64),
            azimuthal=np.asarray(result.azimuthal, dtype=np.float64),
            intensity=np.asarray(result.intensity, dtype=np.float64),
            unit=self._unit,
        )
