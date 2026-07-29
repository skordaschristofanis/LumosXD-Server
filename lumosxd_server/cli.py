#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/cli.py
# ----------------------------------------------------------------------------------
# Purpose:
# Command implementations for the integrate and cake CLI subcommands.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from argparse import Namespace
from logging import getLogger
from pathlib import Path

import fabio
import h5py
import numpy as np

from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import FrameStack, integrate_cake_stack, integrate_stack

logger = getLogger(__name__)

_NPT_FACTORS = {"1d": 1.5, "2d": 2.0}


def _calculate_npt(poni_path: Path, frame_shape: tuple[int, int], mode: str) -> int:
    """Calculate radial integration points from beam center to farthest image corner."""
    ai = AzimuthalIntegrator()
    ai.load(str(poni_path))
    center_y = ai.poni1 / ai.pixel1
    center_x = ai.poni2 / ai.pixel2
    h, w = frame_shape
    max_dist = max(
        np.sqrt((r - center_y) ** 2 + (c - center_x) ** 2)
        for r, c in ((0, 0), (0, w), (h, 0), (h, w))
    )
    return int(max_dist * _NPT_FACTORS[mode])

_NPY_EXTS = {".npy"}
_H5_EXTS = {".h5", ".hdf5", ".nxs", ".nx"}
_IMAGE_EXTS = {".tif", ".tiff", ".edf", ".cbf", ".mar3450", ".img"}
_ALL_EXTS = _NPY_EXTS | _H5_EXTS | _IMAGE_EXTS


def _find_image_datasets(h5file: h5py.File) -> list[str]:
    """Walk an HDF5 file and return paths of datasets with ndim 2 or 3."""
    found: list[str] = []

    def _visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
            found.append(name)

    h5file.visititems(_visit)
    return found


def _load_h5(path: Path, dataset: str | None) -> FrameStack:
    """Load frames from an HDF5 file. Auto-detects the dataset when not specified."""
    with h5py.File(path, "r") as f:
        if dataset:
            ds = f[dataset]
            raw = ds[()]
        else:
            candidates = _find_image_datasets(f)
            if not candidates:
                raise ValueError(f"No 2D/3D datasets found in {path}")
            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple image datasets in {path}: {candidates}. "
                    "Use --h5-dataset to specify one."
                )
            ds_path = candidates[0]
            logger.info("Auto-selected dataset '%s' from %s", ds_path, path)
            raw = f[ds_path][()]

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 2:
        return FrameStack(raw[np.newaxis, ...])
    if raw.ndim == 3:
        return FrameStack(raw)
    raise ValueError(f"Dataset has unsupported shape {raw.shape}")


def _read_frame(path: Path) -> np.ndarray:
    """Read a single 2D detector frame from .npy or any fabio-supported format."""
    if path.suffix.lower() in _NPY_EXTS:
        return np.load(path)
    return fabio.open(str(path)).data


def _load_input(path: Path, h5_dataset: str | None = None) -> FrameStack:
    """Load a single frame file, a 3D .npy stack, an HDF5 file, or a directory of frame files."""
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in _ALL_EXTS)
        if not files:
            raise ValueError(f"No supported image files found in directory: {path}")
        frames: list[np.ndarray] = []
        for f in files:
            if f.suffix.lower() in _H5_EXTS:
                stack = _load_h5(f, h5_dataset)
                frames.extend(stack[i] for i in range(stack.n_frames))
            else:
                frames.append(_read_frame(f))
        bad = [files[i] for i, fr in enumerate(frames) if np.asarray(fr).ndim != 2]
        if bad:
            raise ValueError(f"All files must be 2D frames; bad files: {[str(b) for b in bad]}")
        data = np.stack(frames, axis=0)
        logger.info("Loaded %d frames from %s", len(frames), path)
        return FrameStack(data.astype(np.float64))

    if path.suffix.lower() in _H5_EXTS:
        return _load_h5(path, h5_dataset)

    if path.suffix.lower() in _NPY_EXTS:
        raw = np.load(path)
        if raw.ndim == 3:
            logger.info("Loaded frame stack %s from %s", raw.shape, path)
            return FrameStack(raw.astype(np.float64))
        if raw.ndim != 2:
            raise ValueError(f"Input .npy must be 2D or 3D, got shape {raw.shape}")
        data = raw[np.newaxis, ...]
    else:
        raw = _read_frame(path)
        if raw.ndim != 2:
            raise ValueError(f"Expected a 2D frame from {path}, got shape {raw.shape}")
        data = raw[np.newaxis, ...]

    logger.info("Loaded single frame %s from %s", data.shape[1:], path)
    return FrameStack(data.astype(np.float64))


def _load_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    mask = np.load(path).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask .npy must be 2D, got shape {mask.shape}")
    logger.info("Loaded mask %s from %s", mask.shape, path)
    return mask


def run_tests() -> int:
    """Run the project test suite with pytest. Returns the pytest exit code."""
    try:
        import pytest
    except ImportError:
        logger.error("pytest is not installed. Install the dev group: uv sync --group dev")
        return 1

    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    return pytest.main([str(tests_dir), "-v"])


def run_integrate(args: Namespace) -> int:
    """Execute the integrate command. Returns an exit code."""
    try:
        stack = _load_input(args.input, getattr(args, "h5_dataset", None))
        mask = _load_mask(args.mask)

        npt = args.npt or _calculate_npt(args.poni, stack.frame_shape, args.mode)
        logger.info("npt=%d (%s)", npt, "manual" if args.npt else "auto")

        if args.mode == "1d":
            logger.info("Integrating (1D) %d frame(s) — npt=%d unit=%s workers=%s", stack.n_frames, npt, args.unit, args.workers)
            patterns = integrate_stack(
                poni_path=args.poni,
                stack=stack,
                npt=npt,
                unit=args.unit,
                mask=mask,
                workers=args.workers,
                prefer_opencl=args.opencl,
            )
            radial = patterns[0].radial
            intensity = np.stack([p.intensity for p in patterns], axis=0)
            output: Path = args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output, radial=radial, intensity=intensity, unit=np.bytes_(args.unit))
            logger.info("Saved %d pattern(s) to %s", len(patterns), output)

        else:
            logger.info("Integrating (2D) %d frame(s) — npt=%d npt_azim=%d unit=%s workers=%s", stack.n_frames, npt, args.npt_azim, args.unit, args.workers)
            cakes = integrate_cake_stack(
                poni_path=args.poni,
                stack=stack,
                npt=npt,
                npt_azim=args.npt_azim,
                unit=args.unit,
                mask=mask,
                workers=args.workers,
                prefer_opencl=args.opencl,
            )
            radial = cakes[0].radial
            azimuthal = cakes[0].azimuthal
            intensity = np.stack([c.intensity for c in cakes], axis=0)
            output = args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output, radial=radial, azimuthal=azimuthal, intensity=intensity, unit=np.bytes_(args.unit))
            logger.info("Saved %d cake(s) to %s", len(cakes), output)

    except Exception as exc:
        logger.error("integrate failed: %s", exc)
        return 1
    return 0

