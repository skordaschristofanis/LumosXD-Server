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
from collections.abc import Callable
from itertools import groupby
from logging import getLogger
from pathlib import Path

import fabio
import h5py
import numpy as np

from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.integration import (
    Cake, Pattern,
    integrate_cake_stack, integrate_h5_cake_stack,
    integrate_h5_stack, integrate_stack,
)

logger = getLogger(__name__)

_NPT_FACTORS = {"1d": 1.5, "2d": 2.0}
_NPY_EXTS = {".npy"}
_H5_EXTS = {".h5", ".hdf5", ".nxs", ".nx"}
_IMAGE_EXTS = {".tif", ".tiff", ".edf", ".cbf", ".mar3450", ".img"}
_ALL_EXTS = _NPY_EXTS | _H5_EXTS | _IMAGE_EXTS

_RADIAL_AXIS = {
    "2th_deg": ("two_theta", "degrees"),
    "2th_rad": ("two_theta", "radians"),
    "q_nm^-1": ("q", "nm^-1"),
    "q_A^-1": ("q", "angstrom^-1"),
    "d_nm":    ("d", "nm"),
    "d_A":     ("d", "angstrom"),
}


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


def _find_image_datasets(h5file: h5py.File) -> list[str]:
    """Walk an HDF5 file and return paths of datasets with ndim 2 or 3."""
    found: list[str] = []

    def _visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
            found.append(name)

    h5file.visititems(_visit)
    return found


def _resolve_h5_dataset(f: h5py.File, h5_dataset: str | None, path: Path) -> str:
    if h5_dataset:
        return h5_dataset
    candidates = _find_image_datasets(f)
    if not candidates:
        raise ValueError(f"No 2D/3D datasets found in {path}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple image datasets in {path}: {candidates}. "
            "Use --h5-dataset to specify one."
        )
    logger.info("Auto-selected dataset '%s' from %s", candidates[0], path)
    return candidates[0]


def _load_h5(path: Path, dataset: str | None) -> tuple[np.ndarray, list[str], list[Path]]:
    """Load frames from an HDF5 file. Auto-detects the dataset when not specified."""
    with h5py.File(path, "r") as f:
        ds_path = _resolve_h5_dataset(f, dataset, path)
        raw = f[ds_path][()]

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 2:
        return raw[np.newaxis, ...], [path.stem], [path]
    if raw.ndim == 3:
        n = raw.shape[0]
        return raw, [f"{path.stem}_{i:04d}" for i in range(n)], [path] * n
    raise ValueError(f"Dataset has unsupported shape {raw.shape}")


def _read_frame(path: Path) -> np.ndarray:
    """Read a single 2D detector frame from .npy or any fabio-supported format."""
    if path.suffix.lower() in _NPY_EXTS:
        return np.load(path)
    return fabio.open(str(path)).data


def _load_input(path: Path, h5_dataset: str | None = None) -> tuple[np.ndarray, list[str], list[Path]]:
    """Load a single frame file, a 3D .npy stack, an HDF5 file, or a directory of frame files.

    Returns an ndarray of shape (n_frames, h, w), a list of frame names, and a list of source paths.
    """
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in _ALL_EXTS)
        if not files:
            raise ValueError(f"No supported image files found in directory: {path}")
        frames: list[np.ndarray] = []
        names: list[str] = []
        sources: list[Path] = []
        for f in files:
            if f.suffix.lower() in _H5_EXTS:
                stack, h5_names, h5_sources = _load_h5(f, h5_dataset)
                frames.extend(stack[i] for i in range(stack.shape[0]))
                names.extend(h5_names)
                sources.extend(h5_sources)
            else:
                frames.append(_read_frame(f))
                names.append(f.stem)
                sources.append(f)
        bad = [name for fr, name in zip(frames, names) if np.asarray(fr).ndim != 2]
        if bad:
            raise ValueError(f"All files must be 2D frames; bad files: {bad}")
        data = np.stack(frames, axis=0)
        logger.info("Loaded %d frames from %s", len(frames), path)
        return data.astype(np.float64), names, sources

    if path.suffix.lower() in _H5_EXTS:
        return _load_h5(path, h5_dataset)

    if path.suffix.lower() in _NPY_EXTS:
        raw = np.load(path)
        if raw.ndim == 3:
            logger.info("Loaded frame stack %s from %s", raw.shape, path)
            n = raw.shape[0]
            return raw.astype(np.float64), [f"{path.stem}_{i:04d}" for i in range(n)], [path] * n
        if raw.ndim != 2:
            raise ValueError(f"Input .npy must be 2D or 3D, got shape {raw.shape}")
        data = raw[np.newaxis, ...]
    else:
        raw = _read_frame(path)
        if raw.ndim != 2:
            raise ValueError(f"Expected a 2D frame from {path}, got shape {raw.shape}")
        data = raw[np.newaxis, ...]

    logger.info("Loaded single frame %s from %s", data.shape[1:], path)
    return data.astype(np.float64), [path.stem], [path]


def _load_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    mask = np.load(path).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask .npy must be 2D, got shape {mask.shape}")
    logger.info("Loaded mask %s from %s", mask.shape, path)
    return mask


def _write_h5_nexus(source: Path, group_list: list, mode: str, unit: str) -> None:
    """Write integration results back into an HDF5 file following the NeXus convention."""
    axis_name, axis_units = _RADIAL_AXIS.get(unit, ("radial", unit))
    radial = group_list[0][0].radial
    intensity = np.stack([r.intensity for r, _, _ in group_list], axis=0)
    multi_frame = intensity.shape[0] > 1

    with h5py.File(source, "a") as f:
        entry = f.require_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        process_name = f"integration_{mode}"
        if process_name in entry:
            del entry[process_name]
        process = entry.create_group(process_name)
        process.attrs["NX_class"] = "NXprocess"
        process["program"] = "lumosxd-server"

        nxdata = process.create_group("results")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "intensity"

        if mode == "1d":
            radial_idx = 1 if multi_frame else 0
            nxdata.attrs["axes"] = [axis_name] if not multi_frame else [".", axis_name]
            nxdata.attrs[f"{axis_name}_indices"] = [radial_idx]
            ax = nxdata.create_dataset(axis_name, data=radial)
            ax.attrs["units"] = axis_units
            ds = nxdata.create_dataset("intensity", data=intensity if multi_frame else intensity[0])
            ds.attrs["units"] = "counts"
        else:
            azimuthal = group_list[0][0].azimuthal
            chi_idx, radial_idx = (1, 2) if multi_frame else (0, 1)
            nxdata.attrs["axes"] = ["chi", axis_name] if not multi_frame else [".", "chi", axis_name]
            nxdata.attrs["chi_indices"] = [chi_idx]
            nxdata.attrs[f"{axis_name}_indices"] = [radial_idx]
            chi = nxdata.create_dataset("chi", data=azimuthal)
            chi.attrs["units"] = "degrees"
            ax = nxdata.create_dataset(axis_name, data=radial)
            ax.attrs["units"] = axis_units
            ds = nxdata.create_dataset("intensity", data=intensity if multi_frame else intensity[0])
            ds.attrs["units"] = "counts"

    logger.info("Wrote %d result(s) to %s:/entry/%s", len(group_list), source, process_name)


def _save_split(results: list[Pattern | Cake], names: list[str], sources: list[Path], output_dir: Path, mode: str, unit: str) -> None:
    """Save split output: HDF5 sources write back into the same file; others write .npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, group in groupby(zip(results, names, sources), key=lambda x: x[2]):
        group_list = list(group)
        if source.suffix.lower() in _H5_EXTS:
            _write_h5_nexus(source, group_list, mode, unit)
        else:
            for result, name, _ in group_list:
                out = output_dir / f"{name}_{mode}.npz"
                if mode == "1d":
                    np.savez(out, radial=result.radial, intensity=result.intensity, unit=np.bytes_(unit))
                else:
                    np.savez(out, radial=result.radial, azimuthal=result.azimuthal, intensity=result.intensity, unit=np.bytes_(unit))


def _get_h5_metadata(
    path: Path, h5_dataset: str | None
) -> tuple[int, tuple[int, int], str, list[str], list[Path]]:
    """Return (n_frames, frame_shape, dataset_path, names, sources) without loading frame data."""
    with h5py.File(path, "r") as f:
        ds_path = _resolve_h5_dataset(f, h5_dataset, path)
        ds = f[ds_path]
        if ds.ndim == 2:
            frame_shape = (int(ds.shape[0]), int(ds.shape[1]))
            return 1, frame_shape, ds_path, [path.stem], [path]
        if ds.ndim == 3:
            n = int(ds.shape[0])
            frame_shape = (int(ds.shape[1]), int(ds.shape[2]))
            return n, frame_shape, ds_path, [f"{path.stem}_{i:04d}" for i in range(n)], [path] * n
        raise ValueError(f"Dataset has unsupported shape {ds.shape}")


def _progress_callback(label: str) -> Callable[[int, int], None]:
    def _cb(done: int, total: int) -> None:
        logger.info("%s: frame %d/%d", label, done, total)
    return _cb


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
        mask = _load_mask(args.mask)
        progress = _progress_callback("1D" if args.mode == "1d" else "2D") if args.verbose else None
        h5_dataset = getattr(args, "h5_dataset", None)

        is_single_h5 = not args.input.is_dir() and args.input.suffix.lower() in _H5_EXTS

        if is_single_h5:
            n_frames, frame_shape, dataset, names, sources = _get_h5_metadata(args.input, h5_dataset)
            npt = args.npt or _calculate_npt(args.poni, frame_shape, args.mode)
            logger.info("npt=%d (%s) — streaming %d HDF5 frame(s)", npt, "manual" if args.npt else "auto", n_frames)

            if args.mode == "1d":
                logger.info("Integrating (1D) npt=%d unit=%s workers=%s", npt, args.unit, args.workers)
                patterns = integrate_h5_stack(
                    h5_path=args.input, dataset=dataset, n_frames=n_frames, frame_shape=frame_shape,
                    poni_path=args.poni, npt=npt, unit=args.unit, mask=mask,
                    workers=args.workers, prefer_opencl=args.opencl, progress_callback=progress,
                )
                if args.split:
                    output_dir = args.output or args.input.parent
                    _save_split(patterns, names, sources, output_dir, "1d", args.unit)
                    logger.info("Saved %d pattern(s) to %s", len(patterns), output_dir)
                else:
                    out: Path = args.output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out, radial=patterns[0].radial, intensity=np.stack([p.intensity for p in patterns], axis=0), unit=np.bytes_(args.unit))
                    logger.info("Saved %d pattern(s) to %s", len(patterns), out)

            else:
                logger.info("Integrating (2D) npt=%d npt_azim=%d unit=%s workers=%s", npt, args.npt_azim, args.unit, args.workers)
                cakes = integrate_h5_cake_stack(
                    h5_path=args.input, dataset=dataset, n_frames=n_frames, frame_shape=frame_shape,
                    poni_path=args.poni, npt=npt, npt_azim=args.npt_azim, unit=args.unit, mask=mask,
                    workers=args.workers, prefer_opencl=args.opencl, progress_callback=progress,
                )
                if args.split:
                    output_dir = args.output or args.input.parent
                    _save_split(cakes, names, sources, output_dir, "2d", args.unit)
                    logger.info("Saved %d cake(s) to %s", len(cakes), output_dir)
                else:
                    out = args.output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out, radial=cakes[0].radial, azimuthal=cakes[0].azimuthal, intensity=np.stack([c.intensity for c in cakes], axis=0), unit=np.bytes_(args.unit))
                    logger.info("Saved %d cake(s) to %s", len(cakes), out)

        else:
            stack, names, sources = _load_input(args.input, h5_dataset)
            npt = args.npt or _calculate_npt(args.poni, stack.shape[1:], args.mode)
            logger.info("npt=%d (%s)", npt, "manual" if args.npt else "auto")

            if args.mode == "1d":
                logger.info("Integrating (1D) %d frame(s) — npt=%d unit=%s workers=%s", stack.shape[0], npt, args.unit, args.workers)
                patterns = integrate_stack(
                    poni_path=args.poni, stack=stack, npt=npt, unit=args.unit, mask=mask,
                    workers=args.workers, prefer_opencl=args.opencl, progress_callback=progress,
                )
                if args.split:
                    output_dir = args.output or (args.input if args.input.is_dir() else args.input.parent)
                    _save_split(patterns, names, sources, output_dir, "1d", args.unit)
                    logger.info("Saved %d pattern(s) to %s", len(patterns), output_dir)
                else:
                    out = args.output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out, radial=patterns[0].radial, intensity=np.stack([p.intensity for p in patterns], axis=0), unit=np.bytes_(args.unit))
                    logger.info("Saved %d pattern(s) to %s", len(patterns), out)

            else:
                logger.info("Integrating (2D) %d frame(s) — npt=%d npt_azim=%d unit=%s workers=%s", stack.shape[0], npt, args.npt_azim, args.unit, args.workers)
                cakes = integrate_cake_stack(
                    poni_path=args.poni, stack=stack, npt=npt, npt_azim=args.npt_azim, unit=args.unit, mask=mask,
                    workers=args.workers, prefer_opencl=args.opencl, progress_callback=progress,
                )
                if args.split:
                    output_dir = args.output or (args.input if args.input.is_dir() else args.input.parent)
                    _save_split(cakes, names, sources, output_dir, "2d", args.unit)
                    logger.info("Saved %d cake(s) to %s", len(cakes), output_dir)
                else:
                    out = args.output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out, radial=cakes[0].radial, azimuthal=cakes[0].azimuthal, intensity=np.stack([c.intensity for c in cakes], axis=0), unit=np.bytes_(args.unit))
                    logger.info("Saved %d cake(s) to %s", len(cakes), out)

    except Exception:
        logger.exception("integrate failed")
        return 1
    return 0
