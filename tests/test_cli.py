#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_cli.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for CLI input loading (_load_input) covering .npy, .tif, and .h5 inputs.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from pathlib import Path

import fabio.tifimage
import h5py
import numpy as np
import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from lumosxd_server.cli import _calculate_npt, _load_input
from lumosxd_server.integration import FrameStack


FRAME_SHAPE = (32, 32)


def _make_frame(value: float = 1.0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.float32)


def _write_tif(path: Path, frame: np.ndarray) -> None:
    img = fabio.tifimage.TifImage(data=frame)
    img.write(str(path))


def test_calculate_npt_1d_centered_beam(tmp_path: Path) -> None:
    ai = AzimuthalIntegrator(dist=0.1, poni1=0.05, poni2=0.05, pixel1=1e-4, pixel2=1e-4, wavelength=1e-10)
    poni = tmp_path / "cal.poni"
    ai.save(str(poni))

    npt = _calculate_npt(poni, (1000, 1000), "1d")

    assert 900 < npt < 1200


def test_calculate_npt_2d_uses_larger_factor(tmp_path: Path) -> None:
    ai = AzimuthalIntegrator(dist=0.1, poni1=0.05, poni2=0.05, pixel1=1e-4, pixel2=1e-4, wavelength=1e-10)
    poni = tmp_path / "cal.poni"
    ai.save(str(poni))

    npt_1d = _calculate_npt(poni, (1000, 1000), "1d")
    npt_2d = _calculate_npt(poni, (1000, 1000), "2d")

    assert npt_2d > npt_1d


def test_calculate_npt_off_centre_beam(tmp_path: Path) -> None:
    ai = AzimuthalIntegrator(dist=0.1, poni1=0.01, poni2=0.01, pixel1=1e-4, pixel2=1e-4, wavelength=1e-10)
    poni = tmp_path / "cal.poni"
    ai.save(str(poni))

    npt_corner = _calculate_npt(poni, (1000, 1000), "1d")

    ai2 = AzimuthalIntegrator(dist=0.1, poni1=0.05, poni2=0.05, pixel1=1e-4, pixel2=1e-4, wavelength=1e-10)
    poni2 = tmp_path / "cal2.poni"
    ai2.save(str(poni2))
    npt_centre = _calculate_npt(poni2, (1000, 1000), "1d")

    assert npt_corner > npt_centre


def test_load_npy_single_frame(tmp_path: Path) -> None:
    frame = _make_frame(3.0).astype(np.float64)
    p = tmp_path / "frame.npy"
    np.save(p, frame)

    stack, names, sources = _load_input(p)

    assert isinstance(stack, FrameStack)
    assert stack.n_frames == 1
    assert stack.frame_shape == FRAME_SHAPE
    assert names == ["frame"]
    assert sources == [p]
    np.testing.assert_allclose(stack[0], frame)


def test_load_npy_3d_stack(tmp_path: Path) -> None:
    data = np.arange(3 * 32 * 32, dtype=np.float64).reshape(3, 32, 32)
    p = tmp_path / "stack.npy"
    np.save(p, data)

    stack, names, sources = _load_input(p)

    assert stack.n_frames == 3
    assert names == ["stack_0000", "stack_0001", "stack_0002"]
    assert sources == [p, p, p]
    np.testing.assert_allclose(stack.data, data)


def test_load_npy_rejects_4d(tmp_path: Path) -> None:
    p = tmp_path / "bad.npy"
    np.save(p, np.ones((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="2D or 3D"):
        _load_input(p)


def test_load_tif_single_frame(tmp_path: Path) -> None:
    frame = _make_frame(7.0)
    p = tmp_path / "frame.tif"
    _write_tif(p, frame)

    stack, names, sources = _load_input(p)

    assert stack.n_frames == 1
    assert stack.frame_shape == FRAME_SHAPE
    assert names == ["frame"]
    assert sources == [p]
    np.testing.assert_allclose(stack[0], frame.astype(np.float64))


def test_load_tif_directory(tmp_path: Path) -> None:
    for i in range(3):
        _write_tif(tmp_path / f"frame_{i:03d}.tif", _make_frame(float(i + 1)))

    stack, names, sources = _load_input(tmp_path)

    assert stack.n_frames == 3
    assert names == ["frame_000", "frame_001", "frame_002"]
    assert all(s.parent == tmp_path for s in sources)
    np.testing.assert_allclose(stack[0].mean(), 1.0)
    np.testing.assert_allclose(stack[1].mean(), 2.0)
    np.testing.assert_allclose(stack[2].mean(), 3.0)


def test_load_directory_mixes_npy_and_tif(tmp_path: Path) -> None:
    np.save(tmp_path / "aaa.npy", _make_frame(1.0).astype(np.float64))
    _write_tif(tmp_path / "bbb.tif", _make_frame(2.0))

    stack, names, sources = _load_input(tmp_path)

    assert stack.n_frames == 2
    assert names == ["aaa", "bbb"]


def test_load_directory_skips_unsupported_files(tmp_path: Path) -> None:
    _write_tif(tmp_path / "frame.tif", _make_frame())
    (tmp_path / "Thumbs.db").write_bytes(b"junk")
    (tmp_path / "calibration.poni").write_text("poni_version: 2\n")

    stack, names, _ = _load_input(tmp_path)

    assert stack.n_frames == 1
    assert names == ["frame"]


def test_load_directory_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No supported image files"):
        _load_input(tmp_path)


def test_load_h5_single_2d_dataset(tmp_path: Path) -> None:
    frame = _make_frame(5.0).astype(np.float64)
    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("entry/data", data=frame)

    stack, names, sources = _load_input(p)

    assert stack.n_frames == 1
    assert names == ["data"]
    assert sources == [p]
    np.testing.assert_allclose(stack[0], frame)


def test_load_h5_3d_dataset(tmp_path: Path) -> None:
    data = np.ones((5, 32, 32), dtype=np.float64) * np.arange(5)[:, None, None]
    p = tmp_path / "stack.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("entry/data", data=data)

    stack, names, sources = _load_input(p)

    assert stack.n_frames == 5
    assert names == [f"stack_{i:04d}" for i in range(5)]
    assert sources == [p] * 5
    np.testing.assert_allclose(stack.data, data)


def test_load_h5_explicit_dataset_path(tmp_path: Path) -> None:
    frame = _make_frame(9.0).astype(np.float64)
    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("measurement/detector/frames", data=frame)
        f.create_dataset("measurement/monitor/counts", data=np.array([1.0, 2.0]))

    stack, names, sources = _load_input(p, h5_dataset="measurement/detector/frames")

    assert stack.n_frames == 1
    assert names == ["data"]
    assert sources == [p]
    np.testing.assert_allclose(stack[0], frame)


def test_load_h5_multiple_datasets_requires_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("detector_a", data=_make_frame())
        f.create_dataset("detector_b", data=_make_frame())

    with pytest.raises(ValueError, match="Multiple image datasets"):
        _load_input(p)


def test_load_h5_no_image_datasets_raises(tmp_path: Path) -> None:
    p = tmp_path / "meta.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("scalars", data=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="No 2D/3D datasets"):
        _load_input(p)


def test_load_h5_in_directory(tmp_path: Path) -> None:
    p = tmp_path / "run.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.ones((3, 32, 32), dtype=np.float64))

    stack, names, sources = _load_input(tmp_path)

    assert stack.n_frames == 3
    assert names == ["run_0000", "run_0001", "run_0002"]
    assert sources == [p, p, p]
