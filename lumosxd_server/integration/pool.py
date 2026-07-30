#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/pool.py
# ----------------------------------------------------------------------------------
# Purpose:
# Across-frame parallel 1D and 2D integration for map stacks using a process pool
# and shared memory to avoid pickling full frame arrays.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import get_context, shared_memory
from os import cpu_count, environ
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lumosxd_server.integration.engine import DEFAULT_NPT_AZIM, DEFAULT_UNIT, AzimuthalEngine
from lumosxd_server.integration.results import Cake, Pattern

logger = getLogger(__name__)

# Prefer across-frame processes over nested OpenMP inside each worker.
_WORKER_OMP_THREADS = "1"
# Spawn avoids fork+OpenMP deadlocks after NumPy/pyFAI import in the parent.
_MP_CONTEXT = get_context("spawn")

_WORKER_ENGINE: AzimuthalEngine | None = None
_WORKER_STACK: np.ndarray | None = None
_WORKER_STACK_SHM: shared_memory.SharedMemory | None = None
_WORKER_MASK_SHM: shared_memory.SharedMemory | None = None
_WORKER_DIM: str | None = None

_H5_WORKER_ENGINE: AzimuthalEngine | None = None
_H5_WORKER_FILE: Any = None
_H5_WORKER_DATASET: str | None = None
_H5_WORKER_MASK_SHM: shared_memory.SharedMemory | None = None
_H5_WORKER_DIM: str | None = None


@dataclass(frozen=True, slots=True)
class _WorkerConfig:
    poni_path: str
    npt: int
    npt_azim: int
    unit: str
    prefer_opencl: bool
    dim: str
    stack_name: str
    stack_shape: tuple[int, int, int]
    stack_dtype: str
    mask_name: str | None
    mask_shape: tuple[int, int] | None
    mask_dtype: str | None


@dataclass(frozen=True, slots=True)
class _H5WorkerConfig:
    poni_path: str
    h5_path: str
    dataset: str
    frame_shape: tuple[int, int]
    npt: int
    npt_azim: int
    unit: str
    prefer_opencl: bool
    dim: str
    mask_name: str | None
    mask_shape: tuple[int, int] | None
    mask_dtype: str | None


def _share_ndarray(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, str, tuple, str]:
    contiguous = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
    np.copyto(np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf), contiguous)
    return shm, shm.name, contiguous.shape, str(contiguous.dtype)


def _init_worker(config: _WorkerConfig) -> None:
    global _WORKER_ENGINE, _WORKER_STACK, _WORKER_STACK_SHM, _WORKER_MASK_SHM, _WORKER_DIM
    environ["OMP_NUM_THREADS"] = _WORKER_OMP_THREADS
    _WORKER_DIM = config.dim
    _WORKER_STACK_SHM = shared_memory.SharedMemory(name=config.stack_name)
    _WORKER_STACK = np.ndarray(config.stack_shape, dtype=np.dtype(config.stack_dtype), buffer=_WORKER_STACK_SHM.buf)
    mask: np.ndarray | None = None
    if config.mask_name is not None and config.mask_shape is not None and config.mask_dtype is not None:
        _WORKER_MASK_SHM = shared_memory.SharedMemory(name=config.mask_name)
        mask = np.ndarray(config.mask_shape, dtype=np.dtype(config.mask_dtype), buffer=_WORKER_MASK_SHM.buf)
    engine = AzimuthalEngine.from_poni(
        config.poni_path, config.npt, config.unit,
        prefer_opencl=config.prefer_opencl, npt_azim=config.npt_azim,
    )
    if mask is not None:
        engine.set_mask(mask)
    frame_shape = (int(config.stack_shape[1]), int(config.stack_shape[2]))
    engine.warmup(frame_shape, config.dim)
    _WORKER_ENGINE = engine


def _integrate_index(index: int) -> Pattern | Cake:
    if _WORKER_ENGINE is None or _WORKER_STACK is None:
        raise RuntimeError("Worker not initialized")
    if _WORKER_DIM == "1d":
        return _WORKER_ENGINE.integrate(_WORKER_STACK[index])
    return _WORKER_ENGINE.integrate_cake(_WORKER_STACK[index])


def _init_h5_worker(config: _H5WorkerConfig) -> None:
    global _H5_WORKER_ENGINE, _H5_WORKER_FILE, _H5_WORKER_DATASET, _H5_WORKER_MASK_SHM, _H5_WORKER_DIM
    environ["OMP_NUM_THREADS"] = _WORKER_OMP_THREADS
    _H5_WORKER_DIM = config.dim
    _H5_WORKER_FILE = h5py.File(config.h5_path, "r")
    _H5_WORKER_DATASET = config.dataset
    mask: np.ndarray | None = None
    if config.mask_name is not None and config.mask_shape is not None and config.mask_dtype is not None:
        _H5_WORKER_MASK_SHM = shared_memory.SharedMemory(name=config.mask_name)
        mask = np.ndarray(config.mask_shape, dtype=np.dtype(config.mask_dtype), buffer=_H5_WORKER_MASK_SHM.buf)
    engine = AzimuthalEngine.from_poni(
        config.poni_path, config.npt, config.unit,
        prefer_opencl=config.prefer_opencl, npt_azim=config.npt_azim,
    )
    if mask is not None:
        engine.set_mask(mask)
    if config.dim == "1d":
        engine.warmup(config.frame_shape)
    else:
        engine.warmup_cake(config.frame_shape)
    _H5_WORKER_ENGINE = engine


def _integrate_h5_index(index: int) -> Pattern | Cake:
    if _H5_WORKER_ENGINE is None or _H5_WORKER_FILE is None:
        raise RuntimeError("H5 worker not initialized")
    frame = np.asarray(_H5_WORKER_FILE[_H5_WORKER_DATASET][index], dtype=np.float64)
    if _H5_WORKER_DIM == "1d":
        return _H5_WORKER_ENGINE.integrate(frame)
    return _H5_WORKER_ENGINE.integrate_cake(frame)


def _serial_stack(
    poni_path: str | Path,
    stack: np.ndarray,
    npt: int,
    npt_azim: int,
    unit: str,
    dim: str,
    mask: np.ndarray | None,
    prefer_opencl: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> list[Pattern | Cake]:
    engine = AzimuthalEngine.from_poni(poni_path, npt, unit, prefer_opencl=prefer_opencl, npt_azim=npt_azim)
    if mask is not None:
        engine.set_mask(mask)
    engine.warmup(stack.shape[1:], dim)
    integrate_fn = engine.integrate if dim == "1d" else engine.integrate_cake
    results: list[Pattern | Cake] = []
    for i, frame in enumerate(stack):
        results.append(integrate_fn(frame))
        if progress_callback:
            progress_callback(i + 1, stack.shape[0])
    return results


def _parallel_stack(
    poni_path: str | Path,
    stack: np.ndarray,
    npt: int,
    npt_azim: int,
    unit: str,
    dim: str,
    mask: np.ndarray | None,
    n_workers: int,
    prefer_opencl: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> list[Pattern | Cake]:
    stack_shm, stack_name, stack_shape, stack_dtype = _share_ndarray(stack)
    mask_shm: shared_memory.SharedMemory | None = None
    mask_name: str | None = None
    mask_shape_t: tuple[int, int] | None = None
    mask_dtype_s: str | None = None
    if mask is not None:
        mask_shm, mask_name, m_shape, mask_dtype_s = _share_ndarray(mask)
        mask_shape_t = (int(m_shape[0]), int(m_shape[1]))
    config = _WorkerConfig(
        poni_path=str(poni_path), npt=npt, npt_azim=npt_azim, unit=unit,
        prefer_opencl=prefer_opencl, dim=dim,
        stack_name=stack_name,
        stack_shape=(int(stack_shape[0]), int(stack_shape[1]), int(stack_shape[2])),
        stack_dtype=stack_dtype,
        mask_name=mask_name, mask_shape=mask_shape_t, mask_dtype=mask_dtype_s,
    )
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CONTEXT,
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            futures = {pool.submit(_integrate_index, i): i for i in range(stack.shape[0])}
            results: list[Pattern | Cake | None] = [None] * stack.shape[0]
            done = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if progress_callback:
                    progress_callback(done, stack.shape[0])
        return results  # type: ignore[return-value]
    finally:
        stack_shm.close()
        stack_shm.unlink()
        if mask_shm is not None:
            mask_shm.close()
            mask_shm.unlink()


def _run_h5(
    dim: str,
    h5_path: str | Path,
    dataset: str,
    n_frames: int,
    frame_shape: tuple[int, int],
    poni_path: str | Path,
    npt: int,
    npt_azim: int,
    unit: str,
    mask: np.ndarray | None,
    n_workers: int,
    prefer_opencl: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> list[Pattern | Cake]:
    if n_workers == 1:
        engine = AzimuthalEngine.from_poni(poni_path, npt, unit, prefer_opencl=prefer_opencl, npt_azim=npt_azim)
        if mask is not None:
            engine.set_mask(mask)
        engine.warmup(frame_shape, dim)
        integrate_fn = engine.integrate if dim == "1d" else engine.integrate_cake
        results: list[Pattern | Cake] = []
        with h5py.File(h5_path, "r") as f:
            ds = f[dataset]
            for i in range(n_frames):
                frame = np.asarray(ds[i], dtype=np.float64)
                results.append(integrate_fn(frame))
                if progress_callback:
                    progress_callback(i + 1, n_frames)
        return results

    mask_shm: shared_memory.SharedMemory | None = None
    mask_name: str | None = None
    mask_shape_t: tuple[int, int] | None = None
    mask_dtype_s: str | None = None
    if mask is not None:
        mask_shm, mask_name, m_shape, mask_dtype_s = _share_ndarray(mask)
        mask_shape_t = (int(m_shape[0]), int(m_shape[1]))
    config = _H5WorkerConfig(
        poni_path=str(poni_path), h5_path=str(h5_path), dataset=dataset,
        frame_shape=frame_shape, npt=npt, npt_azim=npt_azim, unit=unit,
        prefer_opencl=prefer_opencl, dim=dim,
        mask_name=mask_name, mask_shape=mask_shape_t, mask_dtype=mask_dtype_s,
    )
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CONTEXT,
            initializer=_init_h5_worker,
            initargs=(config,),
        ) as pool:
            futures = {pool.submit(_integrate_h5_index, i): i for i in range(n_frames)}
            h5_results: list[Pattern | Cake | None] = [None] * n_frames
            done = 0
            for future in as_completed(futures):
                h5_results[futures[future]] = future.result()
                done += 1
                if progress_callback:
                    progress_callback(done, n_frames)
        return h5_results  # type: ignore[return-value]
    finally:
        if mask_shm is not None:
            mask_shm.close()
            mask_shm.unlink()


def integrate_stack(
    poni_path: str | Path,
    stack: np.ndarray,
    npt: int,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
    prefer_opencl: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Pattern]:
    """Integrate all frames in stack to 1D patterns, optionally in parallel."""
    if mask is not None and mask.shape != stack.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {stack.shape[1:]}")
    n_workers = max(1, min(workers if workers is not None else (cpu_count() or 1), stack.shape[0]))
    if n_workers == 1:
        logger.info("Integrating %d frames serially (1D)", stack.shape[0])
    else:
        logger.info("Integrating %d frames with %d workers (1D)", stack.shape[0], n_workers)
    return _serial_stack(poni_path, stack, npt, DEFAULT_NPT_AZIM, unit, "1d", mask, prefer_opencl, progress_callback) if n_workers == 1 else _parallel_stack(poni_path, stack, npt, DEFAULT_NPT_AZIM, unit, "1d", mask, n_workers, prefer_opencl, progress_callback)  # type: ignore[return-value]


def integrate_cake_stack(
    poni_path: str | Path,
    stack: np.ndarray,
    npt: int,
    npt_azim: int = DEFAULT_NPT_AZIM,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
    prefer_opencl: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Cake]:
    """Integrate all frames in stack to 2D cakes, optionally in parallel."""
    if mask is not None and mask.shape != stack.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {stack.shape[1:]}")
    n_workers = max(1, min(workers if workers is not None else (cpu_count() or 1), stack.shape[0]))
    if n_workers == 1:
        logger.info("Cake-integrating %d frames serially (2D)", stack.shape[0])
    else:
        logger.info("Cake-integrating %d frames with %d workers (2D)", stack.shape[0], n_workers)
    return _serial_stack(poni_path, stack, npt, npt_azim, unit, "2d", mask, prefer_opencl, progress_callback) if n_workers == 1 else _parallel_stack(poni_path, stack, npt, npt_azim, unit, "2d", mask, n_workers, prefer_opencl, progress_callback)  # type: ignore[return-value]


def integrate_h5_stack(
    h5_path: str | Path,
    dataset: str,
    n_frames: int,
    frame_shape: tuple[int, int],
    poni_path: str | Path,
    npt: int,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
    prefer_opencl: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Pattern]:
    """Integrate frames from an HDF5 dataset to 1D patterns without loading into memory."""
    if mask is not None and mask.shape != frame_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame_shape}")
    n_workers = max(1, min(workers if workers is not None else (cpu_count() or 1), n_frames))
    label = "serially" if n_workers == 1 else f"with {n_workers} workers"
    logger.info("H5-integrating %d frames %s (1D)", n_frames, label)
    return _run_h5("1d", h5_path, dataset, n_frames, frame_shape, poni_path, npt, DEFAULT_NPT_AZIM, unit, mask, n_workers, prefer_opencl, progress_callback)  # type: ignore[return-value]


def integrate_h5_cake_stack(
    h5_path: str | Path,
    dataset: str,
    n_frames: int,
    frame_shape: tuple[int, int],
    poni_path: str | Path,
    npt: int,
    npt_azim: int = DEFAULT_NPT_AZIM,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
    prefer_opencl: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Cake]:
    """Cake-integrate frames from an HDF5 dataset without loading into memory."""
    if mask is not None and mask.shape != frame_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame_shape}")
    n_workers = max(1, min(workers if workers is not None else (cpu_count() or 1), n_frames))
    label = "serially" if n_workers == 1 else f"with {n_workers} workers"
    logger.info("H5-cake-integrating %d frames %s (2D)", n_frames, label)
    return _run_h5("2d", h5_path, dataset, n_frames, frame_shape, poni_path, npt, npt_azim, unit, mask, n_workers, prefer_opencl, progress_callback)  # type: ignore[return-value]
