#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/pool.py
# ----------------------------------------------------------------------------------
# Purpose:
# Across-frame parallel 1D integration for map stacks using a process pool and
# shared memory to avoid pickling full frame arrays.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import get_context, shared_memory
from os import cpu_count, environ
from pathlib import Path

import numpy as np

from lumosxd_server.integration.engine import DEFAULT_UNIT, AzimuthalEngine
from lumosxd_server.integration.frame_stack import FrameStack
from lumosxd_server.integration.pattern import Pattern

logger = getLogger(__name__)

# Prefer across-frame processes over nested OpenMP inside each worker.
_WORKER_OMP_THREADS = "1"
# Spawn avoids fork+OpenMP deadlocks after NumPy/pyFAI import in the parent.
_MP_CONTEXT = get_context("spawn")

_WORKER_ENGINE: AzimuthalEngine | None = None
_WORKER_STACK: np.ndarray | None = None
_WORKER_STACK_SHM: shared_memory.SharedMemory | None = None
_WORKER_MASK_SHM: shared_memory.SharedMemory | None = None


@dataclass(frozen=True, slots=True)
class _WorkerConfig:
    poni_path: str
    npt: int
    unit: str
    stack_name: str
    stack_shape: tuple[int, int, int]
    stack_dtype: str
    mask_name: str | None
    mask_shape: tuple[int, int] | None
    mask_dtype: str | None


def _init_worker(config: _WorkerConfig) -> None:
    """Attach shared arrays and warm one engine per worker process."""
    global _WORKER_ENGINE, _WORKER_STACK, _WORKER_STACK_SHM, _WORKER_MASK_SHM

    environ["OMP_NUM_THREADS"] = _WORKER_OMP_THREADS

    _WORKER_STACK_SHM = shared_memory.SharedMemory(name=config.stack_name)
    _WORKER_STACK = np.ndarray(
        config.stack_shape,
        dtype=np.dtype(config.stack_dtype),
        buffer=_WORKER_STACK_SHM.buf,
    )

    mask: np.ndarray | None = None
    if config.mask_name is not None and config.mask_shape is not None and config.mask_dtype is not None:
        _WORKER_MASK_SHM = shared_memory.SharedMemory(name=config.mask_name)
        mask = np.ndarray(
            config.mask_shape,
            dtype=np.dtype(config.mask_dtype),
            buffer=_WORKER_MASK_SHM.buf,
        )

    engine = AzimuthalEngine.from_poni(config.poni_path, config.npt, config.unit)
    if mask is not None:
        engine.set_mask(mask)
    engine.warmup((_WORKER_STACK.shape[1], _WORKER_STACK.shape[2]))
    _WORKER_ENGINE = engine


def _integrate_index(index: int) -> Pattern:
    """Integrate one frame by index inside a worker process."""
    if _WORKER_ENGINE is None or _WORKER_STACK is None:
        raise RuntimeError("Worker engine is not initialized")
    return _WORKER_ENGINE.integrate(_WORKER_STACK[index])


def _integrate_serial(
    poni_path: str | Path,
    stack: FrameStack,
    npt: int,
    unit: str,
    mask: np.ndarray | None,
) -> list[Pattern]:
    engine = AzimuthalEngine.from_poni(poni_path, npt, unit)
    if mask is not None:
        engine.set_mask(mask)
    engine.warmup(stack.frame_shape)
    return [engine.integrate(frame) for frame in stack]


def integrate_stack(
    poni_path: str | Path,
    stack: FrameStack,
    npt: int,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
) -> list[Pattern]:
    """Integrate all frames in stack, optionally in parallel across processes."""
    if mask is not None and mask.shape != stack.frame_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {stack.frame_shape}")

    n_workers = workers if workers is not None else (cpu_count() or 1)
    n_workers = max(1, min(n_workers, stack.n_frames))

    if n_workers == 1:
        logger.info("Integrating %s frames serially", stack.n_frames)
        return _integrate_serial(poni_path, stack, npt, unit, mask)

    logger.info("Integrating %s frames with %s workers", stack.n_frames, n_workers)

    stack_array = np.ascontiguousarray(stack.data)
    stack_shm = shared_memory.SharedMemory(create=True, size=stack_array.nbytes)
    shared_stack = np.ndarray(stack_array.shape, dtype=stack_array.dtype, buffer=stack_shm.buf)
    np.copyto(shared_stack, stack_array)

    mask_shm: shared_memory.SharedMemory | None = None
    mask_name: str | None = None
    mask_shape: tuple[int, int] | None = None
    mask_dtype: str | None = None
    if mask is not None:
        mask_array = np.ascontiguousarray(mask)
        mask_shm = shared_memory.SharedMemory(create=True, size=mask_array.nbytes)
        shared_mask = np.ndarray(mask_array.shape, dtype=mask_array.dtype, buffer=mask_shm.buf)
        np.copyto(shared_mask, mask_array)
        mask_name = mask_shm.name
        mask_shape = (int(mask_array.shape[0]), int(mask_array.shape[1]))
        mask_dtype = str(mask_array.dtype)

    config = _WorkerConfig(
        poni_path=str(poni_path),
        npt=npt,
        unit=unit,
        stack_name=stack_shm.name,
        stack_shape=(int(stack_array.shape[0]), int(stack_array.shape[1]), int(stack_array.shape[2])),
        stack_dtype=str(stack_array.dtype),
        mask_name=mask_name,
        mask_shape=mask_shape,
        mask_dtype=mask_dtype,
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CONTEXT,
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            return list(pool.map(_integrate_index, range(stack.n_frames)))
    finally:
        stack_shm.close()
        stack_shm.unlink()
        if mask_shm is not None:
            mask_shm.close()
            mask_shm.unlink()
