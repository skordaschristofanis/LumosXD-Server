#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/cake_pool.py
# ----------------------------------------------------------------------------------
# Purpose:
# Across-frame parallel 2D (cake) integration for map stacks using a process pool
# and shared memory to avoid pickling full frame arrays.
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

from lumosxd_server.integration.cake import Cake
from lumosxd_server.integration.engine import DEFAULT_NPT_AZIM, DEFAULT_UNIT, AzimuthalEngine
from lumosxd_server.integration.frame_stack import FrameStack

logger = getLogger(__name__)

_WORKER_OMP_THREADS = "1"
_MP_CONTEXT = get_context("spawn")

_CAKE_WORKER_ENGINE: AzimuthalEngine | None = None
_CAKE_WORKER_STACK: np.ndarray | None = None
_CAKE_WORKER_STACK_SHM: shared_memory.SharedMemory | None = None
_CAKE_WORKER_MASK_SHM: shared_memory.SharedMemory | None = None


@dataclass(frozen=True, slots=True)
class _CakeWorkerConfig:
    poni_path: str
    npt: int
    npt_azim: int
    unit: str
    prefer_opencl: bool
    stack_name: str
    stack_shape: tuple[int, int, int]
    stack_dtype: str
    mask_name: str | None
    mask_shape: tuple[int, int] | None
    mask_dtype: str | None


def _init_cake_worker(config: _CakeWorkerConfig) -> None:
    """Attach shared arrays and warm one cake engine per worker process."""
    global _CAKE_WORKER_ENGINE, _CAKE_WORKER_STACK, _CAKE_WORKER_STACK_SHM, _CAKE_WORKER_MASK_SHM

    environ["OMP_NUM_THREADS"] = _WORKER_OMP_THREADS

    _CAKE_WORKER_STACK_SHM = shared_memory.SharedMemory(name=config.stack_name)
    _CAKE_WORKER_STACK = np.ndarray(
        config.stack_shape,
        dtype=np.dtype(config.stack_dtype),
        buffer=_CAKE_WORKER_STACK_SHM.buf,
    )

    mask: np.ndarray | None = None
    if config.mask_name is not None and config.mask_shape is not None and config.mask_dtype is not None:
        _CAKE_WORKER_MASK_SHM = shared_memory.SharedMemory(name=config.mask_name)
        mask = np.ndarray(
            config.mask_shape,
            dtype=np.dtype(config.mask_dtype),
            buffer=_CAKE_WORKER_MASK_SHM.buf,
        )

    engine = AzimuthalEngine.from_poni(
        config.poni_path,
        config.npt,
        config.unit,
        prefer_opencl=config.prefer_opencl,
        npt_azim=config.npt_azim,
    )
    if mask is not None:
        engine.set_mask(mask)
    engine.warmup_cake((_CAKE_WORKER_STACK.shape[1], _CAKE_WORKER_STACK.shape[2]))
    _CAKE_WORKER_ENGINE = engine


def _integrate_cake_index(index: int) -> Cake:
    """Integrate one frame to a cake by index inside a worker process."""
    if _CAKE_WORKER_ENGINE is None or _CAKE_WORKER_STACK is None:
        raise RuntimeError("Cake worker engine is not initialized")
    return _CAKE_WORKER_ENGINE.integrate_cake(_CAKE_WORKER_STACK[index])


def _integrate_cake_serial(
    poni_path: str | Path,
    stack: FrameStack,
    npt: int,
    npt_azim: int,
    unit: str,
    mask: np.ndarray | None,
    prefer_opencl: bool,
) -> list[Cake]:
    engine = AzimuthalEngine.from_poni(poni_path, npt, unit, prefer_opencl=prefer_opencl, npt_azim=npt_azim)
    if mask is not None:
        engine.set_mask(mask)
    engine.warmup_cake(stack.frame_shape)
    return [engine.integrate_cake(frame) for frame in stack]


def integrate_cake_stack(
    poni_path: str | Path,
    stack: FrameStack,
    npt: int,
    npt_azim: int = DEFAULT_NPT_AZIM,
    unit: str = DEFAULT_UNIT,
    mask: np.ndarray | None = None,
    workers: int | None = None,
    prefer_opencl: bool = False,
) -> list[Cake]:
    """Integrate all frames in stack to cakes, optionally in parallel across processes."""
    if mask is not None and mask.shape != stack.frame_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {stack.frame_shape}")

    n_workers = workers if workers is not None else (cpu_count() or 1)
    n_workers = max(1, min(n_workers, stack.n_frames))

    if n_workers == 1:
        logger.info("Cake-integrating %s frames serially", stack.n_frames)
        return _integrate_cake_serial(poni_path, stack, npt, npt_azim, unit, mask, prefer_opencl)

    logger.info("Cake-integrating %s frames with %s workers", stack.n_frames, n_workers)

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

    config = _CakeWorkerConfig(
        poni_path=str(poni_path),
        npt=npt,
        npt_azim=npt_azim,
        unit=unit,
        prefer_opencl=prefer_opencl,
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
            initializer=_init_cake_worker,
            initargs=(config,),
        ) as pool:
            return list(pool.map(_integrate_cake_index, range(stack.n_frames)))
    finally:
        stack_shm.close()
        stack_shm.unlink()
        if mask_shm is not None:
            mask_shm.close()
            mask_shm.unlink()

