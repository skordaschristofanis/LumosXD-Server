#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/integration/frame_stack.py
# ----------------------------------------------------------------------------------
# Purpose:
# In-memory multi-frame container for map-style integration inputs.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from collections.abc import Iterator

import numpy as np


class FrameStack:
    """Stack of detector frames with shape (n_frames, height, width)."""

    def __init__(self, data: np.ndarray) -> None:
        array = np.asarray(data)
        if array.ndim != 3:
            raise ValueError(f"Expected shape (n_frames, height, width), got {array.shape}")
        if array.shape[0] < 1:
            raise ValueError("FrameStack must contain at least one frame")
        self._data = array

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def n_frames(self) -> int:
        return int(self._data.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self._data.shape[1]), int(self._data.shape[2])

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, index: int) -> np.ndarray:
        return self._data[index]

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(self.n_frames):
            yield self._data[index]
