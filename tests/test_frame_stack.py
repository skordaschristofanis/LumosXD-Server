#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: tests/test_frame_stack.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for the FrameStack multi-frame container.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

import numpy as np
import pytest

from lumosxd_server.integration import FrameStack


def test_frame_stack_length_and_shape() -> None:
    data = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    stack = FrameStack(data)

    assert len(stack) == 3
    assert stack.n_frames == 3
    assert stack.frame_shape == (4, 5)
    assert stack.data.shape == (3, 4, 5)


def test_frame_stack_indexing_and_iteration() -> None:
    data = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
    stack = FrameStack(data)

    np.testing.assert_array_equal(stack[0], data[0])
    np.testing.assert_array_equal(stack[1], data[1])

    frames = list(stack)
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], data[0])
    np.testing.assert_array_equal(frames[1], data[1])


def test_frame_stack_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="n_frames, height, width"):
        FrameStack(np.ones((10, 10), dtype=np.float64))


def test_frame_stack_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        FrameStack(np.ones((0, 10, 10), dtype=np.float64))
