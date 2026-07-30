#!/usr/bin/env python
# ----------------------------------------------------------------------------------
# Project: LumosXD-Server
# File: lumosxd_server/__init__.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file is used to initialize the lumosxd_server package.
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Christofanis Skordas, The University of Chicago
# ----------------------------------------------------------------------------------

from argparse import ArgumentParser
from logging import ERROR, INFO, basicConfig, getLogger
from pathlib import Path
from sys import exit

from lumosxd_server.cli import run_integrate, run_tests

logger = getLogger(__name__)

_VALID_UNITS = ("2th_deg", "2th_rad", "q_nm^-1", "q_A^-1", "d_nm", "d_A")


def main() -> None:
    """Main entry point for `lumosxd-server` console script."""

    # Configure logging
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)
    getLogger("fabio.TiffIO").setLevel(ERROR)

    parser = ArgumentParser("lumosxd-server", description="LumosXD-Server — pyFAI integration backend")
    parser.add_argument("-t", "--test", action="store_true", help="Run the test suite")
    parser.add_argument("input", type=Path, metavar="INPUT", nargs="?", help="Frame file (.npy/.tif/.h5/…) or directory of frames")
    parser.add_argument("output", type=Path, metavar="OUTPUT", nargs="?", default=None, help="Output path — .npz file, or directory when --split (default: input directory when --split)")
    parser.add_argument("--poni", type=Path, metavar="PONI", help="pyFAI calibration .poni file")
    parser.add_argument("--npt", type=int, default=None, metavar="N", help="Number of radial integration points (default: auto from poni and image size)")
    parser.add_argument("--unit", default="2th_deg", choices=_VALID_UNITS, metavar="UNIT", help=f"Radial unit (default: 2th_deg). Choices: {', '.join(_VALID_UNITS)}")
    parser.add_argument("--mask", type=Path, default=None, metavar="MASK", help="Optional 2D boolean mask .npy file (True = masked out)")
    parser.add_argument("--workers", type=int, default=None, metavar="N", help="Worker processes for parallel integration (default: CPU count)")
    parser.add_argument("--opencl", action="store_true", help="Prefer OpenCL GPU integration when available")
    parser.add_argument("--h5-dataset", default=None, metavar="PATH", dest="h5_dataset", help="HDF5 dataset path (auto-detected when omitted)")
    parser.add_argument("--npt-azim", type=int, default=360, metavar="N", dest="npt_azim", help="Number of azimuthal bins for --2d (default: 360)")
    parser.add_argument("--split", action="store_true", help="Write one .npz per input frame into OUTPUT directory instead of a single stacked file")
    parser.add_argument("--verbose", action="store_true", help="Log per-frame progress during integration")

    dim_group = parser.add_mutually_exclusive_group()
    dim_group.add_argument("--1d", dest="mode", action="store_const", const="1d", help="1D azimuthal integration → radial pattern")
    dim_group.add_argument("--2d", dest="mode", action="store_const", const="2d", help="2D cake integration → azimuthal × radial map")

    args = parser.parse_args()

    if args.test:
        exit(run_tests())

    if args.input is None or args.poni is None or args.mode is None:
        parser.print_help()
        exit(0)

    if not args.split and args.output is None:
        parser.error("OUTPUT is required unless --split is used")

    exit(run_integrate(args))

