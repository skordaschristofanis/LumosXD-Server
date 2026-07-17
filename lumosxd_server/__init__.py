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
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from sys import exit

logger = getLogger(__name__)


def _run_tests() -> int:
    """Run the project test suite with pytest. Returns the pytest exit code."""
    try:
        import pytest
    except ImportError:
        logger.error("pytest is not installed. Install the dev group: uv sync --group dev")
        return 1

    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    return pytest.main([str(tests_dir), "-v"])


def main() -> None:
    """Main entry point for `lumosxd-server` console script."""

    # Configure logging
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Set up CLI parser
    parser = ArgumentParser("LumosXD-Server CLI")

    # List of CLI arguments
    parser.add_argument("-t", "--test", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.test:
        exit(_run_tests())

    parser.print_help()
