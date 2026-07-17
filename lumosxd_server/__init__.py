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


def main() -> None:
    """Main entry point for `lumosxd-server` console script."""
    parser = ArgumentParser("LumosXD-Server CLI")

    # List of CLI arguments
    args = parser.parse_args()

    parser.print_help()

