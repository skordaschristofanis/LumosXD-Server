<h1 align="center">
  &nbsp;LumosXD-Server
</h1>

![License](https://img.shields.io/badge/License-MIT-teal.svg) ![Python](https://img.shields.io/badge/Python-3.13-22558a.svg?logo=python&color=22558a)

Backend for LumosXD live azimuthal integration using pyFAI.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Features
- 1D azimuthal integration via `AzimuthalEngine`
- 2D cake integration via `integrate_cake_stack`
- Multi-frame map stacks via `FrameStack` and `integrate_stack`
- Parallel frame processing via a process pool
- Optional OpenCL acceleration (falls back to Cython CSR)
- Input formats: `.npy`, `.tif`/`.tiff`, `.edf`, `.cbf`, `.h5`/`.hdf5`/`.nxs`

## Setup
```bash
uv sync --group dev
```

## Usage

### Command-line

```bash
# 1D azimuthal integration of a directory of TIF frames
uv run lumosxd-server /path/to/frames output.npz --poni calibration.poni --1d

# 2D cake integration with custom azimuthal bins
uv run lumosxd-server /path/to/frames output.npz --poni calibration.poni --2d --npt-azim 360

# Single frame
uv run lumosxd-server frame.tif output.npz --poni calibration.poni --1d

# HDF5 file — dataset auto-detected if there is only one 2D/3D dataset
uv run lumosxd-server data.h5 output.npz --poni calibration.poni --1d

# HDF5 file with explicit dataset path
uv run lumosxd-server data.h5 output.npz --poni calibration.poni --1d --h5-dataset /entry/data/data
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--1d` / `--2d` | *(required)* | Integration mode |
| `--poni` | *(required)* | pyFAI calibration `.poni` file |
| `--npt N` | auto | Radial points — calculated from beam center to farthest image corner (×1.5 for `--1d`, ×2.0 for `--2d`) |
| `--unit` | `2th_deg` | Radial unit (`2th_deg`, `2th_rad`, `q_nm^-1`, `q_A^-1`, `d_nm`, `d_A`) |
| `--mask` | — | 2D boolean `.npy` mask (True = masked out) |
| `--workers N` | CPU count | Parallel worker processes |
| `--opencl` | off | Prefer OpenCL GPU integration |
| `--h5-dataset PATH` | auto | HDF5 internal dataset path |
| `--npt-azim N` | `360` | Azimuthal bins (`--2d` only) |
| `--split` | off | Write one result per input frame instead of a single stacked file |

**Output `.npz` keys:**
- `--1d`: `radial` (1-D), `intensity` (n_frames × npt), `unit`
- `--2d`: `radial` (1-D), `azimuthal` (1-D), `intensity` (n_frames × npt_azim × npt), `unit`

When `--split` is used, OUTPUT is optional and defaults to the input directory. Each frame is saved next to its source file using the source filename stem and mode suffix (e.g. `D3159_d_001_1d.npz`, `D3159_d_001_2d.npz`), so running both modes into the same directory does not overwrite files. HDF5 sources have their results written back into the same `.h5` file following the NeXus convention under `/entry/integration_1d/results` or `/entry/integration_2d/results` as an `NXprocess`/`NXdata` group with proper `signal`, `axes`, and `units` attributes. Stacked inputs (3D `.npy`, multi-frame `.h5`) use zero-padded indices (e.g. `stack_0000.npz`).

### Python API

```python
from lumosxd_server.integration import AzimuthalEngine, FrameStack, integrate_stack

# Single frame
engine = AzimuthalEngine.from_poni("calibration.poni", npt=2000, prefer_opencl=True)
engine.warmup(image.shape)
pattern = engine.integrate(image)

# Batch / map stack
stack = FrameStack(frames_array)   # shape (N, H, W), float64
patterns = integrate_stack(poni_path="calibration.poni", stack=stack, npt=1000)
```

## Tests
```bash
uv run lumosxd-server -t
```

## Contributing
All contributions to the LumosXD-Server project are welcome! Here are some ways you can help:
- Report a bug by opening a [GitHub](https://github.com/skordaschristofanis/lumosxd-server/issues) or a [GitLab](https://gitlab.com/christofanis.skordas/lumosxd-server/-/boards) issue.
- Add new features, fix bugs or improve documentation by submitting a [GitHub](https://github.com/skordaschristofanis/lumosxd-server/pulls) or a [GitLab](https://gitlab.com/christofanis.skordas/lumosxd-server/-/merge_requests) pull request.

Please adhere to the [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow) model when making your contributions! This means creating a new branch for each feature of bug fix, and submitting your changes as a pull request against the main branch. If you're not sure how to contribute, please open an issue and we'll be happy to help you out.

By contributing to the LumosXD-Server project, you agree that your contributions will be licensed under the MIT License.

## License
LumosXD-Server is distributed under the MIT License. You should have received a [copy](LICENSE) of the MIT License along with this program. If not, see https://mit-license.org/ for additional details.
