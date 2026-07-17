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
- Multi-frame map stacks via `FrameStack` and `integrate_stack`
- Optional OpenCL acceleration (falls back to Cython CSR)

## Setup
```bash
uv sync --group dev
```

## Usage
```python
from lumosxd_server.integration import AzimuthalEngine

engine = AzimuthalEngine.from_poni("calibration.poni", npt=2000, prefer_opencl=True)
engine.warmup(image.shape)
pattern = engine.integrate(image)
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
