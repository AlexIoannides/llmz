# llmz

This is the repository for the llmz Python package.

## Developer Setup

This project uses [uv](https://docs.astral.sh/uv/) for managing dependencies, virtual environments and for building the package into a distributable artefact. Please [install](https://docs.astral.sh/uv/getting-started/installation/) the tool and then run,

```text
uv sync --group docs
```

To install the Python package and all the developer tools used in this project.

### Developer Task Execution with Nox

We use [Nox](https://nox.thea.codes/en/stable/) for scripting developer tasks, such as formatting code, checking types and running tests. These tasks are defined in `noxfile.py`, a list of which can be returned on the command line,

```text
uv run nox --list

Sessions defined in /Users/.../noxfile.py:

* run_tests-3.12 -> Run unit tests.
- format_code -> Lint code and re-format where necessary.
* check_code_formatting -> Check code for formatting errors.
* check_types -> Run static type checking.
- build_and_deploy-3.12 -> Build wheel and deploy to PyPI.

sessions marked with * are selected, sessions marked with - are skipped.
```

Single tasks can be executed easily - e.g.,

```text
uv run nox -s run_tests

nox > Running session run_tests-3.12
nox > Creating virtual environment (virtualenv) using python3.12 in .nox/run_tests-3-10
nox > python -m pip install '.[dev]'
nox > pytest 
...
nox > Session run_tests-3.12 was successful.
```

## CI/CD

This repo comes configured to run two [GitHub Actions](https://docs.github.com/en/actions) workflows:

- **Test Python Package (CI)**, defined in `.github/workflows/python-package-ci.yml`

The CI workflow has been configured to run whenever a pull request to the `main` branch is created.
