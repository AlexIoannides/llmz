"""Developer task automation."""

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "check_format_and_linting",
    "check_types",
    "test_docs_build",
    "run_unit_tests",
    "run_func_tests",
    "compute_test_coverage",
]


@nox.session(python=None)
def run_unit_tests(session: nox.Session):
    """Run unit tests."""
    pytest_args = session.posargs if session.posargs else []
    session.run(
        "pytest",
        "--cov=llmz",
        "--cov-report=",
        "tests/unit",
        *pytest_args,
        external=True,
        env={"COVERAGE_FILE": ".coverage.unit"},
    )


@nox.session()
def run_func_tests(session: nox.Session):
    """Run functional tests."""
    pytest_args = session.posargs if session.posargs else []
    session.run(
        "pytest",
        "--cov=llmz",
        "--cov-report=",
        "tests/functional",
        *pytest_args,
        external=True,
        env={"COVERAGE_FILE": ".coverage.func"},
    )


@nox.session(python=None)
def compute_test_coverage(session: nox.Session):
    """Compute test coverage after unit and functional tests."""
    session.run(
        "coverage", "combine", ".coverage.unit", ".coverage.func", external=True
    )
    session.run("coverage", "report", "--fail-under=95", external=True)


@nox.session(python=None)
def format_and_lint(session: nox.Session):
    """Lint code and re-format where necessary."""
    session.run("ruff", "format", "--config=pyproject.toml", external=True)
    session.run("ruff", "check", "--fix", "--config=pyproject.toml", external=True)


@nox.session(python=None)
def check_format_and_linting(session: nox.Session):
    """Check code for formatting errors."""
    session.run("ruff", "check", "--config=pyproject.toml", external=True)


@nox.session(python=None)
def check_types(session: nox.Session):
    """Run static type checking."""
    session.run("mypy", "src", "tests", "noxfile.py", external=True)


@nox.session(python=False)
def test_docs_build(session: nox.Session):
    """Ensure docs can be built."""
    session.run("mkdocs", "build")
    session.run("rm", "-rf", "docs_build", external=True)


@nox.session(reuse_venv=True)
def build_and_deploy_docs(session: nox.Session):
    """Deploy docs to GitHub Pages."""
    session.run_install(
        "uv",
        "sync",
        env={
            "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
            "UV_LINK_MODE": "copy",
        },
    )
    session.run("mkdocs", "gh-deploy")
    session.run("rm", "-rf", "docs_build", external=True)
