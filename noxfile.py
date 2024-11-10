"""Developer task automation."""

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "check_code_formatting",
    "check_types",
    "test_docs_build",
    "run_tests",
]

PYTHON = "3.12"


@nox.session(python=PYTHON)
def run_tests(session: nox.Session):
    """Run unit tests."""
    session.run_install(
        "uv",
        "sync",
        env={
            "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
            "UV_LINK_MODE": "copy",
        },
    )
    pytest_args = session.posargs if session.posargs else []
    session.run("pytest", *pytest_args)


@nox.session(python=None)
def format_code(session: nox.Session):
    """Lint code and re-format where necessary."""
    session.run("black", "--config=pyproject.toml", ".", external=True)
    session.run("ruff", "check", ".", "--config=pyproject.toml", "--fix", external=True)


@nox.session(python=None)
def check_code_formatting(session: nox.Session):
    """Check code for formatting errors."""
    session.run("black", "--config=pyproject.toml", "--check", ".", external=True)
    session.run("ruff", "check", ".", "--config=pyproject.toml", external=True)


@nox.session(python=None)
def check_types(session: nox.Session):
    """Run static type checking."""
    session.run("mypy", "src", "tests", "noxfile.py", external=True)


@nox.session(python=False)
def test_docs_build(session: nox.Session):
    """Ensure docs can be built."""
    session.run("mkdocs", "build")
    session.run("rm", "-rf", "docs_build", external=True)


@nox.session(python=PYTHON, reuse_venv=True)
def build_and_deploy_docs(session: nox.Session):
    """Deploy docs to GitHub Pages."""
    session.run_install(
        "uv",
        "sync",
        "--group",
        "docs",
        env={
            "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
            "UV_LINK_MODE": "copy",
        },
    )
    session.run("mkdocs", "gh-deploy")
    session.run("rm", "-rf", "docs_build", external=True)
