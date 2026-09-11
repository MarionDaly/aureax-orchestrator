"""Locate bundled operational files and create a writable fleet workspace."""

import shutil
from importlib.resources import files
from pathlib import Path

from orchestrator._resource_manifest import PUBLIC_CONFIGS, PUBLIC_FILES, PUBLIC_RESOURCE_FILES


def resource_root() -> Path:
    """The tree holding configs/, scripts/, k8s/ and docker/.

    Searched in order: the process CWD when it is itself a project root or a
    generated workspace (a ``pyproject.toml`` beside ``scripts/join-node.sh``),
    then the tree this package lives in, then the resources bundled into an
    installed ``orchestrator``. The CWD comes FIRST so that a workspace created
    by ``aureax init --workspace`` is what its own commands read and write once
    the operator is inside it — with the consequence that running from another
    checkout targets that checkout's files, not this package's.
    """
    for candidate in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (candidate / "pyproject.toml").is_file() and (candidate / "scripts/join-node.sh").is_file():
            return candidate
    return Path(str(files("orchestrator").joinpath("_resources")))


def create_workspace(destination: Path) -> None:
    """Copy public resources and installed code into a new, private workspace."""
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        # The copy below is not transactional, so a failure part-way (a full
        # disk, a permission error on one file) leaves this guard tripping on
        # the partial tree. Naming the destination and the remedy is what makes
        # the retry obvious; the CLI surfaces this message verbatim.
        raise ValueError(
            f"Workspace must be new or empty; existing files are never overwritten. "
            f"Remove {destination} and re-run, or choose another path."
        )
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = resource_root()
    for name in (*PUBLIC_RESOURCE_FILES, *PUBLIC_FILES, *(f"configs/{name}" for name in PUBLIC_CONFIGS)):
        if (source / name).is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
    for package in ("orchestrator", "client"):
        shutil.copytree(Path(str(files(package))), destination / package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_resources"))
