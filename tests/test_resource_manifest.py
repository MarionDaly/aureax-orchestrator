"""The manifest of shipped assets has to still describe the tree.

``orchestrator/_resource_manifest.py`` is the single list of everything that is
not Python but still ships: the Dockerfiles, the K8s manifests, every
operational script, the config templates, and the handful of root files a
workspace needs to be buildable. Two copiers read it — ``build_support.BuildPy``
into ``orchestrator/_resources`` in the wheel, and
``orchestrator.resources.create_workspace`` out of that directory into a new
fleet workspace — and **both skip a source that does not exist, in silence**.

That skip is load-bearing and must stay: both Dockerfiles COPY a subset of the
tree and then run ``uv pip install --system --no-deps .``, so ``BuildPy`` runs
routinely with most entries absent and has to tolerate it. The cost is that a
renamed or moved asset drops out of the wheel with no error at build time and no
error at install time — it surfaces only when an operator on an installed copy
reaches for it, e.g. ``aureax start``'s first-time cluster setup failing to find
``scripts/setup-k3d.sh``.

So the guard belongs here, in the checkout, where every entry can be resolved
against the real tree. ``scripts/check-installed-package.py`` asks the same
question from the other end — every entry present under ``resource_root()`` and
in a created workspace — so a file that exists in the checkout but never reaches
the wheel is caught too.
"""

from __future__ import annotations

import runpy

import pytest

from orchestrator._resource_manifest import PUBLIC_CONFIGS, PUBLIC_FILES, PUBLIC_RESOURCE_FILES

from .conftest import PROJECT_ROOT

MANIFEST_ENTRIES = (
    *PUBLIC_RESOURCE_FILES,
    *PUBLIC_FILES,
    *(f"configs/{name}" for name in PUBLIC_CONFIGS),
)


@pytest.mark.parametrize("name", MANIFEST_ENTRIES)
def test_every_manifest_entry_exists_in_the_tree(name: str) -> None:
    """One case per entry, so a failure names the file that moved."""
    path = PROJECT_ROOT / name
    assert path.is_file(), (
        f"{name} is listed in orchestrator/_resource_manifest.py but is not a file in the "
        "repository. Both copiers skip a missing source silently, so this would ship a wheel "
        "and create a workspace without it. Update the manifest, or restore the file."
    )


def test_the_manifest_is_not_quietly_empty() -> None:
    """Guards the parametrize above against an emptied tuple passing as zero cases."""
    assert len(PUBLIC_RESOURCE_FILES) > 15
    assert len(PUBLIC_CONFIGS) > 5
    assert len(PUBLIC_FILES) > 5
    assert len(set(MANIFEST_ENTRIES)) == len(MANIFEST_ENTRIES), "duplicate manifest entry"


def test_the_manifest_ships_no_asset_that_cannot_work_in_a_workspace() -> None:
    """A workspace has no ``tests/`` and no ``.git``, so two guards are excluded.

    ``create_workspace`` copies the manifest plus the ``orchestrator`` and
    ``client`` packages, and nothing else. ``check-public-tree.py`` walks Git
    objects and ``check-installed-package.py`` needs an installed wheel, so
    shipping either into a workspace offers the operator a script that can only
    fail there. CI runs both from the checkout.
    """
    for name in ("scripts/check-public-tree.py", "scripts/check-installed-package.py"):
        assert name not in PUBLIC_RESOURCE_FILES, (
            f"{name} needs a Git checkout or an installed wheel; a workspace has neither"
        )


def test_the_build_time_loader_reaches_the_same_manifest() -> None:
    """``build_support`` loads the manifest by path; this module imports it.

    Two readers exist because ``orchestrator`` is not importable while setuptools
    is building it, so the build-time half cannot use an import statement. It is
    exercised here rather than by importing ``build_support`` itself, which pulls
    in setuptools — a build dependency that is absent from the test environment.
    A path that stopped resolving would leave a wheel with an empty
    ``_resources`` directory and no complaint from the build.
    """
    source = (PROJECT_ROOT / "build_support.py").read_text()
    assert 'with_name("orchestrator") / "_resource_manifest.py"' in source, (
        "build_support.py no longer loads orchestrator/_resource_manifest.py by that path"
    )
    loaded = runpy.run_path(str(PROJECT_ROOT / "orchestrator" / "_resource_manifest.py"))
    assert loaded["PUBLIC_RESOURCE_FILES"] == PUBLIC_RESOURCE_FILES
    assert loaded["PUBLIC_FILES"] == PUBLIC_FILES
    assert loaded["PUBLIC_CONFIGS"] == PUBLIC_CONFIGS


def test_the_wheel_installs_no_generic_top_level_module() -> None:
    """The manifest lives under the package, not at the repo root.

    A top-level ``resource_manifest`` in site-packages is shadowable by any other
    distribution using that name, which would silently change which files a
    workspace is built from. ``build_support.py`` stays at the root — setuptools
    imports the ``build_py`` cmdclass from the project directory — but is a
    resource, never an installed module, so ``pyproject.toml`` declares no
    ``py-modules``.
    """
    assert not (PROJECT_ROOT / "resource_manifest.py").exists()
    assert "py-modules" not in (PROJECT_ROOT / "pyproject.toml").read_text()
    assert "build_support.py" in PUBLIC_FILES
    assert "resource_manifest.py" not in PUBLIC_FILES
