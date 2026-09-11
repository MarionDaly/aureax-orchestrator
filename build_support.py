"""Bundle the public operational assets; never glob operator-owned configs."""

import runpy
import shutil
from pathlib import Path

from setuptools.command.build_py import build_py

# Loaded by path, not imported: setuptools runs this module from the project
# root before anything is built, so `orchestrator` is not yet on the path.
_manifest = runpy.run_path(str(Path(__file__).with_name("orchestrator") / "_resource_manifest.py"))
PUBLIC_CONFIGS = _manifest["PUBLIC_CONFIGS"]
PUBLIC_FILES = _manifest["PUBLIC_FILES"]
PUBLIC_RESOURCE_FILES = _manifest["PUBLIC_RESOURCE_FILES"]


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        destination = Path(self.build_lib) / "orchestrator" / "_resources"
        if destination.exists():
            shutil.rmtree(destination)
        for name in (*PUBLIC_RESOURCE_FILES, *PUBLIC_FILES, *(f"configs/{name}" for name in PUBLIC_CONFIGS)):
            source = root / name
            if source.is_file():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
