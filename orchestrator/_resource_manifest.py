"""Explicit public assets shared by package builds and workspace creation.

Private to the package (leading underscore) and deliberately importable two
ways: ``orchestrator.resources`` imports it normally at runtime, and
``build_support.BuildPy`` loads it by path with ``runpy`` at build time, when
``orchestrator`` is not yet importable. Keeping it under the package rather
than at the repo root means the wheel installs no generic top-level name that
another distribution could shadow.

Every entry is checked against the tree by ``tests/test_resource_manifest.py``:
both copiers skip a missing source silently, because both Dockerfiles COPY a
subset of the tree and then install, so a renamed file would otherwise vanish
from the wheel and from a created workspace without a word.
"""

PUBLIC_CONFIGS = (
    "aureax-network.example.yaml", "aureax-edge.yaml", "aureax-planner.yaml",
    "aureax-shard-head.yaml", "aureax-shard-head-35b.yaml",
    "aureax-shard-worker.yaml", "aureax-shard-worker-35b.yaml",
)
PUBLIC_FILES = ("MANIFEST.in", "pyproject.toml", "uv.lock", "build_support.py", "README.md", "ARCHITECTURE.md",
                "LICENSE", "CONTRIBUTING.md", "SECURITY.md", ".dockerignore", ".gitignore")

# The scripts an operator runs from a workspace. The two publication guards
# (check-public-tree.py, check-installed-package.py) are deliberately absent:
# the first needs a Git repository and the second an installed wheel, so
# neither can work in a workspace. CI runs both from the checkout instead.
PUBLIC_RESOURCE_FILES = (
    'docker/Dockerfile.client',
    'docker/Dockerfile.client.cuda',
    'docker/scripts/start.sh',
    'k8s/daemonset-edge.yaml',
    'k8s/daemonset-shard.yaml',
    'k8s/namespace.yaml',
    'k8s/service-llama.yaml',
    'k8s/service-mcp.yaml',
    'scripts/bench-decode.sh',
    'scripts/build-on-node.sh',
    'scripts/check-rpc-firewall.py',
    'scripts/download-model.sh',
    'scripts/join-node.sh',
    'scripts/lib/common.sh',
    'scripts/parse-config.py',
    'scripts/reset-all.sh',
    'scripts/setup-all.sh',
    'scripts/setup-k3d.sh',
    'scripts/stop-all.sh',
    'scripts/test-all.sh',
    'scripts/test-mcp-tools.py',
    'scripts/update-all.sh',
)
