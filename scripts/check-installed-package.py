"""Run using an installed wheel's Python, with CWD outside the source checkout."""

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

with tempfile.TemporaryDirectory() as directory:
    os.environ['HOME'] = directory
    os.chdir(directory)
    from orchestrator._resource_manifest import PUBLIC_CONFIGS, PUBLIC_FILES, PUBLIC_RESOURCE_FILES
    from orchestrator.dashboard.app import create_app
    from orchestrator.resources import create_workspace, resource_root
    assert 'site-packages' in str(Path(__import__('orchestrator').__file__).resolve())
    # Every manifest entry, not a sample: both copiers skip a missing source in
    # silence, so a renamed asset is invisible until an operator needs it.
    manifest = (*PUBLIC_RESOURCE_FILES, *PUBLIC_FILES, *(f'configs/{name}' for name in PUBLIC_CONFIGS))
    root = resource_root()
    missing = [name for name in manifest if not (root / name).is_file()]
    assert not missing, f'bundled resources missing from the wheel: {missing}'
    client = TestClient(create_app(), base_url='http://localhost:7001')
    response = client.get('/')
    assert response.status_code == 200 and '<html' in response.text.lower()
    assert client.get('/main.js').status_code == 200
    workspace = Path(directory) / 'fleet'
    create_workspace(workspace)
    unwritten = [name for name in manifest if not (workspace / name).is_file()]
    assert not unwritten, f'workspace is missing bundled resources: {unwritten}'
    assert not (workspace / 'configs/aureax-network.yaml').exists()
    assert (workspace / 'client/rpc_firewall.py').is_file()
print('Installed wheel: dashboard, scripts, templates and writable workspace passed.')
