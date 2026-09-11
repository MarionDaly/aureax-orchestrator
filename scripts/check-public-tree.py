#!/usr/bin/env python3
"""Check publication refs for private paths and common secret formats.

Output is locations only. This targeted guard complements review and an advisory
scan; it cannot recognize every possible credential format.
"""

import fnmatch
import re
import subprocess
import sys

PRIVATE = (
    '.env', '.env.local', '.env.production', '.aureax/*', '.claude/*', 'CLAUDE.md',
    'CLAUDE.local.md', 'TODO.md', 'OPEN_SOURCE_AUDIT.md', '.mcp.json', 'configs/aureax-network.yaml',
    'configs/aureax-network.yaml.*', 'configs/*.local.yaml', 'configs/aureax-gpu-edge.yaml',
    '*.db', '*.db-wal', '*.db-shm', '*.sqlite*', '*.gguf', '*.pem', '*.key', '*.crt', '*.p12',
    '.env.*', 'configs/aureax-network.*.yaml', '*.egg-info/*', '.DS_Store', '*/.DS_Store',
    'build/*', 'dist/*', '.venv/*', '*/__pycache__/*', '*.pyc',
)
# Tracked on purpose, mirroring the ``!configs/aureax-network.example.yaml``
# negation in .gitignore: the example matches ``configs/aureax-network.*.yaml``
# above, and fnmatch has no negation of its own.
PUBLIC = ('configs/aureax-network.example.yaml',)
TOKENS = re.compile(rb'(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/]{20})')
PASSWORD = re.compile(rb'^\s*["\']?password["\']?\s*:\s*([^\r\n#]+)', re.M)


def main():
    objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all']).decode().splitlines()
    violations = set()
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for entry in objects:
            oid, _, path = entry.partition(' ')
            if not path:
                continue
            if path not in PUBLIC and any(fnmatch.fnmatch(path, pattern) for pattern in PRIVATE):
                violations.add(f'{oid[:12]}:{path}: private path')
            proc.stdin.write((oid + '\n').encode())
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError('Could not read Git object')
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            if header[1] != b'blob':
                continue
            if TOKENS.search(data):
                violations.add(f'{oid[:12]}:{path}: credential pattern')
            if path.startswith('configs/') and path.endswith(('.yaml', '.yml')):
                for match in PASSWORD.finditer(data):
                    value = match[1].strip().strip(b'\'"')
                    if value and value not in (b'null', b'~') and not value.startswith(b'${'):
                        violations.add(f'{oid[:12]}:{path}: literal password')
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    if violations:
        print('\n'.join(sorted(violations)))
        return 1
    print('Publication refs passed the private-path and targeted credential checks.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
