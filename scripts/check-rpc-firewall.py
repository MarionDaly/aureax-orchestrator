"""Isolated Linux integration check. Requires a disposable Docker network namespace.

Run with NET_ADMIN and iptables installed; never with --network host. No fleet access.
"""

import socket
import subprocess
from pathlib import Path

from client.rpc_firewall import install

if not Path('/.dockerenv').exists():
    raise SystemExit('Run only in a disposable Docker container with its own network namespace')

port = 50052
chain = f'AUREAX_RPC_{port}'
config = {'node': {'role': 'shard_worker'}, 'model': {'rpc_server': {
    'port': port, 'allowed_clients': ['192.0.2.1']}}}
local = socket.gethostbyname(socket.gethostname())
try:
    install(config, local)
    install(config, local)
    for tool in ('iptables', 'ip6tables'):
        rules = subprocess.check_output([tool, '-S', 'INPUT'], text=True).splitlines()
        jumps = [rule for rule in rules if f'-j {chain}' in rule]
        assert len(jumps) == 1, rules
        assert rules[1] == jumps[0], rules
        policy = subprocess.check_output([tool, '-S', chain], text=True).splitlines()
        assert policy[-1] == f'-A {chain} -j DROP'
    for family, address in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
        with socket.socket(family) as server:
            server.bind((address, port))
            server.listen()
            with socket.create_connection((address, port), timeout=2):
                accepted, _ = server.accept()
                accepted.close()
    print('Real IPv4/IPv6 netfilter installation, idempotency, rule order and local RPC probes passed.')
finally:
    for tool in ('iptables', 'ip6tables'):
        for args in (['-D', 'INPUT', '-p', 'tcp', '--dport', str(port), '-j', chain], ['-F', chain], ['-X', chain]):
            subprocess.run([tool, *args], check=False, capture_output=True, timeout=10)
