# Contributing

Use a Git checkout and Python 3.11 or newer. Run `uv sync --frozen` to install the locked
runtime and development dependencies. The orchestrator supports macOS; Linux is supported
for the edge client. Actual inference requires suitable hardware, models, and the native
llama.cpp image, which the unit suite does not provision.

## Getting a change in

Open an issue first for anything that changes behavior, so the design can be agreed before the
work. A small, self-evident fix can go straight to a pull request. Branch from `main`, keep the
branch to one subject, run the four local checks below, and target `main`. Contributions are
accepted under the MIT license in [LICENSE](LICENSE). Report vulnerabilities privately instead —
see [SECURITY.md](SECURITY.md).

Local checks (no fleet access):

```sh
uv run --frozen ruff check .
uv run --frozen pytest -q
uv run --frozen python scripts/check-public-tree.py
uv build
```

Tests use documentation-range addresses, temporary state, and mocked fleet operations.
Some tests need loopback sockets; Linux process tests create and clean up their own processes.
Tests that inspect an operator-owned config skip when it is absent. Do not add tests that
require private configs, contact live nodes, or depend on a model download to this default suite.

`setup-all.sh`, `update-all.sh`, `reset-all.sh`, `test-all.sh`, and CLI lifecycle/deploy commands
can mutate or query the real fleet. They are operator checks, not CI or local unit checks.
Read each command's help and use a dedicated test fleet before running them.

For changes, explain the concrete behavior and relevant verification. Add regression tests for
bugs and security boundaries. Keep credentials in ignored operator state. Keep dependency
updates in the lockfile and audit the exported runtime requirements with pip-audit. Do not
suppress an advisory without a reviewed, dated explanation and an expiry for rechecking it.

## Releasing

Release from a fresh clone containing only reviewed refs. Run the public-tree check with the
full history, build both distributions, and run the installed-wheel smoke check outside the
checkout. Never publish a working-directory archive: ignored private files may be present.
Container releases additionally require building and scanning both CPU and CUDA variants and
testing the firewall and inference on the supported Linux hosts.
