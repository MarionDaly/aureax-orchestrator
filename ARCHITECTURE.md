# Aureax Prototype — Architecture

Centralized AI orchestration for edge deployment: one orchestrator machine and
N Linux edge nodes, each running a llama.cpp server and an MCP server,
coordinated through K3s.

This document covers how the pieces fit together and — more usefully — the
invariants that are load-bearing but not obvious from reading any single file.
Breaking one produces a fleet that looks healthy and serves the wrong thing,
which is why they are written down; §3 names each one beside the check that
refuses it and the test that pins the check. README.md is the operator-facing
companion: setup, commands and walkthroughs live there.

> **Status:** prototype. The orchestrator runs on macOS and clients on Linux.
> Python 3.11 is the supported minimum; dependencies are locked with `uv`.

| Surface | Transport | Access model |
|---|---|---|
| Dashboard on loopback (`:7001`) | HTTP | Trusted local system; no login or client certificate |
| Dashboard LAN ingest/administration (`:7001`) | HTTPS (fleet CA) | Trusted LAN clients; no dashboard credential |
| llama-server and MCP | HTTPS | Separate per-node bearer keys |
| Worker RPC (`:50052`) | Plain TCP | Host firewall permits the group head and local probes only |

The dashboard host and clients allowed to reach its LAN listener must be fully trusted.
Host/Origin checks block foreign browser requests and DNS rebinding; they do not authenticate
programmatic clients. Dashboard API responses use `Cache-Control: no-store`.

---

## 1. Shape of the system

```
    ┌─────────────────────── Orchestrator (macOS) ───────────────────────────┐
    │  aureax CLI   dashboard (HTTPS :7001)   k3d cluster (K3s control plane)│
    │       │                  │                        │                    │
    │   SSH/paramiko      SQLite: metrics,          kubectl                   │
    │                     logs, chat sessions                                 │
    └───────┬──────────────────┬────────────────────────┬────────────────────┘
            │                  │ ▲                      │
     config push          telemetry + logs         K3s agent join
     (+ TLS material)     (HTTPS push, edge→orch)  (DaemonSet scheduling)
            │                  │ │                      │
    ┌───────▼──────────────────▼─┴──────────────────────▼────────────────────┐
    │                     Edge node (Linux, ARM64 or x86_64)                 │
    │   client/entrypoint.py — supervisor                                    │
    │     ├── llama-server (HTTPS :8080) OR rpc-server (plain :50052)        │
    │     ├── MCP server   (HTTPS :9000, NodePort :30900)                    │
    │     └── telemetry + log shipping threads                               │
    └────────────────────────────────────────────────────────────────────────┘
```

Communication is deliberately one-directional for data: **edge pushes to the
orchestrator**. There is no control-plane callback, and nothing on the
orchestrator polls a node's HTTP endpoints for state except reachability
probes.

The orchestrator keeps its local state under `~/.aureax`, created by
`aureax init` (`orchestrator/cli.py::init`): `telemetry/`, plus `ssh-keys/`,
`keys/`, `configs/`, `tls/` and `state/` at mode 0700, because all of them hold
something that must not be world-readable — the private keys the fleet
authenticates with, the per-node llama-server and MCP bearer keys, the resolved
per-node YAMLs that embed both, the fleet CA and every TLS leaf, and the
recorded `node_groups` baseline. `init` also repairs the mode on directories
that already exist, since `mkdir` honours the umask and ignores its own mode
for an existing directory.

Two entry points, from `pyproject.toml`:

- `aureax` — orchestrator CLI (`orchestrator.cli:main`)
- `aureax-harness` — interactive MCP testing (`orchestrator.harness.cli:main`)

### Repository layout

```
configs/          YAML configs (fleet definition + per-node edge configs)
orchestrator/     Orchestrator side: CLI, config, SSH, K3s, dashboard,
                  telemetry, log receiver
client/           Edge side: supervisor, llama.cpp wrapper, MCP server,
                  sensors, telemetry
k8s/              Kubernetes manifests (DaemonSets, Services)
docker/           Dockerfiles for the edge client container
scripts/          Operational scripts (setup, update, reset, test,
                  model download, benchmarking)
tests/            pytest suite
```

**A wheel installs two packages and nothing else at top level.** `orchestrator`
and `client` are the only distributions' worth of importable names; the build
helper stays a build-time module and the list of shipped resources lives at
`orchestrator/_resource_manifest.py`, under the package rather than at the repo
root, so the wheel installs no generic top-level name another distribution could
shadow. That manifest is loaded two ways from one file — imported normally by
`orchestrator.resources` at runtime, and read by path at build time, when
`orchestrator` is not yet importable — so the resources a wheel carries and the
resources `aureax init --workspace` copies can never drift apart. Both copiers
skip a missing source silently, so `tests/test_resource_manifest.py` checks every
entry against the tree. The two publication guards
(`scripts/check-public-tree.py`, `scripts/check-installed-package.py`) are
deliberately not shipped into a workspace: one needs a Git repository and the
other an installed wheel, so neither can work there.

---

## 2. Configuration: per-node files, not one shared template

The fleet definition is `configs/aureax-network.yaml`. It is **operator-owned
and untracked** — it carries real addresses and SSH credentials — so the
repository ships `configs/aureax-network.example.yaml` instead and
`.gitignore` excludes the real name. When the default path is missing, the
loader says so and names the copy command rather than raising a bare
`FileNotFoundError` (`orchestrator/config/loader.py`, `EXAMPLE_NETWORK_CONFIG`).
The CLI loads that file **lazily**, on first touch of
`ctx.obj["network_config"]`: Click runs the group callback before it parses a
subcommand's arguments, so an eager load would break `aureax init` on a fresh
clone and every `aureax <cmd> --help`.

Each SSH target in that file carries a `config_file` naming its own YAML. The
templating engine (`orchestrator/config/templating.py`) resolves `${node_id}`,
`${display_name}`, `${tags}`, `${orchestrator_ip}`, `${rpc_servers}`,
`${mcp_endpoints}`, `${api_key}` and friends **per target**, writing one
resolved file per node to `~/.aureax/configs/`.

**A `node_id` is a filename and a URL host is a URL host**, so both are checked
where the fleet definition is parsed rather than where they are joined. A
`node_id` and a `config_file` must each be a single path segment — they go
straight into `~/.aureax/keys/<node_id>.key` and `configs/<config_file>`, and
`Path("/a") / "/etc/x"` is `/etc/x`. Two ids that differ only in case are a
duplicate, because the stores they name live on the orchestrator's own
filesystem and the default macOS volume is case-insensitive: otherwise the
second node silently runs on the first one's key and leaf. `ca` and
`orchestrator` are reserved in any case — `~/.aureax/tls/<node_id>.pem` is the
issuer's or the dashboard's own certificate, not a node's. And a `host` may not
contain `@`, `/`, `\`, `?` or `#`: it is interpolated into
`https://<host>:<port>` URLs that carry that node's bearer key, and each of
those characters ends the authority early, sending the key somewhere else. None
of these values is purely operator-authored — the fleet definition is editable
over the dashboard's unauthenticated bind.

Adding a node is two edits and no code:

1. An `ssh.targets` entry in the fleet definition with a `config_file`.
2. The config file it names, in `configs/`.

Shipped per-node templates: `configs/aureax-edge.yaml` (a standalone node that
serves its own model), `configs/aureax-planner.yaml` (an MCP-host planner —
one config file used by exactly one target), and the sharded-group pair for
head and worker.

**Exception:** targets that differ *only* by placeholder-resolved values may
share one template, because generation still emits a distinct resolved file per
target. In the example fleet `configs/aureax-shard-worker-35b.yaml` serves every
shard worker and `configs/aureax-shard-head-35b.yaml` serves both group heads.
Give a node its own file the moment it needs a genuine override.

`orchestrator/config/schema.py` is the source of truth for what is valid. The
default SSH user is `aureax`. Set `enabled: false` to keep a node documented
without deploying to it: disabled targets are skipped by every SSH pass, label
pass, pod-count wait and config generation run.

### A frozen rollback target is a config contract, not a spare file

A superseded head/worker template pair can be kept unmodified as a rollback
target — in the example fleet, `configs/aureax-shard-head.yaml` and
`configs/aureax-shard-worker.yaml` for the previous, larger model. Two rules
make that work, and both are easy to break by tidying:

- **Reverting is not just the `config_file` lines.** A frozen head carries a
  `tensor_split` sized for the group shape it ran on, and `validate_groups`
  length-checks that against each group's worker count. Pointing a
  seven-entry split at a three-worker group aborts `aureax generate` before
  anything reaches a node. A revert therefore restores the `node_groups` block
  — group membership and RPC order together, since `tensor_split` indexes
  positionally — *and then* flips the `config_file` lines. The example fleet
  definition carries its own recipe inline on that block.
- **The frozen pair stays auth-off, and stays valid, by schema default.** It
  has no `api_key` line at all, and every auth and hardening field in
  `LlamaServerConfig` defaults to the pre-auth behaviour, so the rollback keeps
  validating and keeps serving unauthenticated without being edited. Editing a
  frozen file to match the current model defeats the point of keeping it. The
  same holds for MCP: neither frozen file carries an `api_key: ${mcp_api_key}`
  line, so a rollback head serves MCP open, exactly as it did before. Both ports
  still speak TLS with the node's own leaf, so what a rollback gives up is the
  bearer gate and not the transport: while one is deployed, that head's
  `:9000`/`:30900` answers any LAN client that trusts the fleet CA, and the LAN
  boundary is the whole control there.
- **The only edit a frozen file may take is a fleet-wide transport change.**
  Both files' telemetry endpoints were rewritten to `${orchestrator_url}` when
  TLS became mandatory, because the dashboard has no plaintext listener a node
  can reach and a rollback that could not reach the orchestrator at all is not a
  rollback; the frozen head's inert `health_check.endpoint` went to `https://`
  in the same pass rather than being exempted, so the plaintext guard's
  allowlist never grew a node-side entry. That is
  the boundary: a change every config in the repository takes for the same
  reason, never anything model-shaped. Each edited line carries that reasoning
  inline.

### Things that look like mechanisms but are not

- **`defaults:` in the fleet definition is documentation.** Nothing merges it
  into a per-node config. Each per-node YAML sets its own values; that is the
  point of the per-node pattern. Editing `defaults:` changes no node.
- **There is no ConfigMap.** A per-node config reaches a pod as a host file:
  `Deployer.deploy_configs()` SSH-pushes the resolved YAML to
  `/etc/aureax/aureax-edge.yaml`, and both DaemonSets mount `/etc/aureax` as a
  read-only hostPath. Anything the image writes under `/etc/aureax` is masked by
  that mount — same for `/opt/aureax`, which is why the Dockerfiles create only
  `/tmp/aureax` and the supervisor creates the rest at runtime. `full_deploy()`
  therefore pushes configs *before* labelling: the mount is `type: Directory`,
  which the kubelet will not create for you.

### SSH authentication

A target may set `password`, or `password_env` naming an environment variable to
read. `password_env` is what the shipped example uses, because it keeps the
secret out of the file entirely; a literal `password:` is accepted and puts the
credential in a file that anything reaching the dashboard's unauthenticated
bind can read (§3).

On the first password-authenticated connect the orchestrator installs
`ssh.bootstrap_public_key` into the node's `authorized_keys`, so the bash
scripts — which only ever speak `ssh -i` — keep working without `sshpass`. That
install is not confined to one command: `aureax bootstrap-keys` does it
deliberately, and `aureax status` does it as a documented side effect on every
reachable password-auth node, so the fleet does not silently drift back to
password-only.

It also clears the group-write bit on `$HOME`. sshd's `StrictModes` silently
refuses keys when the home directory is group-writable, which presents as "key
installed, still denied".

A target that *declares* a password by either spelling but resolves none is
reported as exactly that — an unexported `password_env` variable — rather than
as a key-only node (`orchestrator/cli.py::_no_password_reason`). The two call
for opposite operator actions.

Host keys are trusted on first contact and recorded in `~/.aureax/known_hosts`,
which the Python and the shell paths share. An unreadable trust store fails
before authentication rather than falling back to an unverified connect. A key
that *changes* is logged at ERROR with both SHA256 fingerprints, replaced in the
store, and connected to once more — a second mismatch propagates
(`orchestrator/deploy/ssh.py::get_ssh_client`). Nodes here are wiped and
re-imaged, and setup has to stay automated, so refusing would mean hand-editing
a trust store after every wipe: the loud line is the deliverable, not a veto.
The shell side matches it with `StrictHostKeyChecking=no` — a changed key prints
OpenSSH's own warning and the run continues on key auth, since a changed key
disables password auth and `BatchMode` never offers one — and the next Python
connect rewrites the entry.

`aureax trust-host-key <node> --fingerprint SHA256:...` is the optional
out-of-band check for an operator who wants a fingerprint verified before any
connection is made: it performs a handshake that sends no password, client key
or command, and replaces the stored key only on an exact match. It is never a
required step after a re-image. First contact itself still has to happen on a
trusted provisioning network.

### The network config never reaches a node

The fleet definition can carry SSH credentials, and a Docker build context is
unpacked on whichever node is building. So `configs/` is excluded twice over:
from the build context by `.dockerignore`, and from the code rsync by
`scripts/lib/common.sh`. Nothing under `configs/` is COPYed by either
Dockerfile. `tests/` and `.github/` are excluded the same way, since neither is
a build input, and both ignore files exclude TLS material by extension
(`*.pem`, `*.key`, `*.crt`, `*.p12`) so a certificate or private key dropped
anywhere in the tree can reach neither a build context nor a commit. The only
config that reaches a node is its own resolved per-node YAML, pushed over SSH to
`/etc/aureax/aureax-edge.yaml`.

The tree that *does* reach a node — the code sync each node builds its image
from — is staged in the node account's own home, at `~/.aureax-stage/build`
inside a 0700 directory, beside the staging the join script and every
secret-bearing push use. Not `/tmp`, which is world-writable: what is unpacked
there is built and then run as root. The leaf name is fixed rather than random,
because the private parent is what makes it safe and a per-run name would
accumulate a project tree on the node for every interrupted build;
`update-all.sh` removes it after the image import, and both build paths reclaim
the older `/tmp` path they replace.

### llama-server API-key authentication

Fleet-wide, not sharding-specific, and driven entirely from config: **auth on is
one template line.** `api_key: ${api_key}` under `model.server` in a node's
template enables bearer auth on that node; deleting the line disables it. There
is no separate enabled-flag — the template is the single source of truth,
whether edited by hand or through the dashboard's Access tab, which round-trips
the same file. A *shared* template toggles every node that shares it, though
each still gets its own key, because `${api_key}` resolves per target.

- **Keys are per-node and generated at `aureax generate`** into
  `~/.aureax/keys/<node_id>.key` (`orchestrator/config/keystore.py`), 0600 and
  untracked. The format is the literal prefix `aureax_sk_` followed by 32
  CSPRNG bytes base64url-encoded; the prefix is not cosmetic, it is what makes
  a leaked key greppable in logs and shipped transcripts, and `test-all.sh`
  asserts no shipped log contains it. Because the orchestrator holds the key,
  a wiped and re-imaged node regains it from the next `aureax apply` rather
  than needing anything preserved on the node. Rotation keeps exactly one
  predecessor in `<node_id>.key.prev`, so a client still holding the old key
  can be identified while an apply is pending.
- **The key never touches argv.** The supervisor sets `LLAMA_API_KEY` in the
  child's environment and *pops* it when the config has no key
  (`client/entrypoint.py`). argv is world-readable under `hostPID: true` and is
  captured into shipped logs; `tests/test_llama_auth.py` pins the contract,
  including an allowlist of the flags llama-server may be started with, so a
  new flag has to be added there consciously.
- **The key travels 0600 end to end.** It sits in the resolved config under
  `~/.aureax/configs/`, uploads to a freshly-named staging file at mode 0600
  inside a 0700 directory in the node's login-user home (`ssh.stage_path` — a
  fixed name in a world-writable directory could be pre-created by another
  local account), and is then moved to `/etc/aureax/aureax-edge.yaml` and
  chowned `root:root` 0600. The node's TLS material rides the same push, in the
  same staging directory (**TLS**, below). Root-only is still readable by the
  supervisor, whose container runs as root; adding `runAsNonRoot` to either
  DaemonSet means revisiting that chown. It is also written through
  `orchestrator/config/atomic.write_private_file` — unique temp file, fsync,
  `os.replace`, parent-directory fsync — as are the key files and the apply
  journal, so a crash mid-write never leaves a half-written credential where the
  next read expects a whole one.
- **Editor redaction is strictly BY KEY.** The dashboard's config reads mask
  `password` and `api_key` — at any depth, including `mcp.server.api_key` and
  every `mcp.client.remote_servers[].api_key`, and including a YAML alias of one
  of those values — and a save restores the real value for any field the
  operator left masked, so a redacted-view round trip leaves the file
  byte-identical. What it does not do is scan for secret-looking strings: a
  credential pasted into a differently named field is not masked. A file that
  will not parse can still be *opened* (redacted by a line rule, which keeps
  `password_env` out of it); saving it keeps returning 400 until the syntax is
  fixed and the value re-entered. Saves are serialised by a lock and run off the
  event loop.
- **Two routes return this key in plaintext** —
  `GET /api/access/{node_id}/key` and `GET /api/access/{node_id}/key/previous` —
  both audit-logged with the caller's address. The MCP key has its own pair
  (below), so four routes in total hand out a credential. That audit line is the
  only accountability there is: the dashboard itself is unauthenticated (§3).
- **`aureax apply` is what makes a change live.** It regenerates every config,
  SSH-pushes to the requested nodes plus their whole group, and restarts only
  heads and standalone nodes, with a targeted delete of the one pod on the one
  node. Shard workers are pushed but never restarted: `rpc-server` takes no key,
  and restarting one forces a cold group reload. `kubectl rollout restart` is
  banned by `tests/test_apply.py::test_restart_never_uses_rollout_restart`,
  because a DaemonSet rollout is serial and would cold-reload every head in the
  fleet for a one-node config change. `aureax start` does not push configs and
  is never a substitute.
- **Going fleet-wide is opt-in.** `apply` requires `--node <id>` (repeatable) or
  an explicit `--all`, which are mutually exclusive; the bare form is a
  `UsageError`. `--all` rotates every key it is given and restarts every head,
  and a rotation is recoverable only through the single `.key.prev`.
- **`--wait` is a real gate.** `apply` exits non-zero when a replacement pod
  never becomes Ready *and* when the readiness poll was skipped, because a gate
  that never ran is a failed gate (`orchestrator/cli.py::_apply_outcome_failed`).
- **`--wait` is TWO passes, and that is a deliberate tradeoff.** Every targeted
  pod is deleted first; only then are the replacements polled, each on its own
  budget of `worker_wait_timeout_seconds + health_timeout_seconds + 120`
  (`deployer._apply_wait_budget`, shadowing the startupProbe arithmetic in
  **Probe budgets**). A single interleaved loop would hold each node's delete
  until the previous node was Ready again — a serial fleet-wide reload, the very
  shape `kubectl rollout restart` is banned for. The cost is stated plainly:
  `--all --wait` reloads every head at once, so the whole fleet's endpoints are
  down for one shared window instead of half the fleet serving while the other
  half loads. Without `--wait` nothing changes; the deletes go out back to back
  either way (`test_apply.py::test_every_pod_is_deleted_before_any_wait_begins`).
- **The schema rejects values llama.cpp would silently mangle.** A comma in
  `model.server.api_key` splits it into several valid keys, `#` starts a comment
  in `--api-key-file` format, and whitespace is kept verbatim; `alias` must be
  comma-free for the same reason, since llama-server parses `--alias` as a
  comma-separated list. Both rules live in
  `LlamaServerConfig._check_server_string_shapes`. An empty `api_key: ""` is
  rejected too — it is falsy downstream, so it would ship auth silently
  disabled while looking configured. Auth-off is spelled by omitting the field.
- **`hostNetwork` means the container port is a second door.** llama-server's
  `:8080` and the MCP server's `:9000` are reachable directly on the node's own
  address, beside the `:30800` and `:30900` NodePorts. All four are HTTPS with
  the node's own leaf, and each pair leads into a single process, so one check
  covers both doors of a service: this key gates `:8080` and `:30800`, and the
  MCP key gates `:9000` and `:30900` (below). The one door with neither TLS nor
  a key is a shard worker's `rpc-server` on `:50052` (§3, Security).
- **Auth-exempt endpoints are build-specific.** On the pinned build (**b10375**,
  `LLAMA_CPP_VERSION` in both Dockerfiles) `/health`, `/v1/health`, `/models`,
  `/v1/models` and the web-UI assets are exempt from the key; everything else,
  `/metrics` included, 401s without it. `/health` staying public is what lets
  the kubelet's probes work, so losing that exemption crash-loops the node
  rather than locking out clients. The `/models` exemptions are removed upstream
  immediately after b10375, so a version bump flips them:
  `scripts/test-all.sh` section 7 probes `/v1/models` unauthenticated as the
  tripwire, and treats a flip as a real behaviour change to be reflected in the
  docs, not a broken test.

README's Configuration section is the operator-facing walkthrough for turning
auth on.

### MCP bearer authentication

The MCP server takes a per-node bearer key by the same mechanism and with the
same single switch: **`api_key: ${mcp_api_key}` under `mcp.server` turns it on,
deleting the line turns it off.** There is no enabled-flag, and a shared
template toggles every node that shares it while each still gets its own key.
The shipped `configs/aureax-edge.yaml` and `configs/aureax-shard-head-35b.yaml`
both carry the line; the worker templates must not (`validate_groups` rejects
one that does — a worker runs `rpc-server` and no MCP server, so the key would
be a credential nothing on the node reads), and the frozen rollback pair has no
line and serves MCP open.

- **A second keystore file, and the same prefix.** The key lives at
  `~/.aureax/keys/<node_id>.mcp.key` beside the llama key's `<node_id>.key`,
  with `<node_id>.mcp.key.prev` for the one retained predecessor
  (`orchestrator/config/keystore.py`, `kind=KIND_MCP`). Both kinds use the
  literal `aureax_sk_` prefix deliberately, not two prefixes: the prefix is what
  makes a leaked key greppable, and `test-all.sh`'s leak greps, the dashboard's
  key masking and `chat_service`'s unresolved-placeholder check all key off it.
  A second prefix would multiply four sites, and a missed one is a silent hole.
- **The check is a pure ASGI middleware** (`client/mcp/auth.py`,
  `BearerAuthMiddleware`), wrapped around `mcp.streamable_http_app()` by
  `client/mcp/server.py::build_app`. Pure ASGI, not `BaseHTTPMiddleware`, for
  three reasons that are all fatal here: the transport holds a long-lived SSE
  GET open, POST replies are streamed, and the `lifespan` scope has to reach the
  inner app untouched because that app's lifespan *is* `session_manager.run()`.
  FastMCP's own `AuthSettings` is OAuth-shaped — it *requires* an `issuer_url`
  naming an authorization server and a `resource_server_url`, and pairs with a
  token verifier — which is a different protocol from the static per-node key
  the rest of this fleet uses, so adopting it would mean standing up an
  authorization server to protect one port. The
  comparison is `hmac.compare_digest` over bytes (a non-ASCII token must be a
  401, not a 500), the scheme match is case-insensitive, the 401 body is
  `{"error": "unauthorized"}` with
  `WWW-Authenticate: Bearer realm="aureax-mcp"`, and the first rejection per
  process logs at WARNING with path and client host — never the token — with
  everything after it at DEBUG, so a LAN scanner cannot flush the startup lines
  out of the shipped log.
- **`/health` is exempt, and that exemption is load-bearing.**
  `scripts/lib/common.sh::mcp_health_wait` polls it unauthenticated — no
  bearer header — through `curl_tls`, and `setup-all.sh`, `update-all.sh` and
  `test-all.sh` all wait on it — on a freshly imaged node, before any key is in
  hand. Requiring a key there turns every provisioning run into a manual step
  and hangs all three scripts on a 401. `/health/` is the same route;
  `/healthz` is not exempt.
  `parse-config.py` exports `NODE_MCP_AUTH` per target so `test-all.sh` knows
  which nodes to expect a 401 from.
- **The planner's `${mcp_endpoints}` carries its peers' keys.** Each resolved
  `mcp.client.remote_servers` entry gains an `api_key` when that peer's own
  template references `${mcp_api_key}` and the key exists
  (`templating.mcp_endpoints`, using `read_key` and never `ensure_key` — that
  function is on the Access tab's 20-second poll and must not mint fleet
  credentials). Two consequences: `aureax generate` pre-mints every MCP key
  before it resolves any template, so a planner listed before a head does not
  ship a peer list with that head's key missing; and `aureax apply` fans out —
  when an MCP key changes, every enabled target whose template references
  `${mcp_endpoints}` joins the push set and is restarted, with a line saying
  why, because an unpushed consumer keeps dialling with the old key and 401s on
  every call.
- **Two more plaintext-key routes, audited identically.**
  `GET /api/access/{node_id}/mcp-key` and `.../mcp-key/previous`, and the Access
  tab's toggle and rotate routes take a `kind` of `"llama"` or `"mcp"`.
  `apply --rotate-key` rotates *both* kinds on the nodes it is given: one
  restart applies both, and "re-key this node" means the node. The tab's live
  probe POSTs a literal `{}` to `https://<host>:30900/mcp` rather than a real
  `initialize` — `StreamableHTTPSessionManager` registers a server-side session
  for any POST without an `Mcp-Session-Id` *before* it parses the body, so a
  real handshake from a 20-second poll would leak one session per node per poll
  for the life of the pod. A 401 proves the port is secured; the probe deletes
  any session id it does get back. It also writes one rejected-request line into
  each MCP pod's log per poll, which is expected noise from a probe that proves
  enforcement instead of assuming it.
- **The harness resolves the key from a node id.** `aureax-harness ping <node>`
  takes an address *or* an enabled node id; given an id it reads the address
  from the fleet file and the key from the keystore, so nothing crosses argv.
  Precedence is `--api-key` → `--key-file` → `$AUREAX_MCP_API_KEY` → keystore →
  none (`orchestrator/harness/cli.py::_resolve_key`); explicit beats implicit so
  a rotation caught mid-apply can be worked around by hand. `--api-key` says in
  its own help that it lands in `ps` and shell history.
- **Holding only the MCP key still reaches inference.** `run_inference` proxies
  to llama-server with the node's *own* llama key attached, so the MCP key is a
  door onto key-protected inference, not a second factor in front of it. That is
  intended: the two keys guard two ports, not two layers.

### TLS

Every HTTP listener in the fleet is HTTPS and every client verifies against one
private CA. There is no plaintext mode, no `verify=False` anywhere, and no
flag to turn it off — the fleet is addressed by LAN IP, which no public issuer
will certify, so the orchestrator is the issuer.

**The store is `~/.aureax/tls/`** (0700, created by `aureax init`), managed by
`orchestrator/config/tlsstore.py`:

```
~/.aureax/tls/            0700
├── ca.key                0600   EC P-256, no passphrase
├── ca.pem                0644   CN="Aureax Fleet CA", CA:TRUE pathlen:0
├── orchestrator.key/.pem        the dashboard's own leaf
└── <node_id>.key/.pem           one pair per enabled ssh target
```

Certificates are 0644 on purpose: a certificate is not a secret, and
`curl --cacert ~/.aureax/tls/ca.pem` must not need `sudo`. Four parameters each
prevent a specific failure, and all four are stated in the module docstring:

- **EC P-256, not RSA-2048.** Leaves are minted on every `aureax generate` —
  which is on the interactive path of `deploy` and `apply` — one per enabled
  target plus the orchestrator. P-256 keygen is sub-millisecond; RSA-2048 is
  hundreds of milliseconds each, so a ten-node fleet would pay seconds per apply
  for nothing. Every verifier here handles P-256: OpenSSL 3, Python's `ssl`,
  curl, the macOS keychain, every current browser.
- **397-day leaves.** Safari refuses a certificate whose lifetime exceeds 398
  days when it chains to a keychain-trusted root, and the operator is told to
  trust `ca.pem` in the keychain. A ten-year leaf would work in `curl` and fail
  in the browser with an error that names nothing actionable.
- **One hour of backdate on `notBefore`.** These nodes commonly have no RTC and
  step their clock once time sync converges, sometimes *after* the pod has
  started. A certificate issued while a node's clock is twenty minutes behind is
  `notBefore`-invalid there, and it presents as "certificate not trusted" —
  pointing at the CA rather than at the clock.
- **The CA is never rotated automatically.** `ensure_ca()` creates it iff absent
  and refuses to replace one that exists; within 90 days of expiry it logs
  loudly and `aureax tls status` says so. A rotation that happened by itself
  mid-`apply` would push the new CA to some nodes and leave the rest on the old
  one, splitting the fleet in half with no way to tell which half. Replacing it
  is the explicit `aureax tls regenerate --ca`, which is the five-step
  procedure below.

**The CA is name-constrained, because the operator is told to trust it as a
root.** `security add-trusted-cert -r trustRoot` turns `ca.key` — EC P-256,
passphrase-less, 0600 in a directory every backup tool copies — into a signing
authority for whatever the certificate permits, so an unconstrained CA makes one
leaked file a universal MITM credential for that Mac: a leaf for the operator's
bank would show a valid padlock. `_build_ca_cert` therefore carries a critical
`x509.NameConstraints` built by `fleet_name_constraints(config, orchestrator_ip)`
— the IP ranges a LAN fleet lives in (`10/8`, `172.16/12`, `192.168/16`,
`127/8`, `169.254/16`, `::1/128`, `fc00::/7`, `fe80::/10`), plus any fleet or
orchestrator address outside them as a bare `/32` or `/128` (the shipped example
fleet is on `192.0.2.x`, which is not private and must still generate), and the
DNS subtrees `localhost` plus every hostname-shaped target host. Both name types
are constrained, because RFC 5280 leaves an unconstrained type *unrestricted*: a
DNS-only constraint would still let the key sign any IP SAN. macOS, Chrome and
Firefox all enforce constraints on a locally trusted root, which is what makes
the quick start's `-r trustRoot` a bounded instruction.

The ranges are listed rather than derived from each host's own /24, so
re-addressing a node inside the LAN — routine where nodes are re-imaged — needs
no new root. They are baked at *creation*: `ensure_ca` never rewrites an
existing CA, so a fleet that grows past them is not silently re-certified.
Instead `generate_all_edge_configs` checks every planned SAN set against the CA
it is about to sign under (`san_constraint_violations`) *before* minting
anything, and raises naming both the address and
`aureax tls regenerate --ca --orchestrator-ip <ip>` — a leaf that violates its
issuer's constraints is not a degraded certificate, it is one no verifier
accepts for any of its names. `aureax tls status` prints the constraints for the
same reason: they are the CA's blast radius.

**Rotating the CA is five steps, and the last three are the ones that get
skipped.**

1. `aureax tls regenerate --ca --orchestrator-ip <ip>` mints the new root — the
   address is what its name constraints are computed from — and reissues every
   leaf in the same call, since each one fails the AKI rule the moment the CA
   changes. It prints the four steps below as it finishes, so the procedure
   survives this document being out of date.
2. `aureax apply --all --orchestrator-ip <ip>` pushes `ca.pem` and every node's
   new leaf, and restarts the heads and standalone nodes.
3. `aureax stop && aureax start`. `apply` never restarts a shard worker (§3),
   and a node's telemetry and log reporters snapshot their `SSLContext` at
   construction, so without this every worker goes on verifying the dashboard
   against the old root. Metric push failures are logged at DEBUG, so it is
   invisible at default verbosity: the Overview tab simply stops receiving those
   nodes, indefinitely.
4. Restart `aureax dashboard`. One left running through the rotation still
   serves the leaf the *old* CA signed, which the freshly restarted nodes now
   reject.
5. Re-trust `ca.pem` in the browser, which trusts the old root and not the new
   one.

`ensure_leaf` reissues rather than blindly rewriting: `reissue_reason` returns a
reason string for a missing or unparseable file, a changed SAN set, under 30
days to expiry, a `notBefore` in the future, an authority key id that no longer
matches the current CA, or a certificate whose public key does not match its
key file — which is how a half-written pair repairs itself. Loopback SANs
(`127.0.0.1`, `::1`, `localhost`) are on every leaf and are load-bearing: every
in-pod client dials `127.0.0.1`, and an SSH-tunnelled orchestrator
(`server.dashboard.host: 127.0.0.1`) serves its own leaf on loopback. A
node's leaf carries its configured host and the loopback names and nothing else
— deliberately no `DNS:<node_id>`, because nothing in the fleet dials a node by
id and every extra name only widens what the CA has to permit.

**The orchestrator's leaf has two producers, so its SANs are a union.**
`aureax generate` mints it for the `--orchestrator-ip` the operator typed, which
is also the address stamped into every node's telemetry and log endpoint, while
`aureax dashboard` mints for the LAN address it detects. Taking either verbatim
would let whichever ran last delete the other's name — and since
`reissue_reason` treats any SAN-set difference as staleness, the two would
thrash on every run, silently in the worst direction: the dashboard's own URL
keeps working while every node's push fails verification.
`orchestrator_leaf_sans` therefore unions the requested SANs with those the
certificate already carries. The cost
is that a Mac which moves between networks accumulates addresses; `aureax tls
status` shows them and `aureax tls regenerate --node orchestrator
--orchestrator-ip <ip>` resets the leaf to today's. A unioned name outside the
CA's constraints is refused with the same message as any other violation.

**The node's paths are constants, not per-node names.** The generator *injects*
a block into every resolved config, rather than expecting a template to carry
one — a template cannot forget it, TLS is not a per-node choice, and the frozen
rollback pair needs no new block:

```yaml
tls:
  ca_file: /etc/aureax/tls/ca.pem
  cert_file: /etc/aureax/tls/node.pem
  key_file: /etc/aureax/tls/node.key
```

Fixed `node.pem` / `node.key` rather than `<node_id>.pem`, so renaming a node
cannot strand a file in `/etc/aureax/tls` that no config names any more. Both
sides read the same `tlsstore` constants, so the generator and the deployer move
together. A template that carries its own `tls:` block is overridden with a
warning: it was lying about what it deploys.

**Distribution rides the config push.** `Deployer.deploy_configs()` stages four
files per node — the resolved config, `ca.pem`, `<node_id>.pem`,
`<node_id>.key`, each uploaded 0600 into the node user's 0700 staging directory
over ONE connection and ONE SFTP channel (`ssh.upload_files`; an SSH handshake
against a small edge CPU costs real time, and four of them per node is three
too many) — then one sudo command creates `/etc/aureax/tls` at 0700 and moves them into
place as `ca.pem` 0644, `node.pem` 0644, `node.key` 0600, all `root:root`; the
failure path removes every staged file. A node with no leaf on the Mac is
refused by name, pointing at `aureax generate`. Minting itself happens only in
`generate_all_edge_configs`, never in `generate_edge_config`, because the latter
runs on the Access tab's poll and must not create fleet credentials as a side
effect.

**Three listeners, one CA.**

- **The dashboard** — `orchestrator/dashboard/app.py::run` takes `ssl_certfile`
  and `ssl_keyfile` as keyword-only arguments with *no defaults*, so a caller
  structurally cannot start the fleet's listener in plaintext. `aureax
  dashboard` mints or reissues the orchestrator leaf on every start — from
  `--orchestrator-ip` when given, otherwise from the detected LAN IP, so moving
  the Mac to another network reissues rather than serving a certificate for an
  address it no longer has. It *fails* when it has neither: `0.0.0.0`
  is not a name anything can verify, and a loopback-only certificate would serve
  a padlock to nobody while every node's telemetry POST failed on the hostname.
  The error names `--orchestrator-ip` as the way through, which is also the
  answer on a Mac that is off the fleet's network entirely. `_SECURITY_HEADERS` gains
  `strict-transport-security: max-age=31536000`, deliberately without
  `includeSubDomains` or `preload` — browsers ignore HSTS for IP-literal hosts
  anyway, and pinning every sibling `.local` name would break unrelated LAN
  services.
- **The MCP server** — `client/entrypoint.py::_start_mcp_server` passes the same
  cert and key into `uvicorn.Config`.
- **llama-server** — `client/config/parser.py` emits `--ssl-cert-file` and
  `--ssl-key-file` (paths, never key material; argv is world-readable under
  `hostPID`) immediately after `--port`, and **raises** when the resolved config
  has no `tls.cert_file`/`tls.key_file` rather than building a plaintext command
  line. Both flags are on `tests/test_llama_auth.py`'s argv allowlist. This is
  the one listener whose TLS is a *build* property: llama.cpp's CMake option is
  `LLAMA_OPENSSL` (there is no `LLAMA_SERVER_SSL`), it defaults ON, and
  cpp-httplib's `find_package(OpenSSL)` degrades to a `message(WARNING)` on
  failure — so a build with no OpenSSL succeeds and produces a binary that logs
  "the server is built without SSL support" and exits. Both Dockerfiles
  therefore install `libssl-dev`, pass `-DLLAMA_OPENSSL=ON`, and **assert on the
  artefact**: an `ldd build/bin/llama-server | grep -q libssl.so` (and
  `libcrypto.so`) step that exits 1 with the `ldd` output when either is
  missing. Asserting on the flag would prove nothing.

**The dashboard answers on two listeners, one port, one process.** `run()`
builds the app once and serves it twice: HTTPS on the orchestrator's own LAN
address — the address the leaf was minted for, which is the address in every
node's `${orchestrator_url}` — and plain HTTP on `127.0.0.1` and `::1` at the
same port. The HTTPS one is the fleet's and is the load-bearing bind; the
plaintext one exists because the operator sitting at the Mac should not have to
trust a private CA to open their own dashboard, and it cannot be reached from
the LAN at all. The bind is that one LAN address rather than `0.0.0.0` because
a wildcard socket owns loopback too and would leave the second listener nothing
to bind — so a Mac with several interfaces serves the fleet on exactly the one
the certificate names, and `server.dashboard.host`'s `0.0.0.0` default is read
as "the detected or given orchestrator address" rather than as a literal
wildcard. A host set to some other specific address *is* the HTTPS bind, and a
loopback one is the SSH-tunnel case: HTTPS already holds the port there, so no
plaintext listener is added.

Both listeners are one process, which is what keeps `aureax stop`'s
kill-by-port (`lifecycle._kill_process_on_port`, `lsof -sTCP:LISTEN` on
`server.dashboard.port`) finding a single pid. Only the HTTPS server runs the
app's lifespan; the loopback one is handed a `LifespanOnlyShim` that completes
startup and shutdown itself and forwards every other scope untouched, because a
second lifespan would start a second copy of every watcher, polling `kubectl`
in parallel. Ctrl-C reaches both: uvicorn installs its signal handler per
server and `signal.signal` keeps only the last one, so a poll mirrors
`should_exit` across the pair rather than leaving one listener serving. Every
socket is bound before the app is built, so a busy port is one sentence naming
the pid that holds it — the operator's own earlier dashboard, usually — instead
of `[Errno 48] address already in use`.

**Every client verifies.** One cached `ssl.SSLContext` per CA file rather than
`verify="<path>"` (httpx 0.28 deprecates the string form): `client/tls.py`
builds it on the node side from the config's `tls:` block, `tlsstore`
`ca_ssl_context()` on the orchestrator side. Passing `verify=None` is *not* a
disabled check — httpx falls through to httpcore's default context, which is
`CERT_REQUIRED` with hostname checking — so a caller that forgets the fleet
context gets a hard certificate failure, never a quiet downgrade. The supervisor
makes the whole thing fail early rather than half-encrypted: `_require_tls_files`
is the first statement of `start()` and exits **78** (`EX_CONFIG`) naming the
missing file and `aureax apply --node <id>`. Shard workers run it too — they
ship telemetry over TLS.

**The kubelet's probes are the exception, and only in one direction.** The three
`httpGet` probes in `k8s/daemonset-edge.yaml` carry `scheme: HTTPS`, without
which every check fails against an HTTPS server and the startupProbe burns its
whole budget killing a server that was answering correctly. The kubelet does
*not* verify the certificate it is handed. That is fine for what a probe is:
liveness, not identity. The manifest change and the image rebuild are coupled —
an HTTP probe against an HTTPS server and an HTTPS probe against an HTTP server
fail the same way — so the order on a fleet is rebuild both images, then
generate and deploy, then pods.

`${orchestrator_url}` is the placeholder every template's telemetry endpoint
uses: `https://<host>:<server.dashboard.port>`, IPv6-bracketed, carrying the
scheme *and* the port so no template spells either. `${orchestrator_ip}` still
resolves for anything else. The port matters: a template that spelled 7001
itself would ship a moved dashboard's telemetry to a closed port, with nothing
reporting it.

**The operator surface is three subcommands and one flag.** `aureax tls status`
prints the CA's subject, remaining days and name constraints plus every leaf's
expiry, SANs and reissue reason — and it never mints, so it is safe on a fresh machine (it exits
0 there and says the store is empty). It cannot see SAN *drift*, because it
takes no fleet config and so does not know the intended SAN set — it computes
each leaf's reissue reason against the SANs that leaf already carries. A node
that has moved address therefore reads as `ok` here and is reissued by the next
`aureax generate`, which is the command every stale line points at.
`aureax tls ca-path` prints the path and nothing else, for shell substitution
into `--cacert`. `aureax tls regenerate [--ca --orchestrator-ip <ip>] [--node ID]…` reissues.
`--ca` requires the orchestrator address, because the new root's name
constraints are computed from it and from the fleet file, and it prints the four
remaining steps as it finishes. `--node orchestrator` is the one name that
*resets* rather than reissues, dropping the accumulated SANs back to the address
given. On the shell side `scripts/lib/common.sh` sets `AUREAX_CA`
(default `~/.aureax/tls/ca.pem`) for the sourcing shell and defines a `curl_tls`
wrapper that refuses with exit 77 when the CA is unreadable rather than falling
back to an unverified call, and `aureax-harness` takes `--ca` on every command
and reports a verification failure as exactly that, naming the CA
and the node, instead of as an unreachable endpoint. `tests/test_no_plaintext_urls.py`
is the backstop for all of it: no `http://` constant, no `verify=False`, no
`curl -k`, no `--insecure`, no `check_hostname = False` anywhere in the tree.
Its `PLAINTEXT_ALLOWED` holds exactly one URL — the dashboard's loopback
listener, below — pinned by exact value in five files: the banner
`orchestrator/cli.py` prints, `scripts/test-all.sh` §3,
`scripts/check-installed-package.py`'s in-process wheel smoke test, and the two
operator-facing documents, README and this one. A sixth exemption, or a widened
one, fails the suite.

**What TLS does and does not buy.** It gives confidentiality and *server*
authentication: a passive listener on the LAN cannot read prompts, telemetry or
bearer tokens, and a client that trusts the CA knows the node is the node. It
authorizes nobody. The dashboard's ingest and control routes still take no
credential (§3), which is why each port keeps its own bearer check. The CA
private key at `~/.aureax/tls/ca.key` is the single fleet-wide secret — it
forges every node's identity and the dashboard's — and it is passphrase-less
because every mint runs unattended from `generate_all_edge_configs`. And
`rpc-server` on `:50052` is untouched by any of this: it speaks a bespoke binary
protocol with no TLS at any layer, so it is the only plaintext port left on a
node, and a host firewall — installed by the worker pod's own `rpc-firewall`
init container — remains its only control (§3, Security).

---

## 3. Sharded model groups (llama.cpp RPC)

Several nodes can cooperatively serve **one** model that fits on none of them
individually. A `node_groups` entry names a **head** and its **workers**:

- **head** — runs `llama-server` with `--rpc <w1>:50052,<w2>:50052,…` plus the
  MCP server. Only this node holds the GGUF.
- **workers** — run only llama.cpp's `rpc-server`, holding a share of the
  weights and executing tensor ops for the head.

`node_groups` is a list, so a fleet may run several groups at once; the example
fleet runs two groups of 1 head + 3 workers. Each group is addressed exactly
like a single node, through its head: MCP at `https://<head>:30900/mcp`, llama
at `https://<head>:30800`. `uv run aureax group list` prints both, and
`uv run aureax group doctor` collects build id, the `llama-server`/`rpc-server`
checksums, image digests, memory, NIC speed, RPC listeners, GGUF presence and
recent logs from every member in one pass — and **exits non-zero** when the
members disagree.

### Invariants and how each is enforced

Break any invariant in this list and the fleet stays green while serving the
wrong thing. Each row names the check that refuses it and the test that pins
the check, so a regression shows up in the suite rather than on a node.
Static checks never touch the network; every message names the group, the file
and the field, the consequence, and a pasteable remedy. Several rows are
*ceilings* shared across configs rather than descriptions of the current model
— see **Probe budgets** in §4 for why tightening one arms a trap that only
springs on a rollback.

| Invariant | Enforced by | Test |
| --- | --- | --- |
| Head has `node.role: edge_executor` | `templating.validate_groups` | `test_node_groups.py::test_validate_groups_rejects_head_with_worker_role` |
| Head serves MCP | `validate_groups` | `…::test_validate_groups_rejects_a_head_that_serves_no_mcp` |
| Head sets `gpu_layers > 0` | `validate_groups` | `…::test_validate_groups_rejects_head_without_gpu_layers` |
| Head keeps `${rpc_servers}` rather than a literal list | `validate_groups` | `…::test_validate_groups_rejects_a_head_that_hardcodes_its_rpc_servers`, `…rejects_head_missing_rpc_servers` |
| Head keeps `${worker_wait_timeout_seconds}` | `validate_groups` | `…::test_validate_groups_rejects_a_head_that_hardcodes_its_worker_wait`; `test_probe_budget.py::test_a_group_head_takes_its_worker_wait_from_the_group` |
| `len(tensor_split) == len(workers)` | `validate_groups` | `…::test_validate_groups_rejects_mismatched_tensor_split` |
| Workers have `node.role: shard_worker` | `validate_groups` | `…::test_validate_groups_rejects_worker_with_edge_role` |
| Workers set `mcp.server.enabled: false` (`parse-config.py` exports it to the shell as `NODE_HAS_MCP_SERVER`, so scripts skip MCP checks against them) | `validate_groups` | `…::test_validate_groups_rejects_worker_serving_mcp` |
| No worker template references `${api_key}` or `${mcp_api_key}` | `validate_groups` | `…::test_validate_groups_rejects_a_worker_template_carrying_the_api_key`, `…_the_mcp_api_key` |
| Worker's `model.rpc_server.port` == the group's `rpc_port` | `validate_groups` | `…::test_validate_groups_rejects_worker_port_mismatch` |
| Group membership is well-formed; no node in two groups | `schema.NetworkConfig._check_node_groups` | `…::test_invalid_groups_are_rejected` |
| **Worker order is `--rpc` order: append, or remove from the tail** | **`group_state.order_problems` at generate — fatal for any edit that moves a worker already in the list (a swap, a middle insert, a middle removal); `--accept-worker-reorder` records a deliberate one** | **`test_node_groups.py::test_a_reordered_workers_list_fails_generation`; `test_group_state.py::test_inserting_a_worker_in_the_MIDDLE_is_fatal`** |
| **A planner's `remote_servers` is derived, never listed** | **`templating.validate_planners`** | **`test_node_groups.py::test_validate_planners_rejects_a_hardcoded_remote_servers_list`** |
| **A role change REMOVES the other gating label** | `deployer.label_nodes` / `label_shard_workers` **and `lifecycle._start_phase_label_and_wait_pods`**, both through **`cluster.labels.strip_conflicting_label`** | `test_deployer.py::test_edge_label_pass_strips_the_shard_label_it_replaces`; **`test_lifecycle_groups.py::test_start_strips_the_edge_label_from_a_node_it_labels_as_a_worker`** |
| **A node carrying both labels is loud, and one this start cannot fix aborts it** | **`cluster.labels.double_labelled_nodes` via `lifecycle._preflight_double_labels`; printed in red by `cli._status_cluster`** | **`test_lifecycle_groups.py::test_a_double_labelled_node_this_start_does_not_claim_aborts_the_start`; `test_labels.py::test_a_node_carrying_both_label_keys_is_reported`** |
| **Every peer runs the byte-identical llama.cpp binaries** | **`cluster.buildcheck.compare_group`** over the sha256 of `llama-server` and `rpc-server` read inside each member's image over SSH, one round trip per member — run by **`lifecycle._assert_group_builds_match`** (pre-start, before anything is labelled: mismatch raises, unreadable member warns) and by **`cli._group_problems`** (`group doctor`, exit 1); `test-all.sh` §6 asks the same question of `rpc-server`. **NOT the image digest**: every node builds `aureax-client:dev` natively (`scripts/build-on-node.sh`), so digests differ across a coherent group and a digest gate would refuse every real start | **`test_buildcheck.py::test_a_worker_with_a_different_rpc_server_is_a_problem`, `…::test_an_unreadable_member_is_a_note_not_a_problem`; `test_lifecycle_groups.py::test_a_worker_running_other_binaries_than_the_head_fails_the_start`, `…::test_the_build_gate_runs_before_anything_is_labelled`; `test_cli.py::test_doctor_exits_one_when_a_members_binaries_differ`** |
| `LLAMA_CPP_VERSION` identical in both Dockerfiles | `scripts/test-all.sh` §1 | `test_manifest_tls.py::test_both_dockerfiles_pin_the_same_llama_cpp_build` |
| **The pinned tag still resolves to the pinned commit** | each Dockerfile's build stage compares the cloned `HEAD` against `LLAMA_CPP_COMMIT` and fails the build on the node, naming both shas — a tag is mutable upstream, and a moved one would otherwise hand two nodes different binaries under one version string | `test_manifest_tls.py::test_both_dockerfiles_pin_the_same_llama_cpp_commit` |
| **The group config is valid at START, not only at generate** | **`lifecycle._start_phase_validate_config`** (runs `validate_groups` + `validate_planners` before the cluster is touched) | **`test_lifecycle_groups.py::test_start_validates_the_group_config_before_touching_the_cluster`** |
| **The nodes' configs are not stale relative to the fleet file** | **`group_state.baseline_differs`, warned (never fatal) at start** | **`test_lifecycle_groups.py::test_start_warns_when_node_groups_drifted_from_the_last_generate`** |
| Workers Ready before the head's llama-server starts | deploy/lifecycle phase order; the head's own `_wait_for_rpc_workers` | `test_lifecycle_groups.py::test_shard_workers_are_labelled_and_waited_for_before_the_edges` |
| startupProbe budget ≥ `worker_wait + health_timeout`, rollback head included | `k8s/daemonset-edge.yaml` `failureThreshold` | `test_probe_budget.py::test_probe_budget_covers_every_node_carrying_the_edge_probe`, `::test_probe_budget_covers_the_rollback_target` |
| Workers carry a startupProbe ONLY | `k8s/daemonset-shard.yaml` | `test_probe_budget.py::test_worker_probe_is_startup_only` |
| **Worker memory tracks model size ÷ worker count** | `k8s/daemonset-shard.yaml` requests/limits | **`test_shard_budgets.py::test_every_configured_group_fits_the_worker_memory_request`, `…_limit`, `…::test_worker_memory_requests_do_not_exceed_limits`, `…::test_the_worker_memory_limit_still_covers_the_122b_rollback_at_seven_workers`** |
| **The worker memory limit stays the flash-attention tripwire** | `k8s/daemonset-shard.yaml` limit, set between the FA-on and FA-off peaks | **`test_shard_budgets.py::test_the_worker_memory_limit_is_still_the_flash_attention_tripwire`** |
| **`EDGE_WAIT` / `EDGE_WAIT_SECS` cover a cold load and stay under the probe budget** | `scripts/update-all.sh`, `scripts/setup-all.sh` | **`test_shard_budgets.py::test_every_edge_wait_ceiling_stays_under_the_edge_startup_probe_budget` and its three siblings** |
| **Worker count is derived, never a literal** | both scripts read `$SHARD_WORKER_ROLE` | **`test_shard_budgets.py::test_both_scripts_derive_the_worker_count_from_the_role_constant`** |
| **Groups sharing a worker config file have the same shape** | one shared template per group shape | **`test_shard_budgets.py::test_groups_sharing_a_worker_config_file_have_the_same_shape`** |
| **`docker/scripts/start.sh`'s role literal == `SHARD_WORKER_ROLE`** | the `grep` line in `start.sh` | **`test_shard_budgets.py::test_start_sh_greps_the_shard_worker_role_constant`** |
| `flash_attention` is ALWAYS emitted | `client/config/parser.py::resolve_flash_attn` | `test_rpc_args.py::test_flash_attn_flag_is_always_emitted` |
| `--ubatch-size` is never emitted | the arg builder's deny-list and allowlist | `test_llama_auth.py::test_deny_listed_flags_are_never_emitted` |
| A split GGUF is registered by part 1 | `schema.ModelRegistryEntry._filename_is_first_part` | `test_model_parts.py::test_registry_entry_must_name_part_one` |
| **A head really attached the peers its own on-node config names** | **`client/entrypoint.py::_verify_rpc_topology`, after the health wait, exits non-zero** | **`test_supervisor.py::test_a_tensor_split_that_outnumbers_the_rpc_peers_exits`** |
| Never dial a worker's `:50052` from off-node | `lifecycle._nodes_status`; `test-all.sh` checks the listener on the node | `test_lifecycle_groups.py::test_nodes_status_never_dials_a_workers_rpc_port` |
| `rpc-server` `:50052` stays firewalled to the head | `client/rpc_firewall.py` in the shard pod's `rpc-firewall` init container installs and verifies the `AUREAX_RPC_<port>` chain for every IP family the host has, before RPC starts | `test_rpc_firewall.py::test_rpc_firewall_installs_both_families_with_drop_after_head_and_local`, `…::test_rpc_firewall_failure_blocks_init_completion`, `…::test_rpc_firewall_requires_a_head_before_touching_rules`, `…::test_an_unreadable_rule_table_refuses_rpc_startup` |
| **A worker's RPC allowlist holds IP literals, derived from its group head** | `templating.generate_edge_config` resolves the head on the orchestrator and writes `model.rpc_server.allowed_clients`; `rpc_firewall._peer_address` refuses a non-literal by name, since the init container resolves nothing | `test_rpc_firewall.py::test_a_hostname_in_the_allowlist_is_refused_by_name` |
| **Only a recorded child is ever signalled as an orphan** | `client/process_ownership.reap` — pid, start time and boot id must all match a record whose owning supervisor has exited | `test_process_ownership.py::test_unrecorded_processes_are_never_signalled`, `…::test_reused_pids_and_live_supervisors_are_never_killed`, `…::test_a_live_child_this_supervisor_owns_is_still_a_conflict` |

Four things about that table are worth stating outright.

**A runtime device-count or flash-attention check does not exist, and its
absence is deliberate.** `_verify_rpc_topology` compares the node's own
`tensor_split` against its own `rpc_servers` — a structural check on the file
the node actually has, which is what `validate_groups` cannot see. It stops
there because on the pinned build (b10375) `/props` carries no device list and
no flash-attention field, and `llama-server.log` at the default verbosity prints
neither. A check reading a guessed JSON key or log pattern would be inert while
looking enforced — the worst kind of check, because the fleet reads as green and
nothing is watching. The flash-attention invariant is
enforced instead by the worker memory limit in `k8s/daemonset-shard.yaml`,
which sits between the FA-on and FA-off peaks so losing the setting OOM-kills at
startup.

**An even `tensor_split` is not an even footprint.** llama.cpp divides layers by
`n_layer + 1` and the last device also carries the output tensor; only some
layers are full attention, so the KV cache lands unevenly across peers even at
equal weights. That is why the memory numbers in `daemonset-shard.yaml` are
sized against the *peak* worker rather than the mean, and why
`tests/test_shard_budgets.py` encodes the overhead as an additive per-shape
constant taken from the manifest's own arithmetic rather than as one multiplier
— no single multiplier fits both the 131072-context head and the 4096-context
rollback.

**A split GGUF is registered by its first part, and `size_gb` is decimal.** A
model over ~50 GB is published as `name-00001-of-00003.gguf` under a per-quant
subdirectory in the HF repo. `filename` is the flat *local* basename of part 1
(llama.cpp opens the rest itself); `huggingface_file` is the repo path and may
include that directory; `gguf_part_names()` derives every part from the suffix,
so there is nothing else to configure. `size_gb` is **decimal GB (10⁹)**, which
`download-model.sh`'s completeness check and `tests/test_shard_budgets.py`'s
memory arithmetic both depend on — reading it as GiB overstates every headroom
by 7%.

An entry can also pin *what* it fetches. `revision` names the HuggingFace ref the
download URL is built from (`resolve/<revision>`); unset it is `main`, which
upstream is free to move, so two provisions of one fleet can fetch different
bytes under the same URL. `sha256` records the digest of a single-file GGUF and
`sha256_parts` one digest per part of a split one, positionally paired with the
part list — one or the other, never both, since a single digest cannot describe a
model fetched as three separately-truncatable files. All three are optional, and
declaring no digest keeps the size checks alone. Where a digest *is* declared it
is checked twice, in two different places for two different reasons:
`ModelRegistryEntry._check_digests` checks the **shape** where the registry is
parsed (64 lowercase hex characters; the part count must match the part list),
because a malformed digest discovered at download time fails on every node at
once after the multi-gigabyte fetch; `download-model.sh` checks the **bytes**
with `shasum -a 256` — freshly fetched parts and parts already on disk alike —
before anything is pushed to a node, deleting a mismatched part and exiting
non-zero.

**`flash_attention` is tri-state (`true`/`false`/`"auto"`) and is ALWAYS
emitted.** Omitting the flag hands the decision to llama.cpp's own AUTO probe,
and over RPC `ggml_backend_rpc_device_supports_op` is a bare `return true` — so
AUTO always resolves to *enabled*, and `false` is honoured only because the
flag is emitted on every launch. It matters at long context: without flash
attention llama.cpp reserves an `[n_kv x n_ubatch x n_head_q]` F32 score
tensor, several GiB per worker at 128k
context on nodes with no swap. `llama-server.log`'s "Flash Attention enabled"
line is the only runtime source of truth, for the reason in the first note.

### Security

`rpc-server` is the **only unauthenticated door left on a node, and the only
plaintext one**. It speaks a bespoke binary protocol with no TLS at any layer
and no auth at any layer, and it deserializes arbitrary tensor graphs
(CVE-2026-34159 was an unauthenticated RCE in exactly this path). Port 50052
must stay on a trusted LAN, firewalled to the head's address. NetworkPolicy does
not apply to `hostNetwork` pods, so the host firewall is the only network
control available here. Nothing in the TLS or bearer work above changes that;
closing it means a different transport, not a configuration change.

That firewall is not left to the operator. Each worker pod runs an
`rpc-firewall` init container (`client/rpc_firewall.py`, `NET_ADMIN`, host
network) ahead of `rpc-server`. It builds an `AUREAX_RPC_<port>` chain — accept
loopback, accept the node's own address and the head from
`model.rpc_server.allowed_clients`, `DROP` the rest — jumps to it first from
`INPUT` for that TCP port, and verifies every rule it just wrote. Anything that
does not install and verify refuses startup, so a worker with no policy never
serves. Four properties are worth knowing:

- **The allowlist is derived from the group head**, resolved to IP literals on
  the orchestrator at generate time. The init container resolves no names — it
  has neither cluster DNS nor an mDNS resolver — so a hostname there is a named
  refusal naming `aureax generate` and `aureax apply`, not a crash-loop.
- **Both IP families, failing closed on each one the host has.** A host with no
  IPv6 address at all and an IPv4 RPC bind skips the `ip6tables` pass and says
  so: there is no v6 socket to protect, and this container drops every
  capability but `NET_ADMIN` and mounts no `/lib/modules`, so it could not
  create an absent ip6 filter table. A host that *does* carry IPv6 still fails
  closed.
- **The rules are baked in at pod start** and nothing re-runs them for the life
  of the pod. `apply` never restarts a worker, so a group whose head address
  moved needs `aureax generate` + `aureax apply` and then
  `aureax stop && aureax start`; until the pod is recreated the worker DROPs its
  own head and the group hangs at the RPC handshake.
- **The rules outlive the pod deliberately**, so a shutdown never reopens the
  port. Removing the `INPUT` jump and the chain is safe only once every RPC
  process on that port has stopped, and a firewall manager that flushes them
  while workers run undoes the whole control.

Everything else on a node carries both: HTTPS with a leaf the fleet CA
signed (§2, **TLS**), plus a per-node bearer key on `:8080`/`:30800` and on
`:9000`/`:30900` when the node's template carries the corresponding one-line
`api_key:`. The MCP key is a door onto key-protected inference rather than a
second factor in front of it — `run_inference` proxies to llama-server with the
node's own llama key attached — so a caller holding only the MCP key still
reaches inference. That is intended. The clamps in `client/mcp/server.py`
(`MAX_INFERENCE_TOKENS`, the metrics window, the anomaly truncation) bound what
an *authenticated* caller can cost the pod.

**The dashboard on :7001 is intentionally unauthenticated** — no token, no
login, just `https://<host>:7001/` from the LAN, or `http://localhost:7001/`
from the orchestrator itself. TLS changed the transport and nothing about
the authorization: it proves the dashboard is the dashboard and keeps the LAN
from reading what crosses it, and then serves every route to whoever asks. The
HTTPS bind is the orchestrator's LAN address and that is load-bearing: every
node posts telemetry and logs to that address, so narrowing
`server.dashboard.host` to loopback silently kills fleet ingest (§2, TLS). The
plaintext listener beside it is loopback-only, so it widens nothing: reaching
it already means being on the machine, where every route is readable anyway.
The boundary is therefore the whole authorization story: whoever reaches either
listener can reveal node keys, rewrite configuration and operate the fleet, so
the orchestrator and every client on its LAN listener must be trusted. Browser
Host/Origin validation rejects foreign pages and DNS rebinding and API responses
carry `Cache-Control: no-store`, but neither authorizes a non-browser caller —
the Host check trusts loopback names, this machine's hostname, and the address
the dashboard was started on, which is also the address every node ships to.
SECURITY.md states the same boundary for a deployer.

One consequence of proving auth rather than assuming it: the Access tab's live
probes deliberately send an unauthenticated request first, so while the tab is
open each node logs roughly one rejected llama request per minute and one
rejected MCP request per 20-second poll. Expected noise, not a fault.

---

## 4. Lifecycle: K8s-native suspend, never delete

`aureax stop` patches each DaemonSet's `nodeSelector` to a value no node carries.
K8s gracefully terminates the pods while preserving the DaemonSets, Services and
node labels. It then waits for those pods to be gone, **stops the k3d cluster**
and kills the dashboard (a stop driven *from* the dashboard passes
`skip_self_kill` and leaves it serving) — pod deletion is driven by the API
server, so stopping
k3d first would leave a node that has not yet seen the deletion running its
child with no control plane left to deliver it. Expect `kubectl` to stop
answering after a full stop; `aureax start` brings the cluster back up and
patches the selectors back.

Per-node stop/start adds or removes the gating label on that specific K8s node
and does **not** touch the cluster.

**Never** implement "stop" by deleting K8s resources — that forces a full
redeploy to restart.

When a node belongs to a `node_groups` entry, per-node stop/start **fans out to
the whole group**. A group serves one model across all its members: dropping a
worker makes the head abort mid-inference, and dropping the head leaves workers
idling with their share of the weights resident. The dashboard's confirmation
names every member for this reason.

Start sequences three phases in dependency order
(`_start_phase_label_and_wait_pods`): shard workers first, because a head's
llama-server exits immediately if a peer is unreachable at load time; then edges
and heads, which is the model load itself; then any planner, whose MCP host must
be able to reach the edge MCP servers on first connect or it cold-start
crash-loops.

Ahead of all three, on a fleet with groups, `_assert_group_builds_match` asks
every group member over SSH for the sha256 of `llama-server` and `rpc-server`
inside its image (one round trip each, concurrent, 60 s ceiling) and refuses the
start when a member's pair differs from its head's — naming both nodes, both
short hashes and `./scripts/build-on-node.sh <host>`. Before any label is
written, because nothing should be scheduled on a group whose members cannot
complete the RPC handshake, and because past the edge wait a mismatch costs the
full 3015 s startupProbe budget in silence. A member that could not be read is a
warning, not a refusal.

**Why binaries and not image digests.** Every node builds `aureax-client:dev`
*natively* (`scripts/build-on-node.sh`, once per node), so layer timestamps and
base-image state make the pod `imageID`s differ **by construction** across a
coherent group while the compiled binaries are identical, so a digest gate
would refuse every real start. The RPC wire format is locked to the binaries, so the
binaries are the observable; the digest is not.

**The node wait is an allow-list, not a count.** `_split_hostnames_by_role`
resolves each *enabled* target's K8s hostname over SSH and buckets it by the
role in its own `config_file`; only names in those buckets are labelled or
waited for. Labelling "everything that is not control-plane" instead would hand
an edge pod to targets marked `enabled: false` and to any machine that joined
the cluster for other reasons. Two consequences follow from that design:

- A target whose hostname cannot be resolved is carried in its own bucket under
  its `node_id` as a placeholder rather than dropped. Dropping it would shrink
  the expected pod count to match, so the readiness wait would pass with fewer
  nodes than intended and report success. The placeholder never matches a real
  node name, so the wait for that role fails instead — which is the intended
  outcome for a node that cannot be reached at all.
- A planner has no pod-count wait that a placeholder could fail, so an enabled
  planner whose hostname is not a node in the cluster is a **fatal error** with
  its own message. Without that check the planner is left unlabelled and
  unwaited-for while the run reports success.

An unreadable per-node config aborts the whole start with every offender named:
its role is unknown, and defaulting to `edge` would label a shard worker with
the edge key.

### Probe budgets

A `startupProbe` covers model load, not the readiness delay. Pods look
`0/1 Running` while the model loads — roughly a minute or two for a small local
model, and **minutes** for a sharded head, which streams every tensor to its
peers before `/health` answers. Liveness only begins after the startupProbe
first succeeds; without a startupProbe, a liveness probe of 3 × 30 s kills the
container mid-load, forever, as a CrashLoopBackOff with no obvious cause.

**The edge startupProbe budget must cover the supervisor's whole sequential
startup**, which for a sharded head is two waits back to back:

```
  daemonset-edge.yaml startupProbe budget
      =  initialDelaySeconds + failureThreshold × periodSeconds

  must be ≥  node.worker_wait_timeout_seconds      (wait for every RPC peer)
           + model.server.health_timeout_seconds   (then load the model)

  ... taken over EVERY head config the manifest might serve.
```

With the example fleet's values: the running heads set a 300 s worker wait and a
900 s health timeout (1200 s), the frozen rollback head sets 300 s + 2400 s
(2700 s), and the shipped manifest allows `15 + 200 × 15 = 3015 s`.
`tests/test_probe_budget.py` checks the budget against the rollback target as
well as every deployed node, and separately checks that a head keeps the
`${worker_wait_timeout_seconds}` placeholder rather than hardcoding a number.

**These budgets are CEILINGS shared across configs, not descriptions of the
current model.** The same four knobs are sized for the largest head the fleet
might run, and tightening any of them to fit a smaller current model arms a trap
that only springs on a rollback:

| Knob | Where |
| --- | --- |
| startupProbe `failureThreshold` | `k8s/daemonset-edge.yaml` |
| worker memory limit | `k8s/daemonset-shard.yaml` |
| `EDGE_WAIT` health poll | `scripts/update-all.sh` |
| edge-readiness fallback | `scripts/setup-all.sh` |

`update-all.sh`'s ceiling deliberately exceeds the current heads' own give-up
point, and stays under the manifest budget: polling past the startupProbe budget
would report "MCP not responding" for a pod the kubelet has already begun
restarting — a different fault, reported as this one. `setup-all.sh` derives its
wait from the same two config values rather than hardcoding a copy, and falls
back to the worst-case literal only when that derivation fails. The worker
memory limit is doubly loaded: it is also the flash-attention tripwire described
in §3, so lowering it to fit the current model removes that guard as well.

**Shard workers carry a `startupProbe` only — no readiness, no liveness.**
`rpc-server` serves one client with a listen backlog of 1, so it does not accept
probe connections while loading or generating. A liveness probe reads that as
"dead" and SIGTERMs the worker mid-load, crash-looping the whole group. For the
same reason, never test a worker by dialing :50052 from off-node while the head
is attached — check the listener on the node instead. A genuinely dead worker is
still caught: the supervisor exits its container when the child dies, and the
head independently waits for every peer's RPC port before loading.

#### Measurements from one test bench

Only the numbers the budgets above are derived from are kept here — cold model
load, which is what a `health_timeout_seconds` and a startupProbe budget have to
cover. They come from a single bench (ARM64 single-board computers on 1 GbE,
September 2026) serving a 4-bit MoE model across a group, so they are
hardware-specific: use them to understand the *shape* of the cost and
re-measure rather than trusting the values.

| Quantity | Measured |
| --- | --- |
| Cold sharded load, 15.7 GB over 3 workers | 280 s |
| Cold sharded load, 15.7 GB over 7 workers | 272 s |
| Warm reload (worker tensor cache intact) | ~107 s |
| Cold load, 71.7 GB over 7 workers | ~947 s |

Two things fall out of that table:

- **Cold load is flat in peer count**, because the head ships the same total
  bytes either way. Resizing a group invalidates every worker's tensor cache, so
  the first start after a resize is always a cold one.
- **The `~17 s/GB` planning heuristic** quoted in the config comments comes from
  the 15.7 GB / 272 s point and **overestimates at larger sizes**: it predicts
  ~1220 s for the 71.7 GB model, which actually loaded in ~947 s (~13 s/GB).
  Overestimating is the safe direction for a `health_timeout_seconds`, which is
  a deadline rather than a sleep — the supervisor polls every 2 s and returns on
  first success, so overshooting costs nothing and undershooting crash-loops.
  `tests/test_shard_budgets.py` encodes that heuristic as the floor every
  `EDGE_WAIT` has to clear, for the deployed models and the rollback alike.

Per-worker memory figures are deliberately not in this table: the worker
footprint is not one multiplier over the mean (see **Invariants and how each is
enforced** in §3), and the numbers that govern it live in
`k8s/daemonset-shard.yaml`'s own arithmetic, with
`tests/test_shard_budgets.py` holding the manifest to it.

---

## 5. The edge supervisor

`client/entrypoint.py` is the process supervisor inside each edge pod. It:

1. Opens the local log store and builds the log shipper (in `__init__`, so this
   genuinely is first).
2. **Refuses to start without its TLS material.** `_require_tls_files` is the
   first statement of `start()` — before the signal handlers and before the
   role branch — and exits 78 (`EX_CONFIG`) naming the missing file and
   `aureax apply --node <id>`. Every listener and every client on this node is
   TLS-only, so a node whose certificates never arrived is a node that cannot
   serve, not one that serves differently. Shard workers run it too: they ship
   telemetry over TLS.
3. **Branches on `node.role` before any llama-server logic**, because a shard
   worker serves no HTTP and the health wait below would spawn a pointless
   `llama-server` and then exit when `/health` never answers.
   - `shard_worker` → start log shipping (*before* the listen wait, which can
     exit the process — a worker whose `rpc-server` dies during load is exactly
     what the shipper exists for), spawn `rpc-server`, wait for the address it
     actually bound, and only then start telemetry. Block on the child.
   - otherwise → start telemetry and log shipping first, deliberately: a sharded
     head can spend many minutes between here and a healthy `/health`, and
     starting telemetry afterwards means the node reports nothing at all for
     exactly the window an operator is watching. Then reap orphans, then wait for
     RPC peers, spawn `llama-server` and wait for `/health`.
     - **A recorded child that is still alive ends the start immediately**, at
       ERROR and exit 1 through the reporting path — before the port probe and
       before the peer wait, since neither branch below could proceed and
       waiting five minutes only delays the same failure. That covers a child
       whose owning supervisor is still live and one that outlived its SIGKILL:
       under `hostPID` it is still holding the port, so attaching would front a
       process this supervisor has no handle on and spawning would fail to bind.
     - If the reap left no survivors but the llama port is already in use,
       attach to the running server instead of spawning one (see **Orphan
       reaping** below), and health-check it too.
     - **An attached server that enforces no key is refused**, when *this*
       node's config carries one. `_refuse_unauthenticated_attach` GETs `/props`
       with no `Authorization` header once `/health` answers; only a 2xx acts,
       and it exits 1 rather than fronting an ungated inference port under
       `hostNetwork`. A 401, any other status and a transport error all leave
       the attach alone — the check fires on positive evidence that the port is
       open, never on the absence of evidence. `/health` would prove nothing
       there, being auth-exempt; `/props` is gated on the pinned build.
4. **Checks the topology it actually got**, on both the spawn and the attach
   branch: `_verify_rpc_topology` exits non-zero when this node's own resolved
   config has a `tensor_split` whose length differs from its `rpc_servers`. A
   healthy `/health` proves nothing about that, and `aureax generate` cannot see
   a hand-edited `/etc/aureax/aureax-edge.yaml`. See **Invariants and how each
   is enforced** in §3 for why the check stops at the structural comparison.
5. Branches on MCP mode — see **Process Management Architecture** below.

Orphan reaping is not exclusive to the head path — `_spawn_child` reaps for the
binary it is about to start, which covers the worker arm too. What is specific to
the non-worker arm is reaping *before* the port probe, for the reason below.

### Process Management Architecture

The supervisor owns **exactly one child process** and, separately, **at most one
in-process runtime**. The split matters because only the first is a subprocess:

- **The managed child** is the inference binary — `llama-server` on a head or a
  standalone node, `rpc-server` on a shard worker — spawned by `_spawn_child`,
  tracked on `self._child_proc` / `self._child_log_file`, and torn down in
  `stop()`.
- **The MCP runtime runs in-process on the main thread**, never as a child.
  `mcp.client.enabled` takes the MCP-host path (`_start_planner`: connect
  outbound to each `mcp.client.remote_servers` entry — with that entry's own
  bearer key, and refusing any value still containing `${`, which would mean the
  config was never resolved — and idle until SIGTERM or until llama-server
  exits); otherwise `mcp.server.enabled` runs the FastMCP server, which blocks.
  What uvicorn is handed there is `build_app()`, never
  `mcp.streamable_http_app()` directly: the wrapper is the only enforcement
  point for the MCP bearer key (§2, **MCP bearer authentication**), and both
  `:9000` and `:30900` reach this one app. With both disabled the supervisor
  blocks on the child instead of returning, since returning would exit PID 1
  under a healthy llama-server. While the MCP server blocks, a background
  watcher polls the child so a crash still routes through the exit-reporting
  path — `sys.exit` from a non-main thread ends only that thread.

The planner pod's own client is what owns the MCP host runtime in production;
the dashboard's chat service is a separate quick-test path, not the planner.

**The managed child is llama-server OR rpc-server**, spawned by `_spawn_child`.
Its output is **teed to both the log file and container stdout** — `kubectl logs`
and the orchestrator's crash-log fetchers only ever see stdout, so logging solely
to a file reduces every crash to "exited with code N".

**Orphan reaping** requires `hostPID`. After its pod dies, an orphaned child is
re-parented to host init, in a PID namespace a fresh container cannot otherwise
see. The reap runs *before* the supervisor probes whether the llama port is
already in use — an orphan is precisely what holds that port, and llama.cpp
accepts connections while still loading, so probing first would take the "attach"
branch and never reap.

**The reaper targets an ownership record, not a binary name.** Under `hostPID`
this container sees every process on the node, and "kill anything called
`llama-server`" would reach a process nobody here started. So
`client/process_ownership.py` keeps one record per node id and binary at
`/opt/aureax/run/child-<sha>.json` — `<sha>` being a digest of
`<node_id>:<basename>`, so the two DaemonSets never collide. It holds two
identities, the child's and the owning supervisor's, and each is a triple of
**pid, that pid's start time from `/proc/<pid>/stat`, and the host's boot id**.
Start time defeats PID reuse; the boot id makes a record from before a reboot
unusable rather than misleading. `remember()` writes it after a successful
spawn, through a temp file and `os.replace`; `reap()` reads it before one.

Three rules follow, and they are what keeps the reaper safe:

- **An unrecorded process is never signalled.** A missing, stale or unparseable
  record authorizes nothing and `reap()` returns no survivors. That is the
  deliberate tradeoff: an orphan left by a pod that predates this record — a
  first deploy of this code, or a wiped `/opt/aureax` — is not reaped, and the
  port probe then takes the attach branch, which is why that branch
  health-checks and auth-checks what it finds instead of trusting it.
- **A child that is still owned is reported, not killed.** If the recorded
  owner's identity still matches a live process — this supervisor included —
  the pid comes back as a survivor, so the caller refuses to spawn a second
  inference process over one it already owns.
- **The kill is pinned to that exact process.** `os.pidfd_open` holds the pid
  through the identity re-check, the `SIGKILL` and the wait, so a pid recycled
  mid-reap cannot be hit; a kernel without pidfd fails closed and reports the
  pid as surviving. A `spawn_guard` `flock` beside the record serializes the
  whole check-spawn-record sequence across pod incarnations, so two supervisors
  starting at once cannot both decide the port is free. All of this is
  Linux-only and skipped where `/proc/sys/kernel/random/boot_id` does not
  exist.

**Shutdown spends one budget, not two.** `process.shutdown_timeout_seconds` is
split between uvicorn's graceful drain and the wait on the inference child,
because those two legs are serial and `terminationGracePeriodSeconds` in the
DaemonSets is sized from that single figure. Giving each leg the full value
would double the teardown, the kubelet would SIGKILL, and a `hostPID`
llama-server would survive holding `:8080` for the next pod's reaper to find.
The MCP server is also built by hand rather than through
`mcp.run(transport="streamable-http")` for teardown reasons: uvicorn's
`capture_signals()` would replace the supervisor's SIGTERM handler, and FastMCP
never sets `timeout_graceful_shutdown`, so uvicorn would wait unboundedly on the
long-lived SSE streams every MCP client holds.

Every path that exits the process routes through one place that flushes the
child's crash context and ships pending logs first.

---

## 6. Observability: how a node's failure reaches the orchestrator

`kubectl logs` **cannot work in this topology**. It proxies through each node's
kubelet on :10250, which the k3d server container cannot route to for a
LAN-joined node — and a shard worker serves no HTTP at all. Without the path
below, a worker crash is only visible by SSHing to the node.

```
  child stdout ──▶ ChildLogCapture ──▶ LogStore (SQLite, on the node)
                     │  warn/error stored as they happen
                     │  everything else held in a ring buffer,
                     │  flushed only if the child dies
                     ▼
                  LogReporter ─HTTPS─▶ POST /api/v1/logs
                                             │
                                     orchestrator/logs/receiver.py
                                             │  tagged source="node:<id>"
                                             ▼
                                       SystemLogger.emit_many
                                        ├── LogStore (orchestrator)
                                        └── SSE ──▶ dashboard log console
```

- Filtering is what makes this affordable: llama.cpp emits a timing line every
  few tokens, so shipping everything would swamp the orchestrator and put real
  traffic on a LAN that is already latency-bound.
- Log shipping is **independent of `telemetry.enabled`**. The two share an
  address — the `/api/v1/logs` endpoint is derived from the telemetry one, so
  there is a single place to point a node at its orchestrator — but they are
  separate features: a node with `telemetry.enabled: false` still ships logs,
  because the shipper is gated only on `logging.ship_logs`.
- Ingest goes through `SystemLogger`, not straight to the store, so shipped lines
  reach live SSE subscribers instead of sitting on disk until a restart.
- **`SystemLogger.emit` and `emit_many` serialise four steps under one lock
  hold**: the store append, the line-index claim, the ring-buffer append and the
  broadcast. `line_index` is the store's row id, and both consumers — the SSE
  skip in `orchestrator/dashboard/lifecycle_api.py` and the browser's own dedupe
  — drop any line whose index is not above the highest they have seen. Claiming
  an id outside the lock lets two emitters (a lifecycle worker thread and a pod
  watcher on the event loop) reach the fan-out in the reverse of their id order,
  and the lower line is then gone from the live view for good. Persisting first
  also means a crash during fan-out still leaves the row on disk. `emit_many`
  additionally returns the number of rows *durably stored*, because the edge
  shipper advances its cursor on a 200 — "broadcast, but the write failed" must
  not read as success.
- The supervisor's own logger must be named `aureax.<something>`. The store
  handler attaches to `aureax`, and Python's logger hierarchy splits on `.` only
  — a hyphenated name is a sibling, not a child, and its records never reach the
  handler.

Telemetry follows the same shape: `client/telemetry` samples and pushes to
`POST /api/v1/telemetry`, which the dashboard stores and queries. Both endpoints
are derived from the one `${orchestrator_url}` a node's config carries, and both
are verified against the fleet CA — but neither *authenticates the poster*, so
"anything on the LAN can write rows here" stays true and is why retention prunes
on the receiver's own clock (below).

**Every stored line carries two clocks.** `ts` is the originating clock, copied
verbatim out of the shipped batch after a parseability check; `received_at` is
stamped by the receiving store and no poster can move it (`client/log/store.py`,
which both sides use — an `ALTER TABLE` migration adds the column to a store
created before it). Both are load-bearing:

- **Retention prunes on either clock**, as two single-column DELETEs rather than
  one `OR` (an `OR` across two columns can use neither index). A node with no
  RTC that boots years ahead — or anything on the LAN posting to the
  unauthenticated logs endpoint — writes rows a `ts`-only prune could never
  reach, and retention would stop bounding the table.
- **The dashboard's `last_seen` is a MAX over `received_at`, never over the
  node-supplied timestamp** (`orchestrator/dashboard/api.py`), because the
  "online" dot is a freshness signal about arrival and a node with a wrong clock
  would otherwise show as permanently online or permanently stale.
  `aureax status` reads the same column, so the CLI and the Overview tab cannot
  disagree about which nodes are live.

**Every store on both sides opens through `client/sqlite.py::connection`**, which
is the single place WAL journaling and the 5-second busy timeout are set — the
node's telemetry, log and chat stores and the orchestrator's telemetry and log
stores alike. Several threads of one process write these files (a node's
telemetry loop, its hourly log loop and the supervisor's log handler; the
orchestrator's ingest routes and the dashboard's queries), and in the rollback
journal mode a writer locks readers out for the whole transaction, so a
concurrent read waits out the timeout and raises — a dropped log line with no
symptom. The visible consequence is that each `.db` carries `-wal` and `-shm`
sidecars beside it, under `/opt/aureax` on a node and `~/.aureax` on the
orchestrator; they are part of the database, so a copy or a backup takes all
three.

Retention has **two paths for the log store, and they do not fight**. On a node,
the supervisor's hourly log loop prunes to `logging.retention_days` (default 7).
On the orchestrator there is no per-node config to read — `create_app` opens the
store directly — so `LogStore` also prunes on write, to the
`LOG_RETENTION_DAYS = 30` module constant in `client/log/store.py`, matching
`MetricsStorage`'s own retention. An edge node runs both, but a DELETE of rows
another DELETE already removed is a no-op and an hourly guard means at most one
runs per hour per process. The node's loop is the tighter of the two, so it keeps
deciding retention there; the 30-day one is the backstop on the orchestrator.

### Per-node detail: one document, no new fetches

`GET /api/nodes/{node_id}` backs the dashboard's per-node view (`#node/<id>`,
opened from an Overview telemetry card or a Lifecycle row). It joins four
sources the orchestrator already holds, so an open detail tab costs no kubectl,
no SSH and no call to the node:

- **identity**, from the fleet file and the node's resolved config through the
  mtime-keyed loaders in `orchestrator/config/cache.py`: host, display name,
  tags, role, group (head and workers in `--rpc` order), model name /
  quantization / file / context / GPU layers, MCP on or off. A value still
  spelled `${...}` — a template that was never generated — is reported as null
  rather than shown as a value.
- **k8s**, from the same TTL-cached `kubectl get nodes` / `get pods` payloads
  the Overview polls. A fleet node id is not a Kubernetes node name, and the
  only exact resolver is an SSH `hostname`, so the route matches instead: the
  K3s node whose `InternalIP` equals the target's `host`, else the pods whose
  `status.hostIP` does (both DaemonSets are `hostNetwork`), else a node named
  after the id or the host; `matched_by` says which rule fired and the view
  shows it. Beyond what the Overview renders it surfaces the pod's start time,
  each container's start and last-exit times, the node's kernel / OS /
  architecture / kubelet / runtime, CPU and memory capacity, and any
  `*Pressure` condition. No image digests: native per-node builds make them
  differ by construction (§4).
- **metrics**: the newest value per metric and `last_seen` come from the same
  fleet listing `/api/nodes` serves, so the online rule is shared by
  construction, plus one hour of samples per metric from
  `MetricsQueries.get_metric_history`. The window filters on the node's own
  `timestamp` — the clock that orders samples against each other — while
  `last_seen` is `received_at`; a node whose clock runs behind therefore shows
  an empty series beside a green dot, a contradiction an operator can act on
  rather than a plausible line drawn at the wrong times. The cutoff is built in
  Python for the reason given above, and the existing `(node_id, timestamp)`
  index serves the scan.
- **logs**: the newest 100 rows tagged `source = "node:<id>"`, via
  `LogStore.recent(source=...)`. That reader is why the store now carries an
  `entries_source_id (source, id DESC)` index: a healthy node ships a few lines
  a day, so the unindexed form walked 30 days of rows backwards on every poll.
  The store is shared with the nodes, so the b-tree write lands there too — at
  warn/error volume, negligible.

Each block carries its own `status`, so a stopped cluster or an unopened store
blanks one card instead of failing the page. The sparklines are inline SVG
polylines painted from `style.css` — the dashboard ships no build step and no
charting library, and its CSP has no `unsafe-inline` — with the sample index on
the x axis, since a node-supplied timestamp is only length-checked on ingest.
The Overview card shows a fixed headline set (`HEADLINE_METRICS` in
`static/utils.js`) and a "+N more" line, so a metric added node-side lands in
the detail view rather than lengthening every card.

The collector's set grew with this view: `load_1m`, `uptime_seconds`,
`swap_used_percent` (only where swap exists), and the `kv_cache_tokens` /
`tokens_predicted_total` levels that were already in the `/metrics` scrape. Like
every metric, each is one guarded read that publishes nothing rather than a
placeholder when its source is missing. They reach a node with the next code
push (`scripts/update-all.sh`); shard workers, which `apply` never restarts,
pick the new collector up on the next `aureax stop && aureax start`.

---

## 7. Gotchas

- **K8s directory resolution**: `Deployer.k8s_dir = config_path.parent.parent / "k8s"`
  — the `k8s/` directory is at the project root, not next to `configs/`.
- **`node.role` literals live in `orchestrator/config/schema.py`**
  (`HEAD_ROLE`, `SHARD_WORKER_ROLE`, `PLANNER_ROLE`). Both Dockerfiles copy that
  file into the edge image, so the client imports the same constants rather than
  redeclaring them.
- **MCP service session affinity**: `k8s/service-mcp.yaml` must keep
  `externalTrafficPolicy: Local`. Without it, session IDs break because POSTs and
  background GETs can hit different pods.
- **`parse-config.py model_for_node <N>` has a different argv layout** from the
  other subcommands (index positional, config path optional third arg), and the
  shell scripts call it that way. Don't refactor without preserving it.
  `parse-config.py target <N>` shares that layout and indexes the same filtered
  target list; it is an operator-facing debugging aid with no caller in this
  repository, so a refactor that preserves only `target` breaks the one that is
  actually wired up.
- **`parse-config.py` quotes every shell-array element.** Several arrays are
  indexed positionally in lockstep with `NODE_HOSTS`, and some values are
  legitimately empty (a target with no `config_file` has no role). Bash discards
  an unquoted empty word, which shortens the array and shifts every later index
  onto the wrong node.
- **The manifests hardcode their namespace.** `kubernetes.namespace` is read at
  runtime by stop/start, so the two must agree; `apply_manifests()` refuses to
  deploy when they don't, rather than creating a stack that cannot be stopped.
- **K3s agents follow the server's version at join.** `Deployer.join_nodes`
  reads `kubectl.server_version()` and passes it to `join-node.sh --k3s-version`;
  `scripts/setup-all.sh` reads the same `gitVersion` from the API server's
  `/version` and passes it the same way. Without the pin the installer takes
  latest stable, so nodes joined weeks apart end up with kubelets newer than the
  API server — outside the supported skew, surfacing much later as an
  unexplained NotReady node. A cluster that cannot answer `/version` gets an
  unpinned install, and `join-node.sh` warns on that path.
- **The K3s join token never travels in argv.** `join-node.sh` takes it via
  `--token-file` or `--token-stdin`. An argv word is public twice over:
  `/proc/<pid>/cmdline` is world-readable (both DaemonSets set `hostPID: true`),
  and `sudo` copies the whole command line into the node's auth log, where it
  outlives the join. The token is the one credential that lets any machine join
  the cluster. There is no positional form: a token in that position is refused
  with the usage line, rather than accepted for a hand-run.
  The orchestrator's own join path stages the token as a 0600 file and
  sweeps it even when the join times out (`orchestrator/deploy/k3s_agent.py`).
- **The dashboard's chat clients follow a target that moves.** The per-target
  llama client and MCP pool are cached by node id, which survives a change to
  either the API key or `ssh.targets[].host`, so both are re-checked on every
  use and updated *in place* — cached engines hold their own reference to the
  client object. Without the address check, every chat on that target keeps
  dialling the old host for the life of the process. The pool remembers each
  member as `(url, key)` rather than a bare URL for the same reason: a rotated
  MCP key with an unchanged address has to force a reconnect, or every call 401s
  until the process restarts (`orchestrator/dashboard/chat_service.py`).
- **`aureax generate` writes under `$HOME`, and a test that reaches it must
  redirect `$HOME`.** Generation mints keys into `~/.aureax/keys`, TLS material
  into `~/.aureax/tls` and the `node_groups` baseline into `~/.aureax/state`, so
  any test that calls `generate_all_edge_configs`, `Deployer.generate_configs`
  or `POST /api/config/regenerate` needs the `tmp_aureax_home` fixture or it
  edits the developer's real fleet credentials. `generate_edge_config` alone —
  the per-poll path the Access tab uses — deliberately mints nothing.
- **A chat session's history budget is derived from the target's
  `context_size`**, not from a fleet-wide constant: targets range from a sharded
  head at 131072 down to a planner at 16384 and the standalone edge template at
  4096, and a budget larger than the context never trims, so the session
  overflows and every later turn fails identically.

---

## 8. Hardware assumptions

**This codebase is hardware-agnostic.** Node references in code and comments say
"node" or "edge node", never a specific board. Shell variables use the `NODE_*`
prefix. The client runs on ARM64 or x86_64 Linux, and the test bench happening to
use small ARM64 single-board computers is not something anything depends on.

The **orchestrator** is the one place with a platform assumption: the current
tooling targets macOS. `orchestrator/cluster/k3d.py` detects the LAN IP with
`ipconfig getifaddr` and BSD `route -n get` and returns nothing elsewhere;
`scripts/setup-all.sh` installs `k3d` and `kubectl` through Homebrew; and the
cluster itself is k3d on Docker Desktop.

Where a measurement is quoted in this document it names the bench it came from
(see **Measurements from one test bench** in §4), because these numbers are
hardware-specific and should be re-measured rather than trusted.

---

## Further reading

- [`README.md`](README.md) — Requirements, Quick start, Configuration,
  Operating, Models, Connecting clients, Testing, Sharded model groups, Project
  structure and License.
- `orchestrator/config/schema.py` — the authoritative config schema, commented.
- `configs/aureax-network.example.yaml` — the fleet definition to copy and edit,
  commented field by field.
- `scripts/test-all.sh` — eight sections. Sections 0 (pytest) and 1
  (orchestrator) need no infrastructure, though section 0 skips unless the dev
  dependency group is installed (`uv sync --group dev`); the other six are gated
  on a reachable cluster, dashboard, MCP endpoint, planner target, non-empty
  `node_groups` or an auth-enabled node with a local key.
