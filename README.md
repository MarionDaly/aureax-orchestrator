# Aureax Prototype

Aureax deploys and operates local LLM inference on a fleet of Linux edge nodes from one
orchestrator machine. It reaches each node over SSH, joins it to a K3s cluster, pushes that node's
own resolved configuration, and runs two workloads there: a
[llama.cpp](https://github.com/ggml-org/llama.cpp) HTTP server exposing an OpenAI-compatible API,
and an [MCP](https://modelcontextprotocol.io/) server exposing the node's inference, sensors,
health and telemetry as tools. A web dashboard shows the fleet, edits configs, drives start/stop
and chats with any node. Several nodes can also be grouped to serve **one** model cooperatively
over llama.cpp's RPC backend, for a model that fits on none of them individually.

It is for people running models on hardware they own who want one command to describe the fleet and
one to deploy it. Adding a node is two edits to YAML and no code.

> **Project status: prototype.** Pre-1.0, developed against a small test bench. The orchestrator
> assumes **macOS today** — K3s through k3d on Docker Desktop, `k3d` and `kubectl` from Homebrew,
> its own LAN address from `ipconfig getifaddr` and BSD `route -n get`
> (`orchestrator/cluster/k3d.py::mac_lan_ip`). The edge client runs on ARM64 or x86_64 Linux.

> **Security model: trusted operator system and LAN, with transport-specific protections.**

| Surface | Transport | Access model |
|---|---|---|
| Dashboard on loopback (`:7001`) | HTTP | Trusted local system; no login or client certificate |
| Dashboard LAN ingest/administration (`:7001`) | HTTPS (fleet CA) | Trusted LAN clients; no dashboard credential |
| llama-server and MCP | HTTPS | Separate per-node bearer keys |
| Worker RPC (`:50052`) | Plain TCP | Host firewall permits the group head and local probes only |

The dashboard host and clients allowed to reach its LAN listener must be fully trusted.
Host/Origin checks block foreign browser requests and DNS rebinding; they do not authenticate
programmatic clients. Dashboard API responses use `Cache-Control: no-store`.

See [SECURITY.md](SECURITY.md) for the supported boundary and private reporting.


---

## Requirements

**Orchestrator** (macOS today): [Docker Desktop](https://www.docker.com/products/docker-desktop/)
running; [Homebrew](https://brew.sh/), plus `k3d` and `kubectl`, which `setup-all.sh` installs if
missing; [uv](https://docs.astral.sh/uv/), the one tool it will not; Python 3.11+, which `uv`
provisions. Use `uv sync` / `uv run`, not `pip` or a hand-made `venv`.

**Edge nodes** run Linux on ARM64 or x86_64 and need SSH from the orchestrator — a key, or a
password (see [Fleet file](#fleet-file)) — plus passwordless `sudo`, because every step after the
first runs `sudo` over SSH where a password prompt has no tty. Each needs RAM for what it serves
(its model plus KV cache, or in a group its share of the weights plus the group's context —
derivation in
[ARCHITECTURE.md → "Invariants and how each is enforced"](ARCHITECTURE.md#invariants-and-how-each-is-enforced))
and disk for the GGUF if it loads one; workers hold no model file but keep a tensor cache under
`/opt/aureax/rpc-cache`. Every peer in a group must run the byte-identical llama.cpp build: the RPC
wire format is version-locked, and `aureax start` refuses a group whose members' `llama-server` /
`rpc-server` checksums disagree.

---

## Quick start

```bash
uv sync --group dev                      # pytest and ruff live in the dev group
uv run aureax init                       # ~/.aureax: telemetry/, and ssh-keys/, keys/,
                                         # configs/, tls/, state/ at mode 0700
cp configs/aureax-network.example.yaml configs/aureax-network.yaml
$EDITOR configs/aureax-network.yaml      # describe your fleet
uv run aureax nodes                      # check the orchestrator agrees with you

# Provision everything: cgroups, K3s, model download, image build, deploy. Also mints
# the fleet CA and issues every node's certificate.
./scripts/setup-all.sh

# Trust the fleet CA once, so https:// URLs open without a browser warning.
# Not needed for the dashboard from this machine — see "The dashboard" below.
security add-trusted-cert -r trustRoot \
    -k ~/Library/Keychains/login.keychain-db ~/.aureax/tls/ca.pem

uv run aureax status --verbose
uv run aureax dashboard                  # prints both URLs to open
```

`configs/aureax-network.yaml` is **gitignored**: it carries your nodes' addresses and, optionally,
an SSH password. The shipped example uses the `192.0.2.0/24` documentation range, so it describes
no real machine. Commands that need a fleet load that file on first use and name the example and
the copy command when it is missing; `aureax init` and `--help` work before it exists, as do the
two ungated test sections. `-c/--config` points at a copy elsewhere.

The trust step covers your login account only; for machine-wide trust use `sudo security
add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ~/.aureax/tls/ca.pem`.
Trusting it as a *root* is bounded: the CA is name-constrained to the addresses your fleet uses and
can certify nothing else ([TLS](#tls)). Command-line clients need no keychain — point them at the
CA file instead.

**Upgrading an existing fleet rather than provisioning one?** Rebuild both container images before
this version's first `aureax deploy` — `./scripts/update-all.sh` does it in the right order,
`./scripts/build-on-node.sh <host>` does one node. llama-server's TLS is a *build* property and the
`scheme: HTTPS` probes in `k8s/daemonset-edge.yaml` are the same change, so a pre-TLS image under
the new manifest crash-loops every head with "the server is built without SSL support" once the
startupProbe has burned its whole budget, with nothing naming the cause.

---

## Configuration

### Per-node configs

Each SSH target in `configs/aureax-network.yaml` names its own `config_file` under `configs/`. The
orchestrator resolves that file's placeholders per target and pushes the result to that node alone:
`${node_id}` `${display_name}` `${tags}` `${orchestrator_ip}` `${orchestrator_url}` `${group_id}`
`${rpc_servers}` `${worker_wait_timeout_seconds}` `${mcp_endpoints}` `${api_key}` `${mcp_api_key}`.
Certificate paths are not a placeholder: the generator stamps a `tls:` block naming
`/etc/aureax/tls/{ca.pem,node.pem,node.key}` onto every resolved config, so no template can forget
it. Adding a node is an `ssh.targets` entry with a `config_file` plus the file it names — no code
changes, N nodes dynamically. Targets differing only by placeholder-resolved values may share one
file; give a node its own the moment it needs a real override.

| Shipped config file | What it is |
|------|------------|
| `aureax-edge.yaml` | Standalone node: one small model, llama-server + MCP server |
| `aureax-planner.yaml` | Cognitive planner: an MCP *host* that drives other nodes' MCP servers |
| `aureax-shard-head-35b.yaml` | Head of a sharded group |
| `aureax-shard-worker-35b.yaml` | Worker in a sharded group |
| `aureax-shard-head.yaml`, `aureax-shard-worker.yaml` | A previous group model's pair, kept unmodified as a rollback target |

`orchestrator/config/schema.py` is the authoritative schema, and it is commented — read it rather
than guessing at a field.

### Fleet file

```yaml
ssh:
  key_store: "~/.ssh/"                        # relative key_file names resolve against this
  bootstrap_public_key: "~/.ssh/id_ed25519.pub"
  targets:
    - node_id: "edge-01"                      # repeat per node
      host: 192.0.2.11
      user: "aureax"                          # default if omitted
      key_file: "id_ed25519"                  # relative to key_store; or password_env: "VAR"
      config_file: "aureax-edge.yaml"

node_groups:                                  # omit entirely unless you want sharding
  - name: "qwen35b-a"
    head: "shard-head"
    workers: ["shard-a1", "shard-a2", "shard-a3"]   # RPC0, RPC1, … — APPEND, never reorder
    rpc_port: 50052
    include_head_as_worker: false
    worker_wait_timeout_seconds: 300
```

Set `enabled: false` to keep a node documented without deploying to it: disabled targets still
validate and are skipped by every SSH pass, label pass, pod-count wait and generation run.
`node_groups` is a list, so a fleet can run several groups at once — see
[Sharded model groups](#sharded-model-groups).

**SSH authentication.** A target authenticates with a key (`key_file`) or, when that is all it has,
a password — `password_env` names an environment variable read at connect time, keeping the secret
out of git. On the first password-authenticated connect the orchestrator installs
`ssh.bootstrap_public_key` into the node's `authorized_keys` and clears the group-write bit on its
home directory, because sshd's `StrictModes` refuses keys under a group-writable `$HOME` and that
presents as "key installed, still denied". From then on `key_file` takes over, which is what the
bash scripts need — they only ever speak `ssh -i`. The fleet file itself never reaches a node:
`.dockerignore` and the code rsync exclude `configs/`, because a build context is unpacked on
whichever node is building
([ARCHITECTURE.md → "SSH authentication"](ARCHITECTURE.md#ssh-authentication)).

**Host keys** are trusted on first use and remembered in `~/.aureax/known_hosts`. A key that later
*changes* is logged at error level with both SHA256 fingerprints, replaced in that file, and
connected to once more: nodes get wiped and re-imaged, and hand-editing a trust store afterwards
would be a manual step, which this project treats as a bug. The loud line is the deliverable, not a
veto — **if you did not just re-image that node, stop and investigate.** The bash scripts share the
same store and behave the same way: they pass `StrictHostKeyChecking=no`, so a changed key prints
OpenSSH's "REMOTE HOST IDENTIFICATION HAS CHANGED" warning and the run continues on key auth, and
the next Python connect rewrites the entry.

To check a fingerprint *before* any connection is made, read it at the node's console
(`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`) and record it yourself:

```bash
uv run aureax trust-host-key edge-01 --fingerprint SHA256:THE_VERIFIED_FINGERPRINT
```

That handshake sends no password, no client key and no command, and replaces the stored key only on
an exact match. It is optional — nothing requires it after a re-image.

---

## Operating

### CLI

```bash
uv run aureax init                    # create ~/.aureax (see Quick start)
uv run aureax nodes                   # hosts, config files, tags, both auth states
uv run aureax status [--verbose]      # SSH + cluster health; --verbose adds K3s + telemetry
uv run aureax dashboard [--orchestrator-ip <IP>]   # serve it; prints both URLs

uv run aureax generate --orchestrator-ip <IP>                   # configs, keys, certificates
uv run aureax deploy --orchestrator-ip <IP> [--setup-cluster]   # push them; also joins nodes
uv run aureax apply --orchestrator-ip <IP> --node edge-01 [--rotate-key] [--no-restart]
uv run aureax apply --orchestrator-ip <IP> --all --wait

uv run aureax stop [--node edge-01]   # suspend workloads; bare form also stops the cluster
uv run aureax start [--node edge-01]

uv run aureax tls status              # CA, its name constraints, every leaf's expiry and SANs
uv run aureax tls ca-path             # just the path, for `curl --cacert "$(...)"`
uv run aureax tls regenerate [--ca --orchestrator-ip <IP>] [--node edge-01]

uv run aureax group list              # members, endpoints and auth state per group
uv run aureax group doctor [--name qwen35b-a]

uv run aureax bootstrap-keys          # install the public key on every password-auth node
uv run aureax heal-nodes              # rewrite a stale K3S_URL after the LAN IP changed
uv run aureax trust-host-key <node_id> --fingerprint SHA256:...   # optional, see Host keys
```

- **`apply` needs a target**: `--node <id>` (repeatable) or an explicit `--all`, mutually exclusive,
  and the bare form exits 2. Fleet-wide is opt-in because `--all` restarts every head at once and,
  with `--rotate-key`, rotates every key at once. Applying to any group member pushes to that
  member's whole group.
- **`--wait` is two passes**: every targeted pod is deleted, and only then are the replacements
  polled, each on its own budget of that node's `worker_wait_timeout_seconds` plus
  `health_timeout_seconds` plus 120s. So `--all --wait` reloads every head at once and takes the
  whole fleet's llama and MCP endpoints down for one shared multi-minute window, rather than
  reloading node by node. Without `--wait` the deletes go out back to back either way. It exits
  non-zero when a replacement never becomes Ready, and when the poll was skipped because the
  pre-delete pod snapshot came back empty — a gate that never ran is a failed gate.
- **`status` is not read-only**: it installs the bootstrap public key on every reachable
  password-auth node, exactly as `bootstrap-keys` does.
- **`stop` and `start` are group-atomic.** `--node` on a group member takes the whole group, which
  serves one model across all its members: dropping a worker makes the head abort mid-inference,
  dropping the head leaves workers idling with their share of the weights resident. `stop` suspends
  rather than deletes — it patches each DaemonSet's `nodeSelector`, waits for the pods to go, then
  stops the cluster, so `kubectl` stops answering. `start` pushes no configs; only `deploy` and
  `apply` reconcile a config or role change.
- **`--accept-worker-reorder`** on `generate`, `deploy` and `apply` overrides a refused change to a
  group's worker order (see [Sharded model groups](#sharded-model-groups)).

### Scripts

- `setup-all.sh` — provisioning from scratch through to a deploy; `MAC_IP=<addr>` overrides
  LAN-IP detection.
- `update-all.sh` — fast code push: rsync and rebuild on each node, then `deploy` and `apply`.
- `stop-all.sh` — a wrapper around `uv run aureax stop`; arguments are forwarded.
- `reset-all.sh` — destructive wipe of every node and the cluster; `--keep-models` keeps GGUFs.
- `test-all.sh` — the full suite, see [Testing](#testing).

Everything else in `scripts/` is a helper those five call — model download, single-node build, k3d
setup, benchmarking, `join-node.sh`, and `lib/common.sh` — each with its usage in its header.

`join-node.sh` takes the K3s token via `--token-stdin` or `--token-file <path>`, never as an argv
word: argv is visible in `ps` on both machines, and `sudo` writes the command line into the node's
auth log. There is no positional form — a token in that position is refused. A `LLAMA_CPP_VERSION`
bump in either Dockerfile needs `aureax stop && aureax start` rather than `update-all.sh`, because
`apply` never restarts shard workers — head and peers would sit on different builds and the group
would hang at the RPC handshake.

### Installed packages

A wheel carries the two Python packages, `orchestrator` and `client`, plus the dashboard UI and the
operational resources an operator runs — configs, `k8s/`, `docker/` and `scripts/`. To operate from
an installed package rather than a checkout, create a writable workspace, enter it and configure the
example fleet:

```sh
aureax init --workspace ./my-fleet
cd my-fleet
cp configs/aureax-network.example.yaml configs/aureax-network.yaml
```

The workspace holds public templates and code only; it copies no private configuration from another
checkout, and it does not carry the two publication guards (one needs a Git repository, the other an
installed wheel). Run fleet commands from inside it — the working directory is searched first, so a
command run there reads and writes that workspace's files. Use a Git checkout for development and
tests; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Models

**Model selection is per node.** Each node's own config names what it serves, in `model.filename`;
`model_registry` in the fleet definition selects nothing, it only says where a filename is fetched
from and how big it should be. `./scripts/download-model.sh` downloads each unique model once to
`models/` and SCPs it to the nodes that declare it. Only a head or standalone node holds a GGUF;
workers receive tensors over the network. `target_roles` on a registry entry is recorded but read
by nothing yet.

- `size_gb` is **decimal GB (10⁹)**, and `download-model.sh` judges a download complete at 95% of
  it — a wrong value either re-downloads a complete file forever or accepts a truncated GGUF.
- **A registry entry can pin the bytes it fetches.** Declare `sha256` for a single-file GGUF, or
  `sha256_parts` — one lowercase-hex digest per split part, in part order — and `download-model.sh`
  verifies every declared digest with `shasum -a 256`, for freshly fetched parts and for parts
  already on disk, before any file is pushed to a node. A mismatched part is deleted and the run
  exits non-zero. `shasum -a 256 models/<file>` prints a value in exactly the form these fields
  accept, and the shape is checked when the fleet file is parsed (64 lowercase hex characters, one
  per part, the two fields mutually exclusive) so a malformed digest is refused before any
  multi-gigabyte fetch. Declaring no digest stays supported and keeps the size checks; the run then
  prints `no digest declared for <model>; size check only` once per model.
- **`revision` pins *which* bytes.** It is the HuggingFace ref the download resolves
  (`resolve/<revision>`); unset it is `main`, which upstream is free to move, so two provisions of
  one fleet can otherwise fetch different files under the same URL.
- **A large GGUF arrives split.** Above roughly 50 GB, HuggingFace publishes
  `name-00001-of-00003.gguf` under a per-quant subdirectory. Point `filename` at the local basename
  of part 1 — llama.cpp opens the rest itself — and let `huggingface_file` carry the repo path
  including that directory; `gguf_part_names()` in `orchestrator/config/schema.py` derives every
  part from the suffix, and all of them must reach the node.

---

## Connecting clients

Every endpoint below is HTTPS, served with a certificate the fleet CA signed. Start with
[TLS](#tls) if a client of yours does not trust it yet.

### OpenAI-compatible API

Any OpenAI-compatible client — aider, Continue, Open WebUI, the `openai` SDK — drives a node
directly:

```bash
export OPENAI_BASE_URL=https://<NODE_IP>:30800/v1
export OPENAI_API_KEY=$(cat ~/.aureax/keys/<node_id>.key)
```

Stop at `/v1`: clients append `chat/completions` themselves, so the full path gives a 404.
`OPENAI_API_BASE` is the older LiteLLM/aider-era spelling, and some tools read only that one. The
**model id** is whatever `alias` the node sets under `model.server` — without one, llama-server
reports the GGUF's full path and clients must use that verbatim; an alias may not contain a comma,
because llama.cpp parses `--alias` as a comma-separated list. Ask the node with `curl --cacert
~/.aureax/tls/ca.pem https://<NODE_IP>:30800/v1/models`. llama.cpp's own chat UI is at
`https://<NODE_IP>:30800/`, and the same server is reachable as `:8080` directly on the node,
because the pod runs with `hostNetwork` — `:30800` is not the only door.

### llama-server API keys

Auth for a node is **one line** in that node's config file, under `model.server`:

```yaml
api_key: "${api_key}"
```

There is no separate enabled flag — the line is the switch, whether you edit it by hand or from the
dashboard's **Access** tab, which round-trips the same file. Toggling a *shared* config file
toggles every node using it, though each still gets its own key. Keys are per-node, generated by
`aureax generate` into `~/.aureax/keys/<node_id>.key` (`aureax_sk_` plus 32 CSPRNG bytes, mode
0600, untracked), so a wiped node regains its key from the orchestrator; rotation keeps one
`.key.prev` for a client still holding the old key while an apply is pending
(`orchestrator/config/keystore.py`).

Turning auth on is that line, then `aureax generate` (which creates the key file if absent) and
`aureax apply --node <node_id>` (which pushes the config and restarts the pod). `--rotate-key`
replaces the key in the same step: rotating without applying would leave the server enforcing the
old key while every client sends the new one, which is why rotation is a flag on `apply` rather
than its own command. Delete the line and repeat both commands to turn auth off.

**Exempt endpoints.** On the pinned build (`b10375`, `ARG LLAMA_CPP_VERSION` in both Dockerfiles),
`/health`, `/v1/health`, `/models`, `/v1/models`, `/` and the web-UI assets answer without a key —
the Kubernetes probes and a client's model discovery depend on that. Everything else, `/metrics`
included, returns 401 without `Authorization: Bearer <key>`. The list is build-specific: upstream
narrowed the two `/models` routes right after `b10375`, so a `LLAMA_CPP_VERSION` bump can start
requiring a key there, and `test-all.sh` section 7 checks unauthenticated `/v1/models` as the
tripwire. Details in
[ARCHITECTURE.md → "llama-server API-key authentication"](ARCHITECTURE.md#llama-server-api-key-authentication).

### MCP

Each node's MCP server speaks streamable-http at `https://<NODE_IP>:30900/mcp`, and any
MCP-compatible host can connect directly. Its tools are `run_inference`, `read_sensor`,
`get_health`, `get_sensor_status`, `get_metrics` and `report_anomaly` — `aureax-harness tools
<NODE>` prints their schemas. They clamp what one caller can cost the pod: max output tokens, the
metrics window, anomaly field lengths (`client/mcp/server.py`).

### MCP keys

The MCP port has its own bearer key, set the same way — one line under `mcp.server`,
`api_key: "${mcp_api_key}"`. Keys land in `~/.aureax/keys/<node_id>.mcp.key`, share the
`aureax_sk_` prefix, and rotate with the same `aureax apply --rotate-key`, which rotates a node's
llama and MCP keys together. `/health` stays exempt with a key set, because the deploy scripts and
the test suite wait on it unauthenticated. Holding only the MCP key still reaches key-protected
inference — `run_inference` proxies to llama-server with the node's own llama key attached — so
this is one door, not a second factor. A planner's `${mcp_endpoints}` block carries each peer's MCP
key, which is why `apply` fans out to every consumer of it when a key changes. See
[ARCHITECTURE.md → "MCP bearer authentication"](ARCHITECTURE.md#mcp-bearer-authentication).

### TLS

`aureax generate` mints the fleet CA on first run and issues a certificate for the orchestrator and
every enabled node: `~/.aureax/tls/ca.key` (the single fleet-wide secret, mode 0600), `ca.pem` (the
certificate every client trusts), and one `<name>.pem`/`.key` leaf per node plus `orchestrator.pem`
for the dashboard. `aureax deploy` and `aureax apply` push `ca.pem`, `node.pem` and `node.key` to
`/etc/aureax/tls` alongside the node's config, and the supervisor refuses to start without all
three — it exits 78 (`EX_CONFIG`) rather than opening a plaintext listener. `generate` reissues a
leaf whenever it goes stale (a changed address, an approaching expiry, a CA that no longer
matches); `aureax tls status` reports the CA, its name constraints and every leaf, and
`aureax tls regenerate` forces a reissue. Point command-line clients at the CA file:

```bash
curl --cacert "$(uv run aureax tls ca-path)" https://<NODE_IP>:30800/v1/models
export AUREAX_CA=~/.aureax/tls/ca.pem       # read by the scripts and the harness
```

For browsers, trust the CA once — see [Quick start](#quick-start).

**The CA is name-constrained**, which is what makes trusting it as a root reasonable: it permits
only the private, loopback and link-local ranges a fleet lives in, plus any fleet address or
hostname outside them at the moment it was minted. macOS, Chrome and Firefox enforce that, so a
leaked `ca.key` cannot certify your bank or your SSO host — the blast radius of the one fleet-wide
secret is the fleet. A node addressed outside that set is refused by `aureax generate`; admitting it
means a new root, `aureax tls regenerate --ca --orchestrator-ip <IP>`, and the rotation below.

**The orchestrator's leaf accumulates addresses**, because nodes ship telemetry to the
`--orchestrator-ip` they were generated against while `aureax dashboard` mints for the address it
detects. It carries the union rather than whichever ran last, since dropping the other name ends
fleet ingest silently; `aureax tls regenerate --node orchestrator --orchestrator-ip <IP>` prunes it.

**Rotating the CA is five steps, and stopping early splits the fleet.** Nothing rotates it by
itself: a rotation part-way through an `apply` leaves some nodes on the new root and some on the
old, with no way to tell which is which.

```bash
uv run aureax tls regenerate --ca --orchestrator-ip <IP>   # 1. new root, new leaves for everyone
uv run aureax apply --all --orchestrator-ip <IP>           # 2. push both; restarts heads/standalone
uv run aureax stop && uv run aureax start                  # 3. workers and every shipper reload
```

Step 1 prints the remaining four as it finishes: **4.** restart `aureax dashboard`, which otherwise
keeps serving the leaf the *old* CA signed, and **5.** re-trust `ca.pem` in the browser. Step 3 is
not optional — `apply` never restarts a shard worker, and every node's telemetry and log shippers
hold the CA they read at process start, so skipping it leaves those nodes verifying against the old
root and gone from the Overview tab with only a debug line. (A first deploy of this version also
needs both images rebuilt, above.)

### The testing harness

`aureax-harness` is an interactive CLI for a node's MCP server — send prompts, read sensors, call
tools, inspect schemas — from the orchestrator.

```bash
uv run aureax-harness ping <NODE>
uv run aureax-harness infer <NODE> "What is 2+2?"
uv run aureax-harness sensor <NODE> --type temperature
uv run aureax-harness call <NODE> read_sensor --arg sensor_type=humidity
uv run aureax-harness connect <NODE>                    # interactive REPL
```

`NODE` is an address **or an enabled node id**. Given an id, the harness reads that node's address
from the fleet file and its MCP key from `~/.aureax/keys/<node-id>.mcp.key`, so nothing has to be
typed; given an address, supply the key with `--api-key`, `--key-file` or `$AUREAX_MCP_API_KEY`, in
that order of precedence — `--api-key` lands in shell history and in `ps`, so prefer the other two.
`--config` points at another fleet file, `--ca` at another CA bundle (default `$AUREAX_CA`, else
`~/.aureax/tls/ca.pem`); a certificate the CA did not sign is reported as a TLS verification failure
naming the CA and the node, not as an unreachable endpoint. `--port/-p` is the MCP NodePort (default
`30900`), `call`/`infer`/`connect` take `--timeout` (seconds of silence tolerated on the stream),
`infer` adds `--max-tokens/-m` and `--temperature/-t`, and `--arg` is strictly `key=value`. Every
command exits non-zero on failure with a one-line cause; `AUREAX_HARNESS_TRACEBACK=1` re-raises.

### The dashboard

`uv run aureax dashboard` serves the same dashboard on two listeners, in one process, on one port:

- **From the orchestrator itself: `http://localhost:7001/`.** Plain HTTP, bound to `127.0.0.1` and
  `::1` only. No certificate, no trust store, no padlock — the machine that runs the dashboard is a
  trusted machine, and loopback never reaches the LAN.
- **From the fleet and from any other machine: `https://192.0.2.1:7001/`,** at the orchestrator's
  own LAN address. Trust the fleet CA (`~/.aureax/tls/ca.pem`) in that browser first — the keychain
  command in [Quick start](#quick-start). Firefox keeps its own certificate store, so if the
  keychain trust does not reach it, import the same file under Settings → Privacy & Security →
  Certificates → View Certificates → Authorities → Import and tick "Trust this CA to identify
  websites".

Both URLs are printed at startup. The HTTPS listener binds that one LAN address rather than every
interface, because the loopback listener needs the same port — so a Mac with several interfaces
serves the fleet on exactly the address its certificate names, which is the address in every node's
config. It mints that certificate for the LAN address it detects, takes `--orchestrator-ip` when
that detection is wrong or unavailable, and refuses to start rather than serve a certificate no node
can verify. If the port is busy it says so and names the process holding it. Tabs:
**Overview** (fleet state and telemetry; a telemetry card opens a per-node view at `#node/<node_id>` —
identity, its K3s node and pods, an hour of every metric, and the log lines it shipped — served by
`GET /api/nodes/{node_id}`), **Lifecycle** (start/stop, streamed setup logs),
**Access** (per-node endpoints, masked llama and MCP keys with reveal/copy/rotate, and a Save +
Apply flow), **Chat** (talk to any node), **Configs** (view and edit config files). Four routes
return a plaintext key — `/api/access/{node_id}/key` and `/mcp-key`, plus `/previous` for each —
and each writes an audit line with the client's address, which is the only accountability there is
because the dashboard itself is unauthenticated. Do not narrow `server.dashboard.host` to
`127.0.0.1` to "fix" that: every node ships telemetry and logs to that address, so the HTTPS bind is
load-bearing and narrowing it silently empties the Overview tab (it also removes the plaintext
listener, since HTTPS then holds the port on loopback itself). Its `0.0.0.0` default is not a
wildcard bind — it means "the address the fleet reaches this orchestrator at".

Two things worth knowing before you edit a config there. **A browser page from anywhere else is
rejected** by a Host/Origin check, which trusts loopback names, this machine's hostname and the
address the dashboard was started on — the same address every node ships to, so a fleet addressed by
hostname ingests as well as one addressed by IP. That check stops foreign pages and DNS rebinding;
it authorizes no programmatic caller. **Editor redaction is by key**: `password` and `api_key` are
masked wherever they appear, a save restores whatever you left masked, so a view-and-save round trip
leaves the file byte-identical — but a secret pasted into a differently named field is not masked.
A file that will not parse still opens (redacted); saving keeps returning 400 until the syntax is
fixed and the value re-entered.

---

## Testing

```bash
uv run pytest -q        # unit tests alone (needs `uv sync --group dev`)
./scripts/test-all.sh   # everything
```

`test-all.sh` has eight sections and a gated-out one prints `[SKIP]` rather than passing silently:
0 (pytest, when the dev group is installed) and 1 (orchestrator config and CLI) need no fleet or
cluster, while the six gates are a reachable
`kubectl` for 2 (infrastructure), the dashboard answering on `:7001` for 3, the first
`mcp.server.enabled` node answering `/health` on `:30900` for 4 (MCP and its bearer check), the
planner's own probe for 5 (a planner-role target PASSes, none enabled SKIPs, a probe that could not
run **FAILs**), `kubectl` plus a non-empty `node_groups` for 6 (sharded groups), and `kubectl` plus
a node with `api_key` whose key file exists here for 7 (llama-server auth).

Section 0 also skips in a created workspace, which carries no `tests/` — run the unit suite from a
Git checkout. Section 6 additionally needs coreutils `timeout` on each node, since it bounds the
container it starts there to read a binary's checksum.

Three further checks run after section 7 at file scope, needing no cluster: two grep the shipped
logs for key material, one asserts the generated configs are mode 600. An opt-in lifecycle block
runs last of all — `POST /api/lifecycle/stop` then `/start` against the dashboard — gated on
`AUREAX_TEST_LIFECYCLE=1` because it stops the whole fleet and the k3d cluster; without that
variable both checks `[SKIP]`. Everything past section 1 dials HTTPS through `curl_tls`, so `aureax generate` must have minted the fleet CA first — without
it those checks fail with curl's status 77 and a line naming the command to run. Section 3 is the
exception: it takes the dashboard's loopback listener over plain HTTP, so it needs no CA at all. The suite resolves
its fleet definition once at startup (`configs/aureax-network.yaml`, or the shipped example with a
printed note), so sections 0 and 1 work on a fresh clone. Never run the suite with `bash -x`
against a live fleet: sections 4 and 7 trace the key material they hand to `curl`.

---

## Sharded model groups

Several nodes can cooperatively serve **one** model that fits on none of them individually, using
llama.cpp's RPC backend. A `node_groups` entry names a **head** — which runs `llama-server` with
`--rpc <w1>:50052,<w2>:50052,...` plus the MCP server, and is the only node holding the GGUF — and
its **workers**, which run only llama.cpp's `rpc-server`, holding a share of the weights and
executing tensor ops for the head. The group is addressed exactly like a single node, through the
head:

```bash
uv run aureax group list                                   # members, endpoints, auth
uv run aureax-harness infer <HEAD_NODE_ID> "What is 2+2?"  # MCP    :30900
curl --cacert ~/.aureax/tls/ca.pem https://<HEAD_IP>:30800/v1/models   # llama :30800
```

That `/v1/models` call keeps working with API-key auth on — see
[Exempt endpoints](#llama-server-api-keys). A cognitive planner picks up every group automatically:
its `remote_servers` is derived from `${mcp_endpoints}`, which resolves to every enabled target
whose own config has `mcp.server.enabled`, excluding itself, and carries each peer's MCP key.
Nothing is listed by hand.

- **Sharding buys capacity, not speed.** Splitting a model across machines is slower per token than
  one machine that could hold it, and adding workers makes it slower still: the RPC backend runs
  layer ranges in sequence, so spreading the work wider adds round trips without cutting compute.
- **Worker order IS `--rpc` order**, naming the devices `RPC0, RPC1, ...` that `devices` and
  `tensor_split` index positionally. **Append** new workers, and remove only from the end.
  `aureax generate` refuses any edit that moves a worker already in the list — a swap, a middle
  insert, a middle removal — naming the slots that moved; `--accept-worker-reorder` overrides it,
  and then every split needs re-checking.
- **Long context needs flash attention**, or llama.cpp reserves an `[n_kv × n_ubatch × n_head_q]`
  F32 score tensor on every worker. `flash_attention` is tri-state (`true` / `false` / `"auto"`)
  and `--flash-attn` is passed on every launch, so the configured value is what llama-server acts
  on; `"auto"` defers to llama.cpp's probe, which over RPC always answers yes, so `false` is
  honoured only because the flag is always emitted. `llama-server.log`'s "Flash Attention enabled"
  line is the only runtime source of truth.
- **The first load is cold and takes minutes.** The head streams every tensor to its workers before
  the endpoint answers, so a pod sits `0/1 Running` throughout. Changing a group's worker count
  invalidates every worker's tensor cache, so the first start after a resize is always cold.
- **`rpc-server` on `:50052` is the fleet's one unauthenticated, plaintext port.** The host firewall
  is its only control, and the worker pod installs that firewall itself before RPC starts — see the
  note below.
- **Every peer must run the byte-identical llama.cpp binaries**, or the head connects and then hangs
  at the RPC handshake with no error on either side. `aureax start` and `aureax group doctor` both
  ask each member for the sha256 of `llama-server` and `rpc-server` *inside its image*, over SSH, and
  refuse a group whose members disagree — naming the odd node and the rebuild command. Not the image
  digest: every node builds `aureax-client:dev` natively, so those differ across a perfectly coherent
  group. A member that cannot be read is a warning, never a refusal.

The full list of invariants, and the check that enforces each, is in
[ARCHITECTURE.md → "Invariants and how each is enforced"](ARCHITECTURE.md#invariants-and-how-each-is-enforced).
When something is wrong, `uv run aureax group doctor` collects build ids, binary checksums, memory,
NIC speed, RPC listeners, GGUF presence and recent logs from every member in one pass — and exits
non-zero on a mismatched group, so it composes into a pre-flight check.

> **Security:** `rpc-server` has no authentication, speaks no TLS, and deserializes arbitrary
> tensor graphs. Port 50052 must stay on a trusted LAN and must never reach a routable network.
> Kubernetes `NetworkPolicy` does not apply to `hostNetwork` pods, so the host firewall is the only
> network control here — and the shard pod installs it rather than leaving it to you.

Each worker pod runs an `rpc-firewall` init container (`client/rpc_firewall.py`, `NET_ADMIN`, host
network) that installs and verifies a host allowlist before `rpc-server` starts. It builds a chain
named `AUREAX_RPC_<port>` — accept the loopback interface, accept the node's own address and the
group head, `DROP` everything else — and puts its jump first in `INPUT` for that TCP port. Anything
that does not install and verify blocks startup, so a worker with no firewall never serves.

- **The allowlist comes from the head, not from you.** `aureax generate` resolves the group head to
  IP literals on the orchestrator and writes them into the worker's `model.rpc_server.allowed_clients`.
  The init container never resolves names — it has no cluster DNS and no mDNS — so a hostname there
  is a named refusal pointing at `aureax generate` and `aureax apply`.
- **It runs both IP families**, and fails closed on each family the host actually has. A host with
  no IPv6 address at all and an IPv4 RPC bind logs that it is skipping the `ip6tables` pass, because
  there is no v6 socket to protect and no ip6 filter table this container could create.
- **The rules are baked in at pod start** and nothing re-runs them for the life of the pod. `apply`
  pushes a worker's new config but never restarts a worker, so a group whose **head address changed**
  needs `aureax generate` and `aureax apply`, then `aureax stop && aureax start` to recreate the
  worker pods. Until then the worker DROPs its own head and the group hangs at the RPC handshake.
- **The rules deliberately outlive the pod**, and a firewall manager that flushes them while workers
  run reopens the port. Remove the `INPUT` jump and the chain for both families only after every RPC
  process on that port has stopped, and do not start `rpc-server` by hand outside this pod.
- Diagnose a worker that will not start with
  `kubectl -n aureax logs <worker-pod> -c rpc-firewall`; every refusal there is named.

### Swapping a group's model

Never a one-line change: the head and worker `config_file` lines and the `node_groups` block whose
worker count a head's `tensor_split` must match move together, the previous pair is kept unmodified
as the rollback target, and the ceilings a model size drives are shared with it. The contract is
[ARCHITECTURE.md → "A frozen rollback target is a config contract, not a spare file"](ARCHITECTURE.md#a-frozen-rollback-target-is-a-config-contract-not-a-spare-file),
and the numbers are ["Probe budgets"](ARCHITECTURE.md#probe-budgets).

---

## Project structure

```
configs/          Fleet definition + per-node config files
client/           Edge client: supervisor, llama.cpp wrapper, MCP server, sensors,
                  telemetry, log shipping
orchestrator/     CLI, config schema/loader/templating, TLS and key stores, SSH, cluster,
                  deploy, lifecycle, dashboard, harness, telemetry, logs
k8s/              Manifests: namespace, DaemonSets (edge + shard), Services
docker/           Dockerfile.client (ARM64/x86_64), Dockerfile.client.cuda, scripts/start.sh
scripts/          All operational scripts, plus lib/common.sh
tests/            pytest suite
models/           Downloaded GGUFs. Created by download-model.sh, not tracked
```

There is no ConfigMap. A per-node config reaches a pod as a host file: the deployer SSH-pushes the
resolved YAML to `/etc/aureax/aureax-edge.yaml` (and the node's certificates to `/etc/aureax/tls`),
and both DaemonSets mount `/etc/aureax` as a read-only hostPath. More such detail:
[ARCHITECTURE.md → "7. Gotchas"](ARCHITECTURE.md#7-gotchas).

---

## License

MIT — see [LICENSE](LICENSE). Contributions are covered by
[CONTRIBUTING.md](CONTRIBUTING.md); security reports by [SECURITY.md](SECURITY.md).
