#!/usr/bin/env python3
"""Helper to extract values from aureax-network.yaml for shell scripts.

Usage:
    uv run python scripts/parse-config.py <field> [config_path]

Fields:
    target <n>           — JSON object for target at index n. An operator-facing
                           debugging aid: nothing in this repository calls it,
                           so its argv layout pins no live contract
    groups               — shell-sourceable node_group topology:
                           GROUP_COUNT, GROUP_NAMES, GROUP_HEAD_HOSTS,
                           GROUP_HEAD_IDS (each head's node_id, in lockstep
                           with GROUP_HEAD_HOSTS — the address dials a port,
                           the id finds that node's key), GROUP_RPC_PORTS,
                           GROUP_WORKER_HOSTS (one space-separated host list
                           per array slot)
    k3d                  — shell-sourceable cluster identity:
                           K3D_CLUSTER_NAME, K3D_API_PORT, K3D_SERVERS
    model_for_node <n>   — shell-sourceable model vars for the model the SSH
                           target at index n needs, looked up from that node's
                           config_file (model.filename) against model_registry:
                           MODEL_FILENAME, MODEL_URL, MODEL_SIZE_GB (part 1 of
                           a split GGUF), MODEL_PARTS + MODEL_PART_URLS (every
                           part, in lockstep), MODEL_SHA256 (a single-file
                           GGUF's declared digest) + MODEL_PART_SHA256S (one
                           per part, in the same order as MODEL_PARTS) +
                           MODEL_REVISION (the HuggingFace ref the URLs
                           resolve, `main` when the entry pins none), and
                           MODEL_DRAFT_FILENAME / MODEL_DRAFT_URL /
                           MODEL_DRAFT_SIZE_GB / MODEL_DRAFT_SHA256 for a
                           speculative-decoding draft. All twelve names are
                           always defined —
                           download-model.sh evals them in a loop under
                           `set -u`.
    all                  — every ENABLED target as lockstep NODE_* shell arrays
                           plus NODE_COUNT, SHARD_WORKER_ROLE and PLANNER_ROLE
                           (the role literals) and MCP_NODE_PORT /
                           LLAMA_NODE_PORT (the NodePorts), all four from
                           schema.py; disabled targets are omitted, so the
                           indices are dense
"""

import json
import shlex
import sys
from pathlib import Path

# Add project root to path so we can import orchestrator modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.config.loader import DEFAULT_NETWORK_CONFIG, load_network_config
from orchestrator.config.schema import (
    DEFAULT_MODEL_REVISION,
    LLAMA_NODE_PORT,
    MCP_NODE_PORT,
    PLANNER_ROLE,
    SHARD_WORKER_ROLE,
    mcp_server_enabled,
    node_role,
)


def _print_empty_model_vars(filename: str = "") -> None:
    """Print shell-sourceable MODEL_* vars for a node with no resolvable model.

    Callers still need every name defined so `eval` in the shell scripts doesn't
    leave stale values from a previous node in the loop.
    """
    print(f"MODEL_FILENAME={shlex.quote(filename)}")
    print('MODEL_URL=""')
    print('MODEL_SIZE_GB=""')
    print('MODEL_PARTS=()')
    print('MODEL_PART_URLS=()')
    print('MODEL_SHA256=""')
    print('MODEL_PART_SHA256S=()')
    print('MODEL_REVISION=""')
    _print_empty_draft_vars()


def _print_empty_draft_vars() -> None:
    """Every MODEL_DRAFT_* name, defined and empty.

    Same contract as the MODEL_* vars above: download-model.sh runs under
    `set -u` and evals these in a loop, so a name left undefined for one node
    either aborts the script or silently carries the previous node's draft.
    """
    print('MODEL_DRAFT_FILENAME=""')
    print('MODEL_DRAFT_URL=""')
    print('MODEL_DRAFT_SIZE_GB=""')
    print('MODEL_DRAFT_SHA256=""')


def _print_draft_vars(node_cfg, registry) -> None:
    """Resolve `model.draft.filename` against the registry, if enabled.

    Silent no-op when speculative decoding is off, which is every node except a
    shard head that has opted in.
    """
    draft = ((node_cfg.get("model") or {}).get("draft") or {})
    wanted = draft.get("filename", "") if draft.get("enabled") else ""
    if not wanted:
        _print_empty_draft_vars()
        return
    for m in registry.models:
        if m.filename == wanted:
            urls = m.part_urls()
            print(f"MODEL_DRAFT_FILENAME={shlex.quote(m.filename)}")
            print(f"MODEL_DRAFT_URL={shlex.quote(urls[0] if urls else '')}")
            print(f"MODEL_DRAFT_SIZE_GB={shlex.quote(str(m.size_gb))}")
            # A draft is fetched as ONE file (download-model.sh Step 2b), so
            # only a whole-file digest can apply to it; a split draft's
            # per-part list has no single-file meaning and is not reported.
            print(f"MODEL_DRAFT_SHA256={shlex.quote(m.sha256 or '')}")
            return
    # Loud on stderr, empty on stdout: a draft named but absent from the
    # registry means llama-server will be handed a path that does not exist and
    # will exit at startup. Better to surface it during download than at boot.
    print(f"# draft model {wanted} not found in model_registry", file=sys.stderr)
    _print_empty_draft_vars()


def _print_shell_array(name: str, values) -> None:
    """Emit `name=(...)` with every element quoted.

    Every scalar MODEL_* emitted in this file goes through `shlex.quote` for
    the same reason: these values come from YAML (`huggingface_repo` /
    `huggingface_file` build the URL) and are eval'd by
    scripts/lib/common.sh::load_network_env, so an unquoted `$`, backtick or
    `$(` in a repo name would expand — or run — on the orchestrator.

    Quoting is not cosmetic. Several of these arrays are indexed positionally
    in lockstep with NODE_HOSTS, and some elements are legitimately empty — a
    target with no `config_file` has no role, and `key_file` is optional. Bash
    discards an unquoted empty word, so `NODE_ROLES=( shard_worker)` would be
    length 1 rather than 2 and every later index would refer to a different
    node than the caller thinks. `''` survives as a real element.
    """
    print(f"{name}=({' '.join(shlex.quote(str(v)) for v in values)})")


def _load_node_config(node_id: str, path: Path):
    """Read one per-node config, or exit 1 saying which file and why.

    Exits rather than returning an empty value. An empty role in `all` or empty
    MODEL_* in `model_for_node` is indistinguishable from a target that
    legitimately has no `config_file`, and every shell consumer reads an empty
    role as "ordinary edge node" — so a named-but-unreadable config_file would
    silently promote a shard worker into the edge label pass.
    """
    from client.config.parser import load_edge_config_raw

    if not path.exists():
        print(f"ERROR: {node_id}: config_file not found at {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return load_edge_config_raw(str(path))
    except Exception as e:
        # Broad on purpose: ruamel raises YAMLError, not ValueError, for a
        # malformed document, and an unreadable file raises OSError.
        print(f"ERROR: {node_id}: could not read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _print_model_part_arrays(model) -> None:
    """Emit MODEL_PARTS / MODEL_PART_URLS for a registry entry.

    A large GGUF is published split across several files. Both arrays are in
    part order and the same length, so shell callers can index them together.
    A single-file model yields one element each — no special case downstream.
    """
    _print_shell_array("MODEL_PARTS", model.part_filenames)
    _print_shell_array("MODEL_PART_URLS", model.part_urls())


def _print_model_digest_vars(model) -> None:
    """Emit MODEL_SHA256 / MODEL_PART_SHA256S / MODEL_REVISION.

    What the digests are for: every other gate in download-model.sh compares
    LENGTHS, so any file of the right size passes, and the URLs resolve a ref
    that upstream is free to move. A declared digest is what turns "the right
    number of bytes arrived" into "the right bytes arrived" before the file is
    pushed to a head and handed to llama.cpp's GGUF deserializer.

    Both digest names are always emitted and at most one of them is non-empty:
    the registry declares a whole-file digest for a single-file GGUF or one per
    part for a split one, never both (ModelRegistryEntry validates that). The
    per-part list is an ARRAY for the same reason MODEL_PARTS is one — the
    shell indexes it against MODEL_PARTS positionally.

    MODEL_REVISION mirrors the default in ModelRegistryEntry.part_urls: an
    entry that pins nothing is fetched from `main`, and saying so is the point
    — it tells the operator the bytes behind those URLs can change between two
    provisions of the same fleet.
    """
    print(f"MODEL_SHA256={shlex.quote(model.sha256 or '')}")
    _print_shell_array("MODEL_PART_SHA256S", model.sha256_parts)
    print(f"MODEL_REVISION={shlex.quote(model.revision or DEFAULT_MODEL_REVISION)}")


def main() -> None:
    """Parse aureax-network.yaml for the shell scripts.

    Every subcommand but `target` prints shell-sourceable assignments; `target`
    prints a JSON object.
    """
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <field> [config_path]", file=sys.stderr)
        sys.exit(1)

    field = sys.argv[1]

    # "target" takes an index before the optional config_path, unlike every
    # subcommand except `model_for_node`:
    #   parse-config.py target <index> [config_path]
    # Nothing in this repository calls it — it is a hand-run debugging aid, so
    # the layout is preserved for operators rather than for a live caller.
    if field == "target":
        # Imported here, not at module scope: orchestrator.deploy.ssh pulls in
        # paramiko for what is a pure-pathlib helper, and only this branch and
        # `all` call it. download-model.sh runs `model_for_node` in a fresh
        # interpreter once per node, so a module-scope import would charge every
        # one of those starts for a library none of them use.
        from orchestrator.deploy.ssh import resolve_key_path

        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        config_path = sys.argv[3] if len(sys.argv) > 3 else str(DEFAULT_NETWORK_CONFIG)
        config = load_network_config(config_path)
        t = config.ssh.enabled_targets[idx]
        print(json.dumps({
            "host": t.host,
            "user": t.user,
            "port": t.port,
            "node_id": t.node_id,
            "key_file": resolve_key_path(t.key_file, config.ssh.key_store) or "",
            "display_name": t.display_name,
        }))
        return

    # "model_for_node" — output shell-sourceable MODEL_* vars for the model
    # the SSH target at <index> needs. Resolves the per-node config_file's
    # model.filename and looks it up in model_registry.models.
    if field == "model_for_node":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        config_path = sys.argv[3] if len(sys.argv) > 3 else str(DEFAULT_NETWORK_CONFIG)
        config = load_network_config(config_path)
        target = config.ssh.enabled_targets[idx]
        configs_dir = Path(config_path).parent
        per_node_path = configs_dir / target.config_file if target.config_file else None
        if per_node_path is None:
            print(f'# no config_file for node {target.node_id}', file=sys.stderr)
            _print_empty_model_vars()
            return
        # A config_file that is NAMED but missing or unparseable is an error,
        # not an empty model — _load_node_config exits 1 naming the file, so
        # download-model.sh stops instead of reporting a model it could not
        # resolve from a file it never managed to open.
        node_cfg = _load_node_config(target.node_id, per_node_path)
        wanted = (node_cfg.get("model") or {}).get("filename", "")
        if not wanted:
            print(f'# per-node config has no model.filename for {target.node_id}', file=sys.stderr)
            _print_empty_model_vars()
            return
        for m in config.model_registry.models:
            if m.filename == wanted:
                urls = m.part_urls()
                # MODEL_FILENAME/MODEL_URL stay pointed at part 1 for callers
                # that only need one name (llama.cpp is handed part 1 and finds
                # the rest); MODEL_PARTS/MODEL_PART_URLS carry the whole set.
                print(f"MODEL_FILENAME={shlex.quote(m.filename)}")
                print(f"MODEL_URL={shlex.quote(urls[0] if urls else '')}")
                print(f"MODEL_SIZE_GB={shlex.quote(str(m.size_gb))}")
                _print_model_part_arrays(m)
                _print_model_digest_vars(m)
                _print_draft_vars(node_cfg, config.model_registry)
                return
        print(f'# model {wanted} not found in model_registry', file=sys.stderr)
        _print_empty_model_vars(filename=wanted)
        return

    # All other subcommands: parse-config.py <field> [config_path]
    config_path = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_NETWORK_CONFIG)
    config = load_network_config(config_path)

    # Disabled targets are omitted entirely rather than exported with a flag:
    # every shell loop indexes these arrays positionally, so filtering here
    # means no script has to know about `enabled` for the NODE_* arrays. Note
    # `target <N>` and `model_for_node <N>` index the SAME filtered list, so a
    # caller looping 0..NODE_COUNT-1 stays consistent across those subcommands.
    # The `groups` branch below is the deliberate exception — see its comment.
    targets = config.ssh.enabled_targets
    key_store = config.ssh.key_store

    if field == "k3d":
        # The cluster's identity, for the shell path, from the same config
        # orchestrator/deploy/deployer.py reads. setup-all.sh derives its
        # `grep`, its `docker exec k3d-<name>-server-0` and its API port from
        # these rather than spelling them out, so editing kubernetes.cluster_name
        # or k3d.api_port cannot leave the bash path pointed at the old cluster —
        # a failure that would otherwise surface as `docker exec` missing a
        # container, several steps after the edit that caused it.
        k8s = config.kubernetes
        print(f"K3D_CLUSTER_NAME={shlex.quote(k8s.cluster_name)}")
        print(f"K3D_API_PORT={k8s.k3d.api_port}")
        print(f"K3D_SERVERS={k8s.k3d.servers}")
    elif field == "groups":
        # Shell-sourceable node_group topology. Emits, for each group index i:
        #   GROUP_NAMES[i], GROUP_HEAD_HOSTS[i], GROUP_HEAD_IDS[i]
        #   GROUP_RPC_PORTS[i]
        #   GROUP_WORKER_HOSTS[i]  — space-separated within one array slot
        #
        # GROUP_HEAD_IDS is the head's node_id, in lockstep with
        # GROUP_HEAD_HOSTS, because the two are not interchangeable to every
        # caller: `aureax-harness` resolves a node id through the fleet config
        # and sends that node's MCP bearer key from the key store, while a bare
        # address sends no key at all — so test-all.sh's end-to-end inference
        # check answered 401 on every keyed head while the address was all it
        # had. Hosts stay for the checks that dial a port directly.
        #
        # Members are resolved by node_id against the FULL target list, since
        # config.ssh.targets is simply the id->host map; the enabled-only
        # filtering the NODE_* arrays need is about positional indices, which
        # nothing here uses. It smuggles nothing in either: schema.py's
        # NetworkConfig._check_node_groups rejects a group naming an unknown or
        # a disabled node, so every host emitted here is an enabled target that
        # also appears in the NODE_* arrays.
        host_by_id = {t.node_id: t.host for t in config.ssh.targets}
        names, heads, head_ids, workers, ports = [], [], [], [], []
        for g in config.node_groups:
            names.append(g.name)
            heads.append(host_by_id.get(g.head, ""))
            head_ids.append(g.head)
            workers.append(" ".join(host_by_id.get(w, "") for w in g.workers))
            ports.append(str(g.rpc_port))
        print(f'GROUP_COUNT={len(config.node_groups)}')
        _print_shell_array("GROUP_NAMES", names)
        _print_shell_array("GROUP_HEAD_HOSTS", heads)
        _print_shell_array("GROUP_HEAD_IDS", head_ids)
        _print_shell_array("GROUP_RPC_PORTS", ports)
        # Each element holds a space-separated host list, so quoting keeps one
        # group's workers as one array element.
        _print_shell_array("GROUP_WORKER_HOSTS", workers)
    elif field == "all":
        # Output shell-sourceable variables for nodes + model.
        # Both imports are local for the same reason as in `target` above: they
        # cost real import time and the `k3d` and `groups` subcommands, which
        # run on every script start, never reach here.
        from orchestrator.config.templating import (
            template_references_api_key,
            template_references_mcp_api_key,
        )
        from orchestrator.deploy.ssh import resolve_key_path

        configs_dir = Path(config_path).parent

        hosts = [t.host for t in targets]
        users = [t.user for t in targets]
        # `port` is per-target and defaults to 22 in the schema. It is exported
        # because scripts/lib/common.sh's ssh_cmd/scp_cmd resolve it by host:
        # without it every bash step would dial 22, so a target on a non-22 port
        # would be unreachable from the shell path while paramiko, which does
        # read it, connected fine.
        ports = [str(t.port) for t in targets]
        key_files = [resolve_key_path(t.key_file, key_store) or "" for t in targets]
        node_ids = [t.node_id for t in targets]
        config_files = [t.config_file or "" for t in targets]

        # Per-node role + mcp.server.enabled — let shell scripts skip planners
        # from MCP-server-only checks (planner is an MCP client, not server).
        #
        # NODE_AUTH and NODE_MCP_AUTH ride along in the same pass: whether this
        # node's TEMPLATE turns on llama-server api_key auth, and whether it
        # turns on MCP bearer auth, so shell callers know which endpoints answer
        # 401 unauthenticated. The two are independent — the MCP port and the
        # llama port are separate doors with separate keys. Only ever booleans —
        # NO KEY MATERIAL is exported here, or anywhere in this file. The keys
        # themselves stay in ~/.aureax/keys/<node_id>.key and
        # ~/.aureax/keys/<node_id>.mcp.key and are read by callers at the point
        # of use.
        roles = []
        mcp_servers = []
        auth_enabled = []
        mcp_auth_enabled = []
        for t in targets:
            role = ""
            has_mcp_server = "true"
            has_auth = "false"
            has_mcp_auth = "false"
            per_node_path = configs_dir / t.config_file if t.config_file else None
            if per_node_path is not None:
                # Errors rather than defaulting to role "". Every shell caller
                # reads "" as "ordinary edge node", so a config_file that is
                # named but missing or malformed would silently promote a shard
                # worker into the edge label pass, the multi-gigabyte model push
                # and the MCP health checks, with nothing printed anywhere.
                ncfg = _load_node_config(t.node_id, per_node_path)
                role = node_role(ncfg)
                has_mcp_server = "true" if mcp_server_enabled(ncfg) else "false"
                # Delegated, not re-derived: templating.template_references_api_key
                # is the one place that decides what "this template asks for a
                # key" means, and generation itself (which creates the key file)
                # asks the same question through it. Re-implementing the test
                # here is how the shell path and the Python path drift into
                # disagreeing about which nodes are authenticated.
                has_auth = "true" if template_references_api_key(per_node_path) else "false"
                # Same delegation, same reason: templating decides what
                # "this template asks for an MCP key" means, and the pre-mint
                # pass in generate_all_edge_configs asks through the same
                # function. A re-implementation here is how test-all.sh's
                # section 4 would end up sending a bearer token to a node that
                # enforces none, or none to a node that does.
                has_mcp_auth = (
                    "true" if template_references_mcp_api_key(per_node_path) else "false"
                )
            roles.append(role)
            mcp_servers.append(has_mcp_server)
            auth_enabled.append(has_auth)
            mcp_auth_enabled.append(has_mcp_auth)

        _print_shell_array("NODE_HOSTS", hosts)
        _print_shell_array("NODE_USERS", users)
        _print_shell_array("NODE_PORTS", ports)
        _print_shell_array("NODE_KEY_FILES", key_files)
        _print_shell_array("NODE_IDS", node_ids)
        _print_shell_array("NODE_CONFIG_FILES", config_files)
        _print_shell_array("NODE_ROLES", roles)
        _print_shell_array("NODE_HAS_MCP_SERVER", mcp_servers)
        _print_shell_array("NODE_AUTH", auth_enabled)
        _print_shell_array("NODE_MCP_AUTH", mcp_auth_enabled)
        print(f'NODE_COUNT={len(targets)}')
        # The role literals themselves, derived from schema.py rather than
        # spelled again per call site. Shell sites index NODE_ROLES against
        # $SHARD_WORKER_ROLE and $PLANNER_ROLE, so a rename in schema.py reaches
        # all of them at once — a retyped literal that stopped matching would
        # silently promote every worker into the edge label pass, the model push
        # and the MCP health checks, and schedule the planner alongside the
        # edges instead of after them. Both are always printed: setup-all.sh
        # runs under `set -u` and tests either one unconditionally.
        print(f"SHARD_WORKER_ROLE={shlex.quote(SHARD_WORKER_ROLE)}")
        print(f"PLANNER_ROLE={shlex.quote(PLANNER_ROLE)}")
        # The NodePorts, from the same source. Six shell sites probe
        # $MCP_NODE_PORT / $LLAMA_NODE_PORT rather than 30900 / 30800, so
        # changing a Service's NodePort moves the Python layer and the shell
        # checks together instead of leaving the shell probing the old port.
        # (k8s/service-*.yaml keep their own literals — schema.py documents
        # that hand-synced contract.)
        print(f"MCP_NODE_PORT={MCP_NODE_PORT}")
        print(f"LLAMA_NODE_PORT={LLAMA_NODE_PORT}")
    else:
        print(f"Unknown field: {field}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
