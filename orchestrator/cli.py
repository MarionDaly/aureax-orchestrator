"""Aureax CLI — orchestrator command-line interface."""

from __future__ import annotations

import socket
from pathlib import Path

import click
from ruamel.yaml.error import YAMLError

from orchestrator.cluster._subprocess import CommandError
from orchestrator.cluster.buildcheck import BUILD_PROBE
from orchestrator.config.cache import load_network_config_cached
from orchestrator.config.loader import (
    DEFAULT_NETWORK_CONFIG,
    ConfigError,
    generated_config_dir,
    validate_orchestrator_host,
)
from orchestrator.config.schema import LLAMA_NODE_PORT, MCP_NODE_PORT
from orchestrator.config.templating import (
    # _url_host is the tree's single IPv6-bracketing rule for a URL authority:
    # a bare literal's own colons read as the port separator in every client
    # that parses the URL.
    _url_host,
    generate_all_edge_configs,
)

DEFAULT_CONFIG = str(DEFAULT_NETWORK_CONFIG)


@click.group()
@click.option(
    "--config",
    "-c",
    default=DEFAULT_CONFIG,
    # No exists=True: a missing file must reach the loader, whose error names
    # the shipped example and the copy command. Click's own check fires before
    # any code of ours and says only "Path ... does not exist".
    type=click.Path(dir_okay=False),
    help="Path to aureax-network.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Aureax — AI orchestration to the edge."""
    ctx.obj = _CliState(ctx.obj or {})
    ctx.obj["config_path"] = Path(config)


class _CliState(dict):
    """``ctx.obj``: the config path, plus the parsed config loaded on first use.

    Loaded lazily rather than in the group callback because Click runs that
    callback before it parses the subcommand's own arguments: an eager load
    breaks ``aureax init`` (the first command on a fresh clone, before any
    fleet definition exists) and every ``aureax <cmd> --help`` with the
    missing-config error. Commands that need the fleet read
    ``ctx.obj["network_config"]``, which loads it on first touch.

    The load goes through the (path, mtime) cache: ``deploy`` and ``apply``
    build a Deployer that loads the same file again, and a config edited
    between two runs in one process (the tests, the dashboard) is still
    re-read.
    """

    def __missing__(self, key: str):
        if key != "network_config":
            raise KeyError(key)
        path = self["config_path"]
        try:
            cfg = load_network_config_cached(path)
        except ConfigError as e:
            raise click.ClickException(str(e)) from e
        except (YAMLError, OSError) as e:
            # ConfigError only wraps schema failures; the YAML parse itself
            # happens outside loader's try and YAMLError is not a ValueError,
            # so without this one mistyped line makes every command — nodes,
            # status, stop — die with a ruamel traceback instead of a one-liner.
            raise click.ClickException(f"{path}: {e}") from e
        self[key] = cfg
        return cfg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_orchestrator_ip(ctx: click.Context, _param: click.Parameter,
                              value: str | None) -> str | None:
    """Click adapter for ``loader.validate_orchestrator_host``.

    The rule itself lives in the loader because the dashboard's POST
    /api/config/regenerate takes the same value: one definition is what keeps a
    fleet whose orchestrator is named by an mDNS hostname regenerable from both
    entry points instead of only one.

    ``None`` passes through: Click runs the callback for an omitted option too,
    and ``dashboard``/``tls regenerate`` take this same value optionally.
    """
    if value is None:
        return None
    try:
        return validate_orchestrator_host(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


_orchestrator_ip_option = click.option(
    "--orchestrator-ip",
    required=True,
    callback=_validate_orchestrator_ip,
    help="IP address (or hostname) of this orchestrator on the LAN",
)

# One definition on all three commands that regenerate configs, because a
# refused reorder must be overridable from whichever of them the operator
# reached for — and the message the refusal prints names this exact spelling.
_accept_worker_reorder_option = click.option(
    "--accept-worker-reorder",
    is_flag=True,
    help="Accept a REORDERED workers: list in node_groups and record the new order as "
         "the baseline. Worker order is --rpc order, which names the devices RPC0..n "
         "that tensor_split and devices index positionally, so a reorder silently "
         "re-points a hand-written split at other machines.",
)


def _fan_out(items, work, max_workers: int = 16) -> list:
    """Run ``work(item)`` for every item concurrently, results in ``items`` order.

    Every caller here is a per-node SSH round trip whose cost is latency, not
    CPU: run serially, one powered-off node charges its full connect timeout to
    every node queued behind it.

    ``work`` MUST catch its own exceptions: this returns ``pool.map``'s
    results, which re-raise at iteration time, so one raising worker would
    discard every other node's result.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(len(items), max_workers)) as pool:
        return list(pool.map(work, items))


def _expected_server_url(config) -> str:
    """The K3s server URL edge agents should point at. Raises when undetectable.

    The loopback guard is the load-bearing half: when LAN-IP detection falls
    back to 127.0.0.1 every node "drifts" against an address that is wrong on
    our side, and healing against it would point the whole fleet at the
    orchestrator's own loopback.
    """
    from orchestrator.cluster import k3d

    expected = k3d.get_server_url(config.kubernetes.k3d.api_port, k3d.probe_host(config))
    if expected.startswith(("https://127.", "https://0.")):
        raise RuntimeError(
            f"could not detect the orchestrator LAN IP (got {expected}). "
            "The address is chosen by asking the routing table which interface "
            "reaches the first enabled node — check that you are on the same "
            "network as the nodes and that the orchestrator has a route to "
            "them. Detection is macOS-only (see orchestrator.cluster.k3d)."
        )
    return expected


def _no_password_reason(target) -> str | None:
    """Why a target resolves no SSH password, or None when it does.

    A target configured the documented way (``password_env:
    AUREAX_SSH_PASSWORD``) resolves to None whenever that variable is simply
    not exported in this shell — indistinguishable at the call site from a
    genuinely key-only target. Reporting the second for the first sends the
    operator into an SSH auth failure moments later with nothing naming the
    cause.

    ``password_configured`` is what tells the two apart: it is True whenever
    the target DECLARES a password by either spelling, so "declared but does
    not resolve" is exactly the unexported-variable case.
    """
    if target.resolve_password():
        return None
    if not target.password_configured:
        return "no password — key-only target"
    # Declared but unresolved. `password:` is a literal, so a value that makes
    # password_configured true also makes resolve_password() truthy — the only
    # way to reach here is a password_env naming a variable nobody exported.
    return (
        f"password_env {target.password_env} is not set in this shell "
        "— export it or use `password:`"
    )


# What a failed operation looks like coming out of the deploy/lifecycle stack.
#
# CommandError is the easy one to forget: every kubectl and k3d call goes
# through orchestrator.cluster._subprocess.run, which raises it (a
# CalledProcessError subclass, so NOT a RuntimeError) on a non-zero exit, a
# timeout or a binary missing from PATH. An API server that is down, a manifest
# kubectl rejects or a wedged Docker daemon reaches the operator as a traceback
# unless it is listed here. str() on it carries the captured stderr, which is
# the part that actually says what kubectl objected to.
#
# ValueError/FileNotFoundError cover generate_all_edge_configs (node_groups
# validation, a missing template); RuntimeError is what the deployer and
# lifecycle raise for their own checked failures.
_DEPLOY_ERRORS = (CommandError, RuntimeError, ValueError, FileNotFoundError)
# Lifecycle stop/start reach the same subprocess layer but never the generator.
_LIFECYCLE_ERRORS = (CommandError, RuntimeError)


def _bootstrap_public_key_text(public_key_path: str) -> tuple[str, str]:
    """The bootstrap public key's contents, and why a configured one is unusable.

    Returns (text, reason). ``reason`` is filled in only when a path IS
    configured and cannot be read — an empty ``ssh.bootstrap_public_key`` is a
    deliberate opt-out and has nothing to report.

    Read once per command rather than per node: folding the install into each
    node's fact probe needs the key TEXT, where ``ensure_key_installed`` takes
    the path and re-reads it on every call.

    The reason is returned rather than swallowed because `status` documents the
    key install as a side effect, and the schema default names a path plenty of
    orchestrators do not have: an empty text alone prints a green fleet on
    which nothing was installed, while `ssh -i` keeps failing for the bash
    scripts.
    """
    if not public_key_path:
        return "", ""
    try:
        return Path(public_key_path).expanduser().read_text().strip(), ""
    except OSError as e:
        return "", f"cannot read {public_key_path} ({e.strerror or e})"


@cli.command()
@click.option("--workspace", type=click.Path(path_type=Path), help="Create a new operational workspace from installed resources")
def init(workspace: Path | None = None) -> None:
    """Initialize local Aureax state directory."""
    if workspace is not None:
        from orchestrator.resources import create_workspace
        try:
            create_workspace(workspace)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Created workspace at {workspace.resolve()}")
    aureax_dir = Path.home() / ".aureax"
    secret_dirs = ["ssh-keys", "keys", "configs", "tls", "state"]
    for d in [aureax_dir, aureax_dir / "telemetry", *(aureax_dir / s for s in secret_dirs)]:
        d.mkdir(parents=True, exist_ok=True)
    # Every one of these but state/ holds secrets: ssh-keys/ the private keys
    # the fleet authenticates with, keys/ the per-node llama-server and MCP API
    # keys, configs/ the resolved per-node YAMLs that embed them, tls/ the
    # fleet CA's private key — the one secret every node's trust hangs on.
    # state/ carries only node ids and hosts (the reorder baseline), but goes
    # in the same group so the whole tree has one rule rather than a per-directory
    # one someone has to look up. mkdir honours the umask (0755 typically) and
    # ignores its own mode for a directory that already exists, so the chmod is
    # both the tightening and the repair.
    for s in secret_dirs:
        (aureax_dir / s).chmod(0o700)
    click.echo(f"Initialized Aureax state at {aureax_dir}")


@cli.command("trust-host-key")
@click.argument("node_id")
@click.option("--fingerprint", required=True,
              help="SHA256:... fingerprint obtained out of band (ssh-keygen -lf on the node)")
@click.pass_context
def trust_host_key(ctx, node_id: str, fingerprint: str) -> None:
    """Record a node's SSH host key after checking it against a fingerprint.

    Optional: a changed key is re-trusted automatically (logged at ERROR) on
    the next connect. Use this when the check should happen before any
    connection is made. The handshake sends no credentials.
    """
    import paramiko

    from orchestrator.deploy.ssh import enroll_host_key
    targets = ctx.obj["network_config"].ssh.targets
    target = next((t for t in targets if t.node_id == node_id), None)
    if target is None:
        raise click.ClickException(f"Unknown node: {node_id}")
    try:
        enroll_host_key(target.host, fingerprint, port=target.port)
    except (ValueError, OSError, paramiko.SSHException) as exc:
        raise click.ClickException(str(exc) or type(exc).__name__) from exc
    click.echo(f"Verified and recorded SSH identity for {node_id}")


@cli.command()
@click.pass_context
def nodes(ctx: click.Context) -> None:
    """List configured edge nodes."""
    from orchestrator.config.keystore import KIND_LLAMA, KIND_MCP

    config = ctx.obj["network_config"]
    configs_dir = ctx.obj["config_path"].parent
    click.echo(f"Network: {config.network.name} v{config.network.version}")
    click.echo(f"Nodes ({len(config.ssh.targets)}):")
    click.echo()
    for target in config.ssh.targets:
        tags = ", ".join(target.tags) if target.tags else "none"
        suffix = "" if target.enabled else click.style("  (disabled)", fg="yellow")
        click.echo(f"  {target.node_id}{suffix}")
        click.echo(f"    Host:    {target.user}@{target.host}:{target.port}")
        click.echo(f"    Name:    {target.display_name or target.node_id}")
        click.echo(f"    Config:  {target.config_file or '(none)'}")
        click.echo(f"    Tags:    {tags}")
        # Two lines, one per port: llama-server on :30800 and the MCP server on
        # :30900 carry SEPARATE keys, and one line collapsing them would report
        # a node as "enabled" while the port an operator is actually being
        # refused on has no key at all.
        click.echo(f"    Auth:    llama: {_auth_status(configs_dir, target, KIND_LLAMA)}")
        click.echo(f"             mcp:   {_auth_status(configs_dir, target, KIND_MCP)}")
        click.echo()


def _serves_mcp(template_path: Path) -> bool:
    """Whether a node's TEMPLATE runs an MCP server at all.

    Unreadable or absent reads as "yes", so the caller falls through to the
    key-reference check and reports the same "disabled" it would for a node
    whose template simply carries no `api_key:` line — a template we cannot
    read is not evidence that the node serves nothing.
    """
    from orchestrator.config.cache import load_template_cached
    from orchestrator.config.schema import mcp_server_enabled

    try:
        return mcp_server_enabled(load_template_cached(template_path))
    except Exception:
        return True


def _auth_status(configs_dir: Path, target, kind: str) -> str:
    """One-line auth state for one of a target's two keys (``kind``: llama/mcp).

    NEVER the full key — that would land in terminal scrollback and shell
    history. Parametrized rather than duplicated because the two kinds differ
    only in which template line turns them on and which file the key lands in;
    two copies would drift the moment one of the four states below changes.
    """
    from orchestrator.config import keystore
    from orchestrator.config.templating import (
        template_references_api_key,
        template_references_mcp_api_key,
    )

    if not target.config_file:
        return "disabled (no config_file)"
    template = configs_dir / target.config_file
    if kind == keystore.KIND_MCP and not _serves_mcp(template):
        # Shard workers and the planner run no MCP server. "disabled" would send
        # an operator looking for an `api_key:` line to add under `mcp.server`
        # in a template that has no MCP server to protect.
        return "unsupported (no MCP server)"
    references = (
        template_references_mcp_api_key if kind == keystore.KIND_MCP else template_references_api_key
    )
    if not references(template):
        return "disabled"
    # key_state, not `last4() or "pending"`: the display helpers collapse
    # "no key file" and "a key file I cannot read" to the same None, and the two
    # call for OPPOSITE actions — run `aureax generate`, or fix the file's
    # ownership (a .key left root-owned by a sudo run is the usual cause).
    # Telling an operator to regenerate a key that already exists is how a
    # running node's clients get 401s.
    state = keystore.key_state(target.node_id, kind=kind)
    if state == keystore.KEY_UNREADABLE:
        path = keystore.key_path(target.node_id, kind=kind)
        return f"enabled (key file unreadable — check ownership of {path})"
    if state == keystore.KEY_ABSENT:
        return "enabled (key pending — run `aureax generate`)"
    return f"enabled (…{keystore.last4(target.node_id, kind=kind)})"


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed status including K3s and metrics")
@click.pass_context
def status(ctx: click.Context, verbose: bool) -> None:
    """Show cluster and node health.

    Not read-only: every reachable password-auth node also gets
    ssh.bootstrap_public_key installed in its authorized_keys — the same action
    as `aureax bootstrap-keys` — so the bash scripts' `ssh -i` keeps working.

    Delegates to four section helpers so each fits on one screen; the SSH pass
    hands the drift check the K3S_URLs it collected on the same connections.
    """
    config = ctx.obj["network_config"]
    click.echo(f"Network: {config.network.name} v{config.network.version}")
    click.echo()

    k3s_urls, drift_unreachable = _status_ssh(config, verbose)
    if verbose:
        click.echo()
        _status_cluster(config)
        _status_drift(config, k3s_urls, drift_unreachable)
        click.echo()
        _status_telemetry(config)


def _status_ssh(config, verbose: bool) -> tuple[dict[str, str], list[str]]:
    """SSH pass over the fleet. Returns (node_id -> K3S_URL, drift-gap notes).

    ONE handshake per node carries everything: the reachability check, the
    K3S_URL the drift section compares, and the bootstrap-key install. Split
    over three connections and run serially, a single powered-off node charges
    its full connect timeout to every node behind it.
    """
    from orchestrator.deploy.k3s_agent import AGENT_ENV_PATH, _parse_k3s_url
    from orchestrator.deploy.ssh import authorized_keys_command, conn_kwargs, probe_facts

    pub_key, key_gap = _bootstrap_public_key_text(config.ssh.bootstrap_public_key)

    def probe(target) -> tuple[list[str], str | None, str | None]:
        """One node's whole pass on one worker thread: (lines, k3s_url, gap note).

        Catches everything — _fan_out re-raises a worker's exception at
        iteration time, which would discard every other node's result.
        """
        if not target.enabled:
            skipped = click.style("SKIPPED", fg="yellow")
            return [f"  {target.node_id:20s} {target.host:16s} [{skipped}] disabled in config"], None, None

        wanted = {"hostname": "hostname"}
        if verbose:
            wanted["k3s_env"] = f"sudo cat {AGENT_ENV_PATH}"
        # A reachable password-auth node is the right moment to install our
        # key — the bash scripts only speak `ssh -i`. Folded into this probe
        # rather than a second full handshake per node.
        gap = _no_password_reason(target) if pub_key else None
        if pub_key and gap is None:
            wanted["key_install"] = authorized_keys_command(pub_key)

        conn = conn_kwargs(target, config.ssh.key_store, timeout=config.ssh.connect_timeout_seconds)
        try:
            facts = probe_facts(target.host, wanted, **conn)
        except Exception as e:
            icon = click.style("UNREACHABLE", fg="red")
            detail = str(e) or type(e).__name__
            return (
                [f"  {target.node_id:20s} {target.host:16s} [{icon}] {type(e).__name__}: {detail}"],
                None,
                f"  {target.node_id} ({target.host}): unreachable ({detail})",
            )

        icon = click.style("OK", fg="green")
        lines = [f"  {target.node_id:20s} {target.host:16s} [{icon}] {facts['hostname'] or target.host}"]
        pad = f"  {'':20s} {'':16s}"
        if "key_install" in wanted:
            out = facts["key_install"]
            if "INSTALLED" in out:
                lines.append(f"{pad} SSH key installed")
            elif "ALREADY_PRESENT" not in out:
                # probe_facts discards stderr, so the reason lives one command
                # away rather than being lost.
                lines.append(click.style(
                    f"{pad} SSH key install failed — run `aureax bootstrap-keys` for the reason",
                    fg="yellow",
                ))
        elif gap and target.password_env:
            lines.append(click.style(f"{pad} key bootstrap skipped: {gap}", fg="yellow"))

        if not verbose:
            return lines, None, None
        url = _parse_k3s_url(facts["k3s_env"]) if facts["k3s_env"] else None
        if url is None:
            # Not an all-clear: probe_facts swallows the failure, so a node
            # that never joined or whose `sudo cat` was refused would
            # otherwise appear in neither the drift list nor the gap list.
            return lines, None, f"  {target.node_id} ({target.host}): K3S_URL unreadable (not joined, or sudo failed)"
        return lines, url, None

    click.echo("SSH Connectivity:")
    if key_gap:
        # Named, not silent: without it a fleet on which no key was installed
        # reads exactly like one where every install succeeded, and `ssh -i`
        # keeps failing for the bash scripts with nothing naming the cause.
        click.echo(click.style(f"  key bootstrap disabled: {key_gap}", fg="yellow"))
    k3s_urls: dict[str, str] = {}
    drift_unreachable: list[str] = []
    for target, (lines, url, note) in zip(
        config.ssh.targets, _fan_out(config.ssh.targets, probe), strict=True
    ):
        for line in lines:
            click.echo(line)
        if url is not None:
            k3s_urls[target.node_id] = url
        if note:
            drift_unreachable.append(note)
    return k3s_urls, drift_unreachable


def _status_cluster(config) -> None:
    """K3s node inventory plus the pods in the CONFIGURED namespace."""
    click.echo("K3s Cluster:")
    try:
        from orchestrator.cluster.kubectl import get_nodes, get_pods
        from orchestrator.cluster.labels import double_labelled_nodes

        k8s_nodes = get_nodes()
        for k8s_node in k8s_nodes:
            ready = k8s_node["status"] == "True"
            state = click.style("Ready", fg="green") if ready else click.style("NotReady", fg="red")
            click.echo(f"  {k8s_node['name']:30s} [{state}] {k8s_node['roles']}")

        # Free: the listing above already carries every node's labels, so this
        # is the one place that can surface a double-labelled node without a
        # second API call. It is worth surfacing because NOTHING else notices —
        # a node matching both DaemonSet selectors runs llama-server and
        # rpc-server side by side, each sized for the whole machine, and both
        # pods look perfectly healthy while they compete for its RAM.
        k8s = config.kubernetes
        for name in double_labelled_nodes(k8s_nodes, k8s):
            click.echo(click.style(
                f"  WARNING: {name} carries both {k8s.node_label} and {k8s.shard_label} "
                "— it matches both DaemonSets and runs two pods. Run `aureax deploy`.",
                fg="red",
            ))

        click.echo()
        click.echo("Aureax Pods:")
        # kubernetes.namespace, not a literal: against a renamed namespace a
        # hardcoded one prints "(no pods found)" for a perfectly healthy fleet.
        pods = get_pods(config.kubernetes.namespace)
        if pods:
            for pod in pods:
                ready = click.style("Ready", fg="green") if pod["ready"] else click.style("NotReady", fg="yellow")
                click.echo(f"  {pod['name']:40s} {pod['node']:20s} {pod['status']:10s} [{ready}]")
        else:
            click.echo("  (no pods found)")
    except Exception as e:
        click.echo(f"  Error querying cluster: {e}")


def _status_drift(config, k3s_urls: dict[str, str], unreachable: list[str]) -> None:
    """Compare each node's reported K3S_URL against the orchestrator's LAN IP.

    Silent when everything agrees; the "incomplete" list is what keeps a node
    whose URL could not be read from reading as an all-clear.
    """
    try:
        expected = _expected_server_url(config)
        drift_lines = [
            f"  {node_id}: K3S_URL={configured} (expected {expected}). Run: aureax heal-nodes"
            for node_id, configured in k3s_urls.items()
            if configured != expected
        ]
        if drift_lines:
            click.echo()
            click.echo(click.style("K3s Agent Drift:", fg="red"))
            for line in drift_lines:
                click.echo(click.style(line, fg="red"))
        if unreachable:
            click.echo()
            click.echo(click.style("Drift check incomplete:", fg="yellow"))
            for line in unreachable:
                click.echo(click.style(line, fg="yellow"))
    except Exception as e:
        click.echo(f"  Drift check error: {e}")


def _status_telemetry(config) -> None:
    """Last-seen and sample count per node, from the telemetry store."""
    click.echo("Telemetry:")
    try:
        from orchestrator.telemetry.queries import MetricsQueries

        queries = MetricsQueries(config.server.telemetry.storage.path)
        summary = queries.get_node_health_summary()
        if summary:
            for row in summary:
                click.echo(
                    f"  {row['node_id']:20s} last_seen={row['last_seen']} metrics={row['total_metrics']}"
                )
        else:
            click.echo("  (no metrics received yet)")
    except Exception as e:
        click.echo(f"  Error querying metrics: {e}")


@cli.command("bootstrap-keys")
@click.pass_context
def bootstrap_keys(ctx: click.Context) -> None:
    """Install the orchestrator's public key on every password-auth node.

    The bash scripts only ever speak `ssh -i <key>`, so a node that ships with
    nothing but a factory password is invisible to them until this has run —
    which is why setup-all.sh calls it before its own SSH preflight rather than
    relying on a Python path further down to do it as a side effect.

    Idempotent and best-effort per node: a target with no password is skipped
    (it is already key-only), and one unreachable node does not stop the rest.
    Exits non-zero only if a node that HAS a password could not be bootstrapped,
    since that is the one that will fail the preflight moments later.
    """
    from orchestrator.deploy.ssh import conn_kwargs, ensure_key_installed

    config = ctx.obj["network_config"]
    pub_key = config.ssh.bootstrap_public_key
    if not pub_key:
        click.echo("ssh.bootstrap_public_key is empty — key bootstrapping disabled")
        return

    failures = 0
    for target in config.ssh.enabled_targets:
        gap = _no_password_reason(target)
        if gap:
            click.echo(f"  {target.node_id:14s} skipped ({gap})")
            continue
        ok, detail = ensure_key_installed(
            target.host,
            pub_key,
            **conn_kwargs(target, config.ssh.key_store, timeout=config.ssh.connect_timeout_seconds),
        )
        if ok:
            click.echo(click.style(f"  {target.node_id:14s} {detail}", fg="green"))
        else:
            click.echo(click.style(f"  {target.node_id:14s} FAILED: {detail}", fg="red"))
            failures += 1

    if failures:
        click.echo(click.style(
            f"{failures} node(s) could not be bootstrapped — `ssh -i` will fail for them.",
            fg="red",
        ))
        ctx.exit(1)


@cli.command()
@_orchestrator_ip_option
@_accept_worker_reorder_option
@click.pass_context
def generate(ctx: click.Context, orchestrator_ip: str, accept_worker_reorder: bool) -> None:
    """Generate per-node edge configs from per-node config files."""
    config_path = ctx.obj["config_path"]
    output_dir = generated_config_dir()

    try:
        paths = generate_all_edge_configs(
            config_path, output_dir, orchestrator_ip,
            accept_worker_reorder=accept_worker_reorder,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
        # node_groups validation, unresolved placeholders and a missing
        # template all raise carefully worded messages, and keystore.ensure_key
        # raises RuntimeError on a key file it cannot use (empty, truncated);
        # a traceback would bury any of them.
        raise click.ClickException(str(e)) from e

    click.echo(f"Generated {len(paths)} edge configs:")
    for p in paths:
        click.echo(f"  {p}")


@cli.command()
@_orchestrator_ip_option
@click.option("--setup-cluster", is_flag=True, help="Create k3d cluster and join edge nodes")
@_accept_worker_reorder_option
@click.pass_context
def deploy(ctx: click.Context, orchestrator_ip: str, setup_cluster: bool,
           accept_worker_reorder: bool) -> None:
    """Deploy configs and workloads to edge nodes."""
    import logging

    from orchestrator.deploy.deployer import Deployer

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load the fleet through ctx.obj first, even though nothing here reads
    # it: the Deployer constructor loads the same file itself, and a config
    # that cannot be read or does not validate would surface from there as a
    # raw ConfigError/YAMLError traceback instead of the one-line message
    # every other command prints. The (path, mtime) cache makes the
    # constructor's own load free.
    ctx.obj["network_config"]

    try:
        deployer = Deployer(
            ctx.obj["config_path"], orchestrator_ip,
            accept_worker_reorder=accept_worker_reorder,
        )
        if setup_cluster:
            deployer.full_deploy()
        else:
            click.echo("Generating configs and applying manifests...")
            config_paths = deployer.generate_configs()
            deployer.deploy_configs(config_paths)
            deployer.apply_manifests()
            click.echo("Done.")
    except _DEPLOY_ERRORS as e:
        # Both branches reach generate_all_edge_configs, which raises ValueError
        # for node_groups validation (e.g. a tensor_split whose length does not
        # match a group) and FileNotFoundError for a missing template — neither
        # of them a RuntimeError, and both mid-pipeline.
        click.echo(click.style(f"Deploy failed: {e}", fg="red"))
        ctx.exit(1)


# Substrings of an ``apply_to_nodes`` outcome that mean the node is NOT known
# to be serving its new config. Two of the three carry no FAILED marker:
# "did not become ready" is a head that never re-attached to its RPC workers,
# and "not waiting" is a --wait run whose readiness poll was skipped because
# the pre-delete pod snapshot came back empty. Both must exit non-zero, or a
# script keyed on the exit code reads a broken node as applied.
_APPLY_FAILURE_MARKERS = ("failed", "did not become ready", "not waiting")


def _apply_outcome_failed(outcome: str) -> bool:
    """Whether an apply outcome means the node did NOT end up on its new config.

    Matched on prose because that is what ``apply_to_nodes`` returns. A gate
    that never ran is a failed gate: --wait exists so the caller can be sure
    the replacement pod came back Ready, and reporting success for a skipped
    poll defeats the whole flag.
    """
    lowered = outcome.lower()
    return any(marker in lowered for marker in _APPLY_FAILURE_MARKERS)


@cli.command()
@_orchestrator_ip_option
@click.option("--node", "node_ids", multiple=True,
              help="Node id to apply to (repeatable). A node in a node_groups "
                   "entry applies to its whole group.")
@click.option("--all", "all_nodes", is_flag=True,
              help="Apply to every enabled target. Required to go fleet-wide; "
                   "--node and --all are mutually exclusive.")
@click.option("--no-restart", is_flag=True,
              help="Push configs only; llama-server keeps its OLD settings until restarted")
@click.option("--rotate-key", is_flag=True,
              help="Rotate each selected node's llama AND MCP API keys before pushing")
@click.option("--wait", is_flag=True,
              help="Delete every targeted pod first, then wait for all replacements to become "
                   "Ready. With --all that reloads every head at once.")
@_accept_worker_reorder_option
@click.pass_context
def apply(ctx: click.Context, orchestrator_ip: str, node_ids: tuple[str, ...], all_nodes: bool,
          no_restart: bool, rotate_key: bool, wait: bool, accept_worker_reorder: bool) -> None:
    """Push resolved config to node(s) and restart their llama-server pod.

    The rotate-then-apply sequence is only correct as one atomic action —
    which is why key rotation is a flag here rather than its own command: a
    rotated key that is never applied leaves llama-server enforcing the old
    key while every client sends the new one.

    --rotate-key rotates BOTH of a node's keys, llama-server's and the MCP
    server's. Both protect the same pod and one restart applies both, so
    rotating them separately would buy a second multi-minute head reload for
    nothing: "re-key this node" means the node, not one of its ports.

    Rotating a node's MCP key also pushes every node that DIALS it — the
    planner's remote_servers carries each peer's key — so a re-keyed head
    whose planner was not pushed would leave the planner sending the old
    bearer to a server that has already stopped accepting it.

    Shard workers are pushed but never restarted — rpc-server takes no key,
    and restarting one forces a whole-group reload.
    """
    import logging

    from orchestrator.deploy.deployer import Deployer

    # Fleet-wide is opt-in, never the default: --all rotates every key it is
    # given and restarts every head, so every sharded group reloads at once
    # — and a rotation is recoverable only through the single .key.prev.
    if all_nodes and node_ids:
        raise click.UsageError("--all and --node are mutually exclusive")
    if not all_nodes and not node_ids:
        raise click.UsageError(
            "specify --node <id> (repeatable), or --all to apply to the whole fleet"
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Loaded through ctx.obj first for the same reason as in `deploy`: the
    # Deployer reads the same file, and a config error from inside it is a
    # traceback rather than a one-line message.
    ctx.obj["network_config"]

    try:
        deployer = Deployer(
            ctx.obj["config_path"], orchestrator_ip,
            accept_worker_reorder=accept_worker_reorder,
        )
        outcomes = deployer.apply_to_nodes(
            list(node_ids) or None,
            restart=not no_restart,
            rotate_keys=rotate_key,
            wait=wait,
            log=click.echo,
        )
    except _DEPLOY_ERRORS as e:
        click.echo(click.style(f"Apply failed: {e}", fg="red"))
        ctx.exit(1)

    click.echo()
    failed = False
    for nid, outcome in outcomes.items():
        is_fail = _apply_outcome_failed(outcome)
        failed = failed or is_fail
        click.echo(f"  {nid:20s} {click.style(outcome, fg='red') if is_fail else outcome}")
    if failed:
        ctx.exit(1)


def _cli_logger():
    """SystemLogger that tees emitted lines to click.echo with CLI coloring.

    Levels: error→red, warn→yellow, hint→cyan, info/success→default.
    """
    from orchestrator.lifecycle import SystemLogger

    log = SystemLogger()
    base_emit = log.emit

    # Signature must track SystemLogger.emit exactly, `source` included: many
    # lifecycle call sites pass source="nodes"/"pods", and a tee that does not
    # accept it raises TypeError straight through `aureax start`'s
    # `except RuntimeError`, minutes into a slow cold start.
    def tee(level: str, text: str, source: str = "lifecycle") -> None:
        if level == "error":
            click.echo(click.style(text, fg="red"))
        elif level == "warn":
            click.echo(click.style(text, fg="yellow"))
        elif level == "hint":
            click.echo(click.style(text, fg="cyan"))
        else:
            click.echo(text)
        base_emit(level, text, source)

    log.emit = tee
    return log


_GROUP_FANOUT_NOTE = (
    " A node that belongs to a node_groups entry takes its WHOLE group with it "
    "— the group serves one model and cannot run with a member missing."
)


@cli.command()
@click.option("--node", default=None, help="Stop one node by node_id." + _GROUP_FANOUT_NOTE)
@click.pass_context
def stop(ctx: click.Context, node: str | None) -> None:
    """Stop the entire Aureax stack.

    Stops all node pods, the k3d cluster, and the dashboard.
    All state is preserved — 'aureax start' brings everything back.

    With --node, stops that node's pod by removing its label — and every other
    member of its node_groups entry, since a group serves one model across all
    of them.
    """
    from orchestrator.lifecycle import do_stop, do_stop_node

    config = ctx.obj["network_config"]
    log = _cli_logger()
    try:
        if node:
            do_stop_node(config, node, log, ctx.obj["config_path"])
        else:
            do_stop(config, log, skip_self_kill=False)
    except _LIFECYCLE_ERRORS as e:
        # The message is often the only statement of the cause (e.g. "DaemonSet
        # is not deployed — run `aureax deploy` first"), so it is echoed rather
        # than swallowed into a bare exit 1 with no remedy. A kubectl/k3d
        # failure arrives as CommandError, whose str() carries the captured
        # stderr.
        log.emit("error", str(e))
        ctx.exit(1)


@cli.command()
@click.option("--node", default=None, help="Start one node by node_id." + _GROUP_FANOUT_NOTE)
@click.pass_context
def start(ctx: click.Context, node: str | None) -> None:
    """Start the entire Aureax stack.

    Starts the k3d cluster, waits for edge nodes to connect, restores
    the DaemonSet so pods are scheduled, and waits for them to be ready.

    If no k3d cluster exists yet this instead runs first-time setup end to end
    (scripts/setup-all.sh): image builds on every node and model downloads,
    5-15 min.

    With --node, starts that node's pod by re-adding its label — and every
    other member of its node_groups entry, since a group serves one model
    across all of them.
    """
    from orchestrator.lifecycle import do_start, do_start_node

    config = ctx.obj["network_config"]
    log = _cli_logger()
    try:
        if node:
            do_start_node(config, node, log, ctx.obj["config_path"])
        else:
            do_start(config, log, network_config_path=ctx.obj["config_path"])
    except _LIFECYCLE_ERRORS as e:
        log.emit("error", str(e))
        ctx.exit(1)


@cli.command("heal-nodes")
@click.pass_context
def heal_nodes(ctx: click.Context) -> None:
    """Rewrite stale K3S_URL on edge nodes after an orchestrator IP change.

    Edge node k3s-agents are configured with the orchestrator's IP at setup
    time. When that IP drifts (e.g. DHCP renewal), agents lose the server and
    pods go NotReady. This command SSHes to each node, rewrites
    /etc/systemd/system/k3s-agent.service.env with the current IP, and
    restarts the k3s-agent service.
    """
    from orchestrator.deploy.k3s_agent import reconcile_server_url
    from orchestrator.deploy.ssh import conn_kwargs

    config = ctx.obj["network_config"]

    try:
        expected = _expected_server_url(config)
    except RuntimeError as e:
        click.echo(click.style(f"ERROR: {e}", fg="red"))
        ctx.exit(1)

    click.echo(f"Expected K3s server URL: {expected}")
    targets = config.ssh.enabled_targets

    # Concurrent: each reconcile blocks on a k3s-agent restart, and serialising
    # them stretched the outage the operator is waiting out by one restart per
    # node. reconcile_server_url never raises, so _fan_out's contract holds.
    # connect_timeout, not conn_kwargs' timeout=: reconcile_server_url's own
    # ``timeout`` bounds the install-and-restart and must not be shrunk to a
    # connect budget. It routes the configured value to the short steps that
    # actually carry the connect.
    results = _fan_out(targets, lambda t: reconcile_server_url(
        t.host,
        expected,
        connect_timeout=config.ssh.connect_timeout_seconds,
        **conn_kwargs(t, config.ssh.key_store),
    ))

    failures = 0
    for t, (ok, msg) in zip(targets, results, strict=True):
        color = "green" if ok else "red"
        click.echo(click.style(f"  [{t.node_id}] {msg}", fg=color))
        if not ok:
            failures += 1

    if failures:
        click.echo(click.style(
            f"{failures} node(s) failed to heal. Fix SSH/sudo and retry.",
            fg="red",
        ))
        ctx.exit(1)

    click.echo("Heal complete. Run 'aureax start' to bring workloads back up.")


@cli.command()
@click.option(
    "--port",
    default=None,
    type=int,
    help="Dashboard port (default: server.dashboard.port from the config)",
)
@click.option(
    "--orchestrator-ip",
    default=None,
    callback=_validate_orchestrator_ip,
    help="Address the fleet reaches this orchestrator on, for the dashboard's certificate "
         "(default: detect this Mac's LAN IP). Pass the value `aureax generate` was run with.",
)
@click.pass_context
def dashboard(ctx: click.Context, port: int | None, orchestrator_ip: str | None) -> None:
    """Start the monitoring web dashboard."""
    from orchestrator.cluster import k3d
    from orchestrator.config import tlsstore
    from orchestrator.dashboard.app import LOOPBACK_HOSTS, WILDCARD_HOSTS, run

    config = ctx.obj["network_config"]
    # Config-driven default, because `aureax stop` kills whatever listens on
    # server.dashboard.port: a hardcoded default here would let a changed
    # config point stop at a port the dashboard was never on.
    if port is None:
        port = config.server.dashboard.port
    db_path = config.server.telemetry.storage.path
    host = config.server.dashboard.host
    # The LAN address is both the HTTPS bind (below) and the name the
    # certificate has to carry as a SAN, or every browser and every node's
    # telemetry push fails verification. An explicit --orchestrator-ip wins over
    # detection and skips it: detection is macOS-only and answers with whichever
    # interface reaches the first node, which on a multi-homed Mac (or a Linux
    # orchestrator) need not be the address the fleet's configs name.
    lan_ip = orchestrator_ip or k3d.mac_lan_ip(k3d.probe_host(config))
    if not lan_ip:
        # Refused rather than falling back to the bind host: 0.0.0.0 is not a
        # name anything can verify, and a certificate minted for loopback alone
        # would serve a padlock to nobody on the LAN while every node's
        # telemetry POST failed with a hostname mismatch.
        raise click.ClickException(
            "could not detect this orchestrator's LAN IP, so the dashboard's TLS certificate "
            "cannot be issued for an address the fleet can reach. Detection asks the routing "
            "table which interface reaches the first enabled node (macOS-only, see "
            "orchestrator.cluster.k3d) — pass `--orchestrator-ip <addr>` with the address the "
            "fleet's configs were generated against, or put this machine on the nodes' network."
        )
    # Called on every start, not only the first: the leaf carries the union of
    # every address this orchestrator has been reached at, so a Mac that moved
    # networks gains today's address here instead of serving a certificate that
    # does not carry it. (`aureax tls regenerate --node orchestrator
    # --orchestrator-ip <addr>` is what prunes that union back.)
    try:
        cert_path, key_path = tlsstore.ensure_orchestrator_leaf(lan_ip)
    except (RuntimeError, OSError, ValueError) as e:
        # ValueError is the fleet CA refusing this address: its name
        # constraints do not cover it, and the message names the rotation that
        # fixes it. As a traceback it would read as a crash in the TLS store.
        raise click.ClickException(str(e)) from e
    # server.dashboard.host is the HTTPS bind, and its "0.0.0.0" default means
    # "the address the fleet reaches this orchestrator at" rather than a
    # literal wildcard: the plaintext loopback listener needs the same port,
    # and a wildcard socket owns loopback too. Nodes therefore reach the Mac at
    # exactly this address — the one in every node's ${orchestrator_url} — and
    # a multi-homed Mac serves the fleet on that interface alone.
    bind_host = lan_ip if host in WILDCARD_HOSTS else host
    # An explicitly loopback host is the SSH-tunnel setup the schema describes:
    # HTTPS already owns the port on loopback, so there is no plaintext
    # listener to add beside it.
    tunnelled = bind_host in LOOPBACK_HOSTS
    ca_path = tlsstore.ca_cert_path()
    click.echo(f"Starting Aureax dashboard on {bind_host}:{port}")
    if not tunnelled:
        click.echo(f"  Browse from this machine: http://localhost:{port}/ (no certificate needed)")
    click.echo(f"  Fleet and other machines: https://{bind_host}:{port}/ (fleet CA: {ca_path})")
    if not tunnelled and bind_host != lan_ip:
        # The leaf is minted for lan_ip, so a hand-set server.dashboard.host
        # that is some OTHER address serves a certificate without that name —
        # a hostname mismatch in every browser and every node.
        click.echo(click.style(
            f"  WARNING: server.dashboard.host is {bind_host}, but the certificate was minted "
            f"for {lan_ip}. Re-run with --orchestrator-ip {bind_host}, or clear the setting.",
            fg="yellow",
        ))
    try:
        run(
            port=port,
            metrics_db_path=db_path,
            network_config_path=str(ctx.obj["config_path"]),
            host=bind_host,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            loopback_http=not tunnelled,
        )
    except OSError as e:
        # A busy port is the common one, and app.py::_listen_socket has already
        # turned it into a sentence naming the pid that holds it. As a
        # traceback that sentence is buried under a stack that reads like a
        # crash in the dashboard.
        raise click.ClickException(str(e)) from e


@cli.group("tls")
def tls_group() -> None:
    """Inspect and reissue the fleet's TLS material."""


@tls_group.command("status")
def tls_status() -> None:
    """Report the fleet CA and every issued certificate.

    Never mints anything — the same rule the dashboard's Access tab follows for
    API keys. A status command that created the fleet's root of trust as a side
    effect would hand out a CA no node has ever been given, and every TLS
    failure afterwards would point at the nodes instead of at this Mac.

    Reads no fleet config, so it answers on a machine that has run `aureax
    init` and nothing else.
    """
    from orchestrator.config import tlsstore

    try:
        info = tlsstore.status()
    except (RuntimeError, OSError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"TLS store: {info['store_dir']}")
    ca = info["ca"]
    if ca["present"]:
        state = click.style("present", fg="green")
        click.echo(f"  CA:  [{state}] {ca['subject']}")
        days = ca["expires_in_days"]
        line = f"       expires in {days} days ({ca['not_after']})"
        click.echo(click.style(line, fg="yellow") if ca["expiring_soon"] else line)
        # What this root may certify is the whole security question once the
        # README has had it trusted in a keychain, and nothing else shows it.
        constraints = tlsstore.ca_name_constraints()
        if constraints is None:
            click.echo(click.style(
                "       limits: NONE — this CA can sign any name on the internet. Replace it "
                "with `aureax tls regenerate --ca --orchestrator-ip <ip>`",
                fg="yellow",
            ))
        else:
            permitted_ips, permitted_dns = constraints
            click.echo(f"       limits: IP  {', '.join(permitted_ips) or '(unrestricted)'}")
            click.echo(f"               DNS {', '.join(permitted_dns) or '(unrestricted)'}")
    else:
        click.echo(click.style(f"  CA:  [absent] {ca['problem']}", fg="red"))
    click.echo(f"  CA file: {ca['path']}")

    leaves = info["leaves"]
    click.echo(f"Certificates ({len(leaves)}):")
    if not leaves:
        click.echo("  (none — run `aureax generate --orchestrator-ip <ip>`)")
    for leaf in leaves:
        mark = click.style("ok", fg="green") if leaf["reissue_reason"] is None else click.style(
            "stale", fg="yellow")
        days = leaf["expires_in_days"]
        expiry = f"expires in {days} days" if days is not None else "no certificate"
        click.echo(f"  {leaf['name']:20s} [{mark}] {expiry}")
        click.echo(f"  {'':20s} SANs: {', '.join(leaf['sans']) or '(none)'}")
        if leaf["reissue_reason"]:
            # Named, not just flagged: "stale" alone sends an operator to
            # `regenerate` when the fix is usually `aureax generate`, which
            # reissues from the fleet config and pushes nothing.
            click.echo(click.style(
                f"  {'':20s} reissue: {leaf['reissue_reason']} — run "
                "`aureax generate --orchestrator-ip <ip>`", fg="yellow"))


@tls_group.command("ca-path")
def tls_ca_path() -> None:
    """Print the path of the fleet CA certificate, and nothing else.

    Shaped for substitution — `curl --cacert "$(aureax tls ca-path)" …` — so it
    prints one line with no decoration even when the file does not exist yet.
    """
    from orchestrator.config import tlsstore

    click.echo(str(tlsstore.ca_cert_path()))


def _ca_rotation_steps(orchestrator_ip: str, ca_path: Path) -> list[str]:
    """The whole post-rotation procedure, in the order it has to be run.

    Every step is here because something else caches the old root, and the
    documented pair (`regenerate --ca` then `apply --all`) leaves most of the
    fleet on it: `apply` restarts only heads and standalone nodes — a shard
    worker is pushed but never restarted, because restarting one forces a
    whole-group reload — and every supervisor's telemetry and log reporter
    snapshots its SSLContext when it starts. Their pushes then fail
    verification against a dashboard serving the new leaf, at DEBUG level, so
    the only symptom is those nodes quietly vanishing from the Overview tab.
    """
    return [
        f"1. aureax apply --all --orchestrator-ip {orchestrator_ip}",
        "     pushes the new CA and every leaf; restarts heads and standalone nodes.",
        "2. aureax stop && aureax start",
        "     shard workers are never restarted by apply, and every node's telemetry and log",
        "     reporter holds the CA it read at startup.",
        "3. restart `aureax dashboard`",
        "     it serves its own leaf and caches a CA context of its own.",
        f"4. re-trust {ca_path} in every browser that opens the dashboard.",
    ]


@tls_group.command("regenerate")
@click.option("--ca", "regenerate_ca", is_flag=True,
              help="Replace the fleet CA itself, and with it every certificate under it")
@click.option("--node", "node_ids", multiple=True,
              help="Reissue only this certificate (repeatable; 'orchestrator' is the "
                   "dashboard's). Default: every certificate in the store.")
@click.option("--orchestrator-ip", default=None, callback=_validate_orchestrator_ip,
              help="Address this orchestrator is reached on. Required with --ca (the new CA is "
                   "name-constrained to this fleet) and with --node orchestrator (which resets "
                   "the dashboard's leaf to this one address).")
@click.pass_context
def tls_regenerate(ctx: click.Context, regenerate_ca: bool, node_ids: tuple[str, ...],
                   orchestrator_ip: str | None) -> None:
    """Force-reissue certificates that are otherwise reissued only when stale.

    Reissuing a leaf is safe on its own — the node keeps trusting the same CA,
    so only that node needs the new pair pushed. Replacing the CA is not: every
    leaf in the fleet is signed by it, so nothing verifies until every step of
    the procedure printed below has been run.

    ``--node orchestrator`` is the one reset lever on the dashboard's leaf,
    which otherwise only ever accumulates addresses (ensure_orchestrator_leaf
    unions, so that `aureax generate` and `aureax dashboard` cannot delete each
    other's SAN). It is dropped and minted again for --orchestrator-ip alone.
    """
    from orchestrator.config import tlsstore

    reset_orchestrator = tlsstore.ORCHESTRATOR_NAME in node_ids
    if not orchestrator_ip and (regenerate_ca or reset_orchestrator):
        # A UsageError (exit 2), not a prompt: the address is baked into a
        # certificate the whole fleet then verifies against, so guessing it
        # here would be guessing on the fleet's behalf.
        flag = "--ca" if regenerate_ca else f"--node {tlsstore.ORCHESTRATOR_NAME}"
        raise click.UsageError(
            f"{flag} needs --orchestrator-ip <addr>: the address the fleet's configs were "
            f"generated against, which is what the new certificate has to carry."
        )

    if regenerate_ca:
        click.echo(click.style(
            "Replacing the fleet CA. Every node's certificate is reissued under the new root, "
            "and until the whole procedure below has run the fleet trusts nothing: telemetry "
            "pushes, the harness and the dashboard's own probes all fail verification. Every "
            "browser that trusted the old CA must trust the new one again.",
            fg="red", bold=True,
        ))

    try:
        if regenerate_ca:
            # The constraints are baked at creation and the CA is never rotated
            # silently, so this call is the only chance to scope the new root to
            # the fleet — an unconstrained one is a universal MITM credential on
            # whichever Mac trusts it.
            permitted_ips, permitted_dns = tlsstore.fleet_name_constraints(
                ctx.obj["network_config"], orchestrator_ip)
            tlsstore.regenerate(
                ca=True, names=list(node_ids) or None,
                permitted_ips=permitted_ips, permitted_dns=permitted_dns,
            )
        else:
            # The orchestrator leaf is handled below, by deletion: the store
            # reissues it with its loopback names only — its SAN set is a
            # running union, so keeping it would preserve every address it has
            # ever carried, which is what --node orchestrator exists to drop.
            others = [n for n in node_ids if n != tlsstore.ORCHESTRATOR_NAME]
            if others or not reset_orchestrator:
                tlsstore.regenerate(names=others or None)
        if reset_orchestrator or regenerate_ca:
            # --orchestrator-ip is what puts today's address back, so the
            # dashboard can serve the fleet again before the next `aureax
            # generate` rather than on a loopback-only certificate.
            cert_path, key_path = tlsstore.leaf_paths(
                tlsstore.ORCHESTRATOR_NAME, allow_reserved=True)
            cert_path.unlink(missing_ok=True)
            key_path.unlink(missing_ok=True)
            tlsstore.ensure_orchestrator_leaf(orchestrator_ip)
    except (RuntimeError, OSError, ValueError) as e:
        # ValueError included: a --node value that is not a single path segment
        # is refused at the store's path builder, and that refusal names the
        # value the operator typed. Unhandled it would be a traceback.
        raise click.ClickException(str(e)) from e

    if regenerate_ca:
        click.echo("Reissued under the new CA. The fleet is down until all four steps have run:")
        for line in _ca_rotation_steps(orchestrator_ip, tlsstore.ca_cert_path()):
            click.echo(f"  {line}")
    else:
        ip = orchestrator_ip or "<ip>"
        click.echo(
            f"Reissued. Next: `aureax apply --all --orchestrator-ip {ip}` to push the new "
            "certificates to every node."
        )
    if reset_orchestrator:
        # uvicorn reads the pair once at startup, so the dashboard keeps
        # serving the certificate this command just replaced.
        click.echo("A running `aureax dashboard` serves the old certificate until it is restarted.")


# ---------------------------------------------------------------------------
# aureax wg — the opt-in WireGuard transport for rpc-server
# ---------------------------------------------------------------------------

@cli.group("wg")
def wg_group() -> None:
    """Inspect and rotate the WireGuard keys that carry a group's RPC traffic."""


@wg_group.command("status")
@click.pass_context
def wg_status(ctx: click.Context) -> None:
    """Per group with wireguard enabled: subnet, every member's address and key.

    Never mints anything — the rule `tls status` and the Access tab follow. A
    member with no key is reported, and `aureax generate` is what mints it.
    """
    from orchestrator.config import wgstore

    config = ctx.obj["network_config"]
    click.echo(f"WireGuard store: {wgstore.store_dir()}")
    if not config.node_groups:
        click.echo("No node groups configured.")
        return
    for g in config.node_groups:
        addresses = config.wireguard_addresses(g)
        if addresses is None:
            click.echo(
                click.style(f"{g.name}", bold=True)
                + "  [off] rpc travels the LAN, firewalled to the head "
                "(set node_groups[].wireguard.enabled: true to tunnel it)"
            )
            continue
        net = config.wireguard_network(g)
        wg = g.wireguard
        click.echo(
            click.style(f"{g.name}", bold=True)
            + f"  [on] subnet {net}  udp/{wg.listen_port}  mtu {wg.mtu}  "
            f"keepalive {wg.persistent_keepalive}s"
        )
        for node_id in g.members:
            role = "head" if node_id == g.head else "worker"
            key_path, _pub_path = wgstore.key_paths(node_id)
            try:
                pub = wgstore.public_key(node_id)
            except ValueError as e:
                pub_text = click.style(f"unreadable: {e}", fg="red")
            else:
                pub_text = pub if pub else click.style(
                    "no key — run `aureax generate --orchestrator-ip <ip>`", fg="yellow")
            key_file = (click.style("present", fg="green") if key_path.exists()
                        else click.style("absent", fg="yellow"))
            click.echo(f"  {role:7s} {node_id:14s} {addresses[node_id]:15s} key file {key_file}")
            click.echo(f"  {'':7s} {'':14s} public key {pub_text}")


@wg_group.command("regenerate")
@click.option("--node", "node_ids", multiple=True, required=True,
              help="Rotate this member's WireGuard key pair (repeatable).")
@click.pass_context
def wg_regenerate(ctx: click.Context, node_ids: tuple[str, ...]) -> None:
    """Replace a member's WireGuard key pair; the old private key is kept as .key.prev.

    Only members of a group with wireguard enabled have a pair, so any other
    node id is refused before anything is written. The rotation reaches the
    tunnel only through the two steps printed afterwards.
    """
    from orchestrator.config import wgstore

    config = ctx.obj["network_config"]
    refused = []
    for node_id in node_ids:
        group = config.group_for_node(node_id)
        if group is None or not group.wireguard.enabled:
            refused.append(node_id)
    if refused:
        raise click.ClickException(
            f"{', '.join(refused)}: not a member of a node group with wireguard.enabled — "
            "only those carry a WireGuard key"
        )

    try:
        for node_id in node_ids:
            pub = wgstore.rotate_key(node_id)
            click.echo(f"Rotated {node_id}: public key {pub}")
    except (OSError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    nodes = " ".join(f"--node {n}" for n in node_ids)
    click.echo("Next, in order:")
    click.echo(f"  1. aureax apply {nodes} --orchestrator-ip <ip>")
    click.echo("       pushes the new wg0.conf to these nodes and the rest of their group (every")
    click.echo("       peer carries the rotated public key) and restarts wg-quick@wg0 where it changed.")
    click.echo("  2. aureax stop && aureax start")
    click.echo("       apply never restarts a shard worker, and a worker's rpc-server still holding")
    click.echo("       the old head connection refuses the new head until its pod is recreated.")


@cli.group()
def group() -> None:
    """Inspect sharded node groups."""


@group.command("list")
@click.pass_context
def group_list(ctx: click.Context) -> None:
    """List configured node groups and their members."""
    config = ctx.obj["network_config"]
    if not config.node_groups:
        click.echo("No node groups configured.")
        return

    from orchestrator.config.keystore import KIND_LLAMA, KIND_MCP

    hosts = config.host_by_node_id()
    by_id = {t.node_id: t for t in config.ssh.targets}
    configs_dir = ctx.obj["config_path"].parent
    for g in config.node_groups:
        click.echo(click.style(f"{g.name}", bold=True) + f"  {g.display_name or ''}")
        click.echo(f"  head:    {g.head:14s} {hosts.get(g.head, '?')}")
        for w in g.workers:
            click.echo(f"  worker:  {w:14s} {hosts.get(w, '?')}:{g.rpc_port}")
        # With the tunnel on, the head dials these and not the LAN addresses
        # above — which is exactly what an operator reading `ss` on a worker
        # needs to know.
        wg_addresses = config.wireguard_addresses(g)
        if wg_addresses is not None:
            dials = ", ".join(f"{w} {wg_addresses[w]}:{g.rpc_port}" for w in g.workers)
            click.echo(
                f"  rpc:     over wireguard {config.wireguard_network(g)} "
                f"(head {wg_addresses[g.head]}; dials {dials})"
            )
        # https, and no plaintext spelling anywhere: both listeners on the head
        # serve TLS with the fleet CA, so an http:// URL copied from here fails
        # at the first byte rather than falling back. The host goes through the
        # tree's single IPv6-bracketing rule so a literal address is copyable.
        head_host = _url_host(hosts.get(g.head, "?"))
        click.echo(f"  MCP:     https://{head_host}:{MCP_NODE_PORT}/mcp")
        click.echo(f"  llama:   https://{head_host}:{LLAMA_NODE_PORT}")
        head_target = by_id.get(g.head)
        if head_target is not None:
            # Both keys, for the same reason `nodes` prints both: the two URLs
            # printed above are protected by different keys.
            click.echo(f"  auth:    llama: {_auth_status(configs_dir, head_target, KIND_LLAMA)}")
            click.echo(f"           mcp:   {_auth_status(configs_dir, head_target, KIND_MCP)}")
        click.echo()


# Collected in one SSH round trip per node — the point is to answer "which of
# these machines is the problem" fast, not to be pretty.
#
# The container-runtime probes carry their own ceiling (`timeout 10` each,
# `timeout 20` for the spliced-in BUILD_PROBE, which starts a container and
# hashes two binaries in it), and group_doctor runs the whole script under ONE
# run_command budget that stays above their sum: run_command discards the stdout
# it has already collected the moment its budget expires, so an unbounded probe
# against a wedged docker daemon — the exact state the doctor exists to diagnose
# — takes arch, memory, NIC, GGUF, RPC and log facts for that node down with it.
# A probe added here needs its own ceiling, and group_doctor's budget has to
# stay above the new total (50s today).
_DOCTOR_PROBE = r"""
echo "arch=$(uname -m) cores=$(nproc) mem=$(awk '/MemAvailable/{printf "%.1fGi", $2/1048576}' /proc/meminfo)"
IF=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
echo "nic=${IF} speed=$(cat /sys/class/net/$IF/speed 2>/dev/null || echo '?')Mb"
# The field name and the flags that matter are both per-architecture:
# aarch64 lists dot-product/matmul extensions under `Features`, x86_64 lists
# AVX under `flags`. Reporting one architecture's flags on the other prints a
# bare 0, which reads as "this CPU lacks them" rather than "wrong question".
case "$(uname -m)" in
  aarch64) echo "isa=$(grep -m1 -i '^Features' /proc/cpuinfo | cut -d: -f2- | tr ' ' '\n' | grep -cE '^(asimddp|i8mm|bf16)$') dp/i8mm/bf16 flags" ;;
  x86_64)  echo "isa=$(grep -m1 -i '^flags' /proc/cpuinfo | cut -d: -f2- | tr ' ' '\n' | grep -cE '^(avx2|avx512f|amx_bf16)$') avx2/avx512f/amx_bf16 flags" ;;
  *)       echo "isa=n/a (no ISA probe for $(uname -m))" ;;
esac
""" + BUILD_PROBE + r"""
# Both copies: pods run the image in K3s containerd, which `docker save | ctr
# images import` populates separately. A stale import — the exact drift that
# hangs an RPC handshake — is invisible if you only look at the Docker one.
# (The Docker copy's build record, the two binary checksums and the image's
# architecture come from the spliced-in BUILD_PROBE above — one container start
# for all four, and the same command the pre-start gate runs, so the doctor and
# the gate can never disagree about what they measured. That architecture is
# `image_arch`, not the `arch` on the first line above: that one is the host's.)
echo "llama_build_k3s=$(timeout 10 sudo k3s ctr images ls -q 2>/dev/null | grep -c '^docker.io/library/aureax-client:dev$' | sed 's/^0$/MISSING/;s/^1$/present/')"
echo "image_digest_docker=$(timeout 10 sudo docker image inspect aureax-client:dev --format '{{.Id}}' 2>/dev/null | cut -c8-19 || echo '?')"
echo "image_digest_k3s=$(timeout 10 sudo k3s ctr images ls 2>/dev/null | awk '$1=="docker.io/library/aureax-client:dev"{print $3}' | cut -c8-19)"
echo "gguf=$(ls -la /opt/aureax/models/*.gguf 2>/dev/null | awk '{print $9" ("int($5/1073741824)"GB)"}' | tr '\n' ' ' || echo none)"
echo "rpc_listen=$( (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -c ':50052' )"
echo "rpc_peers=$( (ss -tn 2>/dev/null || netstat -tn 2>/dev/null) | grep ':50052' | awk '{print $5}' | cut -d: -f1 | sort -u | tr '\n' ',' )"
echo "--- last log lines ---"
for f in /opt/aureax/logs/llama-server.log /opt/aureax/logs/rpc-server.log; do
    [ -f "$f" ] && { echo "[$f]"; tail -12 "$f"; }
done
"""


# The one key in a member's fact dict that the node did NOT report. Kept as a
# named set so "was this member probed at all?" stays an honest question: a
# member whose SSH failed still has its address, and treating that as facts
# would make an unreachable node read as a coherent one.
_DOCTOR_INJECTED_KEYS = frozenset({"ssh_host", "wg_host"})

# What the probe prints between its facts and the tail of the node's logs.
_DOCTOR_LOG_SEPARATOR = "--- last log lines ---"


def _parse_doctor_facts(out: str) -> dict[str, str]:
    """``key=value`` lines from ``_DOCTOR_PROBE``, up to the log-lines separator.

    Split on the FIRST ``=`` only: ``rpc_peers`` and ``gguf`` values contain
    none, but ``llama_build_docker``'s fallback text ("unavailable (image
    missing, or docker/sudo failed)") and its JSON payload must survive intact
    — the verdict quotes them back at the operator.

    Stops at the separator rather than filtering, because the log tail below it
    is llama-server output full of ``key = value`` lines ("n_ctx = 131072") that
    would otherwise land in the fact dict and shadow a real probe key.
    """
    facts: dict[str, str] = {}
    for raw in (out or "").splitlines():
        line = raw.strip()
        if line.startswith(_DOCTOR_LOG_SEPARATOR):
            break
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            facts[key] = value.strip()
    return facts


def _probe_reached(facts: dict[str, str]) -> bool:
    """Whether this member reported anything of its own."""
    return bool(set(facts) - _DOCTOR_INJECTED_KEYS)


def _rpc_listen_count(value: str | None) -> int:
    """``rpc_listen`` as a number; anything unparseable counts as zero listeners.

    The probe pipes through ``grep -c``, so a shell that could not run either
    ``ss`` or ``netstat`` yields "" rather than a count — reporting that as a
    listener would clear the one check that catches a worker whose rpc-server
    never bound.
    """
    try:
        return int((value or "").strip())
    except ValueError:
        return 0


def _group_problems(group, facts_by_node: dict[str, dict], head_id: str) -> tuple[list[str], list[str]]:
    """Cross-member verdicts for one group: ``(problems, notes)``.

    ``problems`` are FAIL lines and an empty list means the group is coherent —
    that is what decides ``group doctor``'s exit code. ``notes`` are the OK/SKIP
    lines, which exist because "no output" and "checked, all good" look
    identical otherwise.

    Two lists rather than one, shaped after ``group_state.order_problems``: the
    caller owns colour and exit code, this owns truth. Pure over the fact dicts,
    so every verdict below is unit-testable with no SSH.

    ``facts_by_node`` maps node_id -> the parsed probe facts plus ``ssh_host``
    (see ``_DOCTOR_INJECTED_KEYS``).
    """
    from orchestrator.config.schema import DEFAULT_WORKER_WAIT_SECONDS, parse_llama_build

    problems: list[str] = []
    notes: list[str] = []
    facts_of = {nid: (facts_by_node.get(nid) or {}) for nid in group.members}
    probed = {nid: f for nid, f in facts_of.items() if _probe_reached(f)}

    for nid in group.members:
        if nid not in probed:
            # FAIL, not SKIP: the whole point of the doctor is to answer "do
            # these machines agree", and one member missing makes the answer
            # unknown rather than yes.
            problems.append(
                f"FAIL {nid}: could not be probed, so this group's build consistency is UNKNOWN.\n"
                f"       The member report above says why (unreachable, sudo, or a wedged container\n"
                f"       runtime). Fix that node and re-run `aureax group doctor`."
            )

    # --- one llama.cpp build across every member -----------------------------
    builds: dict[str, int] = {}
    for nid, facts in probed.items():
        raw = facts.get("llama_build_docker", "")
        build = parse_llama_build(raw)
        if build is None:
            problems.append(
                f"FAIL {nid}: llama.cpp build unreported — the probe said {raw or '(nothing)'!r},\n"
                f"       so this group's build consistency cannot be checked. Every peer must run the\n"
                f"       same llama.cpp build. Rebuild this node: ./scripts/update-all.sh"
            )
        else:
            builds[nid] = build
    head_build = builds.get(head_id)
    if head_build is not None:
        for nid, build in builds.items():
            if nid != head_id and build != head_build:
                problems.append(
                    f"FAIL {nid}: llama.cpp build b{build}, head {head_id} runs b{head_build}. The RPC "
                    f"wire\n       format is version-locked — the head connects and then HANGS at the "
                    f"handshake\n       with no error. Rebuild this node: ./scripts/update-all.sh"
                )
        if len(builds) == len(group.members) and len(set(builds.values())) == 1:
            notes.append(f"OK   all {len(builds)} member(s) run llama.cpp build b{head_build}")

    # --- the commit, and the binaries within one architecture ---------------
    # The stronger form of the check above and the one the pre-start gate runs:
    # a build id is a git tag, which can move, so the commit the image recorded
    # is what has to match on every member; and within the head's architecture
    # two members can report the same commit and still hold different binaries
    # (different compiler, different flags, a patched tree), so there the bytes
    # are compared too. Across architectures they never match, and a member of
    # another architecture is compared by commit alone. The digests below cannot
    # answer any of this — every node builds its own image, so they differ
    # across a perfectly coherent group.
    from orchestrator.cluster import buildcheck

    # `probed`, not `facts_by_node`: a member that answered nothing at all
    # already carries its own FAIL above, and a second line saying the build
    # check did not cover it is noise. A member that answered but could not read
    # its own binaries (sudo, a wedged docker) is NOT in that bucket and does
    # get the note — that is the case the note exists for.
    sum_problems, sum_notes = buildcheck.compare_group(group, probed)
    problems.extend(sum_problems)
    notes.extend(sum_notes)

    # --- the Docker image and the containerd copy are the same image ---------
    digest_mismatches = 0
    digests_compared = 0
    for nid, facts in probed.items():
        docker = facts.get("image_digest_docker", "")
        k3s = facts.get("image_digest_k3s", "")
        # An absent digest on either side has its own check with a sharper
        # remedy (the build check above for Docker, the containerd check below
        # for K3s). Comparing here as well would print two lines for one fault,
        # the vaguer one first.
        if not docker or docker == "?" or not k3s:
            continue
        digests_compared += 1
        if docker != k3s:
            digest_mismatches += 1
            problems.append(
                f"FAIL {nid}: the image in Docker (sha256:{docker}) is not the image imported\n"
                f"       into K3s containerd (sha256:{k3s}). Pods run the containerd copy, so a\n"
                f"       rebuild that was never re-imported is invisible. Re-import on that node:\n"
                f"       `sudo docker save aureax-client:dev | sudo k3s ctr images import -`"
            )
    if digests_compared and not digest_mismatches:
        notes.append("OK   image digests match on every member")

    for nid, facts in probed.items():
        if facts.get("llama_build_k3s") == "MISSING":
            problems.append(
                f"FAIL {nid}: aureax-client:dev is absent from K3s containerd, so its pod cannot\n"
                f"       start (imagePullPolicy is Never). Import it: `sudo docker save aureax-client:dev |\n"
                f"       sudo k3s ctr images import -`"
            )

    if _group_is_stopped(group, probed, head_id):
        # Everything below this line asks a runtime question, and a stopped
        # group answers "no" to all of them without being broken.
        notes.append(
            f"SKIP group is stopped — runtime checks skipped: nothing is listening on "
            f":{group.rpc_port} on any worker and head {head_id} holds no RPC peer connection. "
            f"The build, image and digest checks above still ran and still gate the exit code."
        )
        return problems, notes

    # --- every worker is actually listening ----------------------------------
    listening = 0
    checked_workers = 0
    for nid in group.workers:
        facts = probed.get(nid)
        if facts is None:
            continue
        checked_workers += 1
        if _rpc_listen_count(facts.get("rpc_listen")) >= 1:
            listening += 1
        else:
            problems.append(
                f"FAIL {nid}: nothing is listening on :{group.rpc_port}. The head waits out its\n"
                f"       worker_wait_timeout_seconds ({DEFAULT_WORKER_WAIT_SECONDS}s) and then exits 1, "
                f'which reads as "the worker\n       never came up". Check the aureax-shard pod on this '
                f"node."
            )
    if checked_workers and listening == checked_workers:
        notes.append(f"OK   {listening}/{len(group.workers)} workers listening on :{group.rpc_port}")

    problems_peers, notes_peers = _peer_verdict(group, facts_of, probed, head_id)
    problems.extend(problems_peers)
    notes.extend(notes_peers)
    return problems, notes


def _peers_of(facts: dict) -> set[str]:
    """The addresses this member holds :50052 connections to."""
    return {p.strip() for p in (facts.get("rpc_peers") or "").split(",") if p.strip()}


def _group_is_stopped(group, probed: dict[str, dict], head_id: str) -> bool:
    """Whether this group is merely STOPPED rather than broken.

    `aureax stop` terminates every rpc-server pod and the head's llama-server
    but leaves the machines up, so a stopped group is exactly "no worker
    listening on :50052 and no peer connection on the head" while every member
    still answers SSH and reports its build, image and digest facts. Reporting
    one FAIL per worker for that made `group doctor` exit 1 on a fleet nobody
    had started — and a pre-flight that fails on the state it is a pre-flight
    for is one nobody runs.

    Requires the head and at least one worker to have actually been probed: a
    member that did not answer makes the group UNKNOWN, not stopped, and each
    silent member already carries its own FAIL.
    """
    head_facts = probed.get(head_id)
    if head_facts is None or _peers_of(head_facts):
        return False
    workers = [probed[nid] for nid in group.workers if nid in probed]
    return bool(workers) and not any(
        _rpc_listen_count(f.get("rpc_listen")) >= 1 for f in workers
    )


def _host_addresses(host: str, cache: dict[str, set[str] | None]) -> set[str] | None:
    """Every address ``host`` resolves to, or ``None`` when it resolves to none.

    An IP literal resolves to itself without touching the network, so this only
    costs a lookup on a fleet whose targets are names — which is the case it
    exists for. The cache belongs to the caller, so a host that appears twice
    in a group, or one whose lookup has to time out, is paid for once.
    """
    if host not in cache:
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except (OSError, UnicodeError):
            # Best effort: an unresolvable host is compared literally and the
            # verdict says so, because "DNS is down on this Mac" must not read
            # as "the head is attached to the wrong machines".
            cache[host] = None
        else:
            cache[host] = {info[4][0] for info in infos}
    return cache[host]


def _peer_verdict(group, facts_of: dict[str, dict], probed: dict[str, dict],
                  head_id: str) -> tuple[list[str], list[str]]:
    """The head's live RPC peers against the group's configured workers.

    Stronger than a count and free — ``rpc_peers`` is already the list of
    addresses the head holds connections to. An empty peer list is a SKIP and
    never a FAIL: a head still streaming tensors to its peers looks exactly
    like one that never attached. (A group that is merely stopped never gets
    here — ``_group_is_stopped`` skips both runtime checks for it.)
    """
    head_facts = probed.get(head_id)
    if head_facts is None:
        return [], []
    peers = _peers_of(head_facts)
    hosts = {nid: h for nid in group.workers if (h := (facts_of.get(nid) or {}).get("ssh_host"))}
    if not peers:
        return [], [
            f"SKIP head {head_id} is attached to no peers (the head has not finished loading, "
            "or its workers are down)"
        ]
    if not hosts:
        return [], [f"SKIP peer check: no host address is known for {group.name}'s workers"]

    # Resolved, not compared verbatim: the head reports the ADDRESSES its
    # kernel holds connections to, while ssh.targets gives whatever the
    # operator typed. On a fleet named by DNS ("shard-a1.lan") those never
    # match as strings, and a fully attached group was reported lopsided.
    cache: dict[str, set[str] | None] = {}
    unresolved: list[str] = []
    addrs: dict[str, set[str]] = {}
    for nid, host in hosts.items():
        resolved = _host_addresses(host, cache)
        if resolved is None:
            unresolved.append(host)
        addrs[nid] = {host} | (resolved or set())
        # A group on WireGuard: the head holds its connections to the workers'
        # TUNNEL addresses (injected by group_doctor), which no resolver knows.
        wg_host = (facts_of.get(nid) or {}).get("wg_host")
        if wg_host:
            addrs[nid].add(wg_host)

    missing = sorted(host for nid, host in hosts.items() if not (peers & addrs[nid]))
    unexpected = sorted(p for p in peers if not any(p in a for a in addrs.values()))
    if not missing and not unexpected:
        return [], [f"OK   head {head_id} is attached to all {len(hosts)} worker(s)"]

    detail = f" Missing: {', '.join(missing)}." if missing else ""
    if unexpected:
        detail += f" Attached to {', '.join(unexpected)}, which is not in this group."
    if unresolved:
        detail += (
            f" {', '.join(sorted(set(unresolved)))} could not be resolved to an address, so it "
            f"was compared literally — check this Mac's resolver before the node."
        )
    return [
        f"FAIL {head_id}: attached to {len(peers)} RPC peer(s) {', '.join(sorted(peers))} but the\n"
        f"       group has {len(hosts)} worker(s) {', '.join(sorted(hosts.values()))}. The group is "
        f"serving on\n       fewer devices than configured — every probe stays green and the endpoint "
        f"answers,\n       it is just lopsided.{detail}"
    ], []


@group.command("doctor")
@click.option("--name", default=None, help="Only check this group")
@click.pass_context
def group_doctor(ctx: click.Context, name: str | None) -> None:
    """Diagnose every node in a group in one pass.

    Answers "which of these machines is the problem" without hunting through
    kubectl, SSH sessions and node-local log files by hand.
    """
    from orchestrator.deploy.ssh import conn_kwargs, run_command

    config = ctx.obj["network_config"]
    groups = [g for g in config.node_groups if name is None or g.name == name]
    if not groups:
        click.echo("No matching node groups configured.")
        return

    by_id = {t.node_id: t for t in config.ssh.targets}
    members = [(g, node_id) for g in groups for node_id in g.members]

    def probe(item) -> tuple[dict[str, str], list[str]]:
        """One member's (facts, rendered lines). Never raises (_fan_out)."""
        g, node_id = item
        target = by_id.get(node_id)
        role = "head" if node_id == g.head else "worker"
        if target is None:
            return {}, [click.style(f"  {node_id} ({role}): not in ssh.targets", fg="red")]

        # The SSH host travels WITH the facts because the probe cannot report
        # the address the head dialled it on: rpc_peers is a list of addresses
        # and node_groups is a list of node ids, so the peer check needs the
        # mapping between them. _DOCTOR_INJECTED_KEYS records that this one key
        # did not come from the node.
        facts: dict[str, str] = {"ssh_host": target.host}
        # A group on WireGuard is attached over the tunnel, so the head's
        # rpc_peers are tunnel addresses; the peer check needs those too.
        wg_addresses = config.wireguard_addresses(g)
        if wg_addresses is not None:
            facts["wg_host"] = wg_addresses[node_id]
        lines = [click.style(f"\n-- {node_id} ({role}) {target.host}", fg="cyan")]
        try:
            rc, out, err = run_command(
                target.host, _DOCTOR_PROBE,
                # 90s: above the sum of _DOCTOR_PROBE's own ceilings (three
                # `timeout 10` probes plus BUILD_PROBE's `timeout 20`), because
                # run_command discards the stdout it has already collected the
                # moment its own budget expires. A tighter budget than the script
                # can take would erase every fact on exactly the node the doctor
                # was run against.
                **conn_kwargs(target, config.ssh.key_store, timeout=90),
            )
        except Exception as e:
            lines.append(click.style(f"   UNREACHABLE: {type(e).__name__}: {e}", fg="red"))
            return facts, lines
        if rc == 124:
            # run_command's timeout convention. It returns no stdout at all in
            # that case, so say so rather than letting a node with no facts
            # read like a node with nothing to report.
            lines.append(click.style(
                "   PROBE TIMED OUT — no facts collected (a wedged container "
                "runtime or a sudo password prompt on this node)",
                fg="yellow",
            ))
        for line in (out or "").strip().splitlines():
            lines.append(f"   {line}")
            lines.extend(_build_floor_warning(line))
        if err.strip():
            lines.append(click.style(f"   stderr: {err.strip()[:200]}", fg="yellow"))
        facts.update(_parse_doctor_facts(out or ""))
        return facts, lines

    # Concurrent, because the probe starts a container and shells out to
    # `k3s ctr` twice per node — seconds each on an edge node — and an
    # unreachable member would burn its whole 90s budget before the next
    # member was even tried.
    results = _fan_out(members, probe)
    facts_by_node = {
        node_id: facts for (_g, node_id), (facts, _lines) in zip(members, results, strict=True)
    }
    reports = iter(results)
    failed_groups: list[str] = []
    for g in groups:
        click.echo(click.style(f"=== group {g.name} ===", bold=True))
        for _ in g.members:
            for line in next(reports)[1]:
                click.echo(line)
        problems, notes = _group_problems(g, facts_by_node, g.head)
        click.echo(click.style(f"\n  --- verdict: {g.name} ---", bold=True))
        for problem in problems:
            click.echo(click.style(f"  {problem}", fg="red"))
        for note in notes:
            click.echo(click.style(f"  {note}", fg="yellow" if note.startswith("SKIP") else "green"))
        if problems:
            failed_groups.append(g.name)
        click.echo()

    # Non-zero on any group problem, because this is what a script or a
    # pre-flight step keys on: a doctor that always exits 0 turns "the group is
    # split across two llama.cpp builds" into a wall of text somebody has to
    # read. The below-floor build warning deliberately does NOT gate — it is a
    # security advisory about a knowingly-run rollback, not a peer mismatch.
    if failed_groups:
        click.echo(click.style(
            f"{len(failed_groups)}/{len(groups)} group(s) have problems: "
            f"{', '.join(failed_groups)}", fg="red", bold=True))
        ctx.exit(1)
    click.echo(click.style(f"{len(groups)} group(s) coherent.", fg="green"))


def _build_floor_warning(line: str) -> list[str]:
    """Warning lines for a doctor probe line reporting a below-floor build.

    The floor (MIN_LLAMA_BUILD; the pin, b11429, clears it) is above every
    known fixed-in build for the 2026 llama-server/rpc-server CVE set, so a
    member below it is running exploitable code. Reads the DOCKER copy's build id — a stale containerd
    import can still lag it, which is what the image_digest_docker/k3s pair
    beside it exposes.
    """
    if not line.startswith("llama_build_docker="):
        return []
    from orchestrator.config.schema import MIN_LLAMA_BUILD, parse_llama_build

    build = parse_llama_build(line)
    if build is None or build >= MIN_LLAMA_BUILD:
        return []
    return [click.style(
        f"   WARNING: llama.cpp build b{build} is below the security floor "
        f"b{MIN_LLAMA_BUILD} (2026 CVE fixes) — rebuild this node's image "
        "before trusting it on the network",
        fg="red",
    )]


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    cli()


if __name__ == "__main__":
    main()
