"""Full deployment orchestration — ties together config generation,
SSH operations, K3s setup, and kubectl apply."""

from __future__ import annotations

import contextlib
import logging
import shlex
from pathlib import Path, PurePosixPath

from orchestrator.cluster import k3d, kubectl, labels
from orchestrator.config import tlsstore
from orchestrator.config.cache import load_network_config_cached
from orchestrator.config.loader import generated_config_dir, generated_config_path
from orchestrator.config.schema import HEAD_ROLE, SHARD_WORKER_ROLE, node_role
from orchestrator.config.templating import (
    generate_all_edge_configs,
    template_references_mcp_endpoints,
)
from orchestrator.deploy.k3s_agent import join_agent
from orchestrator.deploy.ssh import (
    conn_kwargs,
    ensure_key_installed,
    run_command,
    stage_path,
    upload_files,
)

logger = logging.getLogger("aureax.deployer")

# Bounds one node's install-and-join. join-node.sh fetches and runs the k3s
# installer, and on a GPU node the container toolkit as well, so this is the
# budget for package installs over a slow link, not for an SSH round trip.
JOIN_TIMEOUT_SECONDS = 900

# Node-side TLS paths come from tlsstore rather than being spelled again here:
# templating injects those same three constants into every resolved config as
# the `tls:` block, and the supervisor refuses to start when a file named there
# is missing (exit 78). Installing them anywhere else would leave every node
# reading paths nothing ever wrote — a fleet-wide crash-loop whose only symptom
# is an exit code.
NODE_TLS_DIR = str(PurePosixPath(tlsstore.NODE_CA_FILE).parent)


class Deployer:
    """Orchestrates full deployment to edge nodes."""

    def __init__(
        self, config_path: Path, orchestrator_ip: str, *, accept_worker_reorder: bool = False
    ) -> None:
        """Load network config and initialize deployment state.

        ``accept_worker_reorder`` is the operator's explicit "yes, I meant to
        reorder that group's workers" — generation refuses a reordered
        ``node_groups.workers`` list otherwise, because worker order IS the
        ``--rpc`` order and renaming RPC0..n silently re-points every
        ``tensor_split`` entry at a different machine. Carried here rather than
        read from the config so it stays a per-invocation decision.
        """
        self.config_path = config_path
        self.orchestrator_ip = orchestrator_ip
        self.accept_worker_reorder = accept_worker_reorder
        # Cached: the CLI has already loaded and validated this exact file to
        # build ctx.obj["network_config"], and the dashboard's Apply builds a
        # Deployer per request against a file it also reads for /api/lifecycle.
        # Keyed on (path, mtime), so an operator editing the config between two
        # applies still gets the new one.
        self.config = load_network_config_cached(config_path)
        # k8s/ is at the project root, which is config_path's grandparent
        # (config_path = <project>/configs/aureax-network.yaml)
        from orchestrator.resources import resource_root
        self.k8s_dir = config_path.parent.parent / "k8s"
        if not self.k8s_dir.is_dir():
            self.k8s_dir = resource_root() / "k8s"
        self._key_store = self.config.ssh.key_store
        self._hostnames: dict[str, str] = {}  # node_id -> SSH `hostname`

    def _conn(self, target, timeout: int | None = None) -> dict:
        """SSH connection kwargs (port/user/key_file/password) for a target."""
        return conn_kwargs(target, self._key_store, timeout=timeout)

    @property
    def _targets(self) -> list:
        """Targets to actually contact — `enabled: false` entries are skipped."""
        return self.config.ssh.enabled_targets

    def setup_cluster(self) -> None:
        """Ensure the k3d cluster exists, is running and its API answers.

        Also saves the K3s join token, which join_nodes reads back. Raises
        RuntimeError if the API never becomes reachable.
        """
        k8s = self.config.kubernetes
        logger.info("Setting up k3d cluster '%s'...", k8s.cluster_name)

        if k3d.cluster_exists(k8s.cluster_name):
            # Existing != running: a stopped cluster leaves every later step
            # talking to an unreachable API server, which surfaces as a
            # confusing `kubectl label` failure rather than "cluster is down".
            # Start it here.
            if k3d.cluster_running(k8s.cluster_name):
                logger.info("Cluster '%s' already running", k8s.cluster_name)
            else:
                logger.info("Cluster '%s' exists but is stopped — starting it", k8s.cluster_name)
                k3d.start_cluster(k8s.cluster_name)
        else:
            k3d.create_cluster(k8s.cluster_name, k8s.k3d.api_port, k8s.k3d.servers)
            logger.info("Cluster created")

        if not kubectl.wait_api_ready(timeout=120):
            raise RuntimeError(
                f"k3d cluster '{k8s.cluster_name}' did not become reachable. "
                "Check `k3d cluster list` and Docker Desktop."
            )

        token = k3d.save_token(k8s.cluster_name)
        if token:
            logger.info("K3s token saved to %s", token)

    def join_nodes(self) -> None:
        """Join all edge nodes as K3s agents.

        Per-node failures are logged as they happen and collected; if any node
        failed, the deployment aborts here rather than carrying on to label a
        fleet that is not fully joined and reporting the deploy as a success.
        """
        k8s = self.config.kubernetes
        token = k3d.get_token(k8s.cluster_name)
        if not token:
            raise RuntimeError(
                "Could not get K3s token — is the cluster running? "
                "Check `k3d cluster list`."
            )

        server_url = k3d.get_server_url(k8s.k3d.api_port, k3d.probe_host(self.config))
        logger.info("K3s server URL: %s", server_url)

        # Pin every agent to the version the server already runs, the same way
        # scripts/setup-all.sh does. Unpinned, get.k3s.io installs latest stable
        # while the k3d server stays on its bundled image, so nodes joined weeks
        # apart end up with kubelets newer than the API server — outside the
        # supported skew, and surfacing much later as an unexplained NotReady
        # node. None is not fatal: the join runs unpinned and join-node.sh warns.
        k3s_version = kubectl.server_version()
        if k3s_version:
            logger.info("Pinning K3s agents to the server's version (%s)", k3s_version)
        else:
            logger.warning(
                "Could not read the server's K3s version — agents will install "
                "latest stable, which may be newer than the server"
            )

        failed: list[str] = []
        for target in self._targets:
            logger.info("Joining %s (%s)...", target.node_id, target.host)
            self.bootstrap_key(target)
            # An install-sized budget, not the connect budget: too short a
            # bound reports a join that is still running as a hard failure,
            # and one failure aborts the whole deploy.
            success, msg = join_agent(
                host=target.host,
                server_url=server_url,
                token=token,
                k3s_version=k3s_version,
                **self._conn(target, timeout=JOIN_TIMEOUT_SECONDS),
            )
            if success:
                logger.info("  %s: %s", target.node_id, msg)
            else:
                logger.error("  %s: FAILED — %s", target.node_id, msg)
                failed.append(target.node_id)

        if failed:
            raise RuntimeError(
                f"{len(failed)}/{len(self._targets)} node(s) failed to join: "
                f"{', '.join(failed)} — see the log above for each node's error"
            )

    def label_nodes(self, *, role: str) -> None:
        """Apply aureax.io/role=edge to nodes whose per-node config matches `role`.

        `role=HEAD_ROLE` / `PLANNER_ROLE` filters by the per-node YAML's
        `node.role`. Filtering on HEAD_ROLE is what keeps a shard worker out of
        this pass — otherwise it would also be handed a llama-server pod with
        no model file and crash-loop against the group it serves. full_deploy
        uses the role-filtered form for that reason, and so the planner pod is
        not scheduled until edges are Ready.

        `role` is required on purpose: an unfiltered pass would label shard
        workers as edges — the one failure the rest of this module exists to
        prevent. A groupless fleet just uses HEAD_ROLE, which is every
        standalone node's role.

        Also STRIPS the shard label from every node it labels, because a role
        change has to REMOVE the other pass's label and not merely add its own:
        a node carrying both aureax.io/role=edge and aureax.io/shard=worker
        matches both DaemonSet selectors and runs llama-server and rpc-server
        side by side — the same crash-loop the role filter above prevents,
        reached by a different route. Promoting a worker to a group head is
        exactly when it happens, and it is silent: nothing else notices two
        pods on one node.
        """
        k8s = self.config.kubernetes

        # Build an SSH-target-host -> per-node-role map. _target_hostname is
        # memoized for the deployer's lifetime, which is what matters — it is
        # a paramiko connect per node, and label_shard_workers walks the same
        # targets again. The role read goes through the (path, mtime) config
        # cache, so the template six shard workers share is parsed once.
        target_role_by_hostname: dict[str, str] = {}
        unresolved: list[str] = []
        for target in self._targets:
            ncfg_role = self._target_role(target)
            hostname = self._target_hostname(target)
            if hostname:
                target_role_by_hostname[hostname] = ncfg_role
            else:
                unresolved.append(target.node_id)

        nodes = kubectl.get_nodes()
        if not nodes:
            # A live cluster always has at least the k3d server node, so an
            # empty list is provably an API failure: kubectl.get() collapses
            # every non-zero exit to {}. Unchecked, the loop below simply never
            # runs and full_deploy still logs "Deployment complete" with not a
            # single pod scheduled.
            raise RuntimeError(
                "kubectl returned no nodes — the API server is unreachable; "
                "check `k3d cluster list` and Docker Desktop"
            )

        labelled: set[str] = set()
        for node in nodes:
            if "control-plane" in node.get("roles", ""):
                continue
            role_for_node = target_role_by_hostname.get(node["name"], "")
            if role_for_node != role:
                logger.info("Skipping label for %s (role=%r, want %r)",
                            node["name"], role_for_node, role)
                continue
            logger.info("Labeling node %s as %s=%s", node["name"], k8s.node_label, k8s.edge_label_value)
            kubectl.label_node(node["name"], k8s.node_label, k8s.edge_label_value)
            labelled.add(node["name"])
            self._strip_conflicting_label(node["name"], k8s.shard_label, "aureax-shard")

        # A node the API server does not list, and one whose hostname probe
        # failed, are both skipped above at info level — indistinguishable in
        # the log from the intended shard-worker skip, while the node ends up
        # with no pod at all. Name them.
        skipped = sorted({h for h, r in target_role_by_hostname.items() if r == role} - labelled)
        if skipped or unresolved:
            logger.error(
                "role=%s: not labelled — %s (not in the cluster's node list), %s "
                "(hostname unresolved over SSH). These nodes get no pod.",
                role, ", ".join(skipped) or "none", ", ".join(sorted(unresolved)) or "none",
            )

    def _strip_conflicting_label(self, node_name: str, label: str, selector: str) -> None:
        """Remove the OTHER pass's label from a node this pass just claimed.

        The strip itself lives in ``orchestrator.cluster.labels`` because
        ``aureax start`` needs the identical operation and cannot import this
        module; only the reporting is local. This path LOGS and carries on
        while start RAISES on the same failure — deliberate asymmetry: a deploy
        is normally followed by a start, which gets a second chance at the
        node, whereas start is the last writer before pods schedule.
        """
        ok, detail = labels.strip_conflicting_label(node_name, label)
        if not ok:
            logger.error(
                "Could not strip %s from %s: %s — the node may still match "
                "the %s selector and run a second pod",
                label, node_name, detail, selector,
            )

    def _target_role(self, target) -> str:
        """Read `node.role` from the SSH target's per-node config (best-effort).

        The one place this module reads a raw config's role — every caller here
        goes through it rather than re-indexing the dict.
        """
        if not target.config_file:
            return ""
        per_node_path = self.config_path.parent / target.config_file
        if not per_node_path.exists():
            return ""
        try:
            # Cached on (path, mtime): targets share templates (six shard
            # workers share one file, both heads share another), so without
            # the cache the same YAML is re-parsed once per sharer per pass.
            # The cache hands back a shared read-only dict, which this only
            # reads from.
            from orchestrator.config.cache import load_edge_config_cached
            ncfg = load_edge_config_cached(per_node_path)
            return node_role(ncfg)
        except Exception as e:
            logger.warning("Could not read role for %s: %s", target.node_id, e)
            return ""

    def _target_hostname(self, target) -> str:
        """SSH `hostname` for an SSH target (best-effort, '' on failure).

        Memoized for the deployer's lifetime: label_nodes, label_shard_workers
        and wait_nodes_registered each walk every target, and each miss is a
        paramiko connect for an answer that cannot change mid-deploy. Failures
        are not cached — a node that was briefly unreachable should be retried
        by the next pass.
        """
        if target.node_id in self._hostnames:
            return self._hostnames[target.node_id]
        try:
            _, out, _ = run_command(target.host, "hostname", **self._conn(target, timeout=10))
            hostname = (out or "").strip()
        except Exception as e:
            logger.warning("Could not resolve hostname for %s: %s", target.node_id, e)
            return ""
        if hostname:
            self._hostnames[target.node_id] = hostname
        return hostname

    def bootstrap_key(self, target) -> None:
        """Install the orchestrator's public key on a password-auth node.

        Best-effort and idempotent. The bash scripts only speak ``ssh -i``, so
        without this a password-only node is reachable from the Python paths
        but not from setup/update/download-model/reset.
        """
        pub_key = self.config.ssh.bootstrap_public_key
        if not pub_key or not target.resolve_password():
            return
        ok, msg = ensure_key_installed(target.host, pub_key, **self._conn(target, timeout=15))
        if ok:
            logger.info("  %s: SSH key %s", target.node_id, msg)
        else:
            logger.warning(
                "  %s: could not install SSH key (%s) — bash scripts may not reach this node",
                target.node_id, msg,
            )

    def generate_configs(self) -> list[Path]:
        """Generate per-node edge configs (and the TLS material they name)."""
        output_dir = generated_config_dir()
        paths = generate_all_edge_configs(
            self.config_path, output_dir, self.orchestrator_ip,
            accept_worker_reorder=self.accept_worker_reorder,
        )
        logger.info("Generated %d edge configs", len(paths))
        return paths

    def deploy_configs(self, config_paths: list[Path], targets: list | None = None) -> None:
        """Push each node's config AND its TLS material to that node via SSH.

        Four files per node: the resolved config, the fleet CA, and the node's
        own leaf certificate and private key. They ride the same path because
        they have the same delivery problem — both DaemonSets mount
        ``/etc/aureax`` as a read-only hostPath with ``type: Directory``, which
        the kubelet will not create, so nothing inside the cluster can put a
        file there. SSH before scheduling is the only channel, and a node that
        received its config but not its certificates fails exactly as hard: TLS
        is mandatory fleet-wide, so the supervisor exits 78 rather than serving
        plaintext.

        Paired by node_id, which the generated filenames already encode
        (``aureax-edge-<node_id>.yaml``), rather than by list position. Position
        matching holds only while both sides walk ``enabled_targets`` in the
        same order — an invariant nothing can enforce, and whose failure mode is
        a node silently receiving another node's config. The leaf is looked up
        by the same node_id, so a mispairing cannot hand one node another's key.

        ``targets=None`` pushes to every enabled target (the full-deploy
        behaviour); apply_to_nodes passes the subset it is touching.

        Per-node failures are logged as they happen and the loop continues, the
        shape join_nodes already uses: aborting on the first one would leave the
        rest of a group unpushed AND unnamed, telling an operator about one node
        while the group sits on mixed configs. The raise at the end names every
        failure and every node that did get the new config — apply_to_nodes'
        outcomes are discarded when this raises, so this message is the record.
        """
        push_targets = self._targets if targets is None else targets
        by_node_id = {p.stem.removeprefix("aureax-edge-"): p for p in config_paths}
        missing = [t.node_id for t in push_targets if t.node_id not in by_node_id]
        if missing:
            raise RuntimeError(
                f"no generated config for {missing} — expected "
                f"aureax-edge-<node_id>.yaml, got {sorted(by_node_id)}"
            )

        # Same shape as the missing-config check above, and for the same
        # reason: refuse the whole push before touching a single node rather
        # than discovering half way through that a node has no certificate.
        # Minting belongs to `aureax generate` — doing it here would issue a
        # leaf against a CA the fleet may not have been given yet.
        ca_cert = tlsstore.ca_cert_path()
        no_leaf = [
            t.node_id for t in push_targets
            if not all(p.exists() for p in tlsstore.leaf_paths(t.node_id))
        ]
        if no_leaf or not ca_cert.exists():
            absent = []
            if not ca_cert.exists():
                absent.append(f"the fleet CA ({ca_cert})")
            if no_leaf:
                absent.append(f"a leaf certificate and key for {', '.join(no_leaf)}")
            raise RuntimeError(
                f"missing from {tlsstore.store_dir()}: {' and '.join(absent)}. TLS is "
                "mandatory fleet-wide and there is no plaintext fallback, so run "
                "`aureax generate --orchestrator-ip <ip>` — it mints the CA and one leaf "
                "per enabled target — before deploying."
            )

        pushed: list[str] = []
        failed: list[str] = []
        for target in push_targets:
            config_path = by_node_id[target.node_id]
            leaf_cert, leaf_key = tlsstore.leaf_paths(target.node_id)
            logger.info("Deploying config and TLS material to %s (%s)",
                        target.node_id, target.host)
            conn = self._conn(target, timeout=self.config.ssh.connect_timeout_seconds)
            # A fresh name per push and per file, under a 0700 directory in the
            # login user's home — ssh.stage_path is the one place that
            # convention lives, shared with the k3s_agent staging so the two
            # cannot drift. The resolved config may carry model.server.api_key
            # and the leaf key is a private key, and a fixed name in
            # world-writable /tmp can be pre-created by any local account on
            # the node — the put would then write the secret into a file
            # somebody else owns, before the chmod fails.
            stage, remote_stage = stage_path("aureax-edge", ".yaml")
            ca_stage, remote_ca = stage_path("aureax-ca", ".pem")
            cert_stage, remote_cert = stage_path("aureax-node", ".pem")
            key_stage, remote_key = stage_path("aureax-node", ".key")
            staged = [remote_stage, remote_ca, remote_cert, remote_key]

            try:
                # mode 0600 from the moment they land, too — not only after the
                # install below. The two certificates end up 0644 on the node,
                # but they are staged as tightly as the key: the widening
                # belongs at the end of the install, not to a file sitting in
                # the login user's home.
                #
                # One connection and one SFTP channel for all four, because an
                # SSH handshake against an edge node's small CPU costs real
                # time and this runs once per node per push.
                upload_files(
                    target.host,
                    [
                        (config_path, stage, 0o600, 0o700),
                        (ca_cert, ca_stage, 0o600, 0o700),
                        (leaf_cert, cert_stage, 0o600, 0o700),
                        (leaf_key, key_stage, 0o600, 0o700),
                    ],
                    **conn,
                )

                # One remote command for the whole install, for the same
                # reason, and because the four files are one unit — a node with
                # a config but no key is as dead as a node with neither. Its
                # exit code must be verified: this step exists precisely so a
                # pod is never scheduled onto a node without its config (the
                # hostPath mount is `type: Directory`, which the kubelet will
                # not create), and a swallowed sudo/disk failure here surfaces
                # only much later, as an opaque ContainerCreating hang. The
                # user comes from config and runs in a root shell, hence the
                # quoting.
                #
                # The config is chowned root:root 0600: it may carry the node's
                # API key, and it is mounted read-only into a container whose
                # process runs as ROOT — neither Dockerfile sets a USER and
                # neither DaemonSet sets runAsUser/runAsNonRoot (daemonset-edge
                # .yaml's securityContext only drops capabilities and
                # privilege escalation, which does not change the uid). So
                # root-only is still readable by the supervisor, and by nobody
                # else with a shell on the node. Adding runAsNonRoot to either
                # DaemonSet means revisiting this chown.
                #
                # node.key gets the same treatment for the same reason. The two
                # certificates are deliberately 0644: a certificate is not a
                # secret, and the 0700 directory (chmodded explicitly, because
                # mkdir's mode applies only when mkdir actually creates the
                # leaf) is what keeps them out of reach anyway. The chown of
                # that directory is not redundant either — a /etc/aureax/tls
                # left behind by a hand-run command could be owned by the login
                # user, and then a local account could swap the node's key.
                q_user = shlex.quote(target.user)
                cmd = (
                    f"sudo mkdir -p /etc/aureax {NODE_TLS_DIR} /opt/aureax/models && "
                    f"sudo chmod 0700 {NODE_TLS_DIR} && "
                    f"sudo mv {remote_stage} /etc/aureax/aureax-edge.yaml && "
                    f"sudo mv {remote_ca} {tlsstore.NODE_CA_FILE} && "
                    f"sudo mv {remote_cert} {tlsstore.NODE_CERT_FILE} && "
                    f"sudo mv {remote_key} {tlsstore.NODE_KEY_FILE} && "
                    f"sudo chown root:root /etc/aureax/aureax-edge.yaml {NODE_TLS_DIR} "
                    f"{tlsstore.NODE_CA_FILE} {tlsstore.NODE_CERT_FILE} "
                    f"{tlsstore.NODE_KEY_FILE} && "
                    "sudo chmod 0600 /etc/aureax/aureax-edge.yaml "
                    f"{tlsstore.NODE_KEY_FILE} && "
                    f"sudo chmod 0644 {tlsstore.NODE_CA_FILE} {tlsstore.NODE_CERT_FILE} && "
                    f"sudo chown {q_user}: /opt/aureax/models"
                )
                # The connect budget is not the command budget: sudo commonly
                # stalls 10-30s resolving the host's own name on a node with
                # incomplete DNS, and connect_timeout_seconds (15) would abort
                # the whole deploy with "exited 124" on a node that was slow.
                rc, _, err = run_command(target.host, cmd, **{**conn, "timeout": 60})
                if rc != 0:
                    raise RuntimeError(
                        f"config push to {target.node_id} ({target.host}) failed: "
                        f"`{cmd}` exited {rc}: {err.strip() or 'no stderr'}"
                    )
            except Exception as e:
                logger.error("  %s: %s", target.node_id, e)
                failed.append(target.node_id)
                # Best-effort, and EVERY staged file: a failure anywhere in the
                # chain above (a failed upload, a failed mv, a full disk) can
                # leave any of the four behind, two of which are secrets — the
                # resolved config and the node's private key. A separate call,
                # so it cannot mask the error just recorded.
                with contextlib.suppress(Exception):
                    run_command(target.host, "rm -f " + " ".join(staged), **conn)
                continue
            pushed.append(target.node_id)

        if failed:
            raise RuntimeError(
                f"config push failed for {', '.join(failed)} — see the log above for each "
                f"node's error. Already updated: {', '.join(pushed) or 'none'}"
            )

    def apply_to_nodes(
        self,
        node_ids: list[str] | None = None,
        *,
        restart: bool = True,
        rotate_keys: bool = False,
        wait: bool = False,
        log=None,
    ) -> dict[str, str]:
        """Push freshly generated config to node(s) and restart their llama-server pod.

        Returns node_id -> outcome. ``node_ids=None`` means every enabled target.

        Those outcome strings are a CONTRACT, not just log text:
        ``orchestrator/cli.py::_apply_outcome_failed`` decides ``apply --wait``'s
        exit code by substring-matching them against
        ``_APPLY_FAILURE_MARKERS`` ("failed", "did not become ready", "not
        waiting"). Rewording a failure outcome here, or moving to structured
        results, has to move that tuple and
        ``tests/test_cli.py::test_apply_exit_code_tracks_the_wait_outcome`` in
        the same change, or --wait silently stops gating.

        The three-way asymmetry that makes this primitive worth having:
        - configs regenerate for the WHOLE fleet (generate_all_edge_configs is
          the only caller of validate_groups, so a partial regeneration could
          let a group's configs drift out of agreement — and it is cheap);
        - the PUSH fans out to each requested node's whole group (a group
          serves one model, so member configs must stay consistent);
        - the RESTART set excludes shard workers entirely: rpc-server takes no
          API key, and restarting a worker tears the group down and invalidates
          its tensor cache, turning a ~107s warm head reload into a ~280s cold
          one. `aureax start` is never a substitute for this — it does not push
          configs (see ARCHITECTURE.md, "llama-server API-key
          authentication").

        ``wait`` is TWO passes, not one: every pod is deleted first, and only
        then are the replacements polled. A single interleaved loop holds each
        node's delete until the previous node is Ready again, which is a serial
        fleet-wide reload — minutes of cold load per head, one after another,
        the very shape ``kubectl.delete_pods_on_node`` bans `kubectl rollout
        restart` over. The tradeoff is deliberate and worth stating plainly:
        every head reloads AT ONCE, so `--all --wait` takes the whole fleet's
        endpoints down for one shared window instead of half the fleet serving
        while the other half loads. Without ``wait`` nothing changes here — the
        deletes go out back to back either way.

        ``rotate_keys`` rotates BOTH of a node's keys — llama and MCP. They
        protect the same pod and one restart applies both, so rotating one
        alone would buy a second multi-minute head reload for nothing; after a
        suspected leak "re-key this node" means the node, not one of its ports.

        A changed MCP key also widens the push set beyond the group: a
        planner's resolved config embeds every peer's MCP key (the
        ``${mcp_endpoints}`` block), so a consumer left unpushed keeps dialling
        with the old key and 401s on every call. Those consumers are found by
        reading the templates, not by role, so a second planner — or any other
        template that grew an ``${mcp_endpoints}`` block — is covered without
        anything here learning about it.
        """
        emit = log or (lambda msg: logger.info("%s", msg))
        outcomes: dict[str, str] = {}

        by_id = {t.node_id: t for t in self._targets}
        if node_ids is None:
            requested = list(by_id)
        else:
            unknown = [n for n in node_ids if n not in by_id]
            if unknown:
                raise ValueError(
                    f"unknown or disabled node id(s): {unknown} — "
                    f"enabled targets are {sorted(by_id)}"
                )
            requested = list(dict.fromkeys(node_ids))  # de-dup, keep order

        from orchestrator.config import keystore

        # Taken before rotation AND before generate, because generate MINTS a
        # first MCP key for a node whose template has just grown the
        # ${mcp_api_key} line. Both events have the same consequence for the
        # peers holding that key, so both have to be visible here — comparing
        # only the rotation set would miss the first-mint half silently.
        # fingerprint() is a sha256 prefix, never the key itself, so it is safe
        # in a local and safe if it ever reaches a log.
        mcp_before = {
            t.node_id: keystore.fingerprint(t.node_id, kind=keystore.KIND_MCP)
            for t in self._targets
        }

        if rotate_keys:
            for nid in requested:
                # Only nodes that actually hold a key, per kind: workers and
                # auth-off nodes have none, and rotate-creating one here would
                # enable nothing (their template carries no ${api_key} /
                # ${mcp_api_key} reference). The two kinds are gated
                # independently — a node may serve keyed llama and unkeyed MCP.
                if keystore.has_key(nid):
                    keystore.rotate_key(nid)
                    emit(f"{nid}: llama API key rotated (old key retained as .prev)")
                if keystore.has_key(nid, kind=keystore.KIND_MCP):
                    keystore.rotate_key(nid, kind=keystore.KIND_MCP)
                    emit(f"{nid}: MCP API key rotated (old key retained as .prev)")

        config_paths = self.generate_configs()

        # Push set: requested nodes plus every enabled member of their groups.
        push_ids: list[str] = []
        for nid in requested:
            group = self.config.group_for_node(nid)
            members = [group.head, *group.workers] if group else [nid]
            for member in members:
                if member in by_id and member not in push_ids:
                    push_ids.append(member)

        mcp_changed = [
            nid for nid, fp in mcp_before.items()
            if keystore.fingerprint(nid, kind=keystore.KIND_MCP) != fp
        ]
        if mcp_changed:
            # Found by reading the templates rather than by role: a planner is
            # only the current example of an ${mcp_endpoints} consumer, and a
            # role-based rule would silently miss the next one.
            consumers = [
                t.node_id for t in self._targets
                if t.config_file
                and template_references_mcp_endpoints(self.config_path.parent / t.config_file)
            ]
            added = [n for n in consumers if n not in push_ids]
            push_ids.extend(added)
            if added:
                emit(
                    f"MCP key changed for {', '.join(mcp_changed)} — also pushing and "
                    f"restarting {', '.join(added)}: their ${{mcp_endpoints}} block carries "
                    "those peers' keys, so an unpushed consumer would keep dialling with the "
                    "old key and 401 on every call"
                )

        push_targets = [by_id[n] for n in push_ids]

        self.deploy_configs(config_paths, targets=push_targets)
        for nid in push_ids:
            outcomes[nid] = "config pushed"

        if not restart:
            emit(
                "WARNING: --no-restart — every pushed node still runs with its "
                "OLD llama-server settings. If a key changed, the in-pod clients "
                "now send the new key against a server enforcing the old one and "
                "every /completion and /metrics call 401s until the pod restarts. "
                "A rotated MCP key reaches further: every ${mcp_endpoints} consumer "
                "keeps the OLD peer key in its running process until IT restarts too, "
                "so the staleness is not confined to the node whose key changed."
            )
            return outcomes

        namespace = self.config.kubernetes.namespace

        # Pass 1: every delete is issued before any wait begins. Nodes to poll
        # in pass 2, as (node_id, hostname, pre_names).
        pending: list[tuple[str, str, set[str]]] = []
        for target in push_targets:
            nid = target.node_id
            if self._target_role(target) == SHARD_WORKER_ROLE:
                # Explicit outcome, not a silent drop — the operator asked
                # about this node and deserves to know why nothing restarted.
                outcomes[nid] = "config pushed; restart skipped (shard workers run no llama-server)"
                continue

            hostname = self._target_hostname(target)
            if not hostname:
                outcomes[nid] = "config pushed; restart FAILED (could not resolve hostname over SSH)"
                continue

            # Only built when it is going to be read: this is a full-namespace
            # `kubectl get pods -o json` spawn plus parse, per restarted node,
            # and `wait` is False for every dashboard Apply. It has to be taken
            # BEFORE the delete either way — that is what makes it a snapshot.
            pre_names: set[str] = set()
            if wait:
                pre_names = {
                    p["name"] for p in kubectl.get_pods(namespace) if p["node"] == hostname
                }
            ok, detail = kubectl.delete_pods_on_node(namespace, hostname)
            if not ok:
                outcomes[nid] = f"config pushed; restart FAILED ({detail})"
                continue
            emit(f"{nid}: pod deleted on {hostname}; DaemonSet is recreating it")
            outcomes[nid] = "config pushed; pod restarting"

            if not wait:
                continue
            if not pre_names:
                # get_pods returns [] for "API unreachable" as much as for
                # "no pods" (kubectl.get collapses every failure to {}), and
                # the delete that just succeeded proves there WAS a pod. An
                # empty snapshot is therefore unusable: the wait would take
                # the still-listed, still-ready OLD pod for the replacement
                # and report the new config live.
                outcomes[nid] = (
                    "config pushed; pod restarting (pre-delete pod list "
                    "unavailable — not waiting)"
                )
                continue
            pending.append((nid, hostname, pre_names))

        # Pass 2: poll for the replacements, each on its own budget. The budget
        # is per node but the clock is not: every pod was already deleted in
        # pass 1, so a head that finished loading while an earlier one was
        # being waited on answers on the first poll.
        for nid, hostname, pre_names in pending:
            if self._wait_pod_replaced(namespace, hostname, pre_names,
                                       self._apply_wait_budget(nid), emit):
                outcomes[nid] = "applied (pod Ready on new config)"
            else:
                outcomes[nid] = (
                    "config pushed; pod restarted but did not become Ready in time — "
                    "a sharded head may have failed to re-attach to its RPC workers"
                )
        return outcomes

    def _apply_wait_budget(self, node_id: str) -> int:
        """Ready-wait ceiling for one node, from its own resolved config.

        worker_wait + health_timeout is the same arithmetic the startupProbe
        budget covers (tests/test_probe_budget.py); the slack absorbs image
        pull and scheduling.

        The 600s fallback is a standalone node's rough ceiling and deliberately
        short of a sharded head's worst case (300 + 2400), so an unreadable
        config can make `apply --wait` report a timeout for a pod that is
        loading normally. Only the report is wrong — the pod is unaffected —
        and the probe budgets this shadows must not be shrunk to match it.
        """
        from orchestrator.config.cache import load_edge_config_cached

        path = generated_config_path(node_id)
        try:
            ncfg = load_edge_config_cached(path)
        except Exception as e:
            # Broad on purpose: ruamel raises YAMLError (not a ValueError) for
            # a malformed config, and letting that escape would abort an apply
            # whose push already landed.
            logger.warning("Could not read wait budget for %s (%s) — using 600s", node_id, e)
            return 600
        server = (ncfg.get("model") or {}).get("server") or {}
        node = ncfg.get("node") or {}
        return (
            int(node.get("worker_wait_timeout_seconds", 0) or 0)
            + int(server.get("health_timeout_seconds", 120) or 120)
            + 120
        )

    def _wait_pod_replaced(
        self, namespace: str, hostname: str, pre_names: set[str],
        timeout: int, emit,
    ) -> bool:
        """Wait for a NEW, Ready pod on ``hostname``.

        The pre-delete pod can stay listed (and even `ready`) for a moment
        after the delete, so a plain readiness wait would return instantly
        against the pod just killed and report success for a node still
        running the old config — hence the name-set comparison.
        """
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            pods = [p for p in kubectl.get_pods(namespace) if p["node"] == hostname]
            fresh = [p for p in pods if p["name"] not in pre_names]
            if any(p["ready"] for p in fresh):
                return True
            time.sleep(5)
        emit(f"timed out after {timeout}s waiting for a Ready replacement pod on {hostname}")
        return False

    def wait_nodes_registered(self, timeout: int = 180, interval: int = 5) -> bool:
        """Wait until every enabled target appears in `kubectl get nodes`.

        Matches on hostnames, not counts. A cluster that still remembers a
        decommissioned or renamed node satisfies a count comparison while the
        node we are actually waiting for is still joining — and the labelling
        pass that follows then silently skips it.

        Returns whether the whole fleet registered. full_deploy deliberately
        continues either way — an unregistered node is simply skipped by the
        label passes — and uses the value only to qualify its closing line.
        """
        import time

        # A target whose SSH hostname probe failed cannot be matched against
        # the cluster's node list at all, but it stays counted: dropping it
        # from `expected` would make "All 3 node(s) registered" true of a
        # 4-node fleet with one node that never joined. Collected in the same
        # pass because _target_hostname does not cache failures — asking again
        # costs another paramiko connect per unresolved target.
        expected: set[str] = set()
        unresolved: list[str] = []
        for t in self._targets:
            hostname = self._target_hostname(t)
            if hostname:
                expected.add(hostname)
            else:
                unresolved.append(t.node_id)

        if not expected:
            logger.warning("Could not resolve any node hostnames — skipping registration wait")
            return False

        deadline = time.time() + timeout
        missing = expected
        while time.time() < deadline:
            registered = {
                n["name"] for n in kubectl.get_nodes()
                if "control-plane" not in n.get("roles", "")
            }
            missing = expected - registered
            if not missing:
                if unresolved:
                    logger.error(
                        "%d node(s) never answered the hostname probe (%s) — they cannot be "
                        "matched against the cluster's node list and will be skipped by "
                        "labelling. The fleet is NOT fully registered.",
                        len(unresolved), ", ".join(sorted(unresolved)),
                    )
                    return False
                logger.info("All %d node(s) registered with the cluster", len(expected))
                return True
            logger.info(
                "Waiting for nodes to register (%d/%d, missing: %s)...",
                len(expected) - len(missing), len(expected), ", ".join(sorted(missing)),
            )
            time.sleep(interval)

        logger.warning(
            "%d/%d node(s) still unregistered after %ds (%s) — continuing; they "
            "will be skipped by labeling",
            len(missing), len(expected), timeout, ", ".join(sorted(missing)),
        )
        return False

    def label_shard_workers(self) -> None:
        """Label group workers with the shard key so aureax-shard schedules.

        Uses the same hostname/role resolution as label_nodes(). Workers are
        NOT given the edge label: label_nodes(role=HEAD_ROLE) filters on
        that literal role, so a `shard_worker` is skipped there automatically —
        which is exactly what stops it also being handed a llama-server pod
        with no model file.

        Being skipped by that pass is not enough on its own, though: skipping
        never removes an edge label a node already carries. So this strips it,
        mirroring what label_nodes does with the shard label. Between them, a
        node that changes role in either direction ends up with exactly one
        label instead of two.
        """
        if not self.config.node_groups:
            return

        k8s = self.config.kubernetes
        worker_ids = {w for g in self.config.node_groups for w in g.workers}
        by_id = {t.node_id: t for t in self._targets}

        # Only label nodes the API server already knows about. A K3s agent
        # takes a little while to register after joining, and labelling a
        # not-yet-registered node fails the whole deploy with an opaque
        # `kubectl label` error rather than saying what is actually going on.
        nodes = kubectl.get_nodes()
        if not nodes:
            # Same reasoning as label_nodes: [] means the API failed, and
            # taking it at face value skips every worker as "not registered
            # yet" while the deploy reports success.
            raise RuntimeError(
                "kubectl returned no nodes — the API server is unreachable; "
                "check `k3d cluster list` and Docker Desktop"
            )
        registered = {n["name"] for n in nodes}

        for node_id in sorted(worker_ids):
            target = by_id.get(node_id)
            if target is None:
                continue
            hostname = self._target_hostname(target)
            if not hostname:
                logger.error("Could not resolve hostname for shard worker %s", node_id)
                continue
            if hostname not in registered:
                logger.warning(
                    "Shard worker %s (%s) has not registered with the cluster yet — "
                    "skipping label; re-run deploy or use 'aureax start' once it joins",
                    node_id, hostname,
                )
                continue
            logger.info(
                "Labeling %s as %s=%s (shard worker)",
                hostname, k8s.shard_label, k8s.shard_label_value,
            )
            kubectl.label_node(hostname, k8s.shard_label, k8s.shard_label_value)
            # Mirror of label_nodes: a node demoted from head to worker would
            # otherwise keep aureax.io/role=edge and be scheduled a
            # llama-server pod with no GGUF alongside its rpc-server.
            self._strip_conflicting_label(hostname, k8s.node_label, "aureax-edge")

    def _check_manifest_namespace(self) -> None:
        """Fail if the manifests and `kubernetes.namespace` disagree.

        The manifests hardcode `metadata.namespace`, while every runtime
        operation reads `config.kubernetes.namespace`. Change the config alone
        and deploy still creates everything in the manifest's namespace, while
        `aureax stop`/`start` patch DaemonSets in the configured one — so the
        stack deploys, reports success, and then cannot be stopped.

        Checked rather than templated: this prototype ships static manifests on
        purpose, so the useful thing is to make the coupling explicit and loud
        instead of silently ignoring half the configuration.
        """
        import re

        configured = self.config.kubernetes.namespace
        for manifest in self._MANIFESTS:
            path = self.k8s_dir / manifest
            if not path.exists():
                continue
            text = path.read_text()
            found = set(re.findall(r"^\s*namespace:\s*(\S+)\s*$", text, re.MULTILINE))
            # namespace.yaml declares its namespace as the Namespace object's
            # `metadata.name`, which the pattern above cannot see — without
            # this, that manifest passed the check no matter what it said.
            if re.search(r"^kind:\s*Namespace\s*$", text, re.MULTILINE):
                found.update(re.findall(r"^\s+name:\s*(\S+)\s*$", text, re.MULTILINE))
            mismatched = {n for n in found if n != configured}
            if mismatched:
                raise ValueError(
                    f"kubernetes.namespace is {configured!r} but {manifest} declares "
                    f"{', '.join(sorted(repr(m) for m in mismatched))}. "
                    f"The manifests in {self.k8s_dir} hardcode their namespace — update them "
                    f"to match, or set kubernetes.namespace back to the manifest's value. "
                    f"Leaving them different deploys a stack that stop/start cannot find."
                )

    # Shard DaemonSet BEFORE the edge one: the edge set carries the group's
    # head, whose llama-server exits immediately if an RPC peer is
    # unreachable at load time. Applying the EDGE set first would mean the head
    # pod can be scheduled while no worker pod exists yet.
    #
    # Shared with _check_manifest_namespace so the namespace check can never
    # drift out of sync with what actually gets applied (the Services hardcode
    # `namespace: aureax` too).
    _MANIFESTS = (
        "namespace.yaml",
        "service-llama.yaml",
        "service-mcp.yaml",
        "daemonset-shard.yaml",
        "daemonset-edge.yaml",
    )

    def apply_manifests(self) -> None:
        """Apply the Kubernetes manifests, in the order _MANIFESTS fixes.

        Raises ValueError if a manifest declares a namespace other than the
        configured one, and FileNotFoundError if the shard DaemonSet is absent
        while node_groups is non-empty. Any other missing manifest is a warning.
        """
        self._check_manifest_namespace()
        manifests = list(self._MANIFESTS)
        # A group whose worker DaemonSet was never applied leaves the head
        # crash-looping against peers that were never scheduled, so a missing
        # manifest is only a warning when no group depends on it.
        required = {"daemonset-shard.yaml"} if self.config.node_groups else set()

        for manifest in manifests:
            path = self.k8s_dir / manifest
            if path.exists():
                logger.info("Applying %s", manifest)
                kubectl.apply(str(path))
            elif manifest in required:
                raise FileNotFoundError(
                    f"{manifest} is required because node_groups is non-empty, "
                    f"but it is missing at {path}"
                )
            else:
                logger.warning("Manifest not found: %s", path)

    def full_deploy(self) -> None:
        """Run the complete deployment pipeline.

        Applies both label passes: shard workers get aureax.io/shard, and
        edge_executor nodes get aureax.io/role=edge. The cognitive_planner is
        labeled later by the orchestrator (setup-all.sh Phase B or
        lifecycle warm-start) once edges are Ready, so the planner pod
        doesn't crash trying to reach edge MCP servers that aren't up yet.
        """
        logger.info("=== Aureax Full Deployment ===")

        # 1. Setup K3s cluster
        self.setup_cluster()

        # 2. Generate configs FIRST, before anything is joined or labelled.
        #    generate_configs() is the only caller of validate_groups(), so
        #    running it later means an invalid group config is discovered after
        #    the cluster has already been labelled for it.
        config_paths = self.generate_configs()

        # 3. Join edge nodes
        self.join_nodes()

        # A K3s agent takes a moment to register after the join returns. Both
        # labelling passes below act on the API server's node list, so without
        # this they can silently skip a node that is seconds from appearing.
        # Deliberately not fatal — the rest of the fleet still deploys — but
        # the closing line has to stop claiming a complete deployment.
        all_registered = self.wait_nodes_registered()

        # 4. Push configs and create model dirs BEFORE labelling. Both DaemonSets
        #    mount /etc/aureax with `type: Directory`, which the kubelet refuses
        #    to create — a pod scheduled onto a node that has never received a
        #    config sits in ContainerCreating until someone notices.
        self.deploy_configs(config_paths)

        # 5. Apply K8s manifests (config is mounted from /etc/aureax via hostPath,
        #    populated by deploy_configs())
        self.apply_manifests()

        # 6. Label nodes, which is what actually schedules pods. Shard workers
        #    first: the head's llama-server exits if a peer is unreachable at
        #    load time, so workers must be schedulable before the head's pod
        #    comes up. (The client also waits for its peers, which covers
        #    restarts this ordering doesn't.)
        self.label_shard_workers()
        self.label_nodes(role=HEAD_ROLE)

        if all_registered:
            logger.info("=== Deployment complete ===")
        else:
            logger.warning(
                "=== Deployment finished, but not every node registered with the cluster — "
                "see the errors above. Unlabelled nodes are running no pods. ==="
            )
