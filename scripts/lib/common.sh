#!/usr/bin/env bash
# ==========================================================================
# Aureax — shared helpers for the operational scripts
#
# Sourced by the Mac-side scripts listed below:
#   setup-all.sh  update-all.sh  download-model.sh  reset-all.sh
#   test-all.sh   build-on-node.sh  bench-decode.sh  setup-k3d.sh
#
# Not every Mac-side script is on that list: stop-all.sh runs here too and
# sources nothing, because it uses none of these helpers.
#
# From a script in scripts/:
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
#
# ${BASH_SOURCE[0]}, not $0: $0 names the script that was *invoked*, so a
# sourced file that resolves its own paths from it is wrong the moment anything
# sources anything else. Nothing here depends on the caller's CWD either — the
# two helpers that shell out to Python cd to $AUREAX_ROOT in a subshell.
#
# join-node.sh is deliberately NOT a consumer: it is staged on a node and run
# there, where this file does not exist. Sharing with it goes the other way
# round, through its --cgroups-only mode (see setup-all.sh Step 2).
#
# THE OUTPUT PREFIXES ARE A PARSED PROTOCOL, not formatting.
# orchestrator/lifecycle.py::_classify_setup_line maps each streamed
# setup-all.sh line onto a dashboard log level, in this order:
#   `[FAIL]`, a line starting with `ERROR`, or one containing ` ERROR ` -> error
#   `[WARN]`, or a line starting with `WARNING`                         -> warn
#   `[OK]`                                                              -> success
#   everything else, `[....]` included                                  -> info
# The bracket text is what carries the contract, and its match is a substring
# test, so indentation is free there. ERROR and WARNING are NOT: those two arms
# are PREFIX tests, so a line spelled either way has to start at column 0.
# That is the route for output these helpers never produced — setup-k3d.sh's
# prerequisite errors, which setup-all.sh streams along with its own, and
# join-node.sh, which runs on a node and cannot source this file at all.
# join-node.sh's output goes into a per-node log rather than the stream today,
# so its column-0 spelling is what keeps that true if it is ever streamed.
#
# Bash 3.2 (the macOS default): no associative arrays, no ${var,,}.
# ==========================================================================

# Project root, derived from this file's own location rather than from the
# caller, so `AUREAX_ROOT` is right however the script was invoked. Two levels
# up, not one: this file is scripts/lib/common.sh.
AUREAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===================== LOGGING =====================
#
# `fail` takes TWO arguments (what, why). test-all.sh is the single exception:
# it sources this file and then deliberately re-defines fail/skip/check as
# counting variants, `fail` with a one-argument signature — last definition
# wins. See the matching comment at that override.

ok()   { echo "  [OK]   $1"; }
fail() { echo "  [FAIL] $1 — $2"; }
info() { echo "  [....] $1"; }
warn() { echo "  [WARN] $1"; }

section_banner() {
    # A step heading: blank line, rule, title, rule, blank line.
    #
    #   section_banner "Step 3: ..." [width]
    #
    # setup-all.sh's step_header and test-all.sh's section were the same five
    # echoes twice over, differing only in rule width — so the width is the
    # second argument rather than a value one of them has to give up. It is not
    # part of the classifier protocol above: a rule carries no bracket prefix
    # and lands as "info" whatever its length.
    local title="$1" width="${2:-64}" rule
    rule="$(printf '%*s' "$width" '' | tr ' ' '=')"
    echo
    echo "$rule"
    echo "  $title"
    echo "$rule"
    echo
}

# ===================== CONFIG =====================

load_network_env() {
    # Run `scripts/parse-config.py <subcommand> [args...]` and eval its output
    # into THIS shell. Returns the child's exit status.
    #
    # Every call site used to spell this `eval "$(uv run python
    # scripts/parse-config.py ...)"`, which cannot observe a failure: `eval`
    # reports its own status, and the status of the command substitution is
    # discarded. A traceback therefore left every NODE_* name undefined and the
    # script died several lines later on an unbound NODE_COUNT,
    # while download-model.sh's `if ! eval ...` guard could only ever fire on a
    # bash *syntax* error, never on the failure it names.
    #
    # The eval is NOT wrapped in a subshell on purpose: parse-config.py's
    # _print_shell_array emits bare `NAME=(...)`, so this assigns globals that
    # the caller can index. Child stderr is left alone so tracebacks reach the
    # terminal (and, under the dashboard, the streamed setup log).
    local _out
    _out="$(cd "$AUREAX_ROOT" && uv run python scripts/parse-config.py "$@")" || return $?
    eval "$_out"
}

detect_mac_ip() {
    # Print the Mac LAN address the edge nodes can actually reach us on, or
    # fail loudly. This is the only Mac-IP detector in the shell layer.
    #
    # The value is baked into every agent's K3S_URL and stamped into every
    # generated per-node config as ${orchestrator_ip}, so a wrong answer is a
    # fleet-wide misconfiguration that outlives the run that made it.
    #
    # Three rules, each of which was broken somewhere before:
    #   - ASK THE ROUTING TABLE (orchestrator.cluster.k3d.mac_lan_ip).
    #     Interface names do not identify the link — en0 is Wi-Fi on some Macs
    #     and Ethernet on others — so setup-all.sh's old
    #     `ipconfig getifaddr en0 || en1` scan could hand back an address on a
    #     link the nodes never see, which also made `aureax status` report
    #     every node's K3S_URL as stale right after a clean setup.
    #   - NEVER FALL BACK TO LOOPBACK. update-all.sh degraded to 127.0.0.1 on
    #     any failure with stderr discarded, so a resolution failure presented
    #     as an entire fleet shipping telemetry to its own loopback.
    #   - NEVER PROMPT. The dashboard runs setup-all.sh non-interactively
    #     (lifecycle._run_setup_all), where the old `read -p "Mac IP: "` had
    #     nothing to read from and hung the run forever.
    #
    # $MAC_IP is honoured as an override — that is how lifecycle passes the
    # address it has already resolved — but validated identically, because a
    # loopback or link-local override does exactly the same damage.
    local ip origin why
    if [ -n "${MAC_IP:-}" ]; then
        ip="$MAC_IP"
        origin="\$MAC_IP"
    else
        origin="routing table"
        ip="$(cd "$AUREAX_ROOT" && uv run python -c '
from orchestrator.cluster.k3d import mac_lan_ip, probe_host
from orchestrator.config.loader import load_network_config

cfg = load_network_config("configs/aureax-network.yaml")
print(mac_lan_ip(probe_host(cfg)) or "")
')" || ip=""
    fi
    ip="$(printf '%s' "$ip" | tr -d '[:space:]')"

    case "$ip" in
        "")        why="no interface resolved an address the nodes could reach" ;;
        127.*|0.*) why="'${ip}' is loopback — every node would point at itself" ;;
        169.254.*) why="'${ip}' is link-local (DHCP failed) and routable from nowhere" ;;
        *)         echo "$ip"; return 0 ;;
    esac

    # stdout is the answer — callers read this through a command substitution —
    # so diagnostics go to stderr. lifecycle merges the script's stderr into
    # stdout, so the [FAIL] prefix still reaches the dashboard classifier.
    fail "Mac LAN IP (${origin})" "$why" >&2
    echo "         Fix the network, or re-run with an explicit address:" >&2
    echo "           MAC_IP=192.0.2.1 $0" >&2
    return 1
}

k3s_token_from_cluster() {
    # Print the K3s join token held by a k3d cluster's first server container.
    #
    #   k3s_token_from_cluster "$K3D_CLUSTER_NAME"
    #
    # Empty output and a non-zero status when the container is absent or not
    # running. ONE spelling of the `k3d-<cluster>-server-0` container name and
    # of the node-token path in the shell layer: they were written out twice,
    # in two different styles, so a change to either had to be found in both.
    # (orchestrator/cluster/k3d.py carries the Python spelling.)
    #
    # The token is printed, never echoed into a log or a file: it is the one
    # credential that lets a machine join the cluster.
    local cluster="$1"
    docker exec "k3d-${cluster}-server-0" \
        cat /var/lib/rancher/k3s/server/node-token 2>/dev/null
}

# ===================== SSH =====================

node_index_of() {
    # Index of a host in the lockstep NODE_* arrays, or -1.
    #
    # Callers that know a host but not its index — the group sections in
    # test-all.sh, and node_ssh_port below — need this because `user`,
    # `key_file` and `port` are all PER TARGET in the schema: reaching every
    # member of a group with node 0's credentials fails auth the moment a group
    # spans nodes with different logins.
    #
    # Tolerates the NODE_* arrays being unset entirely, which is the normal
    # state in bench-decode.sh: it takes a host on argv and never reads the
    # network config. build-on-node.sh takes its target the same way but
    # synthesises one-element arrays from argv, because it needs the port.
    local want="$1" i n="${NODE_COUNT:-0}"
    for ((i = 0; i < n; i++)); do
        if [ "${NODE_HOSTS[$i]:-}" = "$want" ]; then echo "$i"; return 0; fi
    done
    echo "-1"
}

node_ssh_port() {
    # The configured SSH port for a host; 22 when we hold no config for it.
    #
    # `port` is per-target in the schema (orchestrator/config/schema.py) and
    # every Python path honours it, but the shell path dropped it silently — so
    # a target on a non-22 port was unreachable from every bash step while
    # paramiko connected to it perfectly well. NODE_PORTS is
    # emitted by `parse-config.py all` in lockstep with NODE_HOSTS.
    local idx
    idx="$(node_index_of "$1")"
    if [ "$idx" -ge 0 ]; then
        echo "${NODE_PORTS[$idx]:-22}"
    else
        echo 22
    fi
}

# BatchMode=yes is load-bearing, not tidiness. Every target may also carry a
# `password` in the network config, so when key auth fails — the group-writable
# $HOME / StrictModes case is the usual one — ssh falls back to an interactive
# password prompt on /dev/tty. In an interactive terminal that prompt blocks
# forever, and setup-all.sh's Step 6 runs several builds at once with their
# output redirected to per-node log files, so the prompt is invisible and `wait`
# never returns. BatchMode turns that into an immediate, logged auth failure.
#
# ServerAlive* bounds the other half: ConnectTimeout only covers the handshake,
# so a node that freezes mid-command (or drops off the LAN) holds the session
# open indefinitely. Four missed 15s probes drop it after ~60s.
#
# StrictHostKeyChecking=no, against the SAME ~/.aureax/known_hosts the Python
# side keeps (UserKnownHostsFile on every ssh/scp/rsync below): a node never
# seen before is recorded, and a node whose key CHANGED — routine here, nodes
# are wiped and re-imaged — gets OpenSSH's loud "REMOTE HOST IDENTIFICATION HAS
# CHANGED" warning and still connects with the key (a changed key only disables
# password auth, which BatchMode never offers). The next `aureax status`
# rewrites the stale entry. accept-new would refuse instead and turn every
# re-image into a hand edit — see ARCHITECTURE.md, "SSH authentication".
SSH_BATCH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no"

# THE REMOTE SHELL IS THE NODE ACCOUNT'S LOGIN SHELL, and nothing here pins it.
# ssh hands the command string to that shell, so what a snippet may use is
# whatever the node happens to run. The fleet is not consistent about this
# today, and both halves are deliberate:
#   - build-on-node.sh's install snippet uses bash arrays, and ssh_facts asks
#     the node for `printf %q`, a bash builtin-printf feature — both assume
#     bash.
#   - reset-all.sh's wipe avoids bash-isms (`ls -d`, not `compgen -G`) because
#     under dash or ash the builtin returns 127, which its `set +e` swallows
#     into a leftover check that passes with the leftovers still on disk.
# Pinning it here (invoking `bash -s` with the snippet on stdin) is the way to
# close that, and would let the defensive spellings go. Until then, a snippet
# added below is bash-only unless it says otherwise.
ssh_cmd() {
    local host="$1" user="$2" key="$3"; shift 3
    ssh -p "$(node_ssh_port "$host")" \
        $SSH_BATCH_OPTS -o UserKnownHostsFile="$HOME/.aureax/known_hosts" -o ConnectTimeout=10 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -i "$key" "${user}@${host}" "$@"
}

_scp_host() {
    # `[user@]host:path` -> host; empty for a local path.
    case "$1" in
        *:*)
            local hostpart="${1%%:*}"
            echo "${hostpart#*@}"
            ;;
        *) echo "" ;;
    esac
}

scp_cmd() {
    # scp spells the port -P, and the host is buried inside a `user@host:path`
    # operand rather than passed separately, so recover it from whichever
    # operand is the remote one.
    #
    # Establishment timeout does not limit the duration of a large transfer.
    local key="$1" src="$2" dest="$3" host
    host="$(_scp_host "$dest")"
    [ -n "$host" ] || host="$(_scp_host "$src")"
    scp -P "$(node_ssh_port "$host")" $SSH_BATCH_OPTS \
        -o UserKnownHostsFile="$HOME/.aureax/known_hosts" \
        -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i "$key" "$src" "$dest"
}

ssh_facts() {
    # Ask a node several questions in ONE round trip.
    #
    #   ssh_facts "$HOST" "$USER" "$KEY" \
    #       "ARCH=uname -m" \
    #       "NVIDIA=command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no"
    #
    # prints, in the order asked:
    #   ARCH=aarch64
    #   NVIDIA=no
    #
    # Worth having because there is no connection multiplexing here: every
    # extra ask is a full SSH handshake with ConnectTimeout=10 against a node
    # that may be on the far side of a slow link, and several steps ask two or
    # three one-line questions in a row.
    #
    # A snippet that produces nothing yields an empty value rather than a
    # missing line, so a caller can rely on getting back every key it asked
    # for. Values are single-line facts by construction — a snippet printing
    # several lines emits several lines under one key.
    #
    # The VALUE is asked for %q-quoted so that a value containing a space or a
    # newline still arrives as one line under one key. That is ALL it buys:
    # the quoting is performed by the node, so it constrains nothing a
    # compromised node chooses to print.
    #
    # CALLERS MUST PARSE THIS, NEVER `eval` IT — with sed, the way
    # detect_node_arch below does, where the reasoning is spelled out in full.
    # (%q is a bash builtin-printf feature, so this ask is one of the bash
    # assumptions listed at ssh_cmd above.)
    local host="$1" user="$2" key="$3"; shift 3
    local pair name snippet remote=""
    for pair in "$@"; do
        name="${pair%%=*}"
        snippet="${pair#*=}"
        # \$( ) is escaped so the substitution runs on the NODE, not here.
        remote="${remote}printf '%s=%q\n' '${name}' \"\$(${snippet})\"
"
    done
    ssh_cmd "$host" "$user" "$key" "$remote"
}

# ===================== FILES + BUILD =====================

file_size() {
    # Size in bytes, 0 when the file is absent.
    #
    # BSD `stat -f%z` answers first, and on the orchestrator it always answers:
    # every caller is orchestrator-side (download-model.sh), and this file is
    # never sourced on a node. The GNU form is a portability fallback for a
    # Linux orchestrator, unreachable today — and it has to stay second, since
    # on Linux `stat -f` means --file-system, where %z is not a valid specifier.
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0
}

# Which Dockerfile builds a node's image. ONE copy: build-on-node.sh (used by
# setup-all.sh) and update-all.sh must agree, or the two produce different
# images for the same machine and a group's llama.cpp version lock breaks —
# and a mismatched RPC build HANGS at the handshake rather than erroring.
#
# The x86-without-GPU case FAILS rather than falling through to the CPU file:
# docker/Dockerfile.client hard-codes an ARM ISA baseline
# (-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod), so building it on x86_64
# yields a broken image, and a broken image on one group member is that same
# silent hang.
#
# Returns 1 rather than exiting, so a caller that builds several nodes
# concurrently decides for itself whether one bad node aborts the run.
pick_dockerfile() {
    local arch="$1" has_nvidia="$2" host="$3"
    case "$arch" in
        aarch64|arm64) echo "docker/Dockerfile.client" ;;
        x86_64)
            if [ "$has_nvidia" = "yes" ]; then
                echo "docker/Dockerfile.client.cuda"
            else
                fail "$host" "x86_64 without an NVIDIA GPU has no Dockerfile — docker/Dockerfile.client is ARM-only (armv8.2-a ISA baseline). Add an x86 CPU Dockerfile before deploying this node." >&2
                return 1
            fi
            ;;
        *)
            fail "$host" "unrecognised architecture '${arch}' — no Dockerfile mapping" >&2
            return 1
            ;;
    esac
}

detect_node_arch() {
    # What a node is, in ONE round trip, left in NODE_ARCH / NODE_HAS_NVIDIA:
    #
    #   detect_node_arch "$HOST" "$USER" "$KEY" || exit 1
    #   DOCKERFILE=$(pick_dockerfile "$NODE_ARCH" "$NODE_HAS_NVIDIA" "$HOST")
    #
    # Returns 1 (with a [FAIL] line on stderr) when the probe cannot run, so a
    # caller building several nodes concurrently decides for itself whether one
    # unreachable node aborts the run — same contract as pick_dockerfile.
    #
    # One copy for the same reason pick_dockerfile is one copy: setup-all.sh
    # builds through build-on-node.sh and update-all.sh builds through its own
    # loop, and two machines answering this question differently is two
    # different images and a version-locked RPC handshake that hangs.
    local host="$1" user="$2" key="$3" facts
    facts="$(ssh_facts "$host" "$user" "$key" \
        "ARCH=uname -m" \
        "HAS_NVIDIA=command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no")" || {
        fail "$host" "could not probe architecture and GPU over SSH" >&2
        return 1
    }
    # Command-substituted into a variable first, so an ssh failure is visible:
    # a command substitution consumed inline reports the consumer's status and
    # discards the child's, which would leave both names empty and blame the
    # architecture.
    #
    # PARSED, never eval'd. This text is whatever the NODE's shell printed, so
    # `eval` would run it here with the operator's privileges — and every shard
    # worker exposes an unauthenticated rpc-server on the LAN (see
    # ARCHITECTURE.md, "Security"), which makes "compromised node escalates to
    # the orchestrator on the next build" a concrete path rather than a
    # theoretical one. ssh_facts %q-quotes each value, but that quoting happens
    # ON THE NODE and so constrains nothing a compromised node chooses to
    # print — parsing is what makes it inert. pick_dockerfile fails closed on
    # an unrecognised arch, so it is also all the validation this needs.
    NODE_ARCH="$(printf '%s\n' "$facts" | sed -n 's/^ARCH=//p')"
    NODE_HAS_NVIDIA="$(printf '%s\n' "$facts" | sed -n 's/^HAS_NVIDIA=//p')"
}

rsync_usable() {
    # True when the project tree can be shipped with rsync rather than the
    # tar+scp fallback both call sites already carry.
    #
    # The key-path check is not paranoia. rsync splits its `-e` string on
    # whitespace ITSELF, with no quoting and no backslash handling, so an SSH
    # key path containing a space cannot be expressed there at all — no amount
    # of local quoting helps, and what reaches ssh is a truncated path and a
    # baffling error. The scp path takes the key as its own argv word and is
    # unaffected, so route those transfers through it instead.
    local key="$1"
    command -v rsync >/dev/null 2>&1 || return 1
    case "$key$HOME" in
        *[[:space:]]*) return 1 ;;
    esac
    return 0
}

rsync_rsh() {
    # The `-e` value for rsync: ssh with our standard options and this host's
    # configured port and shared trust store. rsync_usable guards path whitespace.
    local host="$1" key="$2"
    echo "ssh -p $(node_ssh_port "$host") $SSH_BATCH_OPTS -o UserKnownHostsFile=$HOME/.aureax/known_hosts -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i $key"
}

project_tar() {
    # Build the fallback archive of the project tree at <outfile>.
    #
    #   project_tar /tmp/aureax-update.tar.gz
    #
    # Its own function so a caller that ships ONE archive to a whole fleet can
    # build it once, up front, and still get the same exclude list
    # sync_project_tree's own fallback uses. update-all.sh builds here (its
    # per-node syncs run concurrently in subshells, where an archive built
    # lazily inside one would be invisible to the others and to the EXIT trap
    # that reclaims it); sync_project_tree builds here too when no caller has.
    #
    # `models`, not `models/`: tar treats the trailing slash literally, so the
    # slashed spelling matched nothing and shipped the GGUFs.
    #
    # `tests` and `.github` never reach a node either: nothing on a node runs
    # them, .dockerignore keeps them out of the daemon's context, and the
    # test tree is the largest thing left once models and configs are out.
    local outfile="$1"
    tar czf "$outfile" \
        --exclude='.venv' \
        --exclude='.git' \
        --exclude='build' \
        --exclude='__pycache__' \
        --exclude='*.egg-info' \
        --exclude='models' \
        --exclude='configs' \
        --exclude='tests' \
        --exclude='.github' \
        -C "$AUREAX_ROOT" .
}

stage_token() {
    # 16 hex characters from the kernel CSPRNG. $RANDOM is 15 bits seeded from
    # the pid and the clock, which is guessable by exactly the local account a
    # staged name exists to keep out.
    od -An -N8 -tx1 /dev/urandom | tr -d ' \n'
}

remote_build_dir() {
    # Prepare the node account's private staging directory and print the build
    # context path inside it, relative to the node user's home (where a
    # relative rsync/scp destination lands).
    #
    #   remote_build_dir "$HOST" "$USER" "$KEY"
    #
    # NOT a fixed path under /tmp: the callers run `sudo docker build` against
    # this directory, so any other local account on the node that can
    # pre-create it — `mkdir -p` on an existing directory succeeds silently —
    # or swap a file in it between the sync and the build chooses what root's
    # daemon bakes into the image. The same goes for the tar fallback's
    # <dir>.tar.gz, which lands beside it. Same convention as the join-script
    # staging in setup-all.sh and orchestrator/deploy/ssh.py::stage_path.
    #
    # The leaf name is fixed rather than random because the 0700 parent is
    # already private to the node account, so unpredictability buys nothing
    # here, while a per-run name would accumulate a full project tree on the
    # node for every build that was interrupted before its callers could clean
    # up. The chmod is what makes the private parent hold: it fails — and takes
    # the build with it — if the directory belongs to someone else.
    #
    # /tmp/aureax-build is the path this staging replaces, so it is reclaimed
    # here: nothing else on the node would ever touch it again, and a tree left
    # there keeps its copy of whatever the sync excludes today.
    local host="$1" user="$2" key="$3"
    ssh_cmd "$host" "$user" "$key" "
        mkdir -p ~/.aureax-stage && chmod 700 ~/.aureax-stage || exit 1
        rm -rf /tmp/aureax-build /tmp/aureax-build.new /tmp/aureax-build.tar.gz 2>/dev/null || true
    " >/dev/null || return 1
    printf '%s\n' ".aureax-stage/build"
}

sync_project_tree() {
    # Ship the project tree to <remote_dir> on a node.
    #
    #   sync_project_tree "$HOST" "$USER" "$KEY" "$REMOTE_DIR" [tarball]
    #
    # rsync -az --delete when rsync_usable says so, otherwise tar+scp with an
    # extract-into-.new-and-swap on the node, so both branches converge on the
    # SAME tree: unpacking over the existing directory left every locally
    # deleted file in place and still inside the `docker build` context.
    #
    # [tarball] is the fallback archive's path. When given it is built once and
    # REUSED if it already exists — that is how a caller ships one archive to a
    # whole fleet — and the caller owns its removal. Without it a temp archive
    # is made and deleted inside this call, so nothing is left in /tmp when a
    # transfer fails.
    #
    # WHAT IS EXCLUDED, and why configs/ is on the list: the Dockerfiles COPY
    # only pyproject.toml, uv.lock*, client/, three orchestrator files and
    # docker/scripts/start.sh, so none of this is needed to build the image —
    # and configs/aureax-network.yaml carries the fleet's plaintext SSH
    # passwords, which would otherwise sit in /tmp on every node, handing any
    # one of them the logins for all of them. .dockerignore excludes it from
    # the daemon's context; this excludes it from the node entirely.
    #
    # --delete-excluded, not just --delete: plain --delete leaves a file
    # MATCHING an --exclude untouched on the receiver, so a directory that once
    # held configs/ would keep that copy through every later sync. The tar
    # branch converges by construction (it extracts into <dir>.new and swaps),
    # and this is what makes the rsync branch answer the same way.
    #
    # The source is $AUREAX_ROOT, never `.`: a caller run from somewhere else
    # would otherwise upload whatever directory the operator happened to be in.
    local host="$1" user="$2" key="$3" remote_dir="$4" tarball="${5:-}"
    local rc=0 owned=0
    if rsync_usable "$key"; then
        rsync -az --delete --delete-excluded \
            -e "$(rsync_rsh "$host" "$key")" \
            --exclude '.venv' \
            --exclude '.git' \
            --exclude 'build' \
            --exclude '__pycache__' \
            --exclude '*.egg-info' \
            --exclude 'models/' \
            --exclude 'configs/' \
            --exclude 'tests/' \
            --exclude '.github/' \
            "${AUREAX_ROOT}/" "${user}@${host}:${remote_dir}/"
        return
    fi

    if [ -z "$tarball" ]; then
        tarball="$(mktemp "${TMPDIR:-/tmp}/aureax-sync.XXXXXX")" || return 1
        owned=1
    fi
    if [ ! -s "$tarball" ]; then
        project_tar "$tarball" || rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        scp_cmd "$key" "$tarball" "${user}@${host}:${remote_dir}.tar.gz" || rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        ssh_cmd "$host" "$user" "$key" "
            rm -rf ${remote_dir}.new && mkdir -p ${remote_dir}.new &&
            tar xzf ${remote_dir}.tar.gz -C ${remote_dir}.new &&
            rm -f ${remote_dir}.tar.gz &&
            rm -rf ${remote_dir} && mv ${remote_dir}.new ${remote_dir}
        " || rc=$?
    fi
    if [ "$owned" = 1 ]; then
        rm -f "$tarball"
    fi
    return "$rc"
}

# ===================== EDGE ENDPOINTS =====================

# The fleet CA that `aureax generate` mints into ~/.aureax/tls/ca.pem
# (orchestrator/config/tlsstore.py). Overridable so a run against a second
# fleet, or a CI checkout with its own store, does not have to move the file.
AUREAX_CA="${AUREAX_CA:-$HOME/.aureax/tls/ca.pem}"

curl_tls() {
    # curl with the fleet CA. Every endpoint this repository dials is HTTPS
    # served with a certificate the fleet CA signed, and nothing here has a
    # plaintext fallback: an operator who has not run `aureax init && aureax
    # generate` has no CA, and must be told that rather than handed a green
    # check from a connection nothing authenticated. No -k / --insecure for
    # the same reason — it would turn every one of these checks into a test
    # that the port answers, not that the right machine answered.
    #
    # 77 is curl's own "problem with the SSL CA cert" exit code, so a caller
    # that already classifies curl statuses classifies this one correctly too.
    if [ ! -r "$AUREAX_CA" ]; then
        echo "aureax: no fleet CA at ${AUREAX_CA} — run 'uv run aureax generate --orchestrator-ip <MAC_LAN_IP>'" >&2
        return 77
    fi
    curl --cacert "$AUREAX_CA" "$@"
}

mcp_health_wait() {
    # Wait for a node's MCP server to answer /health. Prints nothing; the
    # status is the answer.
    #
    #   mcp_health_wait "$HOST" "$MCP_NODE_PORT" 0            # one probe
    #   mcp_health_wait "$HOST" "$MCP_NODE_PORT" "$remaining"  # poll to a budget
    #
    # <budget> is SECONDS REMAINING, not a per-node allowance: callers that
    # restarted a whole fleet together share one deadline and pass what is left
    # of it, so a nine-node check cannot multiply the worst case by nine. A
    # budget of 0 (or a negative one, from an already-expired deadline) still
    # costs exactly one probe — the deadline is checked AFTER the curl, so
    # every node is asked at least once.
    #
    # The port is a parameter rather than a literal here: it comes from
    # $MCP_NODE_PORT, which `parse-config.py all` exports from schema.py, so
    # changing the Service's NodePort does not leave the shell checks probing
    # the old one.
    #
    # HTTPS through curl_tls, and unauthenticated on purpose: /health is the
    # one path client/mcp/auth.py exempts from bearer auth, which is what lets
    # this helper (and setup-all.sh, update-all.sh and test-all.sh through it)
    # poll a node that holds a key it was never given.
    #
    # Reporting is left to the caller. The three call sites label their results
    # differently — ok/fail, a plain echo, and test-all.sh's counting `check`
    # — and test-all.sh redefines `fail` with a one-argument signature, so a
    # helper that printed would print wrongly in at least one of them.
    local host="$1" port="$2" budget="${3:-0}" deadline
    deadline=$(( $(date +%s) + budget ))
    while :; do
        if curl_tls -sf --max-time 5 "https://${host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        # `if`, not `[ ... ] && return 1`: the && form leaves the loop body's
        # status at the test's, which is a `set -e` abort waiting to happen if
        # a future edit ever makes it the last statement here.
        if [ "$(date +%s)" -ge "$deadline" ]; then
            return 1
        fi
        sleep 5
    done
}
