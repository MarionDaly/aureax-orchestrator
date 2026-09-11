"""`scripts/parse-config.py` emits shell-sourceable variables.

Six shell scripts source that output and index the NODE_* arrays positionally
against each other, so the arrays agreeing on length is a load-bearing property
rather than a cosmetic one. These tests source the real emitted text with bash
and check that property directly — asserting on the Python-side strings would
miss the whole failure mode, which is bash's word splitting.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from ruamel.yaml import YAML

from tests.conftest import EXAMPLE_NETWORK_CONFIG, PROJECT_ROOT

PARSE_CONFIG = PROJECT_ROOT / "scripts" / "parse-config.py"

# Every array `all` emits that shell callers index in lockstep with NODE_HOSTS.
LOCKSTEP_ARRAYS = [
    "NODE_HOSTS",
    "NODE_USERS",
    "NODE_PORTS",
    "NODE_KEY_FILES",
    "NODE_IDS",
    "NODE_CONFIG_FILES",
    "NODE_ROLES",
    "NODE_HAS_MCP_SERVER",
    "NODE_AUTH",
    "NODE_MCP_AUTH",
]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _emit(config_path, field: str = "all", index: int | None = None) -> str:
    """Run `parse-config.py` and return its stdout.

    Two argv layouts, both load-bearing. Most fields take `<field> <config>`;
    `target` and `model_for_node` take the index POSITIONALLY with the config
    path as an optional THIRD argument (see ARCHITECTURE.md, "7. Gotchas" — a
    caller that passes the config second silently parses it as the index).
    """
    argv = [sys.executable, str(PARSE_CONFIG), field]
    if index is not None:
        argv.append(str(index))
    argv.append(str(config_path))
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    return proc.stdout


def _source(emitted: str, probe: str) -> str:
    """Source `emitted` in bash under `set -eu`, run `probe`, return its stdout.

    The exit status is asserted here rather than delegated to `check=True`:
    bash reports what went wrong on stderr and CalledProcessError does not show
    it, so a broken emit surfaced as a bare "returned non-zero exit status 1"
    with no clue which array or which line caused it.

    `set -e` is itself part of the contract — every script that sources this
    output runs under `set -euo pipefail`. Without it a failing line inside
    `emitted` does not stop bash, the trailing probe returns 0, and the test
    reports lengths for a config parse that had already failed.
    """
    proc = subprocess.run(
        ["bash", "-c", f"set -eu\n{emitted}\n{probe}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"bash exited {proc.returncode} while sourcing parse-config.py output: "
        f"{proc.stderr.strip() or '<no stderr>'}"
    )
    return proc.stdout


def _array_lengths(emitted: str) -> dict[str, int]:
    """Source `emitted` in bash and report each lockstep array's length."""
    probe = "\n".join(f'echo "{name} ${{#{name}[@]}}"' for name in LOCKSTEP_ARRAYS)
    out = {}
    for line in _source(emitted, probe).strip().splitlines():
        name, count = line.split()
        out[name] = int(count)
    return out


def _element(emitted: str, array: str, index: int) -> str:
    """Source `emitted` in bash and return one array element verbatim."""
    return _source(emitted, f'printf "%s" "${{{array}[{index}]}}"')


def _write_network(tmp_path, data):
    """Write a modified network config beside COPIES of the shipped per-node files.

    parse-config.py resolves each target's `config_file` against the network
    config's OWN parent directory, and a `config_file` that is named but
    missing is a hard error rather than an empty role — an empty role reads as
    "ordinary edge node" in every shell consumer, which is how a shard worker
    would end up labelled edge and sent a 22 GB GGUF. So a temp network config
    has to bring the directory with it, which also means these tests exercise
    the real roles/mcp/auth values instead of a fleet of blanks.
    """
    for src in EXAMPLE_NETWORK_CONFIG.parent.glob("*.yaml"):
        # Per-node templates only. A fleet definition is not one, nothing here
        # resolves it, and the operator-owned copy carries SSH credentials.
        if src.name.startswith("aureax-network"):
            continue
        shutil.copy(src, tmp_path / src.name)
    path = tmp_path / "network.yaml"
    with path.open("w") as fh:
        YAML().dump(data, fh)
    return path


def test_shipped_config_arrays_are_all_node_count_long() -> None:
    """The example config: every lockstep array is exactly NODE_COUNT long."""
    emitted = _emit(EXAMPLE_NETWORK_CONFIG)
    expected = int(next(x for x in emitted.splitlines() if x.startswith("NODE_COUNT=")).split("=")[1])
    assert expected > 0
    assert _array_lengths(emitted) == dict.fromkeys(LOCKSTEP_ARRAYS, expected)


def test_the_all_branch_exports_both_role_literals() -> None:
    """setup-all.sh tests NODE_ROLES against $SHARD_WORKER_ROLE and $PLANNER_ROLE.

    It runs under `set -u`, so a literal that is not printed here is not a
    mismatch — it aborts the script at the first role test. Both come from
    schema.py so a rename there reaches every shell site at once.
    """
    from orchestrator.config.schema import PLANNER_ROLE, SHARD_WORKER_ROLE

    emitted = _emit(EXAMPLE_NETWORK_CONFIG).splitlines()
    assert f"SHARD_WORKER_ROLE={SHARD_WORKER_ROLE}" in emitted
    assert f"PLANNER_ROLE={PLANNER_ROLE}" in emitted


def test_target_without_key_or_config_file_keeps_arrays_aligned(tmp_path) -> None:
    """A target with no `key_file`/`config_file` must not shorten any array.

    `key_file` and `config_file` are both optional in the schema, and a target
    with no resolvable config file has no role. Those elements are empty
    strings, which bash drops from an unquoted array literal — collapsing the
    array and silently shifting every later index onto the wrong node.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    first = data["ssh"]["targets"][0]
    first.pop("key_file", None)
    first.pop("config_file", None)

    path = _write_network(tmp_path, data)

    emitted = _emit(path)
    n = len(data["ssh"]["targets"])
    assert _array_lengths(emitted) == dict.fromkeys(LOCKSTEP_ARRAYS, n)

    # The empty elements are present and empty, not absent.
    assert _element(emitted, "NODE_KEY_FILES", 0) == ""
    assert _element(emitted, "NODE_CONFIG_FILES", 0) == ""
    # ...and the node after them still pairs with its own host.
    assert _element(emitted, "NODE_HOSTS", 1) == data["ssh"]["targets"][1]["host"]
    assert _element(emitted, "NODE_IDS", 1) == data["ssh"]["targets"][1]["node_id"]


def test_non_default_port_reaches_the_shell(tmp_path) -> None:
    """`port` is per-target and the shell path used to drop it entirely.

    Every Python path honours it (paramiko connects fine), so a target on a
    non-22 port was unreachable from the bash steps only — with `aureax status`
    reporting it healthy. scripts/lib/common.sh's ssh_cmd/scp_cmd read it back
    out of this array by host.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    data["ssh"]["targets"][0]["port"] = 2222
    # Stated, not inherited: the second target is on the schema default because
    # this test put it there. Relying on the shipped fleet never configuring a
    # port would fail the day an operator moves a node off 22, with no defect
    # in parse-config.py.
    data["ssh"]["targets"][1].pop("port", None)

    path = _write_network(tmp_path, data)

    emitted = _emit(path)
    assert _element(emitted, "NODE_PORTS", 0) == "2222"
    # The neighbouring target keeps the schema default rather than inheriting it.
    assert _element(emitted, "NODE_PORTS", 1) == "22"


def test_k3d_identity_comes_from_the_config_not_the_defaults(tmp_path) -> None:
    """`k3d` emits the cluster identity setup-all.sh used to hardcode.

    setup-all.sh spelled the cluster name, API port and server count as
    literals — `grep -q "aureax"`, `docker exec k3d-aureax-server-0`, `:6443`,
    and a node-readiness target of NODE_COUNT + 1 — while the Python path read
    them from the YAML. Editing either value broke the bash path only. The
    non-default values below are what proves this reads the file.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    data["kubernetes"]["cluster_name"] = "aureax-bench"
    data["kubernetes"]["k3d"]["api_port"] = 7443
    data["kubernetes"]["k3d"]["servers"] = 3

    path = _write_network(tmp_path, data)

    emitted = _emit(path, field="k3d")
    # Read back through bash, like the caller does: K3D_SERVERS lands in
    # arithmetic (NODE_COUNT + K3D_SERVERS) and the name in a container name.
    probe = 'printf "%s|%s|%s" "$K3D_CLUSTER_NAME" "$K3D_API_PORT" "$K3D_SERVERS"'
    assert _source(emitted, probe) == "aureax-bench|7443|3"


def test_k3d_cluster_name_is_quoted_for_the_shell(tmp_path) -> None:
    """A name with a space stays one word rather than splitting."""
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    data["kubernetes"]["cluster_name"] = "two words"

    path = _write_network(tmp_path, data)

    emitted = _emit(path, field="k3d")
    assert _source(emitted, 'printf "%s" "$K3D_CLUSTER_NAME"') == "two words"


def test_values_containing_shell_metacharacters_survive_intact(tmp_path) -> None:
    """Config values are quoted, so spaces and metacharacters do not split."""
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    data["ssh"]["targets"][0]["user"] = "we ird$(echo pwned)"

    path = _write_network(tmp_path, data)

    emitted = _emit(path)
    n = len(data["ssh"]["targets"])
    assert _array_lengths(emitted)["NODE_USERS"] == n
    assert _element(emitted, "NODE_USERS", 0) == "we ird$(echo pwned)"


# --- groups ---------------------------------------------------------------


def test_group_arrays_are_lockstep_and_worker_lists_stay_one_element() -> None:
    """`groups` emits five arrays test-all.sh indexes together.

    GROUP_WORKER_HOSTS is the one array that deliberately contains spaces — a
    whole space-separated host list packed into ONE element, which the consumer
    unpacks with `read -r -a`. If the quoting collapsed, group B's checks would
    silently run against group A's hosts, so the failure mode this file exists
    for (bash word splitting) is checked on the array most exposed to it.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    groups = data.get("node_groups") or []
    assert groups, "the example config is expected to define node_groups"

    emitted = _emit(EXAMPLE_NETWORK_CONFIG, field="groups")
    probe = "\n".join(
        [
            'echo "COUNT ${GROUP_COUNT}"',
            'echo "LEN_NAMES ${#GROUP_NAMES[@]}"',
            'echo "LEN_HEADS ${#GROUP_HEAD_HOSTS[@]}"',
            'echo "LEN_HEAD_IDS ${#GROUP_HEAD_IDS[@]}"',
            'echo "LEN_PORTS ${#GROUP_RPC_PORTS[@]}"',
            'echo "LEN_WORKERS ${#GROUP_WORKER_HOSTS[@]}"',
            'for ((gi = 0; gi < GROUP_COUNT; gi++)); do',
            '  read -r -a W <<< "${GROUP_WORKER_HOSTS[$gi]}" || true',
            '  echo "WORKERS $gi ${#W[@]} ${GROUP_NAMES[$gi]}"',
            "done",
        ]
    )
    out = _source(emitted, probe)
    lines = dict(line.split(" ", 1) for line in out.strip().splitlines() if not line.startswith("WORKERS"))
    n = len(groups)
    assert lines["COUNT"] == str(n)
    for key in ("LEN_NAMES", "LEN_HEADS", "LEN_HEAD_IDS", "LEN_PORTS", "LEN_WORKERS"):
        assert lines[key] == str(n), f"{key} is {lines[key]}, expected {n}: {out!r}"

    seen = [line.split() for line in out.strip().splitlines() if line.startswith("WORKERS")]
    assert len(seen) == n
    for (_, gi, count, name), group in zip(seen, groups, strict=True):
        assert name == group["name"], out
        assert int(count) == len(group["workers"]), (
            f"group {name} unpacked {count} worker host(s), config lists "
            f"{len(group['workers'])} — the space-separated element collapsed"
        )
        assert gi == str(groups.index(group))


def test_group_head_ids_pair_with_group_head_hosts_index_by_index() -> None:
    """GROUP_HEAD_IDS[i] is the node id of the machine GROUP_HEAD_HOSTS[i] addresses.

    test-all.sh section 6 reads the host for the checks that dial a port and the
    id for `aureax-harness infer`, which needs a node id to find that node's MCP
    bearer key in the key store — a bare address sends no key and the head
    answers 401. Two arrays for one machine only help while they stay in step,
    and nothing in bash would notice them drifting: it would simply run the
    inference check for group A against group B's key.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    groups = data.get("node_groups") or []
    assert groups, "the example config is expected to define node_groups"
    host_by_id = {t["node_id"]: t["host"] for t in data["ssh"]["targets"]}

    emitted = _emit(EXAMPLE_NETWORK_CONFIG, field="groups")
    for gi, group in enumerate(groups):
        head_id = _element(emitted, "GROUP_HEAD_IDS", gi)
        head_host = _element(emitted, "GROUP_HEAD_HOSTS", gi)
        assert head_id == group["head"], f"group {group['name']}: head id"
        assert head_host == host_by_id[group["head"]], f"group {group['name']}: head host"


# --- model_for_node -------------------------------------------------------


def _network_with_probe_node(tmp_path, model_filename: str, registry_entry: dict | None):
    """Write a temp network config whose FIRST enabled target uses a probe file.

    The per-node config lands beside the temp network file because
    parse-config.py resolves `config_file` against the network config's own
    parent directory.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    targets = data["ssh"]["targets"]
    first_enabled = next(i for i, t in enumerate(targets) if t.get("enabled", True))
    targets[first_enabled]["config_file"] = "probe-node.yaml"
    if registry_entry is not None:
        data["model_registry"]["models"].append(registry_entry)

    (tmp_path / "probe-node.yaml").write_text(
        "node:\n"
        "  id: probe\n"
        '  role: "edge_executor"\n'
        "model:\n"
        f'  filename: "{model_filename}"\n'
    )
    path = _write_network(tmp_path, data)
    # The probe node is the first ENABLED target, and `model_for_node <n>`
    # indexes the enabled list.
    return path


MODEL_PROBE = (
    'printf "%s|%s|%s|%s|%s|%s|%s|%s" '
    '"$MODEL_FILENAME" "$MODEL_URL" "$MODEL_SIZE_GB" '
    '"${#MODEL_PARTS[@]}" "${#MODEL_PART_URLS[@]}" '
    '"$MODEL_DRAFT_FILENAME" "$MODEL_DRAFT_URL" "$MODEL_DRAFT_SIZE_GB"'
)


def test_split_model_emits_paired_part_arrays(tmp_path) -> None:
    """MODEL_PARTS and MODEL_PART_URLS are the same length, in part order.

    download-model.sh zips them positionally (`read -r -a parts` /
    `read -r -a part_urls`, then `part_urls[$pi]` against `parts[$pi]`), so a
    length mismatch downloads one part's bytes under another part's name. The
    local basenames and the repo paths differ on purpose — a split quant lives
    in a per-quant subdirectory upstream while landing flat in models/.
    """
    path = _network_with_probe_node(
        tmp_path,
        "Probe-Split-Q4_K_M-00001-of-00003.gguf",
        {
            "name": "Probe-Split",
            "filename": "Probe-Split-Q4_K_M-00001-of-00003.gguf",
            "size_gb": 30.0,
            "quantization": "Q4_K_M",
            "huggingface_repo": "acme/probe-gguf",
            "huggingface_file": "Q4_K_M/Probe-Split-Q4_K_M-00001-of-00003.gguf",
        },
    )

    emitted = _emit(path, field="model_for_node", index=0)
    fields = _source(emitted, MODEL_PROBE).split("|")
    filename, url, size_gb, n_parts, n_urls, d_name, d_url, d_size = fields

    assert filename == "Probe-Split-Q4_K_M-00001-of-00003.gguf", "MODEL_FILENAME must be part 1"
    assert n_parts == "3"
    assert n_parts == n_urls
    assert size_gb == "30.0"
    assert url.endswith("/Q4_K_M/Probe-Split-Q4_K_M-00001-of-00003.gguf")
    # No draft configured — every name still defined, because download-model.sh
    # reads all of them under `set -u`.
    assert (d_name, d_url, d_size) == ("", "", "")

    # And the pairing itself, element by element.
    pairs = _source(
        emitted,
        'for i in "${!MODEL_PARTS[@]}"; do printf "%s %s\\n" "${MODEL_PARTS[$i]}" "${MODEL_PART_URLS[$i]}"; done',
    ).strip().splitlines()
    assert len(pairs) == 3
    for part, part_url in (line.split() for line in pairs):
        assert part_url.endswith(f"/Q4_K_M/{part}"), (part, part_url)


def test_model_absent_from_registry_still_defines_every_name(tmp_path) -> None:
    """An unknown model.filename leaves all eight MODEL_* names defined.

    download-model.sh evals this output in a loop under `set -u` and only then
    reports "could not resolve model" against MODEL_FILENAME. An undefined name
    would abort the script before that diagnostic, and a name left over from the
    previous node would assign one node's model to another.
    """
    path = _network_with_probe_node(tmp_path, "Nowhere-In-The-Registry.gguf", None)

    emitted = _emit(path, field="model_for_node", index=0)
    fields = _source(emitted, MODEL_PROBE).split("|")
    filename, url, size_gb, n_parts, n_urls, d_name, d_url, d_size = fields

    # The WANTED name is echoed back so the caller can name the file in its
    # error; everything else is empty rather than absent.
    assert filename == "Nowhere-In-The-Registry.gguf"
    assert (url, size_gb) == ("", "")
    assert (n_parts, n_urls) == ("0", "0")
    assert (d_name, d_url, d_size) == ("", "", "")


# --- model_for_node: the integrity exports ---------------------------------
# download-model.sh verifies each part against its declared digest before any
# file is pushed to a node, so these three names are what carries the registry's
# integrity declaration across the Python/shell boundary.

DIGEST_PROBE = (
    'printf "%s|%s|%s|%s" '
    '"$MODEL_SHA256" "${#MODEL_PART_SHA256S[@]}" '
    '"${MODEL_PART_SHA256S[*]:-}" "$MODEL_REVISION"'
)

# 64 lowercase hex characters each, which is the only form the schema accepts
# and the exact form `shasum -a 256` prints.
PART_DIGESTS = ["a" * 64, "b" * 64, "c" * 64]
FILE_DIGEST = "d" * 64
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _digest_fields(path) -> tuple[str, str, str, str]:
    """`model_for_node 0` on `path`, read back through bash."""
    emitted = _emit(path, field="model_for_node", index=0)
    single, n_parts, joined, revision = _source(emitted, DIGEST_PROBE).split("|")
    return single, n_parts, joined, revision


# `${VAR+defined}` distinguishes "set to the empty string" from "unset", which
# is the whole contract: download-model.sh evals these under `set -u`.
DRAFT_DIGEST_PROBE = 'printf "%s|%s|%s" "${MODEL_DRAFT_SHA256+defined}" "$MODEL_DRAFT_SHA256" "$MODEL_DRAFT_FILENAME"'


def _network_with_draft_probe_node(tmp_path, draft_entry: dict):
    """A probe node that opts into speculative decoding with `draft_entry`'s file."""
    path = _network_with_probe_node(
        tmp_path,
        "Probe-Target.gguf",
        {"name": "Probe-Target", "filename": "Probe-Target.gguf", "size_gb": 4.0,
         "quantization": "Q4_K_M", "huggingface_repo": "acme/probe-gguf",
         "huggingface_file": "Probe-Target.gguf"},
    )
    yaml = YAML()
    data = yaml.load(path.read_text())
    data["model_registry"]["models"].append(draft_entry)
    path = _write_network(tmp_path, data)
    (tmp_path / "probe-node.yaml").write_text(
        "node:\n"
        "  id: probe\n"
        '  role: "edge_executor"\n'
        "model:\n"
        '  filename: "Probe-Target.gguf"\n'
        "  draft:\n"
        "    enabled: true\n"
        f'    filename: "{draft_entry["filename"]}"\n'
    )
    return path


def test_draft_model_digest_reaches_the_shell(tmp_path) -> None:
    """A draft rides along in Step 3's SCP, so it carries its own whole-file
    digest for Step 2b to verify — a draft is always fetched as one file."""
    path = _network_with_draft_probe_node(
        tmp_path,
        {"name": "Probe-Draft", "filename": "Probe-Draft.gguf", "size_gb": 0.5,
         "quantization": "Q4_K_M", "huggingface_repo": "acme/probe-gguf",
         "huggingface_file": "Probe-Draft.gguf", "sha256": FILE_DIGEST},
    )
    emitted = _emit(path, field="model_for_node", index=0)
    defined, digest, name = _source(emitted, DRAFT_DIGEST_PROBE).split("|")
    assert (defined, digest, name) == ("defined", FILE_DIGEST, "Probe-Draft.gguf")


def test_draft_digest_is_defined_and_empty_without_a_draft(tmp_path) -> None:
    path = _network_with_probe_node(tmp_path, "Nowhere-In-The-Registry.gguf", None)
    emitted = _emit(path, field="model_for_node", index=0)
    assert _source(emitted, DRAFT_DIGEST_PROBE).split("|") == ["defined", "", ""]


def test_split_model_digests_reach_the_shell_in_part_order(tmp_path) -> None:
    """MODEL_PART_SHA256S is indexed against MODEL_PARTS positionally.

    A digest list that arrived reordered — or collapsed by bash's word
    splitting, the failure mode the array shape exists to avoid — would verify
    each part against another part's digest and report a mismatch on files that
    are fine.
    """
    path = _network_with_probe_node(
        tmp_path,
        "Probe-Split-Q4_K_M-00001-of-00003.gguf",
        {
            "name": "Probe-Split",
            "filename": "Probe-Split-Q4_K_M-00001-of-00003.gguf",
            "size_gb": 30.0,
            "quantization": "Q4_K_M",
            "huggingface_repo": "acme/probe-gguf",
            "huggingface_file": "Q4_K_M/Probe-Split-Q4_K_M-00001-of-00003.gguf",
            "sha256_parts": list(PART_DIGESTS),
            "revision": COMMIT,
        },
    )

    single, n_parts, joined, revision = _digest_fields(path)
    assert single == "", "a split model declares no whole-file digest"
    assert n_parts == "3"
    assert joined.split() == PART_DIGESTS
    assert revision == COMMIT

    # And the pairing itself, element by element — same contract as
    # MODEL_PART_URLS.
    pairs = _source(
        _emit(path, field="model_for_node", index=0),
        'for i in "${!MODEL_PARTS[@]}"; do '
        'printf "%s %s\\n" "${MODEL_PARTS[$i]}" "${MODEL_PART_SHA256S[$i]}"; done',
    ).strip().splitlines()
    assert len(pairs) == 3
    for (part, digest), expected in zip(
        (line.split() for line in pairs), PART_DIGESTS, strict=True
    ):
        assert digest == expected, part


def test_single_file_model_digest_and_the_default_ref(tmp_path) -> None:
    """One whole-file digest, an empty part list, and `main` when unpinned.

    `main` is not a placeholder for "unknown": it is the ref the URLs actually
    resolve, and naming it is what tells an operator the bytes behind a URL can
    change between two provisions of the same fleet.
    """
    path = _network_with_probe_node(
        tmp_path,
        "Probe-Single-Q4_K_M.gguf",
        {
            "name": "Probe-Single",
            "filename": "Probe-Single-Q4_K_M.gguf",
            "size_gb": 4.0,
            "quantization": "Q4_K_M",
            "huggingface_repo": "acme/probe-gguf",
            "huggingface_file": "Probe-Single-Q4_K_M.gguf",
            "sha256": FILE_DIGEST,
        },
    )

    single, n_parts, joined, revision = _digest_fields(path)
    assert single == FILE_DIGEST
    assert (n_parts, joined) == ("0", "")
    assert revision == "main"


def test_an_entry_declaring_no_digest_still_defines_every_name(tmp_path) -> None:
    """The shipped registry declares none, and that has to keep working.

    download-model.sh evals this output under `set -u` in a loop, so an
    undefined name aborts the run and a name left over from the previous node
    verifies one model's parts against another model's digests.
    """
    path = _network_with_probe_node(
        tmp_path,
        "Probe-Plain-Q4_K_M.gguf",
        {
            "name": "Probe-Plain",
            "filename": "Probe-Plain-Q4_K_M.gguf",
            "size_gb": 4.0,
            "quantization": "Q4_K_M",
            "huggingface_repo": "acme/probe-gguf",
            "huggingface_file": "Probe-Plain-Q4_K_M.gguf",
        },
    )

    single, n_parts, joined, revision = _digest_fields(path)
    assert (single, n_parts, joined) == ("", "0", "")
    assert revision == "main"


def test_a_model_absent_from_the_registry_defines_the_digest_names_too(tmp_path) -> None:
    """The unresolvable-model path defines them empty, like every other name.

    It is reached before download-model.sh's own "could not resolve model"
    diagnostic, so a name missing here aborts the script with an unbound
    variable instead.
    """
    path = _network_with_probe_node(tmp_path, "Nowhere-In-The-Registry.gguf", None)

    single, n_parts, joined, revision = _digest_fields(path)
    assert (single, n_parts, joined, revision) == ("", "0", "", "")


# --- NODE_MCP_AUTH ---------------------------------------------------------


def _network_with_mcp_probe_nodes(tmp_path):
    """Point the first two enabled targets at probe configs, one MCP-keyed.

    The templates are written here rather than borrowed from configs/ so this
    pins parse-config.py's behaviour rather than whichever shipped file happens
    to carry an `api_key:` line this week.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    targets = [t for t in data["ssh"]["targets"] if t.get("enabled", True)]
    assert len(targets) >= 2, "the example config is expected to enable at least two targets"
    targets[0]["config_file"] = "probe-mcp-auth.yaml"
    targets[1]["config_file"] = "probe-mcp-open.yaml"

    (tmp_path / "probe-mcp-auth.yaml").write_text(
        "node:\n"
        "  id: probe-auth\n"
        '  role: "edge_executor"\n'
        "mcp:\n"
        "  server:\n"
        "    enabled: true\n"
        "    port: 9000\n"
        "    api_key: ${mcp_api_key}\n"
    )
    (tmp_path / "probe-mcp-open.yaml").write_text(
        "node:\n"
        "  id: probe-open\n"
        '  role: "edge_executor"\n'
        "mcp:\n"
        "  server:\n"
        "    enabled: true\n"
        "    port: 9000\n"
    )
    return _write_network(tmp_path, data)


def test_node_mcp_auth_follows_the_template_not_the_node_id(tmp_path) -> None:
    """NODE_MCP_AUTH is true exactly for templates carrying ${mcp_api_key}.

    test-all.sh section 4 indexes this array to decide whether to assert a 401
    on POST /mcp. Get it wrong in either direction and the suite either sends a
    bearer token to a node that enforces none (reported as a broken key) or
    declares an unauthenticated MCP port healthy.
    """
    path = _network_with_mcp_probe_nodes(tmp_path)
    emitted = _emit(path)
    assert _element(emitted, "NODE_MCP_AUTH", 0) == "true"
    assert _element(emitted, "NODE_MCP_AUTH", 1) == "false"


def test_node_mcp_auth_is_independent_of_node_auth(tmp_path) -> None:
    """The MCP port and the llama port are separate doors with separate keys.

    A template may key one and not the other, so the two arrays are computed
    from different questions. Deriving one from the other is how a node with
    llama auth on and MCP auth off would get probed for a 401 that never comes.
    """
    path = _network_with_mcp_probe_nodes(tmp_path)
    emitted = _emit(path)
    # Neither probe template mentions the llama key, so NODE_AUTH is false on
    # both while NODE_MCP_AUTH differs between them.
    assert _element(emitted, "NODE_AUTH", 0) == "false"
    assert _element(emitted, "NODE_AUTH", 1) == "false"
    assert _element(emitted, "NODE_MCP_AUTH", 0) == "true"


def test_node_mcp_auth_is_false_for_a_target_with_no_config_file(tmp_path) -> None:
    """An absent `config_file` asks no template anything, so the answer is no.

    It must still be an element: bash drops an unquoted empty word, and section
    4 indexes NODE_MCP_AUTH against NODE_HOSTS.
    """
    yaml = YAML()
    data = yaml.load(EXAMPLE_NETWORK_CONFIG.read_text())
    data["ssh"]["targets"][0].pop("config_file", None)
    path = _write_network(tmp_path, data)

    emitted = _emit(path)
    assert _element(emitted, "NODE_MCP_AUTH", 0) == "false"
    n = len(data["ssh"]["targets"])
    assert _array_lengths(emitted)["NODE_MCP_AUTH"] == n


def test_the_groups_docstring_names_every_array_that_branch_emits() -> None:
    """The module docstring is the shell contract, so it has to be complete.

    GROUP_HEAD_IDS was emitted and used by test-all.sh while the docstring
    listed only the other five, which leaves a reader auditing the contract
    from the top of the file unable to see an array the suite depends on.
    """
    emitted = _emit(EXAMPLE_NETWORK_CONFIG, field="groups")
    names = {m.group(1) for m in re.finditer(r"^(GROUP_[A-Z_]+)=", emitted, re.M)}
    assert names, "the groups branch emitted nothing"

    docstring = PARSE_CONFIG.read_text().split('"""')[1]
    groups_entry = docstring.split("groups ")[1].split("k3d ")[0]
    missing = sorted(n for n in names if n not in groups_entry)
    assert not missing, f"undocumented shell contract: {missing}"
