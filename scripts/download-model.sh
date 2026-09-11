#!/usr/bin/env bash
# ==========================================================================
# Download model GGUF files to the orchestrator (Mac), then SCP each model
# to the nodes that need it. Each SSH target's per-node config declares the
# model.filename it expects; this script reads that, looks up the model in
# model_registry, and downloads/uploads accordingly.
#
# Different node types may need different models (e.g. shard heads run
# Qwen3.5-35B-A3B, the planner runs Qwen3.6-27B). Each unique model is downloaded
# from HuggingFace once locally, then SCP'd only to the nodes that need it.
#
# Usage: ./scripts/download-model.sh
# ==========================================================================
set -euo pipefail

# ok/fail/info, ssh_cmd/scp_cmd, ssh_facts, file_size, load_network_env.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

PROJECT_DIR="$AUREAX_ROOT"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# --- Read node config (NODE_HOSTS, NODE_USERS, NODE_KEY_FILES, NODE_COUNT) ---
if ! load_network_env all; then
    fail "config" "parse-config.py all failed — see the error above"
    exit 1
fi
# `ssh.targets` has no minimum length and every target may be `enabled: false`,
# so NODE_COUNT can be 0. Caught here because BSD `seq 0 -1` prints "0" and
# "-1": each loop below would run once against an empty NODE_* array (unbound
# variable under `set -u`) and once with subscript -1 (bash 3.2 rejects it).
if [ "${NODE_COUNT:-0}" -eq 0 ]; then
    fail "config" "no enabled SSH targets in configs/aureax-network.yaml"
    exit 1
fi

LOCAL_MODEL_DIR="${PROJECT_DIR}/models"
REMOTE_MODEL_DIR="/opt/aureax/models"

# Staging directory for the current parallel fetch, so an interrupt can clean
# it up. Without this a Ctrl-C mid-download leaves a multi-GB
# models/.pfetch.<pid> directory behind: invisible to `models/*.gguf` patterns
# and to the completeness check below, so it survives every later run and its
# gigabytes are never accounted for by anything.
PFETCH_TMPDIR=""
_cleanup_pfetch() {
    [ -n "$PFETCH_TMPDIR" ] && rm -rf "$PFETCH_TMPDIR"
    PFETCH_TMPDIR=""
}
# Split, because a handler that only cleans up does not stop the script: bash
# runs the trap and RESUMES. The signal handlers must therefore exit
# explicitly — otherwise a Ctrl-C mid-fetch deletes the staging dir, the chunk
# verification prints one "[FAIL] chunk N got 0 bytes" per chunk against the
# directory just removed, and the fallback starts a fresh multi-GB
# single-stream download of the file the operator has just cancelled.
# _cleanup_pfetch nulls PFETCH_TMPDIR, so the EXIT trap running afterwards is a
# no-op. 130/143 are the conventional 128+SIGINT / 128+SIGTERM statuses.
trap _cleanup_pfetch EXIT
trap '_cleanup_pfetch; exit 130' INT
trap '_cleanup_pfetch; exit 143' TERM

# How many concurrent range requests to use. 8 measured well; the point is to
# have several connections, not to saturate a number.
DOWNLOAD_CHUNKS="${AUREAX_DOWNLOAD_CHUNKS:-8}"
# Validated here because the value is used only in arithmetic: a non-numeric
# one makes the `-le 1` test below read false and then divides by zero in
# `remain / DOWNLOAD_CHUNKS`, aborting mid-provision with a bash arithmetic
# error instead of a diagnosis. 0 and 1 are fine — both fall through to the
# single-stream branch.
case "$DOWNLOAD_CHUNKS" in
    ''|*[!0-9]*)
        fail "AUREAX_DOWNLOAD_CHUNKS='${DOWNLOAD_CHUNKS}'" "must be a non-negative integer"
        exit 2
        ;;
esac
# Below this, one stream is simpler and the setup cost of ranges is not repaid.
DOWNLOAD_PARALLEL_MIN=$((256 * 1024 * 1024))

gb() {
    # Bytes as DECIMAL GB (10^9): the unit model_registry's size_gb uses and the
    # unit the completeness check below is built on. Dividing by 1073741824
    # (GiB) under a "GB" label makes a complete 22,016,023,168-byte model print
    # as "22.02 GB" upstream and "20.5 GB" locally in the same run — a 7% gap
    # that reads as a truncated download. $2 is the decimal precision (1).
    awk -v b="$1" -v p="${2:-1}" 'BEGIN{printf "%." p "f", b/1e9}'
}

sha256_of() {
    # The sha256 of $1 as lowercase hex, or an empty line when it cannot be
    # taken (no digest tool, unreadable file).
    #
    # `shasum -a 256` first because this script runs on the orchestrator Mac,
    # where it ships with the system; `sha256sum` is the coreutils name, for a
    # Linux orchestrator. Both print "<digest>  <path>", so only the first
    # field is kept — the path would otherwise be compared as part of the
    # digest.
    #
    # Always exits 0 with exactly one line, like remote_size above: the script
    # runs under `set -euo pipefail`, where a non-zero status inside a command
    # substitution aborts the assignment that reads it, so a missing file would
    # kill the run instead of being reported as unverifiable.
    local out=""
    if command -v shasum >/dev/null 2>&1; then
        out="$({ shasum -a 256 "$1" 2>/dev/null || true; } | awk '{print $1}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        out="$({ sha256sum "$1" 2>/dev/null || true; } | awk '{print $1}')"
    fi
    printf '%s\n' "$out"
}

verify_model_parts() {
    # Verify every part of one model against the digests its registry entry
    # declares:
    #
    #   verify_model_parts <model name> <part names> <digests>
    #
    # Both lists are space-separated and paired BY POSITION, the same shape
    # MODEL_PARTS / MODEL_PART_URLS already travel in. Digests are hex and part
    # names carry no spaces, so word-splitting them back is safe.
    #
    # WHY, when every part is already length-checked: a length is not an
    # identity. The 95% floor, the per-part Content-Length compare and the
    # assembled-size backstop all pass any file of the right size, and the
    # download URLs resolve a mutable ref unless the entry pins `revision` — so
    # the exact bytes behind a URL can change between two provisions. These
    # files are then pushed to every head and handed to llama.cpp's GGUF
    # deserializer, which is why this runs before the upload step rather than
    # after it.
    #
    # Declaring no digest stays legal: an operator with none to hand keeps the
    # size checks, which is what every shipped registry entry has always had.
    # It says so once per model rather than silently, because "verified" and
    # "the right length" are not the same claim.
    #
    # `i` is local for the same reason fetch_file's `p` is: it is the index of
    # Step 2's node loop, which is where both call sites are.
    local name="$1" rc=0 i want got
    local -a vparts vdigests
    read -r -a vparts <<< "$2"
    read -r -a vdigests <<< "$3"

    if [ "${#vdigests[@]}" -eq 0 ]; then
        warn "no digest declared for ${name}; size check only"
        return 0
    fi
    # The schema pairs these lists, so a mismatch here means the emitted text
    # and the registry disagree — verifying part N against part M's digest
    # would report a mismatch on a file that is fine.
    if [ "${#vdigests[@]}" -ne "${#vparts[@]}" ]; then
        fail "$name" "${#vdigests[@]} digest(s) declared for ${#vparts[@]} part(s) — the lists are paired by position"
        return 1
    fi

    for i in $(seq 0 $((${#vparts[@]} - 1))); do
        want="${vdigests[$i]}"
        got=$(sha256_of "${LOCAL_MODEL_DIR}/${vparts[$i]}")
        if [ -z "$got" ]; then
            fail "${vparts[$i]}" "could not compute sha256 — no shasum or sha256sum on PATH, or the file is unreadable"
            rc=1
            continue
        fi
        if [ "$got" != "$want" ]; then
            # Deleted, because nothing else would ever re-fetch it: the file is
            # the right length, so every size check above reports the model
            # complete on this run and on every later one.
            rm -f "${LOCAL_MODEL_DIR}/${vparts[$i]}"
            fail "${vparts[$i]}" "sha256 mismatch — expected ${want}, got ${got}; deleted, re-run to re-fetch"
            rc=1
            continue
        fi
        ok "${vparts[$i]}: sha256 verified"
    done
    return $rc
}

remote_size() {
    # Content-Length of the final URL after redirects, or 0. Header case varies
    # between HTTP/1.1 and HTTP/2, so lowercase before matching, and take the
    # LAST value because -I follows redirects and prints a block per hop.
    #
    # Always exits 0 with exactly one number, so a caller can use it in
    # arithmetic under `set -o pipefail` and needs no `|| echo 0` of its own.
    # curl's status is swallowed INSIDE the pipeline for that reason: leaving it
    # to the caller makes the pipeline non-zero after awk has already printed
    # "0", so `total=$(remote_size "$url" || echo 0)` yields the two lines
    # "0\n0" — which fails `[ -le 0 ]` with "integer expression expected" and
    # skips the very fallback that guard exists to reach.
    { curl -sIL -m 60 "$1" 2>/dev/null || true; } \
        | tr -d '\r' | tr 'A-Z' 'a-z' \
        | awk '/^content-length:/ {v=$2} END {print v+0}'
}

single_stream_fetch() {
    # One resumable stream: $1 url, $2 dest, $3 the upstream length the caller
    # already read (0 when it is unknown).
    #
    # `--fail` matters here as much as on the parallel branch. Without it curl
    # writes a 403/429/5xx BODY to the output file and exits 0; with `-C -` that
    # body is appended to the partial GGUF, so the next run resumes from an
    # inflated offset and the garbage is embedded mid-file forever. HuggingFace
    # throttling — the reason parallel ranges exist at all — is exactly how a 429
    # arrives here.
    #
    # The length check is the other half: curl can also exit 0 on a body cut
    # short of a known Content-Length (a connection closed on a chunked
    # response), and an unverified short part flows straight into the SCP step,
    # where llama.cpp rejects it minutes into a cold load. The caller's 95%
    # floor is meant to be the offline fallback, not the only gate.
    local got
    curl -L --fail --retry 5 --retry-delay 3 --progress-bar -C - -o "$2" "$1" || return $?
    [ "${3:-0}" -gt 0 ] || return 0
    got=$(file_size "$2")
    if [ "$got" -ne "$3" ]; then
        fail "$(basename "$2")" "wrote $got bytes, expected $3"
        return 1
    fi
    return 0
}

fetch_file() {
    # Download $1 to $2, resuming, using parallel byte ranges when worthwhile.
    #
    # WHY: HuggingFace throttles a single connection hard and progressively —
    # measured on one test bench at 55 MB/s falling to 2 MB/s on one stream, while
    # 8 concurrent range requests to the SAME file sustained ~53 MB/s and
    # peaked at 86. On the 22.02 GB group model that is the difference between
    # about seven minutes at that sustained rate (four at the 86 MB/s peak) and
    # over an hour, and it is paid on every fresh provision — twice now, since
    # each group head needs its own copy.
    #
    # Falls back to a plain resumable curl whenever ranges cannot be used, so
    # a server without Content-Length or without range support still works.
    local url="$1" dest="$2"
    # `p` is local too: it is the chunk-pid loop variable below, and it is also
    # the PART-NAME variable in all three callers' loops — one of which calls
    # this from inside that loop. A global `p` there silently downloads or
    # pushes a different part than the caller named.
    local total have remain chunk i off end want got p
    local tmpdir pids rc

    # Both helpers already exit 0 with a single number of their own accord
    # (remote_size swallows curl's status inside its pipeline; file_size ends in
    # `|| echo 0`), so no `|| echo 0` here — that construct emits TWO lines
    # under pipefail and breaks every arithmetic test below.
    total=$(remote_size "$url")
    have=$(file_size "$dest")

    # Nothing to compare against when upstream sends no Content-Length, so this
    # is the one branch that cannot verify what it wrote.
    if [ "${total:-0}" -le 0 ]; then
        info "  (no content-length — single stream)"
        single_stream_fetch "$url" "$dest" 0
        return $?
    fi
    if [ "$have" -eq "$total" ]; then
        ok "  already complete ($have bytes)"
        return 0
    fi
    # Longer than upstream is corruption, not completeness: `-C -` against a
    # file that changed upstream appends the new body to the old one, and an
    # error body appended by a throttled resume does the same. Treating that as
    # complete makes every later run a no-op — the caller re-detects the size
    # mismatch, calls back in here, is told "already complete", and SCPs the
    # corrupt part to a node. Truncating is what makes the repair automatic.
    if [ "$have" -gt "$total" ]; then
        info "  ${have} bytes locally vs ${total} upstream — corrupt, re-fetching from scratch"
        : > "$dest"
        have=0
    fi

    remain=$((total - have))
    if [ "$remain" -lt "$DOWNLOAD_PARALLEL_MIN" ] || [ "$DOWNLOAD_CHUNKS" -le 1 ]; then
        single_stream_fetch "$url" "$dest" "$total"
        return $?
    fi

    tmpdir="$(dirname "$dest")/.pfetch.$$"
    mkdir -p "$tmpdir"
    PFETCH_TMPDIR="$tmpdir"   # so the EXIT/INT trap can reclaim it
    chunk=$((remain / DOWNLOAD_CHUNKS))
    info "  $(gb "$remain") GB remaining via ${DOWNLOAD_CHUNKS} parallel ranges"

    pids=""
    for i in $(seq 0 $((DOWNLOAD_CHUNKS - 1))); do
        off=$((have + i * chunk))
        if [ "$i" -eq $((DOWNLOAD_CHUNKS - 1)) ]; then end=$((total - 1)); else end=$((off + chunk - 1)); fi
        ( curl -sL --fail --retry 5 --retry-delay 3 -r "${off}-${end}" -o "$tmpdir/c$i" "$url" ) &
        pids="$pids $!"
    done
    rc=0
    for p in $pids; do wait "$p" || rc=1; done

    # Verify every chunk is EXACTLY its requested length before touching the
    # destination. A short chunk concatenated in order corrupts the file
    # silently — the GGUF would load and emit garbage rather than fail, which
    # is far worse than a failed download.
    for i in $(seq 0 $((DOWNLOAD_CHUNKS - 1))); do
        off=$((have + i * chunk))
        if [ "$i" -eq $((DOWNLOAD_CHUNKS - 1)) ]; then end=$((total - 1)); else end=$((off + chunk - 1)); fi
        want=$((end - off + 1))
        got=$(file_size "$tmpdir/c$i" || echo 0)
        if [ "$got" -ne "$want" ]; then
            fail "chunk $i" "got $got bytes, wanted $want"
            rc=1
        fi
    done

    if [ "$rc" -ne 0 ]; then
        _cleanup_pfetch
        info "  parallel fetch incomplete — falling back to a single stream"
        single_stream_fetch "$url" "$dest" "$total"
        return $?
    fi

    for i in $(seq 0 $((DOWNLOAD_CHUNKS - 1))); do cat "$tmpdir/c$i" >> "$dest"; done
    _cleanup_pfetch

    got=$(file_size "$dest")
    if [ "$got" -ne "$total" ]; then
        fail "$(basename "$dest")" "assembled $got bytes, expected $total"
        return 1
    fi
    return 0
}

# --- Step 1: Resolve which model each node needs ---
echo "=== Aureax Model Download ==="
echo

declare -a NODE_MODEL_FILENAME NODE_MODEL_URL NODE_MODEL_SIZE
# A large GGUF is published split across several files. bash 3.2 (macOS) has no
# nested arrays, so each node's part list is one space-separated string per
# index — the same shape parse-config.py already uses for GROUP_WORKER_HOSTS.
# GGUF filenames contain no spaces, so word-splitting them back is safe.
declare -a NODE_MODEL_PARTS NODE_MODEL_PART_URLS
# What the registry declares about the BYTES: a whole-file digest, or one per
# part in the same order as NODE_MODEL_PARTS, plus the HuggingFace ref the URLs
# resolve. At most one of the two digest strings is ever non-empty, and both
# may be empty — a registry entry that declares no digest is size-checked only.
declare -a NODE_MODEL_SHA256 NODE_MODEL_PART_SHA256S NODE_MODEL_REVISION
# Speculative decoding's draft model, when a node opts in. Kept in its own set
# of arrays rather than appended to NODE_MODEL_PARTS: the completeness check
# below sums the parts against ONE registry size_gb, so folding a second,
# unrelated model into that list would make the total never match.
declare -a NODE_DRAFT_FILENAME NODE_DRAFT_URL NODE_DRAFT_SIZE NODE_DRAFT_SHA256
# Which indices have been resolved through parse-config.py, so a later node
# naming the SAME per-node config can copy the answer instead of paying for it
# again — see the cache lookup in the loop.
declare -a NODE_MODEL_RESOLVED
for i in $(seq 0 $((NODE_COUNT - 1))); do
    NODE_MODEL_RESOLVED[$i]=0
    # Shard workers never receive a GGUF: only the group's head loads the file
    # and streams tensors to its workers over RPC. Pushing the 22.02 GB model
    # to every member would be eight transfers where two are needed.
    if [ "${NODE_ROLES[$i]}" = "$SHARD_WORKER_ROLE" ]; then
        # Every array is indexed in lockstep below, and the script runs under
        # `set -u` — leaving any of them unset for this index would fail later
        # with an unbound-variable error rather than skipping cleanly.
        NODE_MODEL_FILENAME[$i]=""
        NODE_MODEL_URL[$i]=""
        NODE_MODEL_SIZE[$i]=""
        NODE_MODEL_PARTS[$i]=""
        NODE_MODEL_PART_URLS[$i]=""
        NODE_MODEL_SHA256[$i]=""
        NODE_MODEL_PART_SHA256S[$i]=""
        NODE_MODEL_REVISION[$i]=""
        NODE_DRAFT_FILENAME[$i]=""
        NODE_DRAFT_URL[$i]=""
        NODE_DRAFT_SIZE[$i]=""
        NODE_DRAFT_SHA256[$i]=""
        echo "  ${NODE_HOSTS[$i]} → (shard worker — receives tensors over RPC, no local model)"
        continue
    fi

    # A node's model is decided entirely by its per-node config (model.filename,
    # plus model.draft) resolved against one registry, so two targets naming the
    # same config file resolve identically — and targets that differ only in
    # placeholder-resolved values are allowed to share one template (see
    # ARCHITECTURE.md, "2. Configuration: per-node files, not one shared
    # template").
    # Each parse-config.py call is a fresh interpreter, a `uv` resolution and a
    # full parse of the network config, so reuse the first answer for that file.
    # bash 3.2 has no associative arrays; the fleet is small and this scans only
    # indices already resolved.
    #
    # A C-style loop, not `seq 0 $((i - 1))`: BSD seq infers a negative step, so
    # at i=0 that prints "0 -1" and bash 3.2 rejects the negative subscript with
    # "bad array subscript".
    CACHED=-1
    if [ -n "${NODE_CONFIG_FILES[$i]}" ]; then
        for ((j = 0; j < i; j++)); do
            if [ "${NODE_MODEL_RESOLVED[$j]}" = "1" ] &&
               [ "${NODE_CONFIG_FILES[$j]}" = "${NODE_CONFIG_FILES[$i]}" ]; then
                CACHED=$j
                break
            fi
        done
    fi

    if [ "$CACHED" -ge 0 ]; then
        NODE_MODEL_FILENAME[$i]="${NODE_MODEL_FILENAME[$CACHED]}"
        NODE_MODEL_URL[$i]="${NODE_MODEL_URL[$CACHED]}"
        NODE_MODEL_SIZE[$i]="${NODE_MODEL_SIZE[$CACHED]}"
        NODE_MODEL_PARTS[$i]="${NODE_MODEL_PARTS[$CACHED]}"
        NODE_MODEL_PART_URLS[$i]="${NODE_MODEL_PART_URLS[$CACHED]}"
        NODE_MODEL_SHA256[$i]="${NODE_MODEL_SHA256[$CACHED]}"
        NODE_MODEL_PART_SHA256S[$i]="${NODE_MODEL_PART_SHA256S[$CACHED]}"
        NODE_MODEL_REVISION[$i]="${NODE_MODEL_REVISION[$CACHED]}"
        NODE_DRAFT_FILENAME[$i]="${NODE_DRAFT_FILENAME[$CACHED]}"
        NODE_DRAFT_URL[$i]="${NODE_DRAFT_URL[$CACHED]}"
        NODE_DRAFT_SIZE[$i]="${NODE_DRAFT_SIZE[$CACHED]}"
        NODE_DRAFT_SHA256[$i]="${NODE_DRAFT_SHA256[$CACHED]}"
    else
        # Clear first: `eval` of a failed parse-config.py sets nothing, and the
        # MODEL_* names would still hold the PREVIOUS node's values — silently
        # assigning one node's model to another.
        MODEL_FILENAME=""; MODEL_URL=""; MODEL_SIZE_GB=""; MODEL_PARTS=(); MODEL_PART_URLS=()
        MODEL_SHA256=""; MODEL_PART_SHA256S=(); MODEL_REVISION=""
        MODEL_DRAFT_FILENAME=""; MODEL_DRAFT_URL=""; MODEL_DRAFT_SIZE_GB=""; MODEL_DRAFT_SHA256=""
        # load_network_env, not `eval "$(...)"`: written the latter way this
        # guard could only fire on a bash SYNTAX error in the emitted text,
        # because eval reports its own status and the command substitution's is
        # discarded — so a traceback from parse-config.py would reach the "could
        # not resolve model" branch below instead, naming the wrong cause.
        if ! load_network_env model_for_node "$i"; then
            fail "${NODE_HOSTS[$i]}" "parse-config.py model_for_node $i failed"
            exit 1
        fi
        NODE_MODEL_FILENAME[$i]="${MODEL_FILENAME:-}"
        NODE_MODEL_URL[$i]="${MODEL_URL:-}"
        NODE_MODEL_SIZE[$i]="${MODEL_SIZE_GB:-}"
        NODE_MODEL_PARTS[$i]="${MODEL_PARTS[*]:-}"
        NODE_MODEL_PART_URLS[$i]="${MODEL_PART_URLS[*]:-}"
        NODE_MODEL_SHA256[$i]="${MODEL_SHA256:-}"
        NODE_MODEL_PART_SHA256S[$i]="${MODEL_PART_SHA256S[*]:-}"
        NODE_MODEL_REVISION[$i]="${MODEL_REVISION:-}"
        NODE_DRAFT_FILENAME[$i]="${MODEL_DRAFT_FILENAME:-}"
        NODE_DRAFT_URL[$i]="${MODEL_DRAFT_URL:-}"
        NODE_DRAFT_SIZE[$i]="${MODEL_DRAFT_SIZE_GB:-}"
        NODE_DRAFT_SHA256[$i]="${MODEL_DRAFT_SHA256:-}"
    fi
    NODE_MODEL_RESOLVED[$i]=1
    if [ -z "${NODE_MODEL_FILENAME[$i]}" ]; then
        fail "${NODE_HOSTS[$i]}" "could not resolve model from per-node config (check configs/${NODE_CONFIG_FILES[$i]})"
        exit 1
    fi
    read -r -a _parts <<< "${NODE_MODEL_PARTS[$i]}"
    if [ "${#_parts[@]}" -gt 1 ]; then
        echo "  ${NODE_HOSTS[$i]} → ${NODE_MODEL_FILENAME[$i]} (${NODE_MODEL_SIZE[$i]} GB, ${#_parts[@]} parts)"
    else
        echo "  ${NODE_HOSTS[$i]} → ${NODE_MODEL_FILENAME[$i]} (${NODE_MODEL_SIZE[$i]} GB)"
    fi
    if [ -n "${NODE_DRAFT_FILENAME[$i]}" ]; then
        echo "  ${NODE_HOSTS[$i]} → ${NODE_DRAFT_FILENAME[$i]} (${NODE_DRAFT_SIZE[$i]} GB, speculative-decoding draft)"
    fi
done
echo

# --- Step 2: Download each unique model locally if not already present ---
# Use a space-delimited string for "seen" tracking — Mac default bash is 3.2,
# which doesn't support associative arrays (`declare -A`).
mkdir -p "$LOCAL_MODEL_DIR"

SEEN_MODELS=" "
for i in $(seq 0 $((NODE_COUNT - 1))); do
    f="${NODE_MODEL_FILENAME[$i]}"
    [ -z "$f" ] && continue   # shard worker — nothing to download
    case "$SEEN_MODELS" in
        *" $f "*) continue ;;
    esac
    SEEN_MODELS="${SEEN_MODELS}${f} "

    size_gb="${NODE_MODEL_SIZE[$i]}"
    read -r -a parts <<< "${NODE_MODEL_PARTS[$i]}"
    read -r -a part_urls <<< "${NODE_MODEL_PART_URLS[$i]}"

    # The digests to verify this model against, paired with `parts` by
    # position. A split GGUF declares one per part; a single-file one declares
    # a single whole-file digest, which is that same list with one element —
    # the registry refuses both at once, so the fallback cannot mask anything.
    # Empty means the entry declares none, and verify_model_parts says so.
    digests="${NODE_MODEL_PART_SHA256S[$i]}"
    if [ -z "$digests" ]; then digests="${NODE_MODEL_SHA256[$i]}"; fi

    # Checked HERE, before anything expands "${parts[@]}". This is reachable:
    # when a per-node config names a model.filename that model_registry does not
    # define, parse-config.py emits the wanted name as MODEL_FILENAME (so the
    # non-empty guard in Step 1 passes) alongside empty MODEL_PARTS/
    # MODEL_PART_URLS. On bash 3.2 — the Mac default, and the version this file
    # is written for — `"${parts[@]}"` on an empty array is an error under
    # `set -u`, so without this the completeness loop below dies with
    # "parts[@]: unbound variable" instead of printing the diagnostic.
    if [ "${#parts[@]}" -eq 0 ] || [ "${#part_urls[@]}" -eq 0 ]; then
        fail "${f}" "no download URL available — model_registry needs an entry for this filename with huggingface_repo + huggingface_file"
        exit 1
    fi

    # Completeness decides whether this model is skipped forever: an
    # interrupted 22 GB download leaves gigabytes on disk, so anything short of
    # a per-part length check treats it as complete on every later run and
    # pushes a truncated GGUF to the node, where llama-server fails to load it
    # minutes into a cold start.
    #
    # size_gb is DECIMAL GB (10^9), the unit model cards quote — multiplying by
    # 1073741824 (GiB) here would set the bar ~7% ABOVE the real file size and
    # re-download a complete model forever. 95% then allows for rounding.
    #
    # This floor is the OFFLINE FALLBACK ONLY. On its own it accepts a download
    # that stopped 1.1 GB short of the 22.02 GB group model (3.58 GB short of
    # the 71.7 GB rollback target): the part still exists, so `missing` is 0,
    # the skip fires on this and every later run, and Step 3 ships a truncated
    # GGUF that llama.cpp rejects minutes into a cold load. Raising it to 100%
    # is not a substitute — size_gb legitimately exceeds the real byte count
    # (22.02e9 declared vs 22,016,023,168 actual).
    expected_bytes=$(awk -v g="${size_gb:-0}" 'BEGIN{printf "%d", g * 1000000000 * 0.95}')
    [ "${expected_bytes:-0}" -lt 1000000 ] && expected_bytes=1000000

    # So each present part is compared against its real upstream length instead,
    # which remote_size already knows. `exact` drops to 0 for any part whose
    # Content-Length we could not read (offline, or a server that does not send
    # one) and only then does the 95% floor decide. remote_size always exits 0
    # with one number, so neither call can abort this loop under `set -e`.
    have_bytes=0
    missing=0
    short=0
    exact=1
    for pi in $(seq 0 $((${#parts[@]} - 1))); do
        p="${parts[$pi]}"
        if [ ! -f "${LOCAL_MODEL_DIR}/${p}" ]; then
            missing=$((missing + 1))
            continue
        fi
        psz=$(file_size "${LOCAL_MODEL_DIR}/${p}")
        have_bytes=$((have_bytes + psz))
        rs=$(remote_size "${part_urls[$pi]}")
        if [ "${rs:-0}" -le 0 ]; then
            exact=0
        elif [ "$psz" -ne "$rs" ]; then
            short=1
            info "  ${p}: ${psz} bytes locally, ${rs} upstream"
        fi
    done

    if [ "$missing" -eq 0 ] && [ "$short" -eq 0 ] &&
       { [ "$exact" -eq 1 ] || [ "$have_bytes" -ge "$expected_bytes" ]; }; then
        # A model that is skipped here is still pushed in Step 3, so the
        # digests are checked on this path too — a part that was corrupted
        # after it landed, or fetched before the entry declared a digest, is
        # otherwise never looked at again.
        verify_model_parts "$f" "${NODE_MODEL_PARTS[$i]}" "$digests" || exit 1
        ok "Local: ${f} (${#parts[@]} part(s), $(gb "$have_bytes") GB)"
        continue
    fi
    if [ "$missing" -gt 0 ] || [ "$have_bytes" -gt 0 ]; then
        info "Local copy incomplete (${missing} part(s) missing, ${short} short, ${have_bytes} bytes on disk) — fetching"
    fi

    # The ref is named because it decides which bytes arrive: `main` is
    # mutable, so two provisions of the same fleet can fetch different files
    # from the same URL. `revision` on the registry entry pins it.
    info "Downloading ${f} from HuggingFace ref ${NODE_MODEL_REVISION[$i]:-main} (~${size_gb} GB across ${#parts[@]} part(s), this takes a while)..."
    for pi in $(seq 0 $((${#parts[@]} - 1))); do
        p="${parts[$pi]}"
        pdest="${LOCAL_MODEL_DIR}/${p}"
        # Resumes rather than restarting a multi-GB part, and uses parallel
        # byte ranges to get around HuggingFace's per-connection throttle.
        info "  [$((pi + 1))/${#parts[@]}] ${p}"
        fetch_file "${part_urls[$pi]}" "$pdest"
    done

    # Re-verify the whole set as a backstop. fetch_file already compares each
    # part it wrote against its upstream Content-Length; this sum is what is
    # left when upstream sent no length at all, and it is the last gate before
    # Step 3 SCPs these files to a node.
    have_bytes=0
    for p in "${parts[@]}"; do
        psz=$(file_size "${LOCAL_MODEL_DIR}/${p}")
        have_bytes=$((have_bytes + psz))
    done
    if [ "$have_bytes" -lt "$expected_bytes" ]; then
        fail "${f}" "downloaded ${have_bytes} bytes, expected >= ${expected_bytes} — refusing to push a truncated model"
        exit 1
    fi
    # Last gate before Step 3 SCPs these files to a node: a part whose bytes
    # are not the declared ones is deleted and the run stops here, so nothing
    # unverified reaches a head.
    verify_model_parts "$f" "${NODE_MODEL_PARTS[$i]}" "$digests" || exit 1
    ok "Downloaded ${f} (${#parts[@]} part(s))"
done

# --- Step 2b: the same, for any speculative-decoding draft model ---
# Always single-file (a sub-1 GB GGUF is never published split), so this skips
# the multi-part machinery above and checks one size — and one digest, when
# the entry declares `sha256`, on the same two paths as the target model: a
# draft rides along in Step 3's SCP and is handed to the same deserializer.
SEEN_DRAFTS=" "
for i in $(seq 0 $((NODE_COUNT - 1))); do
    df="${NODE_DRAFT_FILENAME[$i]}"
    [ -z "$df" ] && continue
    case "$SEEN_DRAFTS" in
        *" $df "*) continue ;;
    esac
    SEEN_DRAFTS="${SEEN_DRAFTS}${df} "

    # Same rule as the target model above: the exact upstream length decides,
    # and the 95% floor is only what is left when it cannot be read.
    d_expected=$(awk -v g="${NODE_DRAFT_SIZE[$i]:-0}" 'BEGIN{printf "%d", g * 1000000000 * 0.95}')
    [ "${d_expected:-0}" -lt 1000000 ] && d_expected=1000000
    d_local="${LOCAL_MODEL_DIR}/${df}"
    d_have=$(file_size "$d_local")
    d_remote_len=0
    [ -n "${NODE_DRAFT_URL[$i]}" ] && d_remote_len=$(remote_size "${NODE_DRAFT_URL[$i]}")

    d_complete=0
    if [ "${d_remote_len:-0}" -gt 0 ]; then
        if [ "$d_have" -eq "$d_remote_len" ]; then d_complete=1; fi
    elif [ "$d_have" -ge "$d_expected" ]; then
        d_complete=1
    fi
    if [ "$d_complete" -eq 1 ]; then
        ok "Local: ${df} (draft, $(gb "$d_have" 2) GB)"
        verify_model_parts "$df" "$df" "${NODE_DRAFT_SHA256[$i]}" || exit 1
        continue
    fi

    if [ -z "${NODE_DRAFT_URL[$i]}" ]; then
        fail "${df}" "no download URL — model_registry needs huggingface_repo + huggingface_file"
        exit 1
    fi

    info "Downloading draft ${df} (~${NODE_DRAFT_SIZE[$i]} GB)..."
    fetch_file "${NODE_DRAFT_URL[$i]}" "$d_local"

    d_have=$(file_size "$d_local")
    if [ "$d_have" -lt "$d_expected" ]; then
        fail "${df}" "downloaded ${d_have} bytes, expected >= ${d_expected} — refusing to push a truncated draft"
        exit 1
    fi
    ok "Downloaded ${df} (draft)"
    verify_model_parts "$df" "$df" "${NODE_DRAFT_SHA256[$i]}" || exit 1
done
echo

# --- Step 3: SCP each node's model to its remote model dir ---
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"
    f="${NODE_MODEL_FILENAME[$i]}"

    # Skipped in Step 1 — a shard worker holds its share of the weights in RAM,
    # shipped from the head at load time, never on local disk.
    if [ -z "$f" ]; then
        ok "${HOST}: shard worker — no model push needed"
        continue
    fi

    read -r -a parts <<< "${NODE_MODEL_PARTS[$i]}"
    df="${NODE_DRAFT_FILENAME[$i]}"
    info "Pushing ${f} to ${HOST} (${#parts[@]} part(s))..."

    # Ensure the remote directory exists and belongs to the login user.
    #
    # `id -un`/`id -gn` are evaluated ON THE NODE (the \$( ) escape, the same
    # trick ssh_facts uses), so nothing from the config is interpolated into a
    # remote sudo command line — tests/test_parse_config.py pins that a target's
    # `user` reaches the shell verbatim, metacharacters included. It also uses
    # the real primary group instead of assuming it is named after the user.
    #
    # Kept as its own ssh_cmd rather than folded into the batch below: ssh_facts
    # wraps each snippet in \$( ), which discards its exit status, so a failed
    # chown would stop aborting the run.
    ssh_cmd "$HOST" "$USER" "$KEY" "sudo mkdir -p ${REMOTE_MODEL_DIR} && sudo chown \$(id -un):\$(id -gn) ${REMOTE_MODEL_DIR}"

    # Every remote size in ONE round trip. There is no connection multiplexing
    # here, so asking per part costs 1 + n_parts + 1 full handshakes per node at
    # ConnectTimeout=10 — five per node once the 3-part rollback model is the
    # target. Read with sed rather than eval: the text is whatever the node
    # printed, and eval would run it here — ssh_facts %q-quotes on the NODE,
    # which cannot constrain what a compromised node chooses to print.
    #
    # The part names are single-quoted because they come from the config's
    # `model.filename` / `huggingface_file` — typically pasted from a
    # third-party model page — and ssh_facts embeds each snippet inside `$( )`
    # on the remote command line, so an unquoted `$(...)`, backtick or `;`
    # would run on every node as the login user. The schema checks only the
    # split-part numbering, never the character set. (The scp destinations
    # below are a different case and stay unquoted — see there.)
    SIZE_ASKS=()
    for pi in $(seq 0 $((${#parts[@]} - 1))); do
        SIZE_ASKS+=("SZ_${pi}=stat -c%s '${REMOTE_MODEL_DIR}/${parts[$pi]}' 2>/dev/null || echo 0")
    done
    if [ -n "$df" ]; then
        SIZE_ASKS+=("SZ_DRAFT=stat -c%s '${REMOTE_MODEL_DIR}/${df}' 2>/dev/null || echo 0")
    fi
    REMOTE_SIZES=""
    if [ "${#SIZE_ASKS[@]}" -gt 0 ]; then
        REMOTE_SIZES="$(ssh_facts "$HOST" "$USER" "$KEY" "${SIZE_ASKS[@]}")" || {
            fail "$HOST" "could not read remote model sizes over SSH"
            exit 1
        }
    fi

    # Per part: skip only when the remote size matches the local byte-for-byte.
    # Every part has to land — llama.cpp is handed part 1 and opens the rest by
    # name, so one missing part is a load failure minutes into a cold start.
    for pi in $(seq 0 $((${#parts[@]} - 1))); do
        p="${parts[$pi]}"
        p_local="${LOCAL_MODEL_DIR}/${p}"
        local_size=$(file_size "$p_local")
        # `remote_bytes`, not `remote_size` — that name is the URL-length
        # function defined at the top of this file, and one identifier meaning
        # two things depending on the sigil is a trap for the next reader.
        #
        # Digits only: ssh_facts emits the value %q-quoted, so an empty answer
        # arrives as the two characters '' and the arithmetic tests below would
        # abort with "integer expression expected" rather than re-uploading.
        remote_bytes=$(printf '%s\n' "$REMOTE_SIZES" | sed -n "s/^SZ_${pi}=//p" | tr -cd '0-9')

        if [ "${remote_bytes:-0}" = "$local_size" ] && [ "${remote_bytes:-0}" -gt 1000000 ]; then
            ok "${HOST}: ${p} already present"
            continue
        fi

        info "${HOST}: uploading ${p} ($(gb "$local_size") GB)..."
        # Unquoted, deliberately: scp has used the SFTP protocol since OpenSSH
        # 9.0, so the remote operand is sent verbatim and never reaches a
        # remote shell — quoting it would create a file whose NAME contains the
        # quotes. Only the ssh snippet above needs them.
        scp_cmd "$KEY" "$p_local" "${USER}@${HOST}:${REMOTE_MODEL_DIR}/${p}"
    done
    ok "${HOST}: ${f} uploaded (${#parts[@]} part(s))"

    # The draft model rides along to the same directory. llama-server resolves
    # it from model.local_path exactly like the target, so a missing draft on a
    # node that has speculative decoding enabled is a startup failure, not a
    # silent fallback.
    if [ -n "$df" ]; then
        d_local="${LOCAL_MODEL_DIR}/${df}"
        d_size=$(file_size "$d_local")
        d_remote=$(printf '%s\n' "$REMOTE_SIZES" | sed -n 's/^SZ_DRAFT=//p' | tr -cd '0-9')
        if [ "${d_remote:-0}" = "$d_size" ] && [ "${d_remote:-0}" -gt 1000000 ]; then
            ok "${HOST}: ${df} (draft) already present"
        else
            info "${HOST}: uploading draft ${df} ($(gb "$d_size" 2) GB)..."
            scp_cmd "$KEY" "$d_local" "${USER}@${HOST}:${REMOTE_MODEL_DIR}/${df}"
            ok "${HOST}: ${df} (draft) uploaded"
        fi
    fi
done

echo
echo "=== Done ==="
