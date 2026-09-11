"""The sweep that keeps TLS from being *almost* everywhere.

TLS is mandatory fleet-wide (ARCHITECTURE.md, "TLS"): there is no plaintext
listener left on any node or on the orchestrator, and no configuration flag
that turns TLS off. Every other test in this suite checks one participant —
that the generator injects the ``tls`` block, that llama-server gets
``--ssl-cert-file``, that the kubelet probes with ``scheme: HTTPS``. None of
them can see the thing that actually goes wrong at this scale: **one leftover
``http://``**.

A leftover matters more than it looks:

* An ``http://`` string reaching a client is not a downgrade, it is a hang or a
  connection error — the server never speaks plaintext, so the dial waits on a
  TLS handshake that will not start, or resets. The symptom names a timeout,
  never a scheme.
* A URL printed for the operator (``aureax group list``, the Access tab's copy
  button, the dashboard banner) is copied by hand into curl, a client library
  or a browser, where the same failure surfaces with even less context.
* And an ``http://`` that *does* connect is worse than either: it means
  something in the fleet is still serving plaintext, which is the exact state
  this change removed.

So this module reads the tree the way an operator does — as text — and refuses
any ``http://`` in code, in a shipped script, in a manifest, in a config
template, or in the two documents an operator follows. The second sweep is the
mirror image: no client anywhere may *have* TLS and then decline to verify it.
A ``verify=False`` restores exactly the attacker position the fleet CA was
introduced to close, and it is one keyword, in one call, in a file nobody
re-reads.

**What counts as code.** Prose is exempt, deliberately and structurally — the
comments in this codebase explain *why* plaintext is gone, and several of them
have to spell ``http://`` to do it. Python is parsed with ``ast`` and only
string constants are examined, so comments are not merely skipped, they were
never in the tree; docstrings are dropped explicitly. Shell, YAML, Dockerfiles
and JS have no AST here, so comment lines and trailing comments are stripped
by rule. That asymmetry is the point: a scheme that reaches a socket is always
in a string constant or in a command, never in prose.

**The allowlist covers the dashboard loopback interface and its in-process smoke test.** ``PLAINTEXT_ALLOWED``
exists because a genuine exception is imaginable — a third-party URL that is
not ours to serve, or a loopback address that cannot be reached from the LAN at
all. The second kind now exists: ``aureax dashboard`` serves the same app on
two listeners in one process, HTTPS on the orchestrator's LAN address for the
fleet and plain HTTP on 127.0.0.1/::1 for the operator's own browser, so
``http://localhost:7001/`` is printed by the CLI, dialled by
``scripts/test-all.sh`` and documented in both operator docs. Loopback is not
routable: nothing that URL carries crosses the LAN, and every listener that
does is still TLS-only. Nothing else is exempt: a loopback or inert
``health_check.endpoint`` is converted to https rather than allowlisted, the
frozen 122B rollback head included. Any further entry is a hole in the fleet's
transport, so it carries a reason comment on the line above it.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from .conftest import PROJECT_ROOT

# --- The rule ----------------------------------------------------------------

# Repo-relative path -> the exact matched substrings allowed to stay plaintext.
#
# Add an entry only for a URL that genuinely cannot be TLS — put the reason on
# the line above it and say so in the change's summary, because every entry is
# a place the fleet talks in the clear.
#
# The dashboard's LOOPBACK listener, and nothing else. `aureax dashboard` binds
# two listeners in one process on one port: HTTPS on the orchestrator's LAN
# address, which is what every node ships telemetry and logs to, and plain HTTP
# on 127.0.0.1/::1 for a browser on the Mac itself ("the system that runs the
# dashboard is a trusted system"). Loopback is not routable, so this URL never
# reaches the LAN — and it is the one URL an operator can open without first
# trusting the fleet CA, which is exactly why it is printed and documented.
PLAINTEXT_ALLOWED: dict[str, set[str]] = {
    # In-process wheel smoke test of the trusted loopback HTTP interface; no network request.
    "scripts/check-installed-package.py": {"http://localhost:7001"},
    # The banner `aureax dashboard` prints, as the f-string renders here.
    "orchestrator/cli.py": {"http://localhost:{...}/"},
    # Section 3 dials the loopback listener: no --cacert, no trust store.
    "scripts/test-all.sh": {"http://localhost:7001"},
    # "open this from the orchestrator" in the two operator-facing documents.
    "README.md": {"http://localhost:7001/"},
    "ARCHITECTURE.md": {"http://localhost:7001/"},
}

# ``http://`` followed by something that can start a host. The bare token
# ``http://`` on its own — the way prose names the scheme — is not a URL and is
# not matched. The character class runs to the end of the URL rather than
# matching one character, so the failure message can quote the whole thing;
# quotes and backticks end it, which keeps the closing delimiter of a YAML or
# JS string out of the reported match.
PLAINTEXT_RE = re.compile(r"""http://[^\s"'`]+""")

REMEDY = (
    "TLS is mandatory fleet-wide — see ARCHITECTURE.md, 'TLS'. Use https:// (and "
    "verify against ~/.aureax/tls/ca.pem). Add an entry to PLAINTEXT_ALLOWED in "
    "tests/test_no_plaintext_urls.py only with a reason comment on the line above."
)

# Disabled verification, in every spelling this repo could plausibly grow. A
# fleet CA that is not verified is a fleet CA that authenticates nobody, so
# these are worth catching even in files the URL sweep does not read.
INSECURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "verify=False": re.compile(r"\bverify\s*=\s*False\b"),
    "check_hostname = False": re.compile(r"\bcheck_hostname\s*=\s*False\b"),
    "ssl.CERT_NONE": re.compile(r"\bCERT_NONE\b"),
    # curl's -k, in any bundle of short flags, on either the raw command or the
    # curl_tls wrapper in scripts/lib/common.sh. `--cacert` cannot match: the
    # character class does not cross the second hyphen.
    "curl -k": re.compile(r"\bcurl(?:_tls)?\b[^\n]*\s-[A-Za-z]*k\b"),
    "--insecure": re.compile(r"--insecure\b"),
}

# --- What gets read ----------------------------------------------------------

# Python is read through `ast`; everything else line by line. The bare ``*.py``
# is a non-recursive glob over the repo root, where the only Python file is
# ``build_support.py`` — a resource both Dockerfiles COPY into the edge image
# before installing, so a URL in it would reach a node unswept.
PYTHON_GLOBS = (
    "orchestrator/**/*.py",
    "client/**/*.py",
    "scripts/*.py",
    "*.py",
)

# ``configs/*.yaml`` picks up the operator-owned, gitignored files
# (aureax-network.yaml, aureax-gpu-edge.yaml) whenever they are present, which
# is the point: those are the ones that actually reach a node, and they are
# absent from a fresh clone, so this sweep is stricter on the operator's Mac
# than in CI. docker/scripts/start.sh and the Dockerfiles are in scope for the
# same reason — they ship, and a URL in either reaches a node's shell.
LINE_GLOBS = (
    "scripts/*.sh",
    "scripts/lib/*.sh",
    "docker/scripts/*.sh",
    "docker/Dockerfile*",
    "k8s/*.yaml",
    "configs/*.yaml",
    "orchestrator/dashboard/static/**/*.js",
    # index.html is the document the browser loads, and a `<script src>`,
    # `<form action>` or `<a href>` in it is as much a fetch as one in the JS.
    # The `#` comment token is safe here: `_strip_comment` needs whitespace in
    # front of the marker, so `href="#tab"` survives intact.
    "orchestrator/dashboard/static/**/*.html",
)

# The insecure-flag sweep covers everything that ships: it costs nothing to read
# a file for five regexes, and `verify=False` in a Dockerfile healthcheck would
# be just as fatal as one in a client. `tests` is deliberately out of scope —
# this module's own patterns and parametrize cases are written in the spellings
# it hunts for, so a sweep that read them would flag itself.
INSECURE_ROOTS = ("orchestrator", "client", "scripts", "k8s", "docker")
INSECURE_SUFFIXES = frozenset({".py", ".sh", ".yaml", ".yml", ".js"})

# The documents somebody follows by hand. Every URL in them is one that gets
# pasted into a terminal or a browser: the two operator guides, plus the two
# files a contributor reads before touching the repo.
DOC_FILES = ("README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "SECURITY.md")

# The subset of DOC_FILES that must contain at least one https URL, as a check
# that the file was not emptied out from under the sweep. CONTRIBUTING.md and
# SECURITY.md are prose with no URLs at all, so the same assert on them would
# fail on a correct tree.
DOC_FILES_WITH_URLS = ("README.md", "ARCHITECTURE.md")

# Stands in for an f-string's ``{...}`` so that f"http://{host}:8080" is seen as
# one URL rather than as the fragments "http://" and ":8080" — the fragment
# alone would not match, which is precisely how an interpolated plaintext URL
# would slip through an AST walk that only looked at Constants.
FSTRING_HOLE = "{...}"

# The node types ``ast.get_docstring`` accepts — i.e. the only places a string
# literal is prose rather than a value.
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _relative(path: Path) -> str:
    """Repo-relative path for a message; bare name for anything outside it.

    The self-tests at the bottom of this module run the same extraction over
    files written into ``tmp_path``, which is not under the repo — so this has
    to answer for them rather than raising and turning a rule check into an
    error about paths.
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _files(globs: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pattern in globs:
        found += [p for p in PROJECT_ROOT.glob(pattern) if p.is_file()]
    return sorted(set(found))


def _insecure_files() -> list[Path]:
    found: list[Path] = []
    for root in INSECURE_ROOTS:
        for path in (PROJECT_ROOT / root).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in INSECURE_SUFFIXES or path.name.startswith("Dockerfile"):
                found.append(path)
    return sorted(set(found))


# --- Reading Python: string constants only -----------------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    """ids of the Constant nodes that are docstrings, module/class/function."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS) and ast.get_docstring(node) is not None:
            ids.add(id(node.body[0].value))  # type: ignore[attr-defined]
    return ids


class _StringCollector(ast.NodeVisitor):
    """Every string literal that is code, with f-string holes filled in."""

    def __init__(self, docstrings: set[int]) -> None:
        self._docstrings = docstrings
        self.found: list[tuple[int, str]] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self._docstrings:
            self.found.append((node.lineno, node.value))

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append(FSTRING_HOLE)
                # Still descend, so a literal nested inside the interpolation
                # (or a format spec) is not exempt by being inside an f-string.
                self.visit(value)
        self.found.append((node.lineno, "".join(parts)))


def _python_strings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    collector = _StringCollector(_docstring_ids(tree))
    collector.visit(tree)
    return collector.found


# --- Reading everything else: lines, minus their prose -----------------------


def _comment_token(path: Path) -> str:
    return "//" if path.suffix == ".js" else "#"


def _strip_comment(line: str, token: str) -> str:
    """The code half of one line: '' for a comment line, trailing comment cut.

    A comment marker needs whitespace in front of it to end the code, so
    ``https://host`` keeps its ``//`` and a shell string containing ``#`` is
    only ever truncated (a false negative, never a false positive). For JS the
    continuation lines of a ``/* */`` block start with ``*``; they are prose
    too.
    """
    stripped = line.lstrip()
    if stripped.startswith(token):
        return ""
    if token == "//" and stripped.startswith(("*", "/*")):
        return ""
    return re.split(rf"\s{re.escape(token)}", line, maxsplit=1)[0]


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """(lineno, code text) for a non-Python file, comments removed."""
    token = _comment_token(path)
    lines = path.read_text().splitlines()
    return [(n, _strip_comment(line, token)) for n, line in enumerate(lines, 1)]


def _joined_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Fold backslash continuations into one logical line, at its first lineno.

    The insecure patterns match a command and its flags together, and a long
    ``curl`` invocation split across a continuation is the normal shape in this
    tree (``scripts/test-all.sh`` writes several). Without folding, a ``-k`` on
    the second physical line is a flag the sweep can never see.
    """
    out: list[tuple[int, str]] = []
    start: int | None = None
    parts: list[str] = []
    for lineno, text in lines:
        stripped = text.rstrip()
        if start is None:
            start = lineno
        if stripped.endswith("\\"):
            parts.append(stripped[:-1].strip())
            continue
        parts.append(stripped.strip())
        out.append((start, " ".join(part for part in parts if part)))
        start, parts = None, []
    if start is not None:
        out.append((start, " ".join(part for part in parts if part)))
    return out


def _python_code_lines(path: Path) -> list[tuple[int, str]]:
    """(lineno, code text) for a Python file: comments blanked, docstrings dropped.

    Used only by the insecure-flag sweep, which looks for *code* (``verify=False``
    is a keyword argument, not a string), so the AST walk above cannot serve it.
    ``tokenize`` gives the exact column a comment starts at, which beats a regex
    that cannot tell a ``#`` in a string from a comment.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    prose: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS) and ast.get_docstring(node) is not None:
            doc = node.body[0]  # type: ignore[attr-defined]
            prose.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))

    cuts: dict[int, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            cuts[row] = min(cuts.get(row, col), col)

    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        if lineno in prose:
            continue
        out.append((lineno, line[: cuts[lineno]] if lineno in cuts else line))
    return out


# --- The two sweeps ----------------------------------------------------------


def _plaintext_violations(path: Path) -> list[str]:
    """Rendered ``file:line: match`` lines for one file, allowlist applied."""
    texts = _python_strings(path) if path.suffix == ".py" else _code_lines(path)
    allowed = PLAINTEXT_ALLOWED.get(_relative(path), set())
    return [
        f"{_relative(path)}:{lineno}: {match.group(0)}"
        for lineno, text in texts
        for match in PLAINTEXT_RE.finditer(text)
        if match.group(0) not in allowed
    ]


def _insecure_violations(path: Path) -> list[str]:
    if path.suffix == ".py":
        lines = _python_code_lines(path)
    else:
        lines = _code_lines(path)
        # Shell and Dockerfile commands continue across lines; Python's own
        # continuation is already handled by tokenizing rather than by text.
        if path.suffix == ".sh" or path.name.startswith("Dockerfile"):
            lines = _joined_lines(lines)
    return [
        f"{_relative(path)}:{lineno}: {name} in {line.strip()!r}"
        for lineno, line in lines
        for name, pattern in INSECURE_PATTERNS.items()
        if pattern.search(line)
    ]


# --- Tests -------------------------------------------------------------------


def test_no_python_string_constant_carries_a_plaintext_url() -> None:
    """Every URL the orchestrator or the edge client builds is https."""
    violations: list[str] = []
    for path in _files(PYTHON_GLOBS):
        violations += _plaintext_violations(path)
    assert not violations, (
        "plaintext http:// in Python code:\n  " + "\n  ".join(violations) + f"\n{REMEDY}"
    )


def test_no_script_manifest_config_or_dashboard_asset_carries_a_plaintext_url() -> None:
    """The half of the fleet's URLs that lives outside Python.

    Shell scripts curl these, the kubelet probes them, the generator resolves
    the config templates into what a node actually runs, and the dashboard's JS
    fetches them from the browser — where a plaintext fetch from an https page
    is blocked as mixed content before it even fails.
    """
    violations: list[str] = []
    for path in _files(LINE_GLOBS):
        violations += _plaintext_violations(path)
    assert not violations, (
        "plaintext http:// in a script, manifest, config or dashboard asset:\n  "
        + "\n  ".join(violations)
        + f"\n{REMEDY}"
    )


def test_nothing_in_the_tree_disables_certificate_verification() -> None:
    """A private CA that is not verified authenticates nobody.

    Every client in this fleet verifies against ``~/.aureax/tls/ca.pem`` (or
    ``/etc/aureax/tls/ca.pem`` on a node). One ``verify=False`` — or one
    ``curl -k`` in a script an operator copies from — turns the fleet's server
    authentication back into transport encryption against nobody in particular,
    silently, with every test still green.
    """
    violations: list[str] = []
    for path in _insecure_files():
        violations += _insecure_violations(path)
    assert not violations, (
        "certificate verification is disabled here:\n  "
        + "\n  ".join(violations)
        + "\nUse the fleet CA: client.tls.ssl_context(config) on a node, "
        "tlsstore.ca_ssl_context() on the Mac, curl_tls in shell "
        "(scripts/lib/common.sh). See ARCHITECTURE.md, 'TLS'."
    )


def test_the_operator_docs_show_no_plaintext_url() -> None:
    """The shipped documents are copy-paste sources, so their URLs are code.

    An operator does not read ``curl http://<NODE_IP>:30800/v1/models`` as an
    illustration; they run it, and it fails against a TLS listener. A
    contributor copies from CONTRIBUTING.md and SECURITY.md the same way.
    External links are untouched — only ``http://`` is flagged, so
    ``https://github.com`` and the like stay as they are.
    """
    violations: list[str] = []
    for name in DOC_FILES:
        path = PROJECT_ROOT / name
        text = path.read_text()
        # Both Dockerfiles COPY README.md, so a missing file breaks the image
        # build — but an emptied one would just make this loop find nothing.
        if name in DOC_FILES_WITH_URLS:
            assert "https://" in text, f"{name} contains no https:// URL at all — is it still the doc?"
        # Same allowlist as the code sweeps, and for the same one URL: the
        # docs have to be able to say "open http://localhost:7001/ from the
        # orchestrator", which is the loopback listener and not a LAN URL.
        allowed = PLAINTEXT_ALLOWED.get(name, set())
        for lineno, line in enumerate(text.splitlines(), 1):
            violations += [
                f"{name}:{lineno}: {m.group(0)}"
                for m in PLAINTEXT_RE.finditer(line)
                if m.group(0) not in allowed
            ]
    assert not violations, (
        "plaintext http:// in the operator-facing docs:\n  "
        + "\n  ".join(violations)
        + "\nEvery listener in the fleet is TLS-only; a documented http:// URL is an "
        "instruction that cannot work. See ARCHITECTURE.md, 'TLS'."
    )


# --- Tests about the tests: the sweep has to be reading something ------------


def test_the_sweep_actually_reads_the_tree() -> None:
    """Guards every assertion above against passing on an empty file list.

    A glob that stops matching — a renamed directory, a ``**`` that behaves
    differently — turns four for-loops over nothing into four green tests, and
    the guard is gone with no failure anywhere to say so.
    """
    python_files = _files(PYTHON_GLOBS)
    line_files = _files(LINE_GLOBS)
    assert len(python_files) > 40, f"only {len(python_files)} Python files matched {PYTHON_GLOBS}"
    assert len(line_files) > 10, f"only {len(line_files)} files matched {LINE_GLOBS}"
    insecure_files = _insecure_files()
    assert len(insecure_files) > 50, (
        f"only {len(insecure_files)} files under {INSECURE_ROOTS} were read for disabled "
        "verification"
    )

    # Named files, so a glob that silently drops one whole class of file is
    # caught rather than being absorbed by the counts above.
    names = {_relative(p) for p in python_files + line_files}
    for required in (
        "orchestrator/config/templating.py",
        "client/entrypoint.py",
        "scripts/parse-config.py",
        "scripts/lib/common.sh",
        "scripts/test-all.sh",
        "k8s/daemonset-edge.yaml",
        "configs/aureax-edge.yaml",
        "orchestrator/dashboard/static/utils.js",
        "orchestrator/dashboard/static/index.html",
        "build_support.py",
    ):
        assert required in names, f"{required} is no longer covered by the plaintext sweep"

    # The documents a reader copies from, all four of them.
    for name in DOC_FILES:
        assert (PROJECT_ROOT / name).is_file(), f"{name} is scanned but does not exist"


def test_the_tree_is_full_of_https_urls_the_sweep_can_see() -> None:
    """The complement of the test above: reading files is not enough.

    If the extraction were broken — an AST walk that found no constants, a
    comment stripper that blanked every line — the sweep would read every file
    and see nothing. These are the same code paths, asked for the URLs that
    SHOULD be there.
    """
    seen = 0
    for path in _files(PYTHON_GLOBS):
        seen += sum("https://" in text for _n, text in _python_strings(path))
    assert seen > 10, f"only {seen} https:// string constants found across the Python tree"

    seen = 0
    for path in _files(LINE_GLOBS):
        seen += sum("https://" in line for _n, line in _code_lines(path))
    assert seen > 10, f"only {seen} https:// code lines found across scripts/manifests/configs"


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ('url = "http://10.77.0.1:8080/health"', ["http://10.77.0.1:8080/health"]),
        ('url = f"http://{host}:8080"', ["http://{...}:8080"]),
        ('url = "https://10.77.0.1:8080/health"', []),
        ('# http://10.77.0.1:8080/health\nurl = "https://x"', []),
        ('"""Docstring naming http://10.77.0.1 as the old form."""', []),
        ('def f():\n    """Also http://10.77.0.1."""\n    return 1', []),
        # The bare scheme with nothing after it is how prose names it.
        ('url = "http://"', []),
    ],
)
def test_the_python_rule_reads_code_and_not_prose(tmp_path, source, expect) -> None:
    """Pins each half of the Python rule against a file written for it.

    Without this the module's own extraction is only ever exercised on a tree
    that has no violations in it, so "found nothing" and "cannot find anything"
    are indistinguishable.
    """
    path = tmp_path / "sample.py"
    path.write_text(source)
    found = [m.group(0) for _n, text in _python_strings(path) for m in PLAINTEXT_RE.finditer(text)]
    assert found == expect


@pytest.mark.parametrize(
    ("name", "source", "expect"),
    [
        ("s.sh", 'curl "http://127.0.0.1:7001/health"', ["http://127.0.0.1:7001/health"]),
        ("s.sh", "# curl http://127.0.0.1:7001/health", []),
        ("s.sh", 'curl "$URL"   # was http://127.0.0.1:7001', []),
        ("c.yaml", 'endpoint: "http://127.0.0.1:8080/health"', ["http://127.0.0.1:8080/health"]),
        ("c.yaml", "  # endpoint: http://127.0.0.1:8080/health", []),
        ("c.yaml", 'endpoint: "https://127.0.0.1:8080/health"', []),
        ("a.js", 'fetch("http://127.0.0.1:7001/api")', ["http://127.0.0.1:7001/api"]),
        ("a.js", "// fetch http://127.0.0.1:7001/api", []),
        ("a.js", " * fetch http://127.0.0.1:7001/api", []),
        # The `//` of a scheme must not read as a comment marker, and a
        # trailing `//` comment must still be cut.
        ("a.js", 'fetch("https://x")  // not http://y', []),
    ],
)
def test_the_line_rule_reads_code_and_not_comments(tmp_path, name, source, expect) -> None:
    path = tmp_path / name
    path.write_text(source)
    found = [m.group(0) for _n, line in _code_lines(path) for m in PLAINTEXT_RE.finditer(line)]
    assert found == expect


@pytest.mark.parametrize(
    ("name", "source", "flagged"),
    [
        ("m.py", "httpx.get(url, verify=False)", True),
        ("m.py", "httpx.get(url, verify = False)", True),
        ("m.py", "ctx.check_hostname = False", True),
        ("m.py", "ctx.verify_mode = ssl.CERT_NONE", True),
        ("m.py", "# never verify=False here", False),
        ("m.py", '"""Never pass verify=False."""', False),
        ("m.py", "httpx.get(url, verify=ssl_context)", False),
        ("s.sh", "curl -sk https://x", True),
        ("s.sh", "curl --insecure https://x", True),
        ("s.sh", "curl_tls -sk https://x", True),
        ("s.sh", "# no -k / --insecure anywhere", False),
        ("s.sh", 'curl --cacert "$AUREAX_CA" -fsS https://x', False),
        ("s.sh", "curl_tls -fsS --max-time 5 https://x", False),
        # A continued invocation is one command, so its flags are one line.
        ("s.sh", "curl -sfS \\\n  -k https://x", True),
        ("s.sh", "curl_tls -s \\\n  --insecure https://x", True),
        ("s.sh", 'curl_tls -s \\\n  --cacert "$AUREAX_CA" https://x', False),
        ("Dockerfile.x", "RUN curl -fsS \\\n  -k https://x", True),
    ],
)
def test_the_insecure_sweep_catches_each_spelling(tmp_path, name, source, flagged) -> None:
    """One case per pattern, plus the near-misses that must NOT fire.

    ``--cacert`` is the important negative: it contains a ``k``-free short-flag
    shape that a sloppier regex reads as ``-k``, and flagging the correct call
    would push somebody to weaken the pattern.
    """
    path = tmp_path / name
    path.write_text(source)
    assert bool(_insecure_violations(path)) is flagged


def test_an_allowlisted_match_is_the_only_way_past_the_sweep(tmp_path, monkeypatch) -> None:
    """The escape hatch works, and works only for the exact string named.

    Written down because an allowlist nobody has exercised is an allowlist that
    silently exempts more (or less) than its author believed.
    """
    path = tmp_path / "c.yaml"
    path.write_text('a: "http://198.51.100.9/vendor"\nb: "http://198.51.100.9/other"\n')
    monkeypatch.setitem(PLAINTEXT_ALLOWED, "c.yaml", {"http://198.51.100.9/vendor"})
    assert _plaintext_violations(path) == ["c.yaml:2: http://198.51.100.9/other"]


def test_the_allowlist_holds_only_the_loopback_dashboard() -> None:
    """Not a style rule: every entry is a documented plaintext hole.

    Pinned by exact value rather than by count, so a sixth entry — or a
    widened fifth, say ``http://`` plus a LAN address in README — fails here.
    This test failing is not a reason to relax it; it is the review prompt. An
    entry needs a reason comment above it and a line in the change summary
    saying which traffic now crosses the LAN in the clear.

    Everything named below is ``localhost``: the dashboard's loopback listener,
    which is bound to 127.0.0.1 and ::1 only and therefore cannot be reached
    from the LAN at all. An entry naming any other host is a real hole.
    """
    expected = {
        "scripts/check-installed-package.py": {"http://localhost:7001"},
        "orchestrator/cli.py": {"http://localhost:{...}/"},
        "scripts/test-all.sh": {"http://localhost:7001"},
        "README.md": {"http://localhost:7001/"},
        "ARCHITECTURE.md": {"http://localhost:7001/"},
    }
    actual = dict(PLAINTEXT_ALLOWED)
    assert actual == expected, (
        f"PLAINTEXT_ALLOWED changed: {PLAINTEXT_ALLOWED!r}. Each entry is traffic "
        "exempted from mandatory TLS — keep the reason comment and say so in the "
        "change summary."
    )
    for path, urls in PLAINTEXT_ALLOWED.items():
        for url in urls:
            assert url.startswith("http://localhost:"), (
                f"{path} exempts {url!r}, which is not the loopback dashboard — an "
                "exemption for a routable address is a plaintext listener on the LAN"
            )
