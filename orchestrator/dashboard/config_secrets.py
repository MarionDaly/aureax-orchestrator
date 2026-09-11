"""YAML-aware editor redaction and identity-bound secret restoration."""

import re
from copy import deepcopy
from io import StringIO

from fastapi import HTTPException
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError
from ruamel.yaml.tokens import CommentToken

REDACTION_MARKER = "<redacted — see Access tab>"
SECRET_FIELDS = {"password", "api_key"}

# Shortest secret worth scrubbing out of comments. The scrub is by value and
# nothing reverses it on save, so a short secret that also reads as an ordinary
# word — a default user name, a tag, a node id — would permanently rewrite
# prose the operator wrote. A credential this short is not one this file can
# protect anyway; the Access tab's generated keys are far longer.
COMMENT_SCRUB_MIN = 8

# Fallback redaction for a document ruamel cannot parse: `password:` or
# `api_key:` anywhere on a line, block or flow style. The lookarounds keep
# `password_env` (which names an environment variable, not a secret) and
# `mcp_api_key` out — the latter only ever appears as a placeholder.
_SECRET_KEY_RE = re.compile(r"(?<![\w.\-])(?:password|api_key)(?![\w.\-])[ \t]*:[ \t]*")

# An unresolved `${...}` template placeholder is not a secret, and neither is
# an empty value. The parsed path skips both for the same reason.
_KEEP_VALUE_RE = re.compile(r"""^(["']?\$\{[^{}]*\}["']?)?[ \t]*(\#.*)?$""")


def _load(text: str):
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        _walk(YAML(typ="safe").load(text))
        doc = yaml.load(text)
        _walk(doc)  # Reject recursive aliases before traversing the document.
    except (YAMLError, RecursionError, ValueError) as exc:
        raise HTTPException(400, "Config must be valid, non-recursive YAML with unique keys") from exc
    return doc


def _walk(doc, ancestors=frozenset(), visited=None):
    visited = set() if visited is None else visited
    if isinstance(doc, dict | list):
        if id(doc) in ancestors:
            raise ValueError("recursive alias")
        if id(doc) in visited:
            return
        visited.add(id(doc))
        ancestors = ancestors | {id(doc)}
        for value in doc.values() if isinstance(doc, dict) else doc:
            _walk(value, ancestors, visited)


def _dump(doc) -> str:
    """Serialise a round-trip document back to the layout it was loaded from.

    ruamel's defaults are not this project's: `indent(mapping=2, sequence=2,
    offset=0)` re-indents every dash-offset sequence in configs/, `width=80`
    folds the long description and huggingface_file lines, and a None comes
    back as an empty value rather than the explicit `null` the configs use. A
    redact/restore round-trip must hand the operator their own file back, not a
    reformatted one, so every one of those is pinned here.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 10**6
    yaml.representer.add_representer(
        type(None), lambda r, d: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    stream = StringIO()
    yaml.dump(doc, stream)
    return stream.getvalue()


def _redact_lines(text: str) -> tuple[str, bool]:
    """Conservative line-based redaction, for a document that will not parse.

    A file a hand edit has broken is exactly the file an operator opens the
    editor to repair, so a parse failure cannot be the answer on the way OUT.
    It stays the answer on the way IN: `restore` requires both texts to parse,
    so saving this view fails closed rather than guessing where a value went.

    Everything after the key is dropped — trailing comment included — whatever
    quoting or flow punctuation the broken line happens to carry.
    """
    lines = text.split("\n")
    redacted = False
    for index, line in enumerate(lines):
        match = _SECRET_KEY_RE.search(line)
        if match is None or _KEEP_VALUE_RE.match(line[match.end():]):
            continue
        lines[index] = line[:match.end()] + REDACTION_MARKER
        redacted = True
    return "\n".join(lines), redacted


def redact(text: str) -> tuple[str, bool]:
    """Replace every secret with the marker; ``(text, True)`` if anything moved.

    Secrets are located by POSITION — the key they sit under — never by value.
    A fleet whose SSH password happens to equal another scalar in the file (the
    default SSH user, a tag, a node id) would otherwise have that scalar
    rewritten too, and `restore` derives a node's identity from its `user`, so
    a marker landing there locks the operator out of saving the file at all.
    """
    try:
        doc = _load(text)
    except HTTPException:
        return _redact_lines(text)

    # (id(parent), key) for every secret field, so only the value under a
    # `password:`/`api_key:` key is substituted.
    positions = set()
    # Anchored scalars: ruamel hands every alias of one the SAME object, so a
    # secret reachable through an alias has to go wherever that object appears
    # — the definition included. A plain str is a fresh object per occurrence,
    # so identity here means "same node", never "same text".
    aliased = set()
    # Values to scrub out of comment text, which has no position to key on.
    copied = set()
    collected = set()

    def collect(value):
        if id(value) in collected:
            return
        collected.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                if key in SECRET_FIELDS and child is not None:
                    if not isinstance(child, str):
                        raise HTTPException(400, "Secret fields must be strings")
                    if child and not (child.startswith("${") and child.endswith("}")):
                        positions.add((id(value), key))
                        if type(child) is not str:
                            aliased.add(id(child))
                        if len(child) >= COMMENT_SCRUB_MIN:
                            copied.add(str(child))
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(doc)
    if not positions:
        return text, False

    def comments(value):
        result = deepcopy(value)
        def scrub(item):
            if isinstance(item, CommentToken):
                for secret in sorted(copied, key=len, reverse=True):
                    item.value = item.value.replace(secret, "<redacted>")
            elif isinstance(item, list):
                for child in item:
                    scrub(child)
            elif isinstance(item, dict):
                for child in item.values():
                    scrub(child)
        scrub(result)
        return result

    copies = {}
    def replace(value):
        if isinstance(value, dict | list):
            if id(value) in copies:
                return copies[id(value)]
            # Build a fresh representation. Mutating merge sources in place
            # triggers ruamel alias backreferences and can leave stale values.
            result = CommentedMap() if isinstance(value, dict) else CommentedSeq()
            copies[id(value)] = result
            if hasattr(value, "fa") and value.fa.flow_style():
                result.fa.set_flow_style()
            if hasattr(value, "ca"):
                result.ca.comment = comments(value.ca.comment)
                result.ca.items.update(comments(value.ca.items))
                result.ca.end.extend(comments(value.ca.end))
            if isinstance(value, dict):
                for key, child in value.items():
                    result[replace(key)] = (REDACTION_MARKER
                                            if (id(value), key) in positions
                                            else replace(child))
            else:
                result.extend(replace(child) for child in value)
            return result
        if id(value) in aliased:
            return REDACTION_MARKER
        return value

    return _dump(replace(doc)), True


def _copy_scalar(value):
    result = deepcopy(value)
    if hasattr(result, "yaml_set_anchor"):
        result.yaml_set_anchor(None)
    return result


def restore(text: str, original: str) -> str:
    """Restore network passwords by node and destination; other markers fail closed."""
    doc, stored = _load(text), _load(original)

    def targets(value):
        try:
            ssh = value.get("ssh", {})
            nodes = ssh.get("targets", [])
            result = {}
            for node in nodes:
                key = node["node_id"]
                if key in result:
                    raise ValueError("duplicate node")
                identity = (node.get("host"), node.get("port", 22), node.get("user", "aureax"))
                result[key] = (identity, node)
            return result
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            raise HTTPException(400, "Cannot restore secrets without unique node identities") from exc

    originals = targets(stored)
    for key, (identity, node) in targets(doc).items():
        if node.get("password") == REDACTION_MARKER:
            old_identity, old_node = originals.get(key, (None, {}))
            if identity != old_identity or "password" not in old_node:
                raise HTTPException(400, "A redacted node identity or SSH destination changed; enter its password explicitly")
            node["password"] = _copy_scalar(old_node["password"])
    def remaining(value, old):
        if value == REDACTION_MARKER:
            if not isinstance(old, str) or old == REDACTION_MARKER:
                raise HTTPException(400, "Cannot restore an unresolved secret")
            return _copy_scalar(old)
        if isinstance(value, dict):
            if not isinstance(old, dict):
                old = {}
            # Endpoint changes must provide new secrets explicitly.
            if any(value.get(k) == REDACTION_MARKER for k in SECRET_FIELDS):
                for key in ("host", "port", "user", "url", "base_url", "node_id", "name"):
                    if value.get(key) != old.get(key):
                        raise HTTPException(400, "A secret destination changed; enter its value explicitly")
            for key in value:
                value[key] = remaining(value[key], old.get(key))
        elif isinstance(value, list):
            old = old if isinstance(old, list) else []
            for index, child in enumerate(value):
                if not isinstance(child, dict):
                    if REDACTION_MARKER in str(child):
                        raise HTTPException(400, "Cannot restore a secret in an unnamed list")
                    continue
                identity = next((key for key in ("node_id", "name", "id") if key in child), None)
                matches = [item for item in old if isinstance(item, dict) and identity and item.get(identity) == child[identity]]
                previous = matches[0] if len(matches) == 1 else {}
                value[index] = remaining(child, previous)
        return value

    # Anchor definitions outside secret fields must be restored as well. Paths
    # inside named lists use stable identities, never positions.
    doc = remaining(doc, stored)
    result = _dump(doc)
    if REDACTION_MARKER in result:
        raise HTTPException(400, "Unresolved redaction marker; reload the config or enter the value explicitly")
    return result
