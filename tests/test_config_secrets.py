"""The config editor never hands a secret to the browser, and never loses one.

`orchestrator/dashboard/config_secrets.py` backs the dashboard's YAML editor:
every `password`/`api_key` value is replaced by a marker on the way out, and
the marker is re-bound to its original value on the way back in, keyed on the
node's identity and destination rather than on its position in the document.
Anything that cannot be re-bound — a moved node, a changed host, an edited
marker, an ambiguous document — fails closed with a fixed message, because the
alternative is shipping the fleet's SSH password to a page the dashboard serves
without authentication. See ARCHITECTURE.md, "Security".
"""

from io import StringIO

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from ruamel.yaml import YAML

from orchestrator.dashboard import config_api
from orchestrator.dashboard.config_secrets import REDACTION_MARKER, redact, restore
from tests.conftest import EXAMPLE_NETWORK_CONFIG


@pytest.mark.parametrize("fragment", [
    'password: "synthetic-secret"',
    '"password": synthetic-secret',
    'password: |\n      synthetic-secret',
    'password: &password synthetic-secret\n    copy: *password',
])
def test_yaml_secret_forms_are_redacted_and_roundtrip(fragment):
    text = f"ssh:\n  targets:\n  - node_id: a\n    host: 192.0.2.1\n    {fragment}\n"
    hidden, changed = redact(text)
    assert changed and "synthetic-secret" not in hidden
    restored = YAML().load(restore(hidden, text))
    original = YAML().load(text)
    assert restored == original


def test_flow_yaml_and_external_alias_are_redacted():
    text = 'secret: &pw synthetic-secret\nssh: {targets: [{node_id: a, host: 192.0.2.1, password: *pw}]}\n'
    hidden, _ = redact(text)
    assert "synthetic-secret" not in hidden
    assert YAML().load(restore(hidden, text)) == YAML().load(text)


def test_passwords_follow_nodes_after_reorder_or_removal():
    text = 'ssh: {targets: [{node_id: a, host: 192.0.2.1, password: alpha}, {node_id: b, host: 192.0.2.2, password: beta}]}\n'
    hidden, _ = redact(text)
    doc = YAML().load(hidden)
    doc["ssh"]["targets"].reverse()
    stream = StringIO()
    YAML().dump(doc, stream)
    actual = YAML().load(restore(stream.getvalue(), text))["ssh"]["targets"]
    assert [(n["node_id"], n["password"]) for n in actual] == [("b", "beta"), ("a", "alpha")]
    assert "alpha" not in restore(f'ssh: {{targets: [{{node_id: b, host: 192.0.2.2, password: "{REDACTION_MARKER}"}}]}}', text)


@pytest.mark.parametrize("edit", ["host: 192.0.2.2", "node_id: b", "user: root"])
def test_redacted_destination_edits_fail_closed(edit):
    original = "ssh: {targets: [{node_id: a, host: 192.0.2.1, password: secret}]}"
    hidden, _ = redact(original)
    if edit.startswith("user"):
        edited = hidden.replace("host: 192.0.2.1", f"host: 192.0.2.1, {edit}")
    else:
        before = "node_id: a" if edit.startswith("node_id") else "host: 192.0.2.1"
        edited = hidden.replace(before, edit)
    with pytest.raises(HTTPException):
        restore(edited, original)


def test_literal_template_api_key_roundtrip():
    text = 'model: {server: {api_key: synthetic-key, port: 8080}}'
    hidden, _ = redact(text)
    assert "synthetic-key" not in hidden
    assert YAML().load(restore(hidden, text)) == YAML().load(text)


@pytest.mark.parametrize("text", [
    "x: &x {x: *x, password: leaked-gamma}",
    "password: leaked-alpha\npassword: leaked-beta",
])
def test_ambiguous_yaml_is_still_viewable_with_its_secret_lines_hidden(text):
    """A file only a hand edit can have broken is the file the editor exists to
    repair, so an unparseable document is shown rather than refused — with
    every `password:`/`api_key:` line cut back to the key, whatever quoting or
    flow punctuation surrounds it. The inputs carry a distinctive marker:
    asserting only that the document changed would pass on one that still
    carried the value."""
    hidden, redacted = redact(text)
    assert redacted
    assert "leaked-" not in hidden
    assert REDACTION_MARKER in hidden


@pytest.mark.parametrize("text", [
    "x: &x {x: *x, password: leaked-gamma}",
    "password: leaked-alpha\npassword: leaked-beta",
])
def test_ambiguous_yaml_is_rejected_on_save_without_echoing_values(text):
    """Saving is where a parse must succeed: a marker can only be re-bound to
    the value it replaced if both texts parse. The rejection message is fixed,
    so a parser error carrying the offending scalar cannot become the way a
    secret reaches an unauthenticated page."""
    hidden, _ = redact(text)
    with pytest.raises(HTTPException) as exc:
        restore(hidden, text)
    assert exc.value.status_code == 400
    assert text not in exc.value.detail
    assert "leaked-" not in exc.value.detail


def test_a_password_equal_to_another_scalar_redacts_only_the_password():
    """Secrets are found by the key they sit under, never by their value.

    The default SSH user is a plausible bench password, and matching by value
    would rewrite `user:` to the marker as well — after which `restore` reads
    the marker as the node's identity, finds it changed, and refuses every save
    of the file for good."""
    text = 'ssh: {targets: [{node_id: aureax, host: 192.0.2.1, user: aureax, password: aureax}]}\n'

    hidden, redacted = redact(text)

    assert redacted
    node = YAML().load(hidden)["ssh"]["targets"][0]
    assert node["password"] == REDACTION_MARKER
    assert (node["user"], node["node_id"]) == ("aureax", "aureax")
    assert YAML().load(restore(hidden, text)) == YAML().load(text)


def test_a_short_secret_is_left_alone_in_comment_prose():
    """The comment scrub is by value and no save reverses it, so a secret short
    enough to also be an ordinary word must not rewrite what the operator
    wrote. A real credential is far longer, and is still scrubbed."""
    text = 'ssh:\n  targets:\n  - node_id: a  # the aureax bench node\n    user: aureax\n    password: aureax\n'

    hidden, _ = redact(text)

    assert "# the aureax bench node" in hidden
    assert "<redacted>" not in hidden


def test_redacting_and_restoring_the_example_config_is_byte_identical():
    """The operator gets their own file back, not a reformatted one.

    ruamel's dump defaults re-indent every dash-offset sequence in configs/ and
    fold the long description line at 80 columns, so a GET on a fleet file with
    a literal password used to show a rewritten document and a PUT used to
    write one — with no edit made."""
    marker = '      password_env: "AUREAX_SSH_PASSWORD"\n'
    source = EXAMPLE_NETWORK_CONFIG.read_text()
    assert marker in source, "example config no longer has a password_env target"
    text = source.replace(marker, marker + '      password: "synthetic-secret"\n', 1)

    hidden, redacted = redact(text)

    assert redacted and "synthetic-secret" not in hidden
    assert restore(hidden, text) == text


def test_real_config_route_redacts_flow_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(config_api, "CONFIGS_DIR", tmp_path)
    (tmp_path / "aureax-network.yaml").write_text('ssh: {targets: [{node_id: a, host: 192.0.2.1, "password": synthetic-secret}]}')
    app = FastAPI()
    app.include_router(config_api.router)
    response = TestClient(app).get("/api/config/aureax-network.yaml")
    assert response.status_code == 200
    assert "synthetic-secret" not in response.text
    assert response.json()["redacted"]


def test_merged_aliases_roundtrip_without_stale_secret_values():
    text = 'defaults: &d {password: synthetic-secret}\nssh: {targets: [{<<: *d, node_id: a, host: 192.0.2.1}]}\n'
    hidden, _ = redact(text)
    assert 'synthetic-secret' not in hidden
    assert YAML().load(restore(hidden, text)) == YAML().load(text)


def test_redaction_also_covers_copied_secrets_in_comments():
    hidden, _ = redact('model:\n  server:\n    api_key: synthetic-secret # copied: synthetic-secret\n')
    assert 'synthetic-secret' not in hidden
