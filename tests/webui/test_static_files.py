"""Static checks of the frontend modules and the deployment files.

A single syntax error in any ES module stops the whole app - the static menu and
footer still render, the content never does. Node is not available everywhere
the tests run, so the most common such error is caught textually: a quote that
ends a single-quoted string early (e.g. `'the claim's reviews'`).
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
JS_FILES = sorted((ROOT / "webui" / "static" / "js").rglob("*.js"))

DOUBLE = re.compile(r'"(?:\\.|[^"\\])*"')
TEMPLATE = re.compile(r"`(?:\\.|[^`\\])*`")
SINGLE = re.compile(r"'(?:\\.|[^'\\])*'")
#: Regular-expression literals may contain a bare quote in a character class.
REGEX_LITERAL = re.compile(r"/\[[^\]]*\][^/\n]*/[gimsuy]*")


def unbalanced_single_quotes(line: str) -> bool:
    text = line.strip()
    if text.startswith(("//", "*", "/*")):
        return False
    rest = REGEX_LITERAL.sub("//", line)
    rest = SINGLE.sub("''", TEMPLATE.sub("``", DOUBLE.sub('""', rest)))
    return "'" in rest.replace("''", "")


def test_there_are_frontend_modules():
    assert JS_FILES


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_single_quoted_string_ends_early(path):
    offending = [f"{number}: {line.strip()}"
                 for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if unbalanced_single_quotes(line)]
    assert offending == []


def test_the_check_catches_an_apostrophe_in_a_single_quoted_string():
    assert unbalanced_single_quotes("    key: 'None of the claim's reviews.',")
    assert not unbalanced_single_quotes("    key: 'None of the claim\\'s reviews.',")
    assert not unbalanced_single_quotes("    key: \"None of the claim's reviews.\",")


# --- Deployment ---------------------------------------------------------------------

def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yaml").read_text(encoding="utf-8"))


def test_the_tls_front_is_off_unless_its_profile_is_enabled():
    tls = compose()["services"]["webui-tls"]
    assert tls["profiles"] == ["tls"]
    assert "./webui/Caddyfile:/etc/caddy/Caddyfile:ro" in tls["volumes"]
    assert {"80:80", "443:443"} <= set(tls["ports"])


def test_the_plain_http_port_can_be_kept_local():
    [mapping] = compose()["services"]["webui"]["ports"]
    assert mapping.startswith("${WEBUI_BIND:-0.0.0.0}:")


def test_the_caddyfile_serves_the_configured_domain_from_the_ui():
    caddyfile = (ROOT / "webui" / "Caddyfile").read_text(encoding="utf-8")
    assert "{$WEBUI_DOMAIN} {" in caddyfile
    assert "reverse_proxy {$WEBUI_UPSTREAM}" in caddyfile


def test_the_podman_override_reaches_the_ui_on_the_host_loopback():
    override = (ROOT / "docker-compose.podman.yaml").read_text(encoding="utf-8")
    assert "webui-tls:" in override
    assert "WEBUI_UPSTREAM: ${WEBUI_UPSTREAM:-127.0.0.1:${WEBUI_PORT:-8080}}" in override
    assert "WEBUI_HOST: ${WEBUI_BIND:-0.0.0.0}" in override
