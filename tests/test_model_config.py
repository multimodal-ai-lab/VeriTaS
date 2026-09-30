"""Model versions and the scrapeMM client are configured in config.yaml.

Pure tests: no DB, no network, no model calls. Every helper takes the config
section (and, for scrapeMM, the `configure` function and the environment) as an
argument, so nothing here depends on what the local config.yaml says.
"""

import pytest
import pytest_asyncio

from veritas import SCRAPEMM_ENV_VARS, configure_scrapemm
from veritas.models.base import (
    GEMINI_PROVIDERS,
    OPENAI_PROVIDERS,
    configured_model_name,
    matching_singleton,
    split_known_provider,
)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from `tests/conftest.py`: these tests need no DB."""
    yield None


class FakeModel:
    def __init__(self, specifier: str):
        self.specifier = specifier


# --- configured_model_name ---------------------------------------------------

@pytest.mark.parametrize("config", [{}, {"gpt_strong": None}, {"other_key": "gpt-x"}])
def test_missing_key_falls_back_to_the_default(config):
    assert configured_model_name("gpt_strong", "gpt-default", OPENAI_PROVIDERS, config) == "gpt-default"


def test_configured_value_wins_over_the_default():
    config = {"gpt_strong": "gpt-configured"}
    assert configured_model_name("gpt_strong", "gpt-default", OPENAI_PROVIDERS, config) == "gpt-configured"


@pytest.mark.parametrize("value, providers, expected", [
    ("openai:gpt-configured", OPENAI_PROVIDERS, "gpt-configured"),
    ("OpenAI:gpt-configured", OPENAI_PROVIDERS, "gpt-configured"),
    ("gemini:gemini-configured", GEMINI_PROVIDERS, "gemini-configured"),
    ("google:gemini-configured", GEMINI_PROVIDERS, "gemini-configured"),
    ("  gemini:gemini-configured  ", GEMINI_PROVIDERS, "gemini-configured"),
])
def test_a_matching_provider_prefix_is_stripped(value, providers, expected):
    assert configured_model_name("key", "default", providers, {"key": value}) == expected


@pytest.mark.parametrize("value, providers", [
    ("anthropic:claude-opus-5-5", OPENAI_PROVIDERS),
    ("gemini:gemini-3.1-pro-preview", OPENAI_PROVIDERS),
    ("selfhosted:meta-llama/Llama-4", OPENAI_PROVIDERS),
    ("openai:gpt-6.1-sol", GEMINI_PROVIDERS),
    ("anthropic:claude-opus-5-5", GEMINI_PROVIDERS),
])
def test_a_different_provider_fails_loudly(value, providers):
    with pytest.raises(ValueError, match="models.key"):
        configured_model_name("key", "default", providers, {"key": value})


@pytest.mark.parametrize("value", ["", "   ", 42, ["gpt-x"]])
def test_an_invalid_value_fails_loudly(value):
    with pytest.raises(ValueError):
        configured_model_name("key", "default", OPENAI_PROVIDERS, {"key": value})


def test_an_unknown_prefix_is_part_of_the_model_name():
    """OpenAI fine-tunes contain colons; they must not be mistaken for a provider."""
    name = "ft:gpt-4o:my-org:custom:abc123"
    assert configured_model_name("key", "default", OPENAI_PROVIDERS, {"key": name}) == name
    assert split_known_provider(name) == (None, name)


def test_the_real_singletons_use_the_configured_or_default_names():
    """The module-level singletons are built through the helper, so their names are
    whatever the helper returns for the local config."""
    from veritas import models_config
    from veritas.models import gemini_strong, gpt_nano, gpt_strong

    assert gpt_strong.specifier == configured_model_name(
        "gpt_strong", "gpt-6.1-sol", OPENAI_PROVIDERS, models_config)
    assert gpt_nano.specifier == configured_model_name(
        "gpt_nano", "gpt-6-luna", OPENAI_PROVIDERS, models_config)
    assert gemini_strong.specifier == configured_model_name(
        "gemini_strong", "gemini-3.1-pro-preview", GEMINI_PROVIDERS, models_config)


# --- Ensemble members ----------------------------------------------------------

GPT = FakeModel("gpt-strong")
GEMINI = FakeModel("gemini-strong")
SINGLETONS = [(GPT, OPENAI_PROVIDERS), (GEMINI, GEMINI_PROVIDERS)]


@pytest.mark.parametrize("specifier, expected", [
    ("openai:gpt-strong", GPT),
    ("gpt-strong", GPT),
    ("gemini:gemini-strong", GEMINI),
    ("google:gemini-strong", GEMINI),
    ("gemini-strong", GEMINI),
])
def test_a_specifier_of_a_singleton_reuses_it(specifier, expected):
    assert matching_singleton(specifier, SINGLETONS) is expected


@pytest.mark.parametrize("specifier", [
    "anthropic:claude-opus-5-5",
    "openai:gpt-other",
    "selfhosted:gpt-strong",  # Same name, but served by another endpoint
    "gemini:gpt-strong",
])
def test_other_specifiers_match_no_singleton(specifier):
    assert matching_singleton(specifier, SINGLETONS) is None


def test_ensemble_members_reuse_singletons_and_keep_other_specifiers():
    from veritas.ensemble import ensemble_members

    members = ensemble_members(
        ["openai:gpt-strong", "anthropic:claude-opus-5-5", "gemini:gemini-strong"], SINGLETONS)

    assert members[0] is GPT
    assert members[1] == "anthropic:claude-opus-5-5"
    assert members[2] is GEMINI


@pytest.mark.parametrize("configured", [None, []])
def test_ensemble_members_default_to_the_strong_singletons_and_claude(configured):
    from veritas.ensemble import DEFAULT_ENSEMBLE, ensemble_members
    from veritas.models import gemini_strong, gpt_strong

    real_singletons = [(gpt_strong, OPENAI_PROVIDERS), (gemini_strong, GEMINI_PROVIDERS)]
    members = ensemble_members(configured, real_singletons)

    assert len(members) == len(DEFAULT_ENSEMBLE) == 3
    assert members[0] is gpt_strong
    assert members[1] == "anthropic:claude-opus-5-5"
    assert members[2] is gemini_strong


def test_a_single_string_is_one_ensemble_member():
    from veritas.ensemble import ensemble_members

    assert ensemble_members("anthropic:claude-opus-5-5", SINGLETONS) == ["anthropic:claude-opus-5-5"]


# --- scrapeMM -------------------------------------------------------------------

class FakeConfigure:
    """Records the calls `configure_scrapemm` makes in place of `scrapemm.configure`."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


SECTION = {"api_url": "http://scrapemm.test:3005", "api_key": "test-key"}


def test_both_values_are_passed_without_persisting_by_default():
    configure = FakeConfigure()

    passed = configure_scrapemm(SECTION, configure, environ={})

    assert passed == SECTION
    assert configure.calls == [{**SECTION, "persist": False}]


@pytest.mark.parametrize("section", [None, {}, {"api_url": None, "api_key": ""}])
def test_nothing_to_configure_means_no_call(section):
    configure = FakeConfigure()

    assert configure_scrapemm(section, configure, environ={}) == {}
    assert configure.calls == []


def test_a_missing_value_is_skipped():
    configure = FakeConfigure()

    configure_scrapemm({"api_url": "http://scrapemm.test:3005"}, configure, environ={})

    assert configure.calls == [{"api_url": "http://scrapemm.test:3005", "persist": False}]


def test_an_environment_variable_wins_over_the_config():
    """`scrapemm.configure()` would override the environment in-process, so a value
    whose variable is set must not be passed at all."""
    configure = FakeConfigure()
    environ = {SCRAPEMM_ENV_VARS["api_key"]: "key-from-env"}

    passed = configure_scrapemm(SECTION, configure, environ=environ)

    assert passed == {"api_url": SECTION["api_url"]}
    assert configure.calls == [{"api_url": SECTION["api_url"], "persist": False}]


def test_all_values_set_in_the_environment_means_no_call():
    configure = FakeConfigure()
    environ = {var: "from-env" for var in SCRAPEMM_ENV_VARS.values()}

    assert configure_scrapemm(SECTION, configure, environ=environ) == {}
    assert configure.calls == []


def test_an_empty_environment_variable_does_not_count_as_set():
    """scrapeMM ignores empty variables too (`if env := os.getenv(var)`)."""
    configure = FakeConfigure()
    environ = {var: "" for var in SCRAPEMM_ENV_VARS.values()}

    assert configure_scrapemm(SECTION, configure, environ=environ) == SECTION


def test_saving_ignores_the_environment_and_persists():
    """What scripts/configure_scrapemm.py does: save the config values as they are."""
    configure = FakeConfigure()
    environ = {var: "from-env" for var in SCRAPEMM_ENV_VARS.values()}

    passed = configure_scrapemm(SECTION, configure, persist=True,
                                respect_environment=False, environ=environ)

    assert passed == SECTION
    assert configure.calls == [{**SECTION, "persist": True}]


def test_the_api_key_is_never_logged(caplog):
    caplog.set_level("DEBUG", logger="VeriTaS")

    configure_scrapemm(SECTION, FakeConfigure(), environ={})
    configure_scrapemm(SECTION, FakeConfigure(),
                       environ={var: "from-env" for var in SCRAPEMM_ENV_VARS.values()})

    assert "test-key" not in caplog.text
