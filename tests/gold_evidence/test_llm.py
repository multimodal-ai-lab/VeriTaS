"""Model resolution: a configured model must be constructed once, not per call."""

import pytest

from veritas.gold_evidence import extraction as extraction_module
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence import llm as llm_module
from veritas.gold_evidence import retrieval as retrieval_module
from veritas.gold_evidence.llm import clear_cache, get_model, resolve_model


class FakeModel:
    def __init__(self, specifier: str):
        self.specifier = specifier


@pytest.fixture
def constructions(monkeypatch):
    """Counts how often `init_model` is actually reached."""
    calls: list[str] = []

    def fake_init_model(specifier: str):
        calls.append(specifier)
        return FakeModel(specifier)

    # `get_model` imports init_model lazily from veritas.models, so patch it there.
    import veritas.models

    monkeypatch.setattr(veritas.models, "init_model", fake_init_model)
    clear_cache()
    yield calls
    clear_cache()


def test_the_same_specifier_is_constructed_once(constructions):
    first = get_model("openai:gpt-5")
    second = get_model("openai:gpt-5")

    assert first is second
    assert constructions == ["openai:gpt-5"]


def test_different_specifiers_get_different_instances(constructions):
    a = get_model("openai:gpt-5")
    b = get_model("anthropic:claude-sonnet-4-5-20250929")

    assert a is not b
    assert constructions == ["openai:gpt-5", "anthropic:claude-sonnet-4-5-20250929"]


def test_repeated_resolution_reuses_the_instance(constructions):
    default = FakeModel("default")
    models = [resolve_model("openai:gpt-5", default) for _ in range(50)]

    assert len(set(id(m) for m in models)) == 1
    assert constructions == ["openai:gpt-5"]


@pytest.mark.parametrize("specifier", ["auto", "", None])
def test_auto_returns_the_default_without_constructing_anything(constructions, specifier):
    default = FakeModel("default")
    assert resolve_model(specifier, default) is default
    assert constructions == []


def test_clear_cache_forces_reconstruction(constructions):
    get_model("openai:gpt-5")
    clear_cache()
    get_model("openai:gpt-5")
    assert constructions == ["openai:gpt-5", "openai:gpt-5"]


# --- The three call sites --------------------------------------------------

def test_filtering_model_is_constructed_once(constructions, monkeypatch):
    """Stage 2 resolves this twice per evidence item, so it is the hot one."""
    monkeypatch.setattr(filtering_module, "filtering_model", "openai:gpt-5")

    models = [filtering_module._resolve_filtering_model() for _ in range(20)]

    assert len(set(id(m) for m in models)) == 1
    assert constructions == ["openai:gpt-5"]


def test_filtering_falls_back_to_the_shared_gpt_strong(constructions, monkeypatch):
    from veritas.models import gpt_strong

    monkeypatch.setattr(filtering_module, "filtering_model", "auto")
    assert filtering_module._resolve_filtering_model() is gpt_strong
    assert constructions == []


def test_extraction_model_is_constructed_once(constructions, monkeypatch):
    monkeypatch.setattr(extraction_module, "extraction_model", "gemini:gemini-2.5-pro")

    class PromptWithoutVideos:
        def has_videos(self):
            return False

    models = [extraction_module._resolve_model(PromptWithoutVideos()) for _ in range(10)]

    assert len(set(id(m) for m in models)) == 1
    assert constructions == ["gemini:gemini-2.5-pro"]


def test_extraction_auto_still_picks_by_modality(constructions, monkeypatch):
    from veritas.models import gemini_strong, gpt_strong

    monkeypatch.setattr(extraction_module, "extraction_model", "auto")

    class Prompt:
        def __init__(self, videos):
            self._videos = videos

        def has_videos(self):
            return self._videos

    assert extraction_module._resolve_model(Prompt(True)) is gemini_strong
    assert extraction_module._resolve_model(Prompt(False)) is gpt_strong
    assert constructions == []


def test_dating_model_is_constructed_once(constructions, monkeypatch):
    monkeypatch.setattr(retrieval_module, "dating_model", "openai:gpt-5-nano")

    models = [llm_module.resolve_model(retrieval_module.dating_model, FakeModel("d"))
              for _ in range(10)]

    assert len(set(id(m) for m in models)) == 1
    assert constructions == ["openai:gpt-5-nano"]
