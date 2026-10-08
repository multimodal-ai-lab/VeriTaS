"""A model omitted from one call due to a bad request takes part in later calls again."""

from types import SimpleNamespace

import pytest

from veritas.ensemble import Ensemble, ModelResponse


@pytest.fixture
def ensemble(monkeypatch):
    """An ensemble of three stub models, where `stub:b` fails the first call
    with an error that leads to its omission."""
    ens = Ensemble()
    for name in ["stub:a", "stub:b", "stub:c"]:
        ens.register(SimpleNamespace(specifier=name))
    calls = []

    async def fake_call_models(models, prompt, response_format=None, **kwargs):
        calls.append(list(models))
        return [ModelResponse(model=ens._members[name], output="ok",
                              error=(Exception("Audio file processing failed")
                                     if name == "stub:b" and len(calls) == 1 else None))
                for name in models]

    monkeypatch.setattr(ens, "_call_models", fake_call_models)
    ens.calls = calls
    return ens


@pytest.mark.asyncio
async def test_an_omitted_model_is_called_again_in_the_next_assessment(ensemble):
    first = await ensemble.generate("prompt")
    assert len(first) == 2
    assert ensemble.model_names == ["stub:a", "stub:b", "stub:c"]

    second = await ensemble.generate("prompt")
    assert len(second) == 3
    assert ensemble.calls[-1] == ["stub:a", "stub:b", "stub:c"]


@pytest.mark.asyncio
async def test_omission_does_not_alter_the_callers_model_list(ensemble):
    requested = ["stub:a", "stub:b"]
    await ensemble.generate("prompt", models=requested)
    assert requested == ["stub:a", "stub:b"]
