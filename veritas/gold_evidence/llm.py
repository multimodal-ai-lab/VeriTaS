"""Model resolution for the Gold Evidence pipeline.

`veritas.models.init_model` constructs a fresh provider - and with it a fresh
API client and connection pool - on every call. The stages here resolve their
model per evidence item (twice per item in Stage 2), so resolving through this
module instead keeps exactly one instance per specifier alive for the process.

The models the codebase already exposes as singletons (`gpt_strong`,
`gemini_strong`, `gpt_nano`, ...) are passed in as defaults and reused as they are.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.models import Model

#: Config value meaning "let the stage pick a sensible default".
AUTO = "auto"


@lru_cache(maxsize=None)
def get_model(specifier: str) -> "Model":
    """Returns the model for the given specifier, constructing it at most once.

    Safe to call from concurrent coroutines: there is no `await` between the
    cache lookup and the insertion, so the event loop cannot interleave two
    constructions of the same specifier."""
    from veritas.models import init_model

    return init_model(specifier)


def resolve_model(specifier: str | None, default: "Model") -> "Model":
    """The configured model, or `default` when the config says `auto` (or nothing).

    `default` is expected to be one of the module-level singletons in
    `veritas.models`, so both branches return a reused instance."""
    if specifier and specifier != AUTO:
        return get_model(specifier)
    return default


def clear_cache() -> None:
    """Drops the cached models. Only needed by tests that change the config."""
    get_model.cache_clear()
