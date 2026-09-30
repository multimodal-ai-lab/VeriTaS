"""Gold Evidence Reconstruction & Temporal Analysis.

Reconstructs the evidence that the original professional fact-check used to
establish its verdict - together with the rationale that turns that evidence into a
verdict - filters invalid/leaked evidence, and validates whether what remains
suffices to recover the VeriTaS gold verdict.

The gold verdict is never modified. Instances whose argument cannot be
reconstructed are *rejected*, which is recorded exclusively in the new
`claims.gold_evidence_*` columns and the new tables `evidence`,
`verdict_rationales` and `gold_evidence_results`. No pre-existing DB value is ever
overwritten.
"""

from veritas import globals

_cfg: dict = globals.get("gold_evidence") or {}


def _get(key: str, default):
    value = _cfg.get(key, default)
    return default if value is None else value


# --- Models -------------------------------------------------------------------

#: Model used for Stage 1 evidence extraction of articles without videos.
#: "auto" selects `gpt_strong` (as in stage 5).
extraction_model: str = _get("extraction_model", "auto")

#: Model used for Stage 1 evidence extraction of articles containing videos. Must be
#: able to read videos natively. "auto" selects `gemini_strong` (as in stage 5).
extraction_video_model: str = _get("extraction_video_model", "auto")

#: Model used for the Stage 2 faithfulness and temporal assessments.
filtering_model: str = _get("filtering_model", "auto")

#: Cheap model used for the (optional) LLM fallback that determines `available_since`.
dating_model: str = _get("dating_model", "auto")

#: Cheap model that trims a retrieved source down to its main content (the page's
#: own text and media, without navigation, ads, cookie notices, comments ...).
#: "auto" -> `gpt_nano`, as for the fact-checking articles in the main pipeline.
cleaning_model: str = _get("cleaning_model", "auto")

#: Ensemble members of the Stage 3 sufficiency validator. None -> the global
#: ensemble singleton, whose members are configured under `models.ensemble`.
ensemble_models: list[str] | None = _cfg.get("ensemble_models")

# --- Reasoning effort ---------------------------------------------------------
# Effort scales with task complexity: reading off a date is not reasoning-heavy,
# recovering a verdict from evidence alone is.

#: `None` means: do not pass a reasoning parameter at all (provider default).
#: Reading a publication date off a page needs none; recovering a verdict from
#: evidence alone needs the most.
reasoning_effort_extraction: str | None = _cfg.get("reasoning_effort_extraction", "medium")
reasoning_effort_dating: str | None = _cfg.get("reasoning_effort_dating", None)
reasoning_effort_cleaning: str | None = _cfg.get("reasoning_effort_cleaning", None)
reasoning_effort_faithfulness: str | None = _cfg.get("reasoning_effort_faithfulness", "low")
reasoning_effort_temporal: str | None = _cfg.get("reasoning_effort_temporal", "medium")
reasoning_effort_sufficiency: str | None = _cfg.get("reasoning_effort_sufficiency", "high")

# --- Thresholds ---------------------------------------------------------------

#: Maximum admissible distance between predicted and gold verdict (see `is_close`).
proximity_threshold: float = float(_get("proximity_threshold", 0.3))

#: Minimum faithfulness assessment for an evidence item to remain admissible.
#: 1/3 corresponds to "entailed (rather uncertain)" on the codebase's rating scale.
faithfulness_threshold: float = float(_get("faithfulness_threshold", 1 / 3))

#: Minimum extraction confidence for a candidate to enter Stage 2.
min_extraction_confidence: float = float(_get("min_extraction_confidence", 0.0))

#: How long an evidence item waits before it is retried after its source rate-limited
#: us. A throttled source is a temporary condition, not a verdict on the item.
defer_hours: int = int(_get("defer_hours", 24))

#: How to treat evidence whose source has no determinable publication time (t_e).
#: See `veritas.gold_evidence.admissibility.UNDATED_POLICIES`. The default keeps
#: only tools, because evidence that cannot be dated cannot be shown to predate
#: the cutoff, and keeping it would bias the analysis towards "recoverable".
undated_policy: str = _get("undated_policy", "tool_only")

# --- Sufficiency validator ----------------------------------------------------

#: 'integrity' -> one ensemble call per claim and condition, predicting integrity.
#: 'full'      -> the full stage-6 property cascade (authenticity, contextualization,
#:                veracity, context coverage), i.e. 4-6x the calls.
ensemble_mode: str = _get("ensemble_mode", "integrity")

#: Minimum number of ensemble members that must return a usable rating.
min_ensemble_ratings: int = int(_get("min_ensemble_ratings", 3))

# --- Budgets ------------------------------------------------------------------

max_evidence_per_claim: int = int(_get("max_evidence_per_claim", 20))
max_reviews_per_claim: int = int(_get("max_reviews_per_claim", 2))
max_source_content_length: int = int(_get("max_source_content_length", 30_000))
#: How much of a retrieved source the cleaning model is shown, in characters. Larger
#: than `max_source_content_length`: the noise it removes is what made pages long.
max_cleaning_input_length: int = int(_get("max_cleaning_input_length", 100_000))
#: At most this many of a source's media are attached to one citation - and hence
#: injected into the proposition it supports - so that media-heavy pages (galleries,
#: feeds) cannot flood the sufficiency prompt.
max_media_per_citation: int = int(_get("max_media_per_citation", 4))
max_article_length: int = int(_get("max_article_length", 50_000))

# --- Concurrency --------------------------------------------------------------
# The first two nest: `claim_concurrency` claims are reconstructed at once, and each
# of them filters up to `evidence_concurrency` of its evidence items at once, so the
# peak number of in-flight source retrievals and LLM calls is their product.

#: Claims reconstructed simultaneously (`pipeline.reconstruct_claims`).
claim_concurrency: int = int(_get("claim_concurrency", 6))

#: Evidence items of one claim filtered simultaneously (`filtering.filter_evidence`).
evidence_concurrency: int = int(_get("evidence_concurrency", 8))

#: Claims the temporal analysis processes simultaneously. Independent of the two
#: above: the analysis reads the stored results and only re-runs the sufficiency
#: ensemble when asked to (`scripts/gold_evidence/run_temporal_analysis.py`).
analysis_concurrency: int = int(_get("analysis_concurrency", 10))

#: Passed to scrapeMM when downloading evidence source media. Falls back to the
#: main pipeline's limit. Read from the config directly rather than importing
#: `veritas.pipeline`, so that the pure modules (models, admissibility, analysis)
#: stay importable without pulling in the benchmark pipeline.
max_video_size: int | None = _cfg.get("max_video_size")
if max_video_size is None:
    max_video_size = ((globals.get("pipeline") or {}).get("global") or {}).get("max_video_size")

# --- Statuses -----------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_EXTRACTED = "extracted"
STATUS_FILTERED = "filtered"
STATUS_DEFERRED = "deferred"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

#: Statuses a re-run picks up again, i.e. work that is incomplete rather than
#: decided. `deferred` is among them: the claim waits for a rate-limited source,
#: it was not rejected.
RESUMABLE_STATUSES = ["unprocessed", STATUS_PENDING, STATUS_EXTRACTED,
                      STATUS_FILTERED, STATUS_DEFERRED]

#: The two evidence cutoff conditions studied in the temporal analysis.
CONDITION_CLAIM = "claim"  # t_e <= t_c
CONDITION_FACT_CHECK = "fact_check"  # t_e <= t_f
CONDITIONS = (CONDITION_CLAIM, CONDITION_FACT_CHECK)
