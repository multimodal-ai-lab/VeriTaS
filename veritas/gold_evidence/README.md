# Gold Evidence Reconstruction & Temporal Analysis

Reconstructs the evidence the original professional fact-check used, filters
invalid and leaked evidence, and validates whether what remains suffices to
recover the VeriTaS gold verdict — in order to answer whether evidence that only
became available *during* the fact-checking period is necessary for that.

**The gold verdict is never changed.** Nothing here writes to `verdicts` or to any
pre-existing claim field. Instances whose evidence cannot be reconstructed are
*rejected*, recorded only in the additive `claims.gold_evidence_*` columns;
`claims.dismissed` stays untouched.

See [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) for the methodological choices and
their rationale.

## Stages

| Stage | Module | What it does |
| --- | --- | --- |
| 1 | `extraction.py` | An MLLM reads the stored fact-check article and returns candidate `Evidence` items (an atomic proposition plus an independently locatable source) **and** the `VerdictRationale` that turns those propositions into a verdict. |
| 2 | `filtering.py`, `retrieval.py` | Per **source**: re-retrieve it through scrapeMM and date it (§3.1), check it still supports the proposition (§3.2), compute the two cutoffs and the verdict-leak check (§3.3). Per **item**, once its sources are dated: does the proposition rest on a change of the world that happened only after `t_c`? Tools and offline evidence skip retrieval and faithfulness; a rate-limited source defers itself. |
| 3 | `sufficiency.py`, `closeness.py` | A cross-family ensemble predicts a verdict from claim + retained evidence + rationale, and `is_close` compares it to the gold verdict. |
| — | `analysis.py` | Aggregation for the temporal analysis, including the paired recoverability test. |

`admissibility.py` holds the decision rule, `models.py` the data model, `llm.py`
the cached model resolution, and `pipeline.py` wires the stages together per claim.

## Verdict rationale

Not every verdict rests on external sources. Some claims are settled by arithmetic,
by a contradiction inside the claim itself, or by what the claim's own image plainly
shows. The **verdict rationale** captures that step: it is a multimodal text,
extracted in the same call as the evidence, that bridges the gap between the
propositions and the verdict.

It may build **only on the items marked `essential`** and on the claim itself, and
it may only combine, compare or transform what they already state — it must not
introduce a new externally verifiable factual premise (that would be evidence, and
belongs in `evidence` where it can be dated and re-checked), and it never states the
verdict. Binding it to essential evidence is what keeps it honest: the rationale
becomes invalid exactly when an essential item is lost, which is exactly when the
instance is discarded. It is handed to the sufficiency ensemble alongside
the evidence, which is what makes an **empty evidence set** analyzable instead of
looking like a failed reconstruction.

## Evidence, sources, and what disqualifies an instance

An **evidence item** is one proposition; an **evidence source** is a place that
proposition can be read. Fact-checks cite redundantly — two outlets for one fact, a
register plus a screenshot of it — so an item carries *all* the sources the article
gives for it.

The split follows the questions: everything Stage 2 asks is about a source (can it
still be retrieved, when did it become available, does it still say this?), while
the proposition and its role belong to the item. A source the article cites without
linking is kept too, with an empty locator, and is settled as `inaccessible`
straight away - unless it is a tool or offline evidence, which need no locator. An item therefore **survives as
long as one of its sources does**, and losing a source costs the reconstruction
nothing as long as another still reports the proposition.

`t_e` of an item is the **earliest** `available_since` among its sources — the moment
from which the proposition could be read somewhere. It is what the later-event check
is gated on: evidence that already existed at `t_c` cannot rest on anything that
happened afterwards, so that check runs only for items that appeared later, once per
item rather than once per source.

`role` is judged against the rationale: an item is **essential** when the rationale
breaks without the proposition it asserts. An instance is disqualified **not** when
its evidence set ends up empty, but when an essential item loses *every* source
(`essential_evidence_lost`).

The same rule decides the two conditions without asking the ensemble: if an
essential item has no source inside `E_c`, that condition cannot support the
rationale, so it is recorded as insufficient directly. That is a finding about the
claim, not a defect of the reconstruction.

## Reference times

- `t_c` = `claims.date`
- `t_f` = the latest `reviews.published` among the claim's non-dismissed reviews.
  `reviews.modified` is never used — it is the last-edit time and would push the
  cutoff arbitrarily far into the future.

A claim missing either time is rejected before Stage 1 (`no_claim_time` /
`no_fact_check_time`): without `t_f` there is no evidence cutoff, and without
`t_c` the two conditions collapse into the same set.

Claims with `t_f > t_c` are reconstructed first, ahead of released ones: they are
the only claims with a non-empty interval `t_c < t_e <= t_f`, and hence the only
ones whose outcome can differ between the two conditions.

## Source retrieval

All retrieval runs through scrapeMM — nothing fetches a URL on its own, and each
source is retrieved exactly once:

1. `retrieve(url, output_format="multimodal")`
2. publication time from the meta tags of the raw HTML that the *same* response
   carries in `response.content.html` (reuses `stage_3.extract_date_meta`)
3. only if step 2 found nothing: a cheap model reads a stated publication time off
   the retrieved content, and returns nothing rather than guessing

scrapeMM produces every format preceding the requested one, so `content.multimodal`
and `content.html` come out of one call — no second retrieval, no separate
HTML-to-sequence conversion. Retrieval is therefore not restricted to the
HTML-capable backends: sources served by an API integration (social media,
archiving services, video platforms) are retrieved like any other and, having no
HTML page, are dated in step 3.

A source that only *rate-limited* us is not inaccessible: the item is deferred for
`defer_hours` and its claim ends on status `deferred`, which a later run picks up
again. An exhausted scrapeMM quota, like an exhausted model quota, aborts the run
instead of being recorded against the claim that hit it.

## Running it

```bash
# Stages 1-3; range, limit and batch size come from gold_evidence.reconstruction
python -m scripts.gold_evidence.run_reconstruction

# Export claim-level and evidence-level results plus aggregate tables
python -m scripts.gold_evidence.run_temporal_analysis

# Figures from an exported result directory (no DB access)
python -m scripts.gold_evidence.plot_temporal_analysis exports/gold_evidence/<timestamp>
```

The reconstruction has no command-line interface: it reads
`gold_evidence.reconstruction` from `config.yaml` (`start`, `end`, `limit`,
`batch_size`, `claim_ids`, `dry_run`, ...). It is resumable — evidence already
stored in the DB is reused and claims are picked up again while their status is
incomplete (`pending`, `extracted`, `filtered`, `deferred`) — and `re_extract` /
`re_filter` force the corresponding stage to run again. `re_extract` drops the
claim's stored evidence first, so nothing survives that the new run no longer
produces.

## Reasoning traces

Model reasoning is read from the provider's dedicated field — OpenAI reasoning
items, Anthropic thinking blocks, Gemini thought parts — via
`Model.generate(..., return_reasoning=True)`, never parsed out of the answer text.
Each judgement therefore stores two distinct things:

- `justification` — the short reason the prompt asked the model to state
- `reasoning` — the trace the API reported, empty when the model reported none
  (no reasoning effort configured, or a non-reasoning model)

## Configuration

All knobs live under `gold_evidence:` in `config.yaml` — models, ensemble members
and mode, reasoning effort per task, thresholds, the undated-source policy, the
per-claim budgets and the concurrency limits. Every key has a default, so an
un-updated `config.yaml` still works.

The concurrency limits nest: `claim_concurrency` claims are reconstructed at once
and each filters up to `evidence_concurrency` of its evidence items at once, so the
peak number of in-flight retrievals and LLM calls is their product.
`analysis_concurrency` is separate and applies to the temporal-analysis script.

Two settings change the reported numbers and should be stated in any write-up:

- `proximity_threshold` (default `0.3`) — how close the predicted verdict must be.
- `undated_policy` (default `tool_only`) — how sources without a determinable
  publication time are treated. The default keeps the source kinds that are not
  publications at all (`tool`, `offline`), which need neither a locator nor a
  timestamp; `permissive` keeps every undated source and `strict` keeps none.
  Re-running with `strict` gives a one-flag sensitivity analysis.

## Database

Additive only:

- `evidence` — one row per reconstructed evidence item (proposition, role, and the
  outcome derived from its sources), with a `full_evidence` JSONB blob as the
  authoritative representation of the item itself.
- `evidence_sources` — one row per source, with everything Stage 2 decided about it
  and a `full_source` blob. The web UI joins the two: `evidence` for the
  proposition and its outcome, `evidence_sources` for everything it filters on.
- `verdict_rationales` — one row per fact-checking article that yielded a rationale.
  Kept apart from `evidence` because it asserts no externally verifiable fact, so
  none of the Stage-2 criteria apply to it.
- `gold_evidence_results` — one row per (claim, condition, ensemble mode), holding
  the predicted verdict, every ensemble member's rating, stated justification,
  reasoning trace and answer text, the property distances and the closeness
  decision.
- `claims.gold_evidence_status` / `_reason` / `_updated_at` — the per-instance
  outcome.

## Browsing the results

A read-only web UI shows the processed claims, every stored detail of each
reconstructed evidence item (with its media rendered inline) and the aggregate
statistics:

```bash
docker compose up webui   # -> http://localhost:8080
```

It is a separate service that never imports this package and never writes to the
database. See [`webui/README.md`](../../webui/README.md).

## Tests

```bash
pytest tests/gold_evidence
```

These are pure: no database, no network, no LLM calls. They cover the
admissibility rule, the closeness function, response parsing for all three LLM
stages, DB round-tripping, prompt rendering (including that the faithfulness
prompt cannot see the claim or the verdict), the aggregation, the retrieval order,
deferral of rate-limited sources, the propagation of quota and rate-limit errors,
and the pipeline's accept/reject logic.
