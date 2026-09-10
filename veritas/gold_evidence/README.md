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
| 1 | `extraction.py` | An MLLM reads the stored fact-check article and returns candidate `Evidence` items: an atomic proposition plus an independently locatable source. |
| 2 | `filtering.py`, `retrieval.py` | Per item: re-retrieve the source through scrapeMM and date it (§3.1), check the source still supports the proposition (§3.2), check the cutoff, verdict leakage and later-event contamination (§3.3). Tools and offline evidence skip retrieval and faithfulness, but are still validated temporally once they carry a `t_e`; a rate-limited source defers the item. |
| 3 | `sufficiency.py`, `closeness.py` | A cross-family ensemble predicts a verdict from claim + retained evidence only, and `is_close` compares it to the gold verdict. |
| — | `analysis.py` | Aggregation for the temporal analysis, including the paired recoverability test. |

`admissibility.py` holds the decision rule, `models.py` the data model, `llm.py`
the cached model resolution, and `pipeline.py` wires the stages together per claim.

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

All retrieval runs through scrapeMM — nothing fetches a URL on its own:

1. `retrieve(url, format="html")`
2. publication time from the HTML's meta tags (reuses `stage_3.extract_date_meta`)
3. that same HTML → `MultimodalSequence` via scrapeMM's `to_multimodal_sequence`
4. only if step 2 found nothing: a cheap model reads a stated publication time off
   the converted content, and returns nothing rather than guessing

scrapeMM serves `format="html"` only through Firecrawl and Decodo, and step 1 is
restricted to those two. Sources it handles through an API integration (social
media, archiving services, video platforms) come back unsuccessful and are recorded
as inaccessible — there is no second attempt in the default `multimodal_sequence`
format. See the limitations in `DESIGN_DECISIONS.md`.

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

- `evidence` — one row per reconstructed evidence item, with flat columns for
  querying (including `deferred_until`) and a `full_evidence` JSONB blob as the
  authoritative representation.
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
