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
| 1 | `extraction.py` | An MLLM reads the stored fact-check article and returns candidate `Evidence` items (an atomic proposition plus a `Citation` of every source that reports it) **and** the `VerdictRationale` that turns those propositions into a verdict. |
| 2 | `filtering.py`, `retrieval.py`, `dating.py`, `cleaning.py` | Per **source**, i.e. once per URL: re-retrieve it through scrapeMM, date it (§3.1), trim it to its main content, and check the publisher registry for a verdict leak (§3.3). Per **citation**: check the source still supports the proposition (§3.2), name the source's media that show it, and compute the two cutoffs of the citing claim (§3.3). Per **item**, once its citations are dated: does the proposition rest on a change of the world that happened only after `t_c`? Tools and offline evidence skip retrieval and faithfulness; a rate-limited or Archive.today-gated source defers every item citing it. |
| 3 | `sufficiency.py`, `closeness.py` | A cross-family ensemble predicts a verdict from claim + retained evidence + rationale, and `is_close` compares it to the gold verdict. |
| — | `analysis.py` | Aggregation for the temporal analysis, including the paired recoverability test. |

`admissibility.py` holds the decision rule, `models.py` the data model, `llm.py`
the cached model resolution, and `pipeline.py` wires the stages together per claim.

## Verdict rationale

Not every verdict rests on external sources. Some claims are settled by arithmetic,
by a contradiction inside the claim itself, or by what the claim's own image plainly
shows. The **verdict rationale** captures that step: it is a multimodal text,
extracted in the same call as the evidence, that bridges the gap between the claim
and its evidence on the one side and the verdict on the other — arithmetic, logical
argumentation, comparisons, visual analysis.

It carries **reasoning and commonsense knowledge only**. It must not restate the
evidence, must not refer to specific evidence items, and must not introduce any
externally available information (that would be evidence, and belongs in
`evidence` where it can be dated and re-checked); it never states the verdict.
Keeping the evidence out of it is what keeps it honest: nothing the rationale says
can leak a proposition into a condition that lacks it, so it stays valid whichever
items a condition supplies. If the verdict follows trivially from the evidence, the
rationale says just that. It is handed to the sufficiency ensemble alongside the
evidence, which is what makes an **empty evidence set** analyzable instead of
looking like a failed reconstruction.

## Evidence, sources, and roles

An **evidence item** is one proposition; a **source** is a publication, identified
by its URL; a **citation** records that an item cites a source for its proposition.
Fact-checks cite redundantly — two outlets for one fact, a register plus a screenshot
of it — so an item carries a citation of *every* source the article gives for it.

The three levels follow the questions Stage 2 asks:

- **Source** — can it still be retrieved, what does it say, when did it become
  available, is it a professional fact-check? These depend on the URL alone, so a
  source is **global**: stored once, however many items of however many claims cite
  it, and retrieved and dated exactly once.
- **Citation** — the source's name and kind *as cited*, its proximity to the
  proposition, whether it still supports *this* proposition (faithfulness), and
  where it sits relative to the citing claim's cutoffs.
- **Item** — the proposition, its role, and whether it rests on a later event.

A source the article cites without linking becomes a citation without a source and
is settled as `locator_missing` straight away - unless it is a tool or offline
evidence, which need no locator. That is kept apart from `inaccessible`, which means
a retrieval was tried and failed (link rot), not that the article never said where
the material came from. An item therefore **survives as long as one of its
citations does**, and losing a source costs the reconstruction nothing as long as
another still reports the proposition.

Sources are **write-once**: a retrieved source is reused, never refetched
implicitly, because other claims' decisions rest on what was stored. `re_retrieve`
fetches the sources of the processed claims anew; every citation judged against the
previous content then counts as *stale* and is judged again the next time its claim
is processed.

`t_e` of an item is the **earliest** `available_since` among its citations — the moment
from which the proposition could be read somewhere. It is what the later-event check
is gated on: evidence that already existed at `t_c` cannot rest on anything that
happened afterwards, so that check runs only for items that appeared later, once per
item rather than once per citation.

`role` is assigned once the article's evidence list is complete, by asking what
removing the item from that list would do to the gold verdict:

- **key** — establishes a central factual premise underlying the verdict; removing
  it likely breaks the verdict.
- **auxiliary** — corroborates, qualifies, or strengthens the main justification
  without being its principal evidential basis; removing it would not break the
  verdict.
- **background** — context for understanding the claim or its circumstances,
  without directly contributing to the justification.

Only one or two items — or none — are typically `key`. The roles are **recorded,
not enforced**: an instance is not discarded because a key item lost every citation,
and a condition lacking a key item is not recorded as insufficient in advance.
Whether the remaining evidence still carries the verdict is decided by one judge
only, the sufficiency ensemble. The exports report how many key items each instance
lost (`n_key_lost`) and how many each condition lacked (`n_key_missing`), and
`share_accepted_despite_lost_key` says how often the ensemble recovered the verdict
anyway.

Neither is an empty extraction a rejection. An article that yields no evidence and
no rationale leaves the ensemble to judge the claim on its own. Only two outcomes of
Stage 1 are not analyzed: a claim without any readable fact-checking article is
rejected (`no_fact_check_article`), and a claim for which the extractor *failed* on
every article (an error or an unusable response - not an empty result) stays
`pending` (`extraction_failed`) and is extracted again on the next run.

Rows written before the rename carry the role `essential` and the rejection reason
`essential_evidence_lost`. They are left untouched in the database and read as `key`
and `key_evidence_lost` (`models.LEGACY_ROLE_ALIASES`, `LEGACY_REASON_ALIASES`).
`key_evidence_lost` and `nothing_extracted` are no longer assigned; claims processed
before carry them until they are reconstructed again.

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
source is retrieved exactly once, however many citations point at it. Before a
source is retrieved, its stored state is re-read under a per-URL lock, so a URL that
another claim - even one running concurrently - already retrieved is adopted
instead of fetched again:

1. `retrieve(url, output_format="multimodal")`
2. publication time from the markup of the raw HTML that the *same* response
   carries in `response.content.html` (`dating.extract_publication_time`): JSON-LD,
   Open Graph, microdata and other publication meta tags - never modified or
   updated fields, and never a partial or implausible date
3. only if step 2 found nothing: a cheap model reads a stated publication time off
   the retrieved content, shown together with the HTML's remaining `<time>` markup,
   and returns nothing rather than guessing

scrapeMM produces every format preceding the requested one, so `content.multimodal`
and `content.html` come out of one call — no second retrieval, no separate
HTML-to-sequence conversion. Retrieval is therefore not restricted to the
HTML-capable backends: sources served by an API integration (social media,
archiving services, video platforms) are retrieved like any other and, having no
HTML page, are dated in step 3.

A source that only *rate-limited* us is not inaccessible: it is deferred for
`defer_hours`, and so is every item citing it; their claims end on status
`deferred`. **Every reconstruction run starts by retrying all deferred claims** -
independent of its date range, claim IDs and `limit`, so also with `limit: 0` - and
clears the deferral windows of the sources they wait for, so that the retry actually
retrieves them. A source that is still throttled simply defers again. An exhausted
scrapeMM quota, like an exhausted model quota, aborts the run instead of being
recorded against the claim that hit it.

A source gated behind Archive.today's access check is deferred the same way -
scrapeMM buffers the URL and answers it from a persistent cache once a human
passes the CAPTCHA, which can take much longer than `defer_hours`. Once that has
happened, `python -m scripts.retry_deferred_archive_today` clears the deferral
of every affected source (and every affected Stage 4 appearance) and retries it
right away, instead of waiting for the next scheduled run to notice.

## Source media in the evidence

Fact-checks often only *link* the post, photo or video a proposition is about, so
the medium is in the source, not in the article Stage 1 reads. Stage 1 therefore
stays article-only, and the media are added in Stage 2, where a proposition first
meets its sources:

1. Each retrieved source is **cleaned** once (`cleaning.py`): a cheap model names
   the line ranges of the page's main content - title, byline, text, its own media
   and captions - and everything else (navigation, ads, consent notices, related
   articles, comments) is cut. Lines are never rewritten, so media references
   survive verbatim. The result is `Source.cleaned_content`; `raw_content` is kept,
   and a cleaning that fails or would lose substance leaves the raw content in use.
   Sources retrieved before are cleaned from their stored content, without a refetch.
2. The **faithfulness judge**, which sees one proposition next to one source's
   (cleaned) content, also names the source's media that show what the proposition
   states (`Faithfulness.media`, at most `max_media_per_citation`). A reference that
   does not occur in the content the judge was shown is dropped.
3. Once a citation is **admissible**, its media references are put in front of the
   proposition (`models.prepend_media`), so every later step - the later-event check,
   the sufficiency ensemble, the web UI - sees the proposition with its media.
   Prepending is idempotent: judging an item again does not repeat a medium.

Because the media become part of the proposition, they are shown in every condition
the item is in. An item in `E_c` through an early citation also shows the media of a
source that only appeared later (see DESIGN_DECISIONS §24).

Existing citations get media when they are next judged (`re_filter: True`); no
source needs to be fetched again.

## Running it

```bash
# Stages 1-3; range, limit and batch size come from gold_evidence.reconstruction
python -m scripts.gold_evidence.run_reconstruction

# Export claim-level and evidence-level results plus aggregate tables
python -m scripts.gold_evidence.run_temporal_analysis

# Figures from an exported result directory (no DB access)
python -m scripts.gold_evidence.plot_temporal_analysis exports/gold_evidence/<timestamp>

# After solving Archive.today's access check, retry everything deferred behind it
python -m scripts.retry_deferred_archive_today
```

The reconstruction has no command-line interface: it reads
`gold_evidence.reconstruction` from `config.yaml` (`start`, `end`, `limit`,
`batch_size`, `claim_ids`, `dry_run`, ...). It is resumable — evidence already
stored in the DB is reused and claims are picked up again while their status is
incomplete (`pending`, `extracted`, `filtered`, `deferred`) — and `re_extract` /
`re_filter` force the corresponding stage to run again. `re_extract` drops the
claim's stored evidence and citations first, so nothing survives that the new run
no longer produces; the global sources stay. `re_filter` judges every citation
again against the stored sources, without refetching them; `re_retrieve` refetches
them as well.

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

Model *versions* are set once for all of VeriTaS, in the top-level `models:`
section (`gpt_strong`, `gpt_cheap`, `gpt_nano`, `gpt_transcribe`, `text_embedder`,
`gemini_strong`, `gemini_cheap`, and the members of the global `ensemble`). A
`gold_evidence` model key set to `auto` uses those: `extraction_model` → `gpt_strong`,
`extraction_video_model` (articles with videos; must read video natively) →
`gemini_strong`, `filtering_model` → `gpt_strong`, `dating_model` → `gpt_nano`, and
`ensemble_models: null` → `models.ensemble`. Note that `models.ensemble` lists its
members explicitly, so changing `gpt_strong` alone does not change the ensemble.

The scrapeMM server is set in the top-level `scrapemm:` section (`api_url`,
`api_key`) and configured for the running process on import. The environment
variables `SCRAPEMM_API_URL` / `SCRAPEMM_API_KEY` take precedence over it.
`python -m scripts.configure_scrapemm` saves the same values as scrapeMM's
persistent client configuration, for tools that do not import VeriTaS.

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
- `sources` — one row per URL, global: the retrieved content, its cleaned main
  content (`cleaned_content`), `available_since` and how it was dated,
  accessibility, the registry's fact-check finding and any deferral, with a
  `full_source` blob. Unique on a SHA-1 of the normalized URL.
- `citations` — one row per (evidence item, source): name, kind and proximity as
  cited, faithfulness, the cutoffs of the citing claim and the admissibility, with a
  `full_citation` blob. A citation of a source the article never located has no
  `source_id`.
- `citation_rows` — a read-only view joining the two into one flat row per
  citation, which the web UI reads.
- `evidence_sources` — the per-item predecessor of `sources` and `citations`. Kept
  as it is, but no longer read or written. A claim whose evidence was stored in that
  format has items without citations, and is re-extracted when next processed.
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
