# Gold Evidence Reconstruction — Design Decisions

Methodological decisions taken while implementing the pipeline, with the reasoning
behind each, for reporting in the paper. Purely engineering choices (file layout,
CLI flags, caching) are omitted.

Notation: `t_c` = claim release time, `t_f` = fact-check publication time,
`t_e` = time the evidence became publicly available. The interval under study is
`t_c < t_e <= t_f`.

---

## 1. Evidence unit and representation

**Decision.** An evidence item is one *atomic factual proposition* attributed to one
*independently locatable source*, represented natively as an `ezMM.MultimodalSequence`
with images, video and audio inline. The source carries its name, a type from a
closed vocabulary, a stable locator, a proximity level, and the content as
re-scraped at analysis time.

**Rationale.** Separating the proposition from the source lets faithfulness be
tested (does the source still support the proposition?) independently of
availability (can the source still be reached?) and of timing (when did it become
available?). Bundling them would conflate three distinct failure modes that the
paper needs to report separately.

**Consequence for reporting.** Counts are per *proposition*, not per URL. One
source can contribute several evidence items; the exports carry both counts.

## 2. Source proximity as a three-level scale

**Decision.** `PRIMARY` (first-hand material produced by a party directly involved:
the original post, an eyewitness recording, an official register, a tool's own
output), `SECONDARY` (reporting, analysis or aggregation of primary material),
`TERTIARY` (compilations of secondary material).

**Rationale.** The distinction is standard in source criticism and is what makes
"the fact-checker reached the original source" measurable. It also gives the
sufficiency prompt an explicit instruction to weight primary material higher,
rather than leaving that to model idiosyncrasy.

## 3. Evidence role as a three-level scale

**Decision.** `ESSENTIAL` (the verdict does not hold without it), `AUXILIARY`
(corroborates or refines essential evidence), `BACKGROUND` (context only, no
probative weight).

**Rationale.** Reconstruction recall is not uniform: losing a background item is
inconsequential, losing an essential one should predict a failed reconstruction.
Reporting the role distribution of *rejected* items therefore says more about the
severity of evidence decay than a raw rejection rate does.

## 4. Extraction operates on the stored fact-check article only

**Decision.** Stage 1 reads the article already scraped by the main pipeline and
performs no web access. Every extracted locator must occur *verbatim* in that
article, and locators on the fact-checker's own domain are discarded unless the
item is a tool.

**Rationale.** Two threats are closed at once. Requiring the locator to appear in
the article makes hallucinated sources impossible rather than merely unlikely — the
same device stage 4 already uses for claim appearances. Excluding the
fact-checker's own domain enforces the requirement that evidence point at the
original source rather than at the fact-check quoting it, which would otherwise
allow the verdict in through the back door. Tools are exempt because a
fact-checker may host the tool it used.

**Limitation to state.** Evidence the fact-checker used but did not link is
unrecoverable by construction. Reconstruction recall is therefore bounded by the
outlet's citation practice, and the reported evidence sets are lower bounds.

## 5. Reference times come from the stored records; `t_f` is the latest review

**Decision.** `t_c` is `claims.date`. `t_f` is the **latest** `reviews.published`
among the claim's non-dismissed reviews — the publication time the fact-checking
organization itself states, as already stored in the DB. When a claim has several
reviews, evidence is extracted from all of them (capped at two, earliest first).

**Rationale.** The fact-checking period of a claim ends when the last professional
fact-check of it appears; taking the earliest would truncate the interval under
study and understate how much evidence falls inside it. Extracting from several
reviews increases recall, since outlets cite partly disjoint sources.

**`modified` is not a fallback.** `reviews.modified` records the last time the
publisher edited the article, which can be years after publication. Using it when
`published` is missing would push the evidence cutoff arbitrarily far into the
future and admit material the fact-checker never saw. A review without a
`published` time simply does not contribute to `t_f`.

**Missing reference times reject the instance.** If no review carries a
`published` time, `t_f` is unknown and there is no evidence cutoff at all; if
`claims.date` is missing, `t_c` is unknown and the two conditions collapse into
the same set. Either case is rejected before Stage 1 runs
(`no_fact_check_time` / `no_claim_time`), rather than analyzed without a cutoff.

**Guard.** If `t_f < t_c` (inconsistent metadata), `t_f` is clamped to `t_c`, so
the studied interval is empty rather than negative. This biases towards "no gain
from the fact-checking period", i.e. the conservative direction.

## 6. Faithfulness is validated blind

**Decision.** The faithfulness validator receives *only* the proposition and the
freshly retrieved source content. It never sees the claim, the gold verdict, or the
fact-checker's reasoning. It answers on the codebase's existing
category-plus-certainty scale, yielding a score in `{-1, -2/3, -1/3, 0, 1/3, 2/3, 1}`.

**Rationale.** Circular validation is prevented structurally rather than by
instruction: information that is not in the prompt cannot leak into the judgement.
The shared rating scale keeps faithfulness commensurable with every other rating
in VeriTaS.

**Threshold.** An item is retained at `>= 1/3`, i.e. at least "entailed (rather
uncertain)". `0` denotes "neutral / not enough information" and is *not* retained:
a source that no longer says anything about the proposition is no longer evidence
for it.

**Two records per judgement.** `justification` is the short reason the prompt asks
the model to state; `reasoning` is the model's actual reasoning trace, read from
the dedicated field the provider's API returns it in (see §18). They are stored
separately because they are different evidence about the judgement: the first is
what the model claims, the second is how it got there.

## 7. Verdict leakage is decided by the registry, later events by one LLM call

**Decision.** The two purely temporal comparisons (`t_e <= t_c`, `t_e <= t_f`) are
computed, not predicted. Whether a source is a professional fact-check is decided by
VeriTaS' own publisher registry: a locator whose publisher is an IFCN or EFCSN
signatory is recorded as `SourceKind.FACT_CHECK` and rejected as a verdict leak. The
one remaining judgement — does the source report a post-`t_c` event that changed the
factual basis? — is a single LLM call, made only for items inside the studied
interval `t_c < t_e <= t_f`, where the dates alone cannot rule that contamination out.

**Rationale.** The registry is a curated, auditable list, so the leakage criterion
does not depend on a model's opinion about what counts as a fact-check. Restricting
the LLM call to the interval also removes the cost for the majority of items: an
item that already existed when the claim was made cannot report on anything that
happened afterwards, and one that postdates `t_f` is out regardless.

**Answer polarity.** The prompt asks for the field that is stored — `later_event` —
rather than for its complement, so no answer is inverted between the model and the
record. An unparseable answer leaves `later_event` at its default `false` and the
item is decided by the remaining criteria.

**Not asked: concurrency.** An earlier design also asked whether a fact-check
addressed *the same* claim, and kept non-concurrent fact-checks. That judgement was
dropped: every professional fact-check is now treated as a leak, whatever it covers
and whenever it appeared. This is the conservative direction — it can only shrink
the evidence sets — and it removes a model judgement that was hard to audit.

## 8. Admissibility is evaluated once, against the loose cutoff

**Decision.** An item is admissible iff it is accessible, faithful above threshold,
datable under the undated policy, available by `t_f`, not itself a professional
fact-check, and not a later-event report. `E_factcheck` is the admissible set;
`E_claim` is the pure sub-filter `t_e <= t_c`.

**Rationale.** `t_c <= t_f` always, and the leakage and later-event criteria are
anchored at `t_c` regardless of the cutoff, so the strict condition is a subset of
the loose one by construction. This guarantees `E_claim ⊆ E_factcheck` — the paired
comparison in §13 would be uninterpretable otherwise — and halves the filtering
cost, since no item is judged twice.

**Reason ordering.** An item violating several criteria is attributed to the first
in a fixed order (verdict leak → inaccessible → undated → unfaithful → after cutoff
→ later event), so the reported rejection reasons partition the rejected items
rather than double-counting them.

**Sources that cannot be retrieved.** Tools and offline evidence (a phone call, an
interview) are not publications: they have nothing to re-read and need not carry a
locator. Accessibility and faithfulness therefore do not apply to them, and they are
admitted on the extraction alone. Judging them by criteria they cannot satisfy would
have discarded every tool-derived finding and every fact-checker interview
categorically, rather than on their merits.

**But a known `t_e` is binding.** The temporal criteria are *not* waived: as soon as
such an item carries a publication time, it is placed on the timeline like any other
evidence — both cutoffs are computed from it, and an item inside the interval is
asked the later-event question, with the proposition and the source metadata standing
in for the content that cannot be retrieved. Only the *absence* of a date is
tolerated (§9), never a date that violates the cutoff.

## 9. Undated sources are excluded unless they are not publications

**Decision.** Default policy `tool_only`: an item whose source has no determinable
publication time is admissible only if its source kind is `TOOL` or `OFFLINE`.
Configurable to `permissive` (keep all) or `strict` (keep none, including those two).

**Rationale.** Evidence that cannot be dated cannot be shown to predate the cutoff.
Keeping it would silently inflate both evidence sets and bias the analysis toward
"gold verdict recoverable" — precisely the direction that would weaken the paper's
central claim if it were an artefact. Tools and offline evidence are exempt because
they are instruments and conversations rather than publications: a geolocation
service has no meaningful release time, and neither has a phone call to an expert.
For them, "undated" is not a gap in the record but a property of the source type,
and excluding them would measure the source type rather than the evidence.

**Consequence, and it cuts the other way.** An undated item counts as satisfying
both cutoffs (§8), so exempted tools and offline evidence enter `E_claim` as well —
including an interview the fact-checker conducted *during* the fact-checking period,
which by construction did not exist at `t_c`. This applies only while such an item
carries no date: as soon as one is known, the cutoffs are computed from it like for
any other evidence (§8). It is nonetheless the one place where the default policy is
generous towards `E_claim`, i.e. against finding that the fact-checking period was
necessary. The `strict` policy removes exactly these items and is the sensitivity
analysis to report alongside.

**Reporting.** The exports carry `n_undated` per claim and `undated_source` as a
rejection reason, so the size of this decision is quantified. Re-running with
`permissive` or `strict` gives a one-flag sensitivity analysis; reporting the
default and `strict` is advisable.

## 10. Publication time is read from the page, not inferred

**Decision.** `t_e` is taken from the page's standard publication meta tags where
present. Otherwise a cheap model reads an explicitly stated publication time off
the retrieved content, and is instructed to return "none" rather than guess. Times
of last modification, access and of the *described events* are explicitly excluded.

**Rationale.** Meta tags are the publisher's own machine-readable assertion and are
preferred over any model reading. Forbidding inference is what makes the undated
category meaningful: an item is undated because the source states no date, not
because the model was unsure.

## 11. Retrieval is HTML-first, entirely through scrapeMM

**Decision.** Every source is retrieved through scrapeMM; nothing fetches a URL on
its own. The order is: request the source in scrapeMM's `html` format; read the
publication time off that HTML's meta tags; convert the *same* HTML into the
multimodal content with scrapeMM's own `to_multimodal_sequence`; and only if the
meta tags carried no date, have a cheap model read one off the converted content.

**Rationale.** One retrieval per source settles accessibility, dating and content
together. Going through scrapeMM throughout means evidence sources are fetched by
exactly the same stack — anti-bot handling, archive resolution, media download —
that the benchmark already uses for claim appearances, so accessibility here is
comparable to accessibility there rather than being a different measurement.

**No platform fallback.** scrapeMM serves its `html` format only through the
Firecrawl and Decodo backends; sources it handles through a dedicated API
integration (social media, archiving services, video platforms) return an
unsuccessful response and are recorded as inaccessible. There is no second attempt
in scrapeMM's default `multimodal_sequence` format. Since fact-checks cite
social-media posts heavily, this biases the accessibility statistics against
exactly that source type, and the `inaccessible` rejection rate must be read with
that in mind (see the limitations below).

**Rate limits are not inaccessibility.** A source that only throttled us is left
entirely unjudged and retried after `defer_hours`; the claim it belongs to is put on
status `deferred` rather than rejected. Recording a temporary condition as a
permanent one would inflate the reported `inaccessible` share and reject instances
on an incomplete evidence set.

**Media are downloaded.** Images and video referenced by an evidence source are
retrieved and inlined, so the sufficiency validator sees what the fact-checker saw.

## 12. Sufficiency validation: ensemble, evidence-only, two modes

**Decision.** An ensemble of strong models from different families receives *only*
the claim and the retained evidence, with high reasoning effort, and predicts a
VeriTaS verdict. Two modes:

- `integrity` (default): one ensemble call predicting the claim's integrity, the
  property that is decisive for the gold verdict.
- `full`: the whole property cascade of the annotation pipeline (per-medium
  authenticity and contextualization, then veracity, then context coverage, with
  the same short-circuits), i.e. four to six times the calls.

The prompt forbids relying on recollection of the case or of any fact-check of it.
Every member's rating, stated justification, reasoning trace (§18) and complete
answer text is stored.

**Rationale.** Cross-family ensembling is the codebase's existing safeguard against
single-model idiosyncrasy and is retained here. The two modes trade cost against
resolution: `integrity` answers the recoverability question at one call per claim
per condition; `full` additionally shows *which* property fails to survive the
evidence cutoff. Storing raw responses makes the validator auditable after the
fact — necessary because it, not a human, decides which instances enter the
analysis.

**Effort scaling.** Reasoning effort is matched to task difficulty: none for
reading a date, low for faithfulness, medium for extraction and the temporal
judgements, high only for the sufficiency validator.

## 13. Closeness: maximum distance over shared properties

**Decision.**

```
is_close(predicted, gold) :≡ max over shared properties |score_pred − score_gold| ≤ θ,   θ = 0.3
```

Shared properties are those present in both verdicts; media are matched by
reference. An empty overlap is never close. Integrity is compared as the gold
verdict's own compromising property.

**Rationale.** The maximum, rather than the mean, means every property must be
recovered: an instance is not counted as reconstructed if one property is right and
another badly wrong. This is the conservative direction — it can only lower the
reported recoverability rates, so a positive finding is not an artefact of a
permissive criterion. `θ = 0.3` sits just below the `1/3` spacing of the underlying
7-point rating scale, so it tolerates uncertainty-level disagreement while
rejecting a change of certainty class.

**Float tolerance.** The comparison carries a `1e-9` tolerance so that a distance
of conceptually exactly `θ` is not rejected by floating-point representation.

## 14. Acceptance is decided by `E_factcheck` alone

**Decision.** The instance is *accepted* iff the ensemble recovers the gold verdict
from `E_factcheck`. The `E_claim` outcome is recorded for every instance but never
affects acceptance.

**Rationale.** Sufficiency asks whether the reconstruction captured what the
fact-checker actually had. `E_claim` failing is not a reconstruction failure — it is
the phenomenon under study, namely that the fact-checking period contributed
necessary evidence. Letting it gate acceptance would delete exactly the instances
that carry the paper's finding.

## 15. The gold verdict is never revised, and rejected instances are not dismissed

**Decision.** Nothing in the pipeline writes to `verdicts` or to any pre-existing
claim field. Rejection is recorded in three additive columns
(`gold_evidence_status`, `gold_evidence_reason`, `gold_evidence_updated_at`); the
claim's `dismissed` flag is untouched.

**Rationale.** The ensemble is a sufficiency validator, not a second annotator. An
instance we cannot reconstruct evidence for is still a valid VeriTaS claim with a
valid gold verdict; it is only unusable *for this analysis*. Keeping the two
judgements in separate columns preserves the benchmark and lets the analysis be
re-run under different thresholds without any destructive step.

## 16. Claim selection: released first

**Decision.** Candidates are non-dismissed claims with a current verdict whose
reviews completed the verdict stage (6, or 7 for rectified claims). Released claims
are processed first; once none remain in the requested range, the remaining
verdict-complete claims follow. The date range is user-specified.

**Rationale.** Released claims are the published benchmark, so results on them are
externally checkable and directly citable. Requiring a completed verdict stage
excludes claims whose gold verdict is not yet final, which would otherwise be
compared against a moving target.

## 17. Reported statistics

Per claim: `t_c`, `t_f`, `t_f − t_c`, candidate/admissible/in-window counts,
undated count, per-condition evidence-set sizes, per-condition recoverability and
maximum property distance, gold scores, status and rejection reason.

Per evidence item: source name, type, proximity, role, locator, `t_e`, `t_e − t_c`,
`t_e − t_f`, in-window flag, modality, faithfulness, the temporal flags,
admissibility, rejection reason and whether the item is currently deferred.

Aggregate: share of claims with post-claim/pre-fact-check evidence; number and
fraction of items in the interval; distributions of `t_e − t_c`, `t_e − t_f`,
`t_f − t_c`; source type, proximity, role and modality distributions; share of
rejected candidates and the reason breakdown; share of instances rejected for
insufficient evidence.

**Headline test.** The `E_claim × E_factcheck` recoverability contingency table with
a two-sided **exact McNemar test** on the discordant pairs. The paired design is the
right one because both conditions are evaluated on the same claim with the same
gold verdict and the same ensemble; only the evidence cutoff differs. A significant
excess of "recoverable only from `E_factcheck`" is the evidence that material
appearing *during* the professional fact-checking period is necessary to reconstruct
the gold verdict.

## 18. Reasoning is captured from the provider APIs, not from the answer text

**Decision.** Wherever a model's reasoning is recorded, it is taken from the
dedicated field the provider's API returns it in — OpenAI's reasoning items on the
Responses API, Anthropic's thinking blocks, Gemini's thought parts — and never
parsed out of the answer. `Model.generate(..., return_reasoning=True)` surfaces it
alongside the response, and the ensemble carries it on every member response.

**Rationale.** Current models do not put their reasoning in the answer; it is a
separate channel, and asking a model to restate its reasoning in the answer yields
a post-hoc rationalization rather than the trace that produced the judgement. For
an analysis whose inclusion decisions are made by an LLM ensemble, the auditable
artefact has to be the real trace.

**Consequence.** A model called without a reasoning effort, or a non-reasoning
model, reports no trace, and the field stays empty — the stated justification is
then the only record. Reasoning availability therefore varies by stage (§12:
none for dating, low for faithfulness, medium for extraction and the temporal
judgements, high for sufficiency) and by ensemble member.

**Not everywhere.** `Evidence.extraction_reasoning` remains the per-item reason
the extractor states in its JSON output: one Stage-1 call yields many evidence
items, so the call-level trace cannot be attributed to an individual item.

---

## Known limitations to state in the paper

1. **Citation-bounded recall.** Evidence the fact-checker used but did not link
   cannot be recovered (§4). All evidence-set sizes are lower bounds.
2. **Link rot is time-asymmetric.** Sources are re-retrieved today, so older claims
   lose proportionally more evidence to inaccessibility. Any trend over claim date
   must be read against the `inaccessible` rejection rate, which the exports report.
3. **Dating coverage.** `t_e` is only as good as publishers' meta tags and explicit
   datelines; the undated share (§9) bounds this and should be reported.
4. **Validator, not oracle.** Recoverability is judged by an LLM ensemble. It is
   cross-family and its reasoning traces are stored (§18), but it is not a human
   annotator, and its errors are not guaranteed independent of the claim's
   difficulty. Provider reasoning summaries are also abridged rather than
   verbatim traces, so they support auditing but not exact replay.
5. **`t_f` from review metadata.** Publication times come from the outlets' own
   metadata as stored in `reviews.published`; silent post-publication edits are
   not visible, and claims whose reviews carry no publication time are excluded
   (§5), which may not be missing at random across outlets.
6. **Sources that cannot deliver HTML are lost.** Retrieval is HTML-only (§11), so
   social-media posts, archiving services and video platforms count as inaccessible
   even when scrapeMM could reach them through an API integration. This depresses
   the admissible share for exactly the source type fact-checks cite most, and
   `inaccessible` therefore mixes genuine link rot with this backend restriction.
7. **Later-event judgement is the hardest call.** Distinguishing "the source
   describes pre-existing facts, published later" from "the source reports a new
   event that settles the claim" requires world knowledge; the stored reasoning
   makes this auditable, and a manual audit of a sample is recommended.
