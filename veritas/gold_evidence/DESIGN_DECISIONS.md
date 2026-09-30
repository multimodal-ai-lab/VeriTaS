# Gold Evidence Reconstruction — Design Decisions

Methodological decisions taken while implementing the pipeline, with the reasoning
behind each, for reporting in the paper. Purely engineering choices (file layout,
CLI flags, caching) are omitted.

Notation: `t_c` = claim release time, `t_f` = fact-check publication time,
`t_e` = time the evidence became publicly available. The interval under study is
`t_c < t_e <= t_f`.

---

## 1. Evidence unit and representation

**Decision.** An evidence item is one *atomic factual proposition*, represented
natively as an `ezMM.MultimodalSequence` with images, video and audio inline,
together with a **citation of every source that reports it**. Three records, each
holding what depends on it alone:

- a **source** is a publication identified by its URL — its content as re-scraped
  at analysis time, its publication time, whether it can still be retrieved and
  whether it is a professional fact-check;
- a **citation** relates an item to a source — the source's name and type (from a
  closed vocabulary) as the article cites it, its proximity to the proposition,
  whether it still supports the proposition, and where it sits relative to the
  citing claim's cutoffs;
- the **item** holds the proposition, its probative role and the later-event
  judgement.

**Why the split.** Every filtering question is about either the page (can it still
be retrieved, when did it become available?) or the page *as cited for a
proposition* (does it still say this?), while the proposition and its role are
properties of the evidence. Putting the source-level state on the item forced one
item per source, which made a fact-check's redundant citations look like
independent single points of failure. With citations as their own records, the item
is the aggregation layer: it survives as long as one of its citations does.

**Why sources are global.** Fact-checks cite the same page for several propositions,
and different claims — a rectified claim and its original in particular — cite the
same pages. Stored per citation, one URL was retrieved and dated as often as it was
cited, and could come out accessible for one item and inaccessible for another, or
dated differently by the model fallback (§10). Nothing about retrieval or dating
depends on the citing claim, so a source is stored once per URL and every citation
shares the outcome. The cutoffs *do* depend on the claim (`t_c`, `t_f`) and are
therefore computed per citation; so is faithfulness, which is relative to the
proposition.

**Write-once.** Other claims' decisions rest on what was stored about a source, so
a retrieved source is never refetched implicitly: a re-filtering re-judges the
citations against the stored content. An explicit re-retrieval replaces it, and
every citation judged before the new retrieval then counts as stale and is judged
again when its claim is next processed. Concurrent claims citing one URL retrieve it
once: its stored state is re-read under a per-URL lock before any retrieval.

**Rationale.** Separating the proposition from the source lets faithfulness be
tested (does the source still support the proposition?) independently of
availability (can the source still be reached?) and of timing (when did it become
available?). Bundling them would conflate three distinct failure modes that the
paper needs to report separately.

**Consequence for reporting.** Counts exist at three levels and mean different
things: a *source* failure is link rot or a leak, a *citation* rejection can also be
unfaithfulness or a cutoff, while an *evidence* loss is a hole in the argument. The
exports are per citation, with the item's outcome and the `source_id` on every row;
`share_sources_rejected_fatally` separates rejections that cost the item from those
that did not, and `n_distinct_sources` states how many distinct URLs were cited.

## 2. Source proximity as a three-level scale

**Decision.** `PRIMARY` (first-hand material produced by a party directly involved:
the original post, an eyewitness recording, an official register, a tool's own
output), `SECONDARY` (reporting, analysis or aggregation of primary material),
`TERTIARY` (compilations of secondary material).

**Rationale.** The distinction is standard in source criticism and is what makes
"the fact-checker reached the original source" measurable. It also gives the
sufficiency prompt an explicit instruction to weight primary material higher,
rather than leaving that to model idiosyncrasy.

## 3. Evidence role as a three-level scale, relative to the gold verdict

**Decision.** Roles are assigned once the evidence list of an article is complete:
`KEY` (establishes a central factual premise underlying the gold verdict; removing
it from the list likely breaks the verdict), `AUXILIARY` (corroborates, qualifies,
or strengthens the main justification without being its principal evidential basis;
removing it would not break the verdict), `BACKGROUND` (context for understanding
the claim or its circumstances, without directly contributing to the justification).
The prompt states that only one or two items — or none — are typically `key`.

**Rationale.** Reconstruction recall is not uniform: losing a background item is
inconsequential, losing a key one should predict a failed reconstruction. Reporting
the role distribution of *rejected* items therefore says more about the severity of
evidence decay than a raw rejection rate does.

**Why relative to the verdict, not to the rationale.** An earlier definition — "the
rationale breaks without it" — was judged against a rationale the extractor had just
written from all its strong evidence, and the prompt told it to mark whatever that
rationale leaned on as essential. Every item the rationale mentioned became
essential, so a fact-check that proved one point three independent ways had three
single points of failure. The rationale no longer contains evidence at all (§19), so
the role is judged against the verdict directly: removing a key item from the list
likely breaks it; removing an auxiliary one does not, whether because another item
establishes the same point or because it only corroborates. Losing a key item is
what disqualifies (§20); losing the others is not decided by a rule at all but by the
sufficiency ensemble (§12), which decides acceptance anyway.

**Why not record the alternatives.** Which item can replace which is not stored —
neither as a group label (§20) nor as alternative sufficient sets. The one case the
roles cannot decide on their own — every alternative for a point lost at once —
reaches the ensemble with whatever evidence is left, which is the question the
ensemble exists to answer.

**Errors fail safe.** An auxiliary item mislabeled key makes the rule stricter than
necessary. A key item mislabeled auxiliary costs an ensemble call, and nothing
leaks: the rationale contains no evidence (§19). The extractor states in each item's
`reasoning` how it bears on the verdict and the outcome of the leave-one-out test,
so the labels can be audited.

**Legacy rows.** Items extracted before the rename carry the role `essential`, and
instances rejected for losing one carry the reason `essential_evidence_lost`. The
stored rows are not rewritten; the model (`EvidenceRole._missing_`,
`normalize_reason`) and the web UI's queries read them as `key` and
`key_evidence_lost`.

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

## 7. Verdict leakage is decided by the registry, later events by one call per item

**Decision.** The two temporal comparisons (`t_e <= t_c`, `t_e <= t_f`) are computed
per source, not predicted. Whether a source is a professional fact-check is decided
by VeriTaS' own publisher registry: a locator whose publisher is an IFCN or EFCSN
signatory is recorded as `SourceKind.FACT_CHECK` and rejected as a verdict leak. The
one judgement left to a model — does the evidence rest on a change of the world that
happened only after `t_c`? — is asked **once per evidence item**, and only for items
whose `t_e` lies after `t_c`.

**Rationale.** The registry is a curated, auditable list, so the leakage criterion
does not depend on a model's opinion about what counts as a fact-check. The
later-event question, in contrast, is about the *proposition*: several sources
reporting the same fact either all describe a state that already held or all describe
a later change. Asking per source multiplied the cost by the number of sources and
invited inconsistent answers about one and the same fact. And evidence that could
already be read at `t_c` cannot rest on anything that happened afterwards, so the
gate on `t_e > t_c` removes the call for the majority of items.

**The confusion this check exists to avoid.** "The information appeared later" is not
"the facts changed later". A fact-checking unit publishing a correction after `t_c`
about a video filmed before it has reported on the world, not changed it; a company
filing for bankruptcy after `t_c`, cited as proof that it was bankrupt at `t_c`, has
changed it. The prompt names that distinction first, gives worked examples of both
directions — including an authority issuing a finding, which is the case models get
wrong most often — and asks the decisive question counterfactually: *had this
evidence been available at `t_c`, would the factual basis it describes already have
been true?*

**Answer polarity.** The prompt asks for the field that is stored —
`change_detected` — rather than for its complement, so no answer is inverted between
the model and the record. An unparseable answer leaves it at its default `false` and
the item is decided by the remaining criteria.

**Where the outcome lives.** On `Evidence.later_event`, not on a source: it condemns
the proposition however many sources still report it, so no amount of redundancy
saves an item that rests on a later event.

**Not asked: concurrency.** An earlier design also asked whether a fact-check
addressed *the same* claim, and kept non-concurrent fact-checks. That judgement was
dropped: every professional fact-check is now treated as a leak, whatever it covers
and whenever it appeared. This is the conservative direction — it can only shrink
the evidence sets — and it removes a model judgement that was hard to audit.

## 8. Admissibility is evaluated once, against the loose cutoff

**Decision.** A *citation* is admissible iff its source is accessible, datable under
the undated policy, available by the citing claim's `t_f` and not a professional
fact-check, and the source supports the proposition above the faithfulness
threshold. An *item* is admissible iff at least one of its citations is and it does
not rest on a later event (§7). `E_f` is the admissible set; `E_c` is the
pure sub-filter `t_e <= t_c`.

**Rationale.** `t_c <= t_f` always, and the leakage and later-event criteria are
anchored at `t_c` regardless of the cutoff, so the strict condition is a subset of
the loose one by construction. This guarantees `E_c ⊆ E_f` — the paired
comparison in §13 would be uninterpretable otherwise — and halves the filtering
cost, since no item is judged twice.

**Reason ordering.** An item violating several criteria is attributed to the first
in a fixed order (verdict leak → inaccessible → undated → unfaithful → after cutoff
→ later event), so the reported rejection reasons partition the rejected items
rather than double-counting them.

**`t_e` of an item.** The earliest `available_since` among its citations: from that
moment on the proposition could be read somewhere. Condition membership is still
decided per citation (an item enters `E_c` through whichever source predates `t_c`),
but the later-event gate and the exports use this single item-level `t_e`.

**Sources that cannot be retrieved.** Tools and offline evidence (a phone call, an
interview) are not publications: they have nothing to re-read and need not carry a
locator. Accessibility and faithfulness therefore do not apply to them, and they are
admitted on the extraction alone. Judging them by criteria they cannot satisfy would
have discarded every tool-derived finding and every fact-checker interview
categorically, rather than on their merits.

**Sources the article never located.** A fact-check sometimes cites material it
neither links nor names. Such a source is *extracted* - with an empty locator, and
with the extraction reasoning stating that the article provided none - and then
settled as `inaccessible` without any retrieval attempt, unless it is a tool or
offline evidence, which are not expected to carry a locator anyway. Dropping those
sources at extraction time would have hidden them: as records they show up in the
rejection statistics, which is where "the fact-check did not say where this came
from" belongs.

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
both cutoffs (§8), so exempted tools and offline evidence enter `E_c` as well —
including an interview the fact-checker conducted *during* the fact-checking period,
which by construction did not exist at `t_c`. This applies only while such an item
carries no date: as soon as one is known, the cutoffs are computed from it like for
any other evidence (§8). It is nonetheless the one place where the default policy is
generous towards `E_c`, i.e. against finding that the fact-checking period was
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

## 11. One retrieval per URL, entirely through scrapeMM

**Decision.** Every source is retrieved through scrapeMM; nothing fetches a URL on
its own, and a URL is retrieved once however many citations point at it (§1). The order is: request the source in scrapeMM's `multimodal` output format;
read the publication time off the meta tags of the raw HTML that the *same* response
carries; and only if the meta tags carried no date, have a cheap model read one off
the retrieved content.

**Rationale.** One retrieval per source settles accessibility, dating and content
together. scrapeMM produces the formats preceding the requested one along the way,
so asking for `multimodal` yields the page's raw HTML for free whenever the used
method had access to it — the dating never costs a second scrape. Going through
scrapeMM throughout means evidence sources are fetched by exactly the same stack —
anti-bot handling, archive resolution, media download — that the benchmark already
uses for claim appearances, so accessibility here is comparable to accessibility
there rather than being a different measurement.

**No backend restriction.** Every scrapeMM backend serves the `multimodal` format,
so the method is left to scrapeMM's per-domain choice and sources handled by a
dedicated API integration (social media, archiving services, video platforms) are
retrieved like any other. Those have no HTML page, hence no meta tags: they are
dated by the model or stay undated, which the undated share (§9) already reports.

**Rate limits are not inaccessibility.** A source that only throttled us is left
entirely unjudged and retried after `defer_hours`; the claim it belongs to is put on
status `deferred` rather than rejected. Recording a temporary condition as a
permanent one would inflate the reported `inaccessible` share and reject instances
on an incomplete evidence set.

**Archive.today's access check is deferred, not inaccessible, too.** Archive.today
gates snapshot pages with a CAPTCHA that scrapeMM does not solve; it buffers the
gated URL and answers it from a persistent cache once a human passes the check
(possibly hours later - a captcha is solved on human time, not on `defer_hours`'
schedule). `retrieval.SourceRetrieval.gated` marks this outcome and `filter_source`
defers the source exactly like a rate limit. `scripts/retry_deferred_archive_today.py`
clears the deferral and retries immediately once the check has been passed, instead
of waiting for `defer_hours` to elapse on its own.

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

## 14. Acceptance is decided by `E_f` alone

**Decision.** The instance is *accepted* iff the ensemble recovers the gold verdict
from `E_f`. The `E_c` outcome is recorded for every instance but never
affects acceptance.

**Rationale.** Sufficiency asks whether the reconstruction captured what the
fact-checker actually had. `E_c` failing is not a reconstruction failure — it is
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

## 16. Claim selection: a non-empty interval first, then released

**Decision.** Candidates are non-dismissed claims with a current verdict whose
reviews completed the verdict stage (6, or 7 for rectified claims). They are
processed in this order: claims whose fact-check appeared *after* the claim
(`t_f > t_c`) first, then released claims, then by claim ID. The date range is
user-specified.

**Rationale.** A claim with `t_f = t_c` has an empty interval `t_c < t_e <= t_f`, so
no evidence can fall into it and `E_c = E_f` by construction: the claim
contributes a concordant pair to the paired test no matter what the evidence says,
and reconstructing it cannot change the headline result. Spending the API budget on
claims with a non-empty interval therefore buys information, while the others only
buy denominator. Claims whose stored `t_f` *precedes* `t_c` are treated like the
empty case, since §5 clamps them to `t_c`.

Released claims come second because they are the published benchmark, so results on
them are externally checkable and directly citable. Requiring a completed verdict
stage excludes claims whose gold verdict is not yet final, which would otherwise be
compared against a moving target.

**Not applied to the analysis.** The temporal-analysis script switches this priority
off. It reads what the reconstruction stored, and with a `--limit` the priority would
select precisely the claims that carry window evidence — an artefact in every share
the analysis reports.

## 17. Reported statistics

Per claim: `t_c`, `t_f`, `t_f − t_c`, candidate/admissible/in-window counts,
undated count, rationale count, key evidence items and how many of them were
lost, how many auxiliary items were lost, per-condition evidence-set sizes,
per-condition recoverability and maximum property distance, gold scores, status and
rejection reason.

Per citation: the proposition and role of the item it belongs to, whether that item
survived, how many citations it has, plus the source's ID, name, type, proximity,
locator, `t_e` and how it was dated, `t_e − t_c`, `t_e − t_f`, in-window flag,
faithfulness, the temporal flags, admissibility, rejection reason and whether the
source is currently deferred.

Aggregate: share of claims with post-claim/pre-fact-check evidence; number and
fraction of items in the interval; distributions of `t_e − t_c`, `t_e − t_f`,
`t_f − t_c`; source type, proximity, role and modality distributions; share of
rejected candidates and the reason breakdown; share of instances rejected for
insufficient evidence.

**Headline test.** The `E_c × E_f` recoverability contingency table with
a two-sided **exact McNemar test** on the discordant pairs. The paired design is the
right one because both conditions are evaluated on the same claim with the same
gold verdict and the same ensemble; only the evidence cutoff differs. A significant
excess of "recoverable only from `E_f`" is the evidence that material
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

## 19. The verdict rationale

**Decision.** Stage 1 extracts, in the same call as the evidence, a *verdict
rationale*: the reasoning that bridges the gap between the propositions and the
verdict. It is a `MultimodalSequence`, stored in its own table, and handed to the
sufficiency ensemble alongside the evidence.

**Rationale.** A fact-check does not only gather facts, it argues from them, and
some arguments need no external fact at all: an arithmetic error, a date that
contradicts another date in the same claim, a caption that contradicts what the
image shows. Without the rationale those instances look evidence-less and would be
discarded as unreconstructible — which would systematically remove exactly the
claims whose verification is *not* retrieval-bound, biasing the corpus towards
search-solvable claims.

**Two constraints make it safe to pass on.** The rationale must not introduce a new
externally verifiable factual premise — such a premise is evidence, and belongs in
`evidence` where it is retrieved, dated and checked for faithfulness — and it must
not state the verdict, which is what the ensemble has to recover on its own. The
prompt states both explicitly, and the judge prompt tells the ensemble that the
rationale carries no facts of its own and is to be followed only as far as the
evidence supports it.

**No evidence in it.** The rationale carries reasoning and commonsense knowledge
only: it must not restate the evidence, must not refer to specific evidence items,
and must not introduce externally available information; if the verdict follows
trivially from the evidence, it says so. This closes the leak that would otherwise
affect `E_c`: whatever items a condition lacks, the rationale cannot carry their
content into it, so the ensemble never sees a rationale that rests on evidence the
condition it is judging does not have, and whenever both conditions reach the
ensemble they see the same rationale. The extractor assigns the roles *before*
writing the rationale — the output schema lists the evidence first — so the role
judgement is not shaped by the argument it is about to write.

**Residual risk, to report.** The rationale is still the fact-checker's own
reasoning, so the ensemble is not reasoning wholly independently: a rationale that
smuggles in a premise, or telegraphs the verdict through its phrasing, makes recovery
easier than the evidence alone would. Both constraints are prompt-enforced rather
than machine-checkable. The rationale is stored verbatim so a sample can be audited,
and `gold_evidence_results.with_rationale` records which predictions saw one, so the
recoverability rates can be reported with and without them.

## 20. Redundancy lives inside an evidence item

**Decision.** All sources a fact-check gives for one proposition are cited by one
evidence item. The item is discarded only when every one of its citations was, and
an instance is disqualified only when a *key* item is discarded.

**Rationale.** Fact-checks cite redundantly on purpose: two outlets for one fact, a
register plus a screenshot of it. Treating each citation as a separate evidence item
made each of them a single point of failure, and biased the result — the more
carefully a fact-checker corroborated a proposition, the more chances the instance
had to be discarded. Modelling sources as alternatives for one proposition restores
the intended semantics: the *proposition* must survive, not each of its sources.

**Why not a grouping label.** An earlier design kept one item per source and tagged
interchangeable items with a shared "corroboration group". That worked, but it
carried the grouping as metadata a model had to assign consistently, scoped to a
single article, and it left `available_since`, `accessible`, `faithfulness` and the
temporal validation on an item that could have several of each. Making citations
first class removes the label, the scoping question and the duplication at once;
making sources global (§1) removes the remaining duplication across items.

**Merging across articles.** Two fact-checks of the same claim that state the same
proposition are merged into one item carrying both sets of citations, and the stricter
role wins: if one article's verdict breaks without the proposition, losing it
breaks that article's case.

**Redundancy between propositions.** Different propositions that prove the same
point — a dated news photo, a reverse-image hit, the photographer's statement — are
not merged: they are different facts, each checked against its own sources. Their
redundancy is expressed by the role instead (§3): none of them is key, so
losing any of them costs the instance nothing unless the ensemble finds that what
remains no longer carries the verdict.

## 21. A condition without its key evidence needs no ensemble call

**Decision.** If a key item has no source inside a condition's evidence set, that
condition is recorded as insufficient (`is_close = False`) without querying the
ensemble.

**Rationale.** A key item establishes a central premise of the verdict (§3), so a
condition that cannot supply one cannot support the verdict the fact-check reached.
Asking the ensemble anyway would measure how well four models guess a verdict with a
premise missing — an answer that says nothing about whether the evidence of that
period sufficed. It also saves the majority of `E_c` calls on precisely the claims
the analysis is about.

**Only key items short-circuit.** An auxiliary item that appeared during the
fact-checking period does not carry the verdict by definition, so it must not make
`E_c` insufficient: such an `E_c` goes to the ensemble with whatever evidence
predates `t_c`. Under the earlier, rationale-relative definition, a redundant item
did short-circuit, and every such claim landed in the "recoverable only from `E_f`"
cell — inflating exactly the effect the paper reports.

**Consequence for reporting.** `E_c` failures split into two kinds: a key item that
only appeared during the fact-checking period (recorded in `error`), and a genuine
ensemble failure to recover the verdict. Both count as "not recoverable from `E_c`"
in the headline test; the distinction is available in the stored results.

## 22. An empty evidence set is a result, not a failure

**Decision.** A claim whose reconstruction yields no evidence is analyzed normally,
as long as a rationale carries the argument. The instance is rejected only if Stage 1
produced nothing at all (`nothing_extracted`) or if a key evidence item lost every
source (`key_evidence_lost`).

**Rationale.** The earlier rule — reject when the admissible set is empty — conflated
"the fact-checker needed no external source" with "we failed to reconstruct the
sources". The first is a legitimate, and interesting, category of instance; the
second is a failure. The rejection reasons now separate them, and the exports carry
`share_claims_without_admissible_evidence` so the size of the first category is
visible rather than hidden in a rejection count.

## 23. A rectified claim is assembled counterfactually, not reconstructed

**Decision.** Rectified claims — the corrected variants produced to balance the
dataset, which share their original's fact-checking articles — enter the
reconstruction like any other claim. Stage 1, however, is told that the article
checked a *different* claim, is shown both, and is asked to assemble the case that
bears on the rectified claim instead of reproducing the fact-checker's own
reasoning. It states the difference between the two claims first, extracts only
evidence that speaks to the rectified one, and writes the rationale about it. For
an original claim the prompt renders byte for byte as it did before.

**Rationale.** Without this, three things go wrong, in increasing severity. The
prompt opens on a false premise ("this article establishes a verdict about the
Claim shown below"), which the model has to silently reconcile. Only part of the
article's evidence bears on the corrected statement, so the candidate set is
padded with propositions that exist only to refute the original. And the rationale
— the worst of the three — would argue *against* the original claim, while the gold
verdict of a rectified claim is intact by construction: `validate_verdict` dismisses
any rectified claim that does not come out positive. Since the rationale is handed
to the Stage 3 ensemble as a Reasoning Aid (§19), reconstructing the article's
argument would point the ensemble at the opposite of the gold verdict for the whole
intact half of the dataset.

**Why it also distorts the roles.** `key` is defined relative to the verdict
(§3). An extraction scoped to the wrong claim yields the wrong key set, and
that set decides whether an instance is disqualified (§22) and whether a condition
is sent to the ensemble at all (§21). The error would not stay local to the prompt;
it would reach the headline numbers.

**Verdict leakage.** The rectified prompt forbids stating or implying what the
fact-checker concluded about the claim they checked. That conclusion is the verdict
of a different assertion, and it is the *opposite* of this claim's gold verdict, so
reproducing it would be a leak in the sense of §7 — arriving through the rationale
rather than through a source.

**Not a reconstruction.** For an original claim, Stage 1 recovers a case the
fact-checker actually made. For a rectified claim no such case exists: the
fact-checker never assembled evidence for the corrected statement. The instance
therefore answers a slightly different question — *could the fact-checker's findings
have supported the corrected statement?* rather than *did the evidence they cited
suffice?* — and a thin or empty evidence set is a correspondingly ordinary outcome
here (§22).

**Where it does not apply.** Stage 3 needs no equivalent treatment: it never sees
the article, only the evidence, the rationale and the claim. Once extraction is
scoped to the right claim there is nothing left to disambiguate downstream.

**Limitation to state.** The two populations are not strictly comparable and should
be reported separately; the analysis rows carry `is_rectified` for exactly that.

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
6. **Sources without an HTML page cannot be dated from meta tags.** Sources served
   by scrapeMM's API integrations (social-media posts, archiving services, video
   platforms) are retrieved like any other (§11) but carry no meta tags, so their
   `t_e` rests on the model reading a stated date off the content, or they stay
   undated. The undated share is therefore not uniform across source types.
7. **Rectified claims are a different question.** For corrected claim variants the
   fact-checker never built a case, so their reconstruction is counterfactual
   (§23): it measures whether the article's findings *could* support the corrected
   statement. Evidence sets are expected to be thinner than for original claims,
   and the two populations should be reported separately rather than pooled.
8. **Roles are a model judgement.** The roles (§3) are assigned
   by the extractor, per article. When every alternative for a point is lost at
   once, no rule catches it and the instance rests on the ensemble's judgement of
   the remaining evidence. The exports report `n_auxiliary_lost` and
   `share_accepted_despite_lost_evidence`, so the size of that reliance is visible.
   Extractions made under earlier definitions (`essential`, read as `key`) and
   their rationales, which could still restate evidence, should be re-extracted
   before results are pooled.
9. **Later-event judgement is the hardest call.** Distinguishing "the source
   describes pre-existing facts, published later" from "the source reports a new
   event that settles the claim" requires world knowledge; the stored reasoning
   makes this auditable, and a manual audit of a sample is recommended.
