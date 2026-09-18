/* The claim detail view: claim, gold verdict, timeline and every evidence item. */

import { api } from '../api.js';
import { renderMultimodal } from '../media.js';
import { E_c, E_f, math, typeset } from '../math.js';
import {
    badge, copyText, date, decimal, el, highlightJson, humanize, infoTip, num, PROXIMITY_STYLE,
    REASON_HELP, ROLE_STYLE, SOURCE_KIND_ICON, STATUS_STYLE, truncate, truncated,
} from '../util.js';

export async function renderClaim(root, claimId) {
    root.append(el('div', { class: 'skeleton', style: { height: '160px', marginBottom: '16px' } }));
    root.append(el('div', { class: 'skeleton', style: { height: '320px' } }));

    const detail = await api.claim(claimId);
    root.replaceChildren();

    root.append(head(detail));
    root.append(claimPanel(detail));
    root.append(timelinePanel(detail));
    root.append(rationalePanel(detail));
    root.append(sufficiencyPanel(detail));
    root.append(evidencePanel(detail));

    // Deep link: `#/claims/12/evidence/34` scrolls to and opens that item.
    const target = window.location.hash.match(/\/evidence\/(\d+)/);
    if (target) focusEvidence(Number(target[1]));
}

/** Highlights one evidence card and scrolls it into view. */
export function focusEvidence(evidenceId) {
    const node = document.getElementById(`evidence-${evidenceId}`);
    if (!node) return;
    // A deep link points at one item, so open it rather than leaving the reader
    // to expand the card they were just sent to.
    node.expand?.();
    node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    node.classList.add('hot');
    setTimeout(() => node.classList.remove('hot'), 2000);
}

/* -------------------------------------------------------------------- head */

function head(detail) {
    const { claim, neighbours } = detail;
    const style = STATUS_STYLE[claim.status] ?? { tone: 'plain', icon: 'fa-circle' };

    return el('div', { class: 'detail-head rise' }, [
        el('div', { class: 'titles' }, [
            el('a', { class: 'eyebrow', href: '#/claims' },
                [el('i', { class: 'fa-solid fa-arrow-left' }), ' Back to claims']),
            el('h1', {}, [`Claim #${claim.id}`]),
            el('div', { class: 'chips', style: { marginTop: '10px' } }, [
                badge(claim.status ?? 'unprocessed', style),
                claim.reason
                    ? badge(claim.reason, { tone: 'plain', icon: 'fa-circle-info', title: REASON_HELP[claim.reason] ?? '' })
                    : null,
                claim.released ? badge('released', { tone: 'info', icon: 'fa-box-open' }) : null,
                claim.is_rectified ? badge('rectified', { tone: 'plain', icon: 'fa-pen-nib' }) : null,
                claim.language ? badge(claim.language, { tone: 'plain', icon: 'fa-language' }) : null,
            ]),
        ]),
        el('div', { class: 'spacer', style: { flex: 1 } }),
        el('div', { class: 'chips' }, [
            neighbours?.previous_id
                ? el('a', { class: 'btn', href: `#/claims/${neighbours.previous_id}` },
                    [el('i', { class: 'fa-solid fa-chevron-left' }), 'Previous'])
                : null,
            neighbours?.next_id
                ? el('a', { class: 'btn', href: `#/claims/${neighbours.next_id}` },
                    ['Next', el('i', { class: 'fa-solid fa-chevron-right' })])
                : null,
        ]),
    ]);
}

/* ------------------------------------------------------------------- claim */

function claimPanel(detail) {
    const { claim, verdict, reviews } = detail;

    const facts = el('dl', { class: 'kv' });
    const fact = (label, value, title) => {
        facts.append(el('dt', { title: title ?? null }, [].concat(label)));
        facts.append(el('dd', {}, [value]));
    };
    fact([math('t_c'), ' (claim)'],
        el('span', { text: date(claim.t_c, { time: true }) }), 'claims.date');
    fact([math('t_f'), ' (fact-check)'],
        el('span', { text: date(claim.t_f, { time: true }) }),
        'latest published time among the non-dismissed reviews');
    fact([math('t_f - t_c', 't_f − t_c')], el('span', {
        text: claim.t_c && claim.t_f
            ? `${decimal((new Date(claim.t_f) - new Date(claim.t_c)) / 86400000, 1)} days`
            : '—',
    }), 'length of the fact-checking period');
    fact('Evidence', el('div', { class: 'chips' }, [
        badge(`${num(claim.n_evidence)} candidates`, { tone: 'plain', icon: 'fa-layer-group' }),
        badge(`${num(claim.n_admissible)} admissible`, { tone: 'ok', icon: 'fa-circle-check' }),
        claim.n_inadmissible ? badge(`${num(claim.n_inadmissible)} discarded`, { tone: 'bad', icon: 'fa-circle-xmark' }) : null,
        claim.n_unfiltered ? badge(`${num(claim.n_unfiltered)} unfiltered`, { tone: 'warn', icon: 'fa-hourglass-half' }) : null,
        claim.n_in_window ? badge(`${num(claim.n_in_window)} in window`, { tone: 'warn', icon: 'fa-clock' }) : null,
    ]));
    fact('Last processed', el('span', { text: date(claim.updated_at, { time: true }) }),
        'claims.gold_evidence_updated_at');

    return el('section', { class: 'section fade' }, [
        el('div', { class: 'grid cols-2' }, [
            el('div', { class: 'card' }, [
                el('div', { class: 'card-head' }, [
                    el('i', { class: 'fa-solid fa-quote-left', style: { color: 'var(--accent)' } }),
                    el('h2', { text: 'Claim' }),
                    el('span', { class: 'hint', text: claim.content?.n_media ? `${claim.content.n_media} media` : 'text only' }),
                ]),
                el('div', { class: 'card-body' }, [
                    // One medium gets the large treatment; several tile instead,
                    // so a claim with five images does not become a long column.
                    renderMultimodal(claim.content, {
                        textClass: 'claim-quote',
                        wide: (claim.content?.n_media ?? 0) <= 1,
                    }),
                ]),
            ]),
            el('div', {}, [
                el('div', { class: 'card' }, [
                    el('div', { class: 'card-head' }, [
                        el('i', { class: 'fa-solid fa-clock', style: { color: 'var(--accent)' } }),
                        el('h2', { text: 'Reference times' }),
                    ]),
                    el('div', { class: 'card-body' }, [facts]),
                ]),
                verdictPanel(verdict),
                reviewsPanel(reviews),
            ]),
        ]),
    ]);
}

function verdictPanel(verdict) {
    if (!verdict) return null;
    const full = verdict.full_verdict ?? {};
    const rating = (name, value) => {
        if (!value) return null;
        const score = value.score ?? value;
        const tone = score >= 0.334 ? 'ok' : score <= -0.334 ? 'bad' : 'warn';
        return el('div', { class: 'bar-row', style: { marginBottom: '6px' } }, [
            el('span', { class: 'lbl', text: humanize(name) }),
            el('span', { class: 'track' }, [
                el('span', {
                    class: 'fill',
                    style: { width: `${((score + 1) / 2) * 100}%`, '--tone': `var(--${tone})` },
                }),
            ]),
            el('span', { class: 'num', text: decimal(score, 2) }),
        ]);
    };

    const body = el('div', {}, [
        rating('veracity', full.veracity ?? (verdict.veracity !== null ? { score: verdict.veracity } : null)),
        rating('context coverage', full.context_coverage ?? (verdict.context_coverage !== null ? { score: verdict.context_coverage } : null)),
        verdict.integrity !== null && verdict.integrity !== undefined
            ? rating('integrity', { score: verdict.integrity }) : null,
    ]);

    const explanations = [];
    for (const key of ['veracity', 'context_coverage']) {
        const explanation = full[key]?.explanation;
        if (explanation) {
            explanations.push(el('details', { class: 'disclose' }, [
                el('summary', {}, [
                    el('i', { class: 'fa-solid fa-comment-dots lead' }),
                    `${humanize(key)} explanation`,
                    el('i', { class: 'fa-solid fa-chevron-right chev' }),
                ]),
                el('div', { class: 'content' }, [el('div', { class: 'prose', text: explanation })]),
            ]));
        }
    }

    return el('div', { class: 'card', style: { marginTop: '16px' } }, [
        el('div', { class: 'card-head' }, [
            el('i', { class: 'fa-solid fa-award', style: { color: 'var(--accent)' } }),
            el('h2', { text: 'Gold verdict' }),
            infoTip('The verdict the professional fact-checks established. The reconstruction reads it, compares against it, and never writes to it.'),
        ]),
        el('div', { class: 'card-body' }, [body]),
        ...explanations,
        (verdict.media_verdicts ?? []).length
            ? el('details', { class: 'disclose' }, [
                el('summary', {}, [
                    el('i', { class: 'fa-solid fa-photo-film lead' }),
                    `Media verdicts (${verdict.media_verdicts.length})`,
                    el('i', { class: 'fa-solid fa-chevron-right chev' }),
                ]),
                el('div', { class: 'content' }, [mediaVerdicts(verdict.media_verdicts)]),
            ])
            : null,
    ]);
}

function mediaVerdicts(mediaVerdicts) {
    const payload = {
        text: '',
        segments: [],
        media: mediaVerdicts.map((entry) => entry.media).filter(Boolean),
    };
    const container = el('div', {}, [renderMultimodal(payload, { wide: false })]);
    const table = el('table', { class: 'fields' });
    for (const entry of mediaVerdicts) {
        table.append(el('tr', {}, [
            el('td', { class: 'key', text: entry.reference }),
            el('td', { class: 'val' }, [
                el('div', { class: 'chips' }, [
                    badge(`authenticity ${decimal(entry.authenticity?.score, 2)}`, { tone: 'plain' }),
                    badge(`contextualization ${decimal(entry.contextualization?.score, 2)}`, { tone: 'plain' }),
                ]),
            ]),
        ]));
    }
    container.append(table);
    return container;
}

/** The publisher's own rating. Some publishers state a whole sentence, so the
 *  badge shows the first `RATING_LENGTH` characters and the rest on hover. */
const RATING_LENGTH = 40;

function ratingBadge(rating) {
    const text = String(rating).trim();
    if (text.length <= RATING_LENGTH) {
        return badge(text, { tone: 'accent', icon: 'fa-gavel' });
    }
    return el('span', {
        class: 'badge accent tip',
        'data-tip': text,
        tabindex: '0',
    }, [
        el('i', { class: 'fa-solid fa-gavel' }),
        `${text.slice(0, RATING_LENGTH).trimEnd()}…`,
    ]);
}


function reviewsPanel(reviews) {
    if (!reviews?.length) return null;
    const list = el('div', {});
    for (const review of reviews) {
        list.append(el('div', { style: { padding: '10px 0', borderTop: '1px solid var(--border)' } }, [
            el('div', { class: 'chips' }, [
                badge(review.raw_publisher_name ?? 'unknown publisher', { tone: 'plain', icon: 'fa-building-columns' }),
                review.raw_rating ? ratingBadge(review.raw_rating) : null,
                badge(date(review.published), { tone: 'plain', icon: 'fa-calendar' }),
                review.dismissed ? badge('dismissed', { tone: 'bad', icon: 'fa-ban', title: review.dismissed_reason ?? '' }) : null,
            ]),
            el('div', { style: { marginTop: '6px', fontSize: '.85rem' } }, [
                el('a', { href: review.url, target: '_blank', rel: 'noreferrer noopener' },
                    [truncate(review.article_title || review.url, 90)]),
            ]),
            review.article_id
                ? el('div', { style: { color: 'var(--muted)', fontSize: '.78rem', marginTop: '3px' },
                    text: `article #${review.article_id} · ${num(review.article_length)} chars extracted` })
                : null,
        ]));
    }
    return el('div', { class: 'card', style: { marginTop: '16px' } }, [
        el('div', { class: 'card-head' }, [
            el('i', { class: 'fa-solid fa-newspaper', style: { color: 'var(--accent)' } }),
            el('h2', { text: 'Professional fact-checks' }),
            el('span', { class: 'hint', text: `${reviews.length}` }),
        ]),
        el('div', { class: 'card-body', style: { paddingTop: '0' } }, [list]),
    ]);
}

/* ---------------------------------------------------------------- timeline */

function timelinePanel(detail) {
    const timeline = detail.timeline ?? {};
    const points = timeline.points ?? [];
    if (!points.length && !timeline.t_c) return el('div');

    const stamps = [
        ...points.map((point) => new Date(point.available_since).getTime()),
        timeline.t_c ? new Date(timeline.t_c).getTime() : null,
        timeline.t_f ? new Date(timeline.t_f).getTime() : null,
    ].filter((value) => value !== null && !Number.isNaN(value));

    const low = Math.min(...stamps);
    const high = Math.max(...stamps);
    const span = high - low || 1;
    // 6% padding on both sides so markers at the extremes stay readable.
    const position = (value) => 6 + ((new Date(value).getTime() - low) / span) * 88;

    const track = el('div', { class: 'timeline-track' }, [el('div', { class: 'timeline-line' })]);

    if (timeline.t_c && timeline.t_f) {
        const left = position(timeline.t_c);
        track.append(el('div', {
            class: 'timeline-window',
            style: { left: `${left}%`, width: `${Math.max(position(timeline.t_f) - left, 0.4)}%` },
            title: 'Studied interval: t_c < t_e ≤ t_f',
        }));
    }
    if (timeline.t_c) {
        track.append(el('div', {
            class: 'timeline-marker', 'data-label': 't_c',
            style: { left: `${position(timeline.t_c)}%` },
            title: `t_c = ${date(timeline.t_c, { time: true })}`,
        }));
    }
    if (timeline.t_f) {
        track.append(el('div', {
            class: 'timeline-marker tf', 'data-label': 't_f',
            style: { left: `${position(timeline.t_f)}%` },
            title: `t_f = ${date(timeline.t_f, { time: true })}`,
        }));
    }

    points.forEach((point, index) => {
        const dot = el('div', {
            class: `timeline-dot ${point.zone}${point.admissible ? '' : ' inadmissible'}`,
            style: { left: `${position(point.available_since)}%`, '--i': index },
            'data-evidence': point.evidence_id,
            // A dot is one source; the title says whether its item survived it.
            title: `source #${point.source_id} of evidence #${point.evidence_id}`
                + ` · ${date(point.available_since, { time: true })} · ${humanize(point.zone)}`
                + `${point.admissible ? '' : point.evidence_admissible
                    ? ' · discarded, item survives' : ' · inadmissible'}`,
            onclick: () => focusEvidence(point.evidence_id),
        });
        dot.addEventListener('mouseenter', () => {
            document.getElementById(`evidence-${point.evidence_id}`)?.classList.add('hot');
        });
        dot.addEventListener('mouseleave', () => {
            document.getElementById(`evidence-${point.evidence_id}`)?.classList.remove('hot');
        });
        track.append(dot);
    });

    return el('section', { class: 'section fade' }, [
        el('div', { class: 'card' }, [
            el('div', { class: 'card-head' }, [
                el('i', { class: 'fa-solid fa-timeline', style: { color: 'var(--accent)' } }),
                el('h2', { text: 'Evidence timeline' }),
                el('span', {
                    class: 'hint',
                    text: timeline.n_undated ? `${timeline.n_undated} undated item(s) not shown` : 'every dated item',
                }),
            ]),
            el('div', { class: 'card-body timeline' }, [
                track,
                el('div', { class: 'timeline-legend' }, [
                    legendEntry('--ok', ['available before the claim (',
                        math('t_e \\le t_c', 't_e ≤ t_c'), ')']),
                    legendEntry('--warn', ['only during fact-checking (',
                        math('t_c < t_e \\le t_f', 't_c < t_e ≤ t_f'), ')']),
                    legendEntry('--bad', ['after the fact-check (',
                        math('t_e > t_f'), ')']),
                    el('span', { style: { opacity: .7 } }, ['faded = inadmissible']),
                ]),
            ]),
        ]),
    ]);
}

const legendEntry = (colour, label) =>
    el('span', {}, [el('i', { style: { background: `var(${colour})` } }), ...[].concat(label)]);

/* --------------------------------------------------------------- rationale */

/** The reasoning that bridges the evidence and the verdict. Shown next to the
 *  evidence because the roles below are judged against it: an item is essential
 *  exactly when this reasoning breaks without it. */
function rationalePanel(detail) {
    const rationales = detail.rationales ?? [];
    if (!rationales.length) return el('div');

    const section = el('section', { class: 'section fade' });
    section.append(el('h2', {}, [
        el('i', { class: 'fa-solid fa-diagram-project' }), 'Verdict rationale',
        infoTip('How the fact-check argued from its evidence to its verdict. It is reasoning, not a source: it contributes no external facts of its own.'),
    ]));

    for (const rationale of rationales) {
        const body = el('div', { class: 'card-body' }, [
            renderMultimodal(rationale.content, { textClass: 'prose-text' }),
        ]);
        if (rationale.extraction_reasoning) {
            body.append(el('div', { class: 'rationale-reasoning' }, [
                el('h4', { text: 'Extraction reasoning' }),
                el('div', { class: 'prose reasoning', text: rationale.extraction_reasoning }),
            ]));
        }
        section.append(collapsible({
            head: [
                badge(`review ${rationale.review_id ?? '—'}`, { tone: 'plain', icon: 'fa-file-lines' }),
                rationale.content?.n_media
                    ? badge(`${rationale.content.n_media} media`, { tone: 'violet', icon: 'fa-photo-film' })
                    : null,
                el('span', { class: 'spacer' }),
                el('span', { class: 'preview', text: truncate(plainText(rationale.content?.text), 90) }),
            ],
            body,
        }));
    }
    return section;
}


/** Media references are noise in a one-line preview; the body keeps them. */
const plainText = (text) =>
    String(text ?? '').replace(/<(image|video|audio):\d+>/g, '[media]').trim();


/**
 * A card whose body is hidden until its header is clicked.
 *
 * Used for the rationale, the evidence items and their sources: a claim can hold
 * dozens of each, and all of them expanded at once is unreadable. `onFirstOpen`
 * runs once, which is where a lazily fetched body gets filled in.
 */
/**
 * A card whose body slides open when the card is clicked.
 *
 * Used for the rationale, the evidence items and their sources: a claim can hold
 * dozens of each, and all of them expanded at once is unreadable. `onFirstOpen`
 * runs once, which is where a lazily fetched body gets filled in.
 *
 * The animation is a `0fr -> 1fr` grid row, so it runs to the content's own
 * height without anyone having to measure it - which matters here, because a
 * body is often filled in after the card was built.
 */
function collapsible({ head, body, open = false, className = '', id = null, attrs = {},
                       onFirstOpen = null } = {}) {
    const inner = body ?? el('div');
    // The clipping wrapper carries no padding of its own: a `0fr` grid row is
    // floored by the item's padding box, so a padded body would never collapse
    // all the way.
    const shell = el('div', { class: 'collapsible-body' }, [
        el('div', { class: 'collapsible-clip' }, [inner]),
    ]);
    const chevron = el('i', { class: 'fa-solid fa-chevron-right chev' });
    let opened = open;

    const setOpen = (nowOpen) => {
        card.classList.toggle('open', nowOpen);
        card.dataset.open = String(nowOpen);
        header.setAttribute('aria-expanded', String(nowOpen));
        // While the row is animating the body must clip; once it has settled,
        // let tooltips and sticky bits inside it escape again.
        if (!nowOpen) delete shell.dataset.settled;
        if (nowOpen && !opened) {
            opened = true;
            onFirstOpen?.(inner);
        }
    };
    const toggle = () => setOpen(!card.classList.contains('open'));

    shell.addEventListener('transitionend', (event) => {
        if (event.propertyName !== 'grid-template-rows') return;
        if (card.classList.contains('open')) shell.dataset.settled = '';
    });

    const header = el('div', {
        class: 'collapsible-head',
        role: 'button',
        tabindex: '0',
        'aria-expanded': String(open),
        onkeydown: (event) => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
        },
    }, [...[].concat(head), chevron]);

    const card = el('article', {
        ...attrs,
        class: `card collapsible ${className}`.trim(),
        id,
        'data-open': String(open),
        // Closed, the whole card is the button - there is nothing else to click.
        // Open, only the header collapses it again, so the body behaves like
        // ordinary content: text selects, nothing folds away under the cursor.
        // Either way a link, a control, a disclosure, a nested card or a
        // half-finished text selection wins over the toggle.
        onclick: (event) => {
            if (event.target.closest(NON_TOGGLING)) return;
            if (event.target.closest('.collapsible') !== card) return;
            const isOpen = card.classList.contains('open');
            if (isOpen && event.target.closest('.collapsible-head') !== header) return;
            if (window.getSelection?.()?.toString()) return;
            toggle();
        },
    }, [header, shell]);
    if (open) {
        card.classList.add('open');
        shell.dataset.settled = '';
    }

    // Exposed so deep links can open the card they scrolled to.
    card.expand = () => { if (!card.classList.contains('open')) setOpen(true); };
    return card;
}

/** Clicks on these never collapse the card they are in. */
const NON_TOGGLING = 'a, button, input, select, textarea, label, video, audio, '
    + 'details, summary, .media-figure, .ref, .tip, .raw-json, .fields';


/* ------------------------------------------------------------- sufficiency */

function sufficiencyPanel(detail) {
    const results = detail.results ?? [];
    if (!results.length) return el('div');

    const cards = results.map((result) => {
        const predicted = result.predicted_verdict ?? {};
        const members = result.member_responses ?? {};

        return el('div', { class: 'card' }, [
            el('div', { class: 'card-head' }, [
                el('i', {
                    class: `fa-solid ${result.is_close ? 'fa-circle-check' : 'fa-circle-xmark'}`,
                    style: { color: `var(--${result.is_close ? 'ok' : 'bad'})` },
                }),
                el('h3', {}, [result.condition === 'claim' ? E_c() : E_f()]),
                el('span', {
                    class: 'hint',
                    text: `${humanize(result.ensemble_mode)} · ${num(result.n_evidence)} items`,
                }),
            ]),
            el('div', { class: 'card-body' }, [
                el('dl', { class: 'kv' }, [
                    el('dt', { text: 'Verdict recovered' }),
                    el('dd', {}, [result.is_close === null
                        ? badge('no result', { tone: 'warn', icon: 'fa-question' })
                        : badge(result.is_close ? 'yes' : 'no', {
                            tone: result.is_close ? 'ok' : 'bad',
                            icon: result.is_close ? 'fa-check' : 'fa-xmark',
                        })]),
                    el('dt', { text: 'Max property diff' }),
                    el('dd', {}, [`${decimal(result.max_property_diff, 3)} (`,
                        math('theta'), ` = ${decimal(result.threshold, 2)})`]),
                    el('dt', { text: 'Property diffs' }),
                    el('dd', {
                        text: Object.entries(result.property_diffs ?? {})
                            .map(([key, value]) => `${humanize(key)} ${decimal(value, 3)}`)
                            .join(' · ') || '—',
                    }),
                    el('dt', { text: 'Ensemble' }),
                    el('dd', { text: (result.model_specifiers ?? []).join(', ') || '—' }),
                    result.error ? el('dt', { text: 'Error' }) : null,
                    result.error ? el('dd', { style: { color: 'var(--bad-ink)' }, text: result.error }) : null,
                ]),
            ]),
            Object.keys(predicted).length
                ? disclosure('fa-robot', 'Predicted verdict', rawJson(predicted))
                : null,
            Object.keys(members).length
                ? disclosure('fa-users-line',
                    `Ensemble member responses (${Object.values(members).flat().length})`,
                    rawJson(members))
                : null,
        ]);
    });

    return el('section', { class: 'section fade' }, [
        el('h2', {}, [
            el('i', { class: 'fa-solid fa-scale-unbalanced' }), 'Sufficiency validation',
            infoTip('Whether an ensemble, shown only the claim and this evidence, '
                + 'predicts a verdict close enough to the gold verdict.'),
        ]),
        el('div', { class: 'grid cols-2' }, cards),
    ]);
}

/* ---------------------------------------------------------------- evidence */

/** Whether an evidence item's proposition references a medium of that kind. */
const hasKind = (item, kind) =>
    (item.proposition?.media ?? []).some((medium) => medium.kind === kind);


function evidencePanel(detail) {
    const evidence = detail.evidence ?? [];
    const section = el('section', { class: 'section fade' });

    const filters = { view: 'all' };
    const list = el('div', { class: 'evidence-list' });

    // Each view is its own predicate, so the pill count and the filtered list can
    // never drift apart.
    const VIEWS = [
        ['all', 'All', null, () => true],
        ['admissible', 'Admissible', 'fa-circle-check', (item) => item.admissible === true],
        ['inadmissible', 'Discarded', 'fa-circle-xmark', (item) => item.admissible === false],
        ['partial', 'Lost a source', 'fa-link-slash',
            (item) => (item.sources ?? []).some((source) => source.admissible === false)],
        ['image', 'With images', 'fa-image', (item) => hasKind(item, 'image')],
        ['video', 'With videos', 'fa-film', (item) => hasKind(item, 'video')],
    ];

    const pills = el('div', { class: 'pill-group' });
    for (const [value, label, icon, predicate] of VIEWS) {
        pills.append(el('button', {
            class: filters.view === value ? 'active' : '',
            onclick: (event) => {
                filters.view = value;
                [...pills.children].forEach((child) => child.classList.remove('active'));
                event.currentTarget.classList.add('active');
                draw();
            },
        }, [
            icon ? el('i', { class: `fa-solid ${icon}` }) : null,
            ` ${label}`,
            el('span', {
                style: { opacity: .55, marginLeft: '6px' },
                text: String(evidence.filter(predicate).length),
            }),
        ]));
    }

    const draw = () => {
        const predicate = (VIEWS.find(([value]) => value === filters.view) ?? VIEWS[0])[3];
        const visible = evidence.filter(predicate);
        list.replaceChildren(...(visible.length
            ? visible.map((item, index) => evidenceCard(item, index))
            : [el('div', { class: 'empty' }, [
                el('i', { class: 'fa-solid fa-inbox' }), 'Nothing in this category.'])]));
    };

    section.append(el('h2', {}, [
        el('i', { class: 'fa-solid fa-layer-group' }), 'Reconstructed evidence',
        el('span', { class: 'hint' }, [pills]),
    ]));
    section.append(list);
    draw();
    return section;
}

/** One evidence item: the proposition, its role, and every source reporting it.
 *
 *  Collapsed by default. Collapsed, a card shows only what identifies it - the
 *  proposition with its media, the tags, the ID and the copy-link button - so a
 *  claim with dozens of items stays scannable. Expanding one fetches its full
 *  record (reasoning traces, scraped source documents, the JSONB blobs), which
 *  the claim endpoint deliberately does not ship.
 */
function evidenceCard(item, index) {
    const admissible = item.admissible;
    const roleStyle = ROLE_STYLE[item.role] ?? { tone: 'plain', icon: 'fa-circle' };
    const sources = item.sources ?? [];
    const kept = sources.filter((source) => source.admissible === true).length;

    const tags = [
        admissible === true ? badge('admissible', { tone: 'ok', icon: 'fa-circle-check' })
            : admissible === false
                ? badge(item.inadmissibility_reason ?? 'discarded',
                    { tone: 'bad', icon: 'fa-circle-xmark', title: REASON_HELP[item.inadmissibility_reason] ?? '' })
                : badge('not filtered', { tone: 'warn', icon: 'fa-hourglass-half' }),
        badge(item.role ?? 'auxiliary', roleStyle),
        item.later_event?.change_detected
            ? badge('later event', { tone: 'bad', icon: 'fa-hourglass-half',
                title: REASON_HELP.later_event ?? '' })
            : null,
        badge(sources.length === 1 ? '1 source' : `${kept} of ${sources.length} sources`, {
            tone: sources.length > 1 ? 'info' : 'plain', icon: 'fa-clone',
            title: 'Every source reports the same proposition, so the item survives '
                + 'as long as one of them does.',
        }),
        item.proposition?.is_multimodal
            ? badge(`${item.proposition.n_media} media`, { tone: 'violet', icon: 'fa-photo-film' })
            : null,
        item.dismissed ? badge('dismissed', { tone: 'bad', icon: 'fa-ban', title: item.dismissed_reason ?? '' }) : null,
    ];

    const head = [
        el('div', { class: 'evidence-tags' }, tags),
        el('span', { class: 'spacer' }),
        el('span', { class: 'eid', text: `#${item.id}` }),
        el('button', {
            class: 'icon-btn small',
            title: 'Copy a link to this evidence item',
            onclick: () => copyText(
                `${window.location.origin}/#/claims/${item.claim_id}/evidence/${item.id}`),
        }, [el('i', { class: 'fa-solid fa-link' })]),
    ];

    // Always visible: the proposition is what the card is about.
    const proposition = renderMultimodal(item.proposition, { textClass: 'proposition' });

    const body = el('div', { class: 'evidence-body' }, [
        el('div', { class: 'loading-row' }, [
            el('span', { class: 'spinner' }), el('span', { text: 'Loading the full record…' }),
        ]),
    ]);

    const card = collapsible({
        head,
        body,
        className: 'evidence',
        id: `evidence-${item.id}`,
        attrs: { 'data-admissible': String(admissible), style: { '--i': Math.min(index, 8) } },
        onFirstOpen: (node) => loadEvidenceDetail(item, node),
    });
    // The proposition sits between the header and the collapsible body.
    card.insertBefore(proposition, card.lastChild);
    card.classList.add('rise');
    return card;
}


/** Fetches the item's full record and renders it into the expanded body. */
async function loadEvidenceDetail(item, node) {
    let full = item;
    try {
        full = await api.evidenceItem(item.id);
    } catch (error) {
        node.replaceChildren(el('div', { class: 'error', style: { padding: '18px' } }, [
            el('i', { class: 'fa-solid fa-triangle-exclamation' }),
            el('span', { text: `Could not load the full record: ${error.message}` }),
        ]));
        return;
    }

    const sources = full.sources ?? [];
    node.replaceChildren();

    node.append(el('div', { class: 'evidence-facts' }, [
        el('span', { title: 't_e — the earliest time any source made the proposition available' },
            [el('i', { class: 'fa-solid fa-calendar-day' }),
                math('t_e'), ` ${date(full.available_since, { time: true })}`]),
        el('span', { title: 'Extraction confidence (Stage 1)' },
            [el('i', { class: 'fa-solid fa-wand-magic-sparkles' }),
                `confidence ${decimal(full.extraction?.confidence, 2)}`]),
    ]));

    if (full.extraction?.reasoning) {
        node.append(disclosure('fa-wand-magic-sparkles', 'Extraction reasoning',
            el('div', { class: 'prose reasoning', text: full.extraction.reasoning })));
    }

    if (sources.length) {
        node.append(el('div', { class: 'source-head' }, [
            el('i', { class: 'fa-solid fa-clone' }),
            sources.length === 1 ? '1 source' : `${sources.length} sources`,
        ]));
        node.append(el('div', { class: 'source-grid' },
            sources.map((source) => sourceCard(source))));
    }

    node.append(disclosure('fa-table-list', 'All stored fields', fieldTable(full.columns)));
    node.append(disclosure('fa-code', 'Raw evidence object (full_evidence)',
        rawJson(full.full_evidence)));

    typeset(node);
}


/** The part of a URL after the host, shortened - the host is shown separately. */
function pathOf(locator) {
    try {
        const url = new URL(locator);
        const rest = `${url.pathname}${url.search}`.replace(/^\/$/, '');
        return truncate(decodeURI(rest), 38);
    } catch {
        return '';
    }
}


/** One source of an evidence item, collapsed to its identity and verdict. */
function sourceCard(source) {
    const proximityStyle = PROXIMITY_STYLE[source.source?.proximity] ?? { tone: 'plain', icon: 'fa-circle' };

    const head = [
        el('i', {
            class: `fa-solid ${SOURCE_KIND_ICON[source.source?.kind] ?? 'fa-circle-question'} kind`,
        }),
        el('span', { class: 'name' }, [truncated(source.source?.name ?? 'unknown source', 34)]),
        source.admissible === true ? badge('admissible', { tone: 'ok', icon: 'fa-circle-check' })
            : source.admissible === false
                ? badge(source.inadmissibility_reason ?? 'discarded',
                    { tone: 'bad', icon: 'fa-circle-xmark',
                        title: REASON_HELP[source.inadmissibility_reason] ?? '' })
                : badge(source.deferred_until ? 'deferred' : 'not filtered',
                    { tone: 'warn', icon: 'fa-hourglass-half',
                        title: source.deferred_until
                            ? `Waiting out a rate limit until ${date(source.deferred_until, { time: true })}`
                            : '' }),
        el('span', { class: 'spacer' }),
    ];

    const body = el('div', { class: 'card-body' });

    body.append(el('div', { class: 'chips' }, [
        badge(humanize(source.source?.kind), { tone: 'plain' }),
        badge(source.source?.proximity ?? 'secondary', proximityStyle),
    ]));

    // The locator is the one thing a reader most often wants to act on, so it is
    // a button rather than a line of small print.
    body.append(source.source?.locator
        ? el('a', {
            class: 'btn source-link', href: source.source.locator,
            target: '_blank', rel: 'noreferrer noopener', title: source.source.locator,
        }, [
            el('i', { class: 'fa-solid fa-arrow-up-right-from-square' }),
            el('span', { class: 'host', text: source.source.domain ?? 'open source' }),
            el('span', { class: 'path', text: pathOf(source.source.locator) }),
        ])
        : el('span', { class: 'locator muted', text: 'no locator' }));

    body.append(el('div', { class: 'evidence-facts' }, [
        el('span', { title: 't_e — when this source became publicly available' },
            [el('i', { class: 'fa-solid fa-calendar-day' }),
                math('t_e'), ` ${date(source.available_since, { time: true })}`]),
        el('span', { title: 'Whether the source could be re-retrieved (§3.1)' },
            [el('i', { class: `fa-solid ${source.accessible ? 'fa-link' : 'fa-link-slash'}` }),
                source.accessible === null ? 'accessibility unknown'
                    : source.accessible ? 'accessible' : 'inaccessible']),
        source.faithfulness
            ? el('span', { title: 'Does the source still support the proposition? (§3.2)' },
                [el('i', { class: 'fa-solid fa-check-double' }),
                    `faithfulness ${decimal(source.faithfulness.assessment, 2)}`])
            : null,
        el('span', { title: 'When the pipeline retrieved this source' },
            [el('i', { class: 'fa-solid fa-clock-rotate-left' }), date(source.accessed_at)]),
    ]));

    body.append(temporalRow(source.temporal_validation));

    if (source.faithfulness) {
        body.append(disclosure('fa-check-double',
            `Faithfulness · ${decimal(source.faithfulness.assessment, 2)}`,
            judgementBody(source.faithfulness)));
    }
    // The two cutoffs are computed from t_e and already shown as badges above;
    // there is no model judgement behind them to disclose.
    if (source.source?.content?.text) {
        body.append(disclosure('fa-file-lines',
            `Retrieved source content (${num(source.source.content.text.length)} chars`
            + `${source.source.content.n_media ? `, ${source.source.content.n_media} media` : ''})`,
            renderMultimodal(source.source.content, {
                containerClass: 'source-content', markdown: true,
            })));
    }
    body.append(disclosure('fa-table-list', 'All stored source fields', fieldTable(source.columns)));
    body.append(disclosure('fa-code', 'Raw source object (full_source)', rawJson(source.full_source)));

    return collapsible({
        head,
        body,
        className: 'evidence-source-card',
        id: `source-${source.id}`,
        attrs: { 'data-admissible': String(source.admissible) },
    });
}

function temporalRow(temporal) {
    if (!temporal) return el('div');
    return el('div', { class: 'evidence-foot' }, temporalFlags(temporal).map(({ label, value, tone, title }) =>
        badge(label, { tone: value === null || value === undefined ? 'plain' : tone, icon: flagIcon(value), title })));
}

const flagIcon = (value) =>
    value === true ? 'fa-check' : value === false ? 'fa-xmark' : 'fa-question';

function temporalFlags(temporal) {
    return [
        { label: 't_e ≤ t_c', value: temporal.before_claim, tone: temporal.before_claim ? 'ok' : 'warn',
            title: 'Was the source available before the claim was made?' },
        { label: 't_e ≤ t_f', value: temporal.before_fact_check, tone: temporal.before_fact_check ? 'ok' : 'bad',
            title: 'Was the source available before the fact-check was published?' },
    ];
}

/** Justification + provider reasoning trace + rater, shared by both judgements. */
function judgementBody(judgement, flags = null) {
    const body = el('div', {});
    if (flags) {
        body.append(el('div', { class: 'chips', style: { marginBottom: '10px' } },
            flags.map(({ label, value, tone, title }) =>
                badge(`${label}: ${value === null || value === undefined ? 'unknown' : value}`,
                    { tone: value === null || value === undefined ? 'plain' : tone, title }))));
    }
    if (judgement.assessment !== undefined && judgement.assessment !== null) {
        body.append(el('dl', { class: 'kv' }, [
            el('dt', { text: 'Assessment' }),
            el('dd', { text: `${decimal(judgement.assessment, 3)}  (−1 contradicts … +1 entails)` }),
        ]));
    }
    if (judgement.justification) {
        body.append(el('h4', { style: { margin: '10px 0 4px', color: 'var(--muted)', fontSize: '.78rem' },
            text: 'Stated justification' }));
        body.append(el('div', { class: 'prose', text: judgement.justification }));
    }
    if (judgement.reasoning) {
        body.append(el('h4', { style: { margin: '12px 0 4px', color: 'var(--muted)', fontSize: '.78rem' },
            text: 'Model reasoning trace (as reported by the provider)' }));
        body.append(el('div', { class: 'prose reasoning', text: judgement.reasoning }));
    } else {
        body.append(el('p', { style: { color: 'var(--muted)', fontSize: '.8rem', marginTop: '10px' },
            text: 'No reasoning trace was reported for this judgement.' }));
    }
    if (judgement.rater) {
        body.append(el('div', { class: 'chips', style: { marginTop: '10px' } },
            [badge(judgement.rater, { tone: 'plain', icon: 'fa-microchip' })]));
    }
    return body;
}

/** Every column of the evidence row, so nothing stored stays hidden. */
function fieldTable(columns) {
    const table = el('table', { class: 'fields' });
    for (const [key, value] of Object.entries(columns ?? {})) {
        table.append(el('tr', {}, [
            el('td', { class: 'key', text: key }),
            el('td', { class: 'val', text: formatValue(value) }),
        ]));
    }
    return table;
}

function formatValue(value) {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value);
    const text = String(value);
    return text.length > 600 ? `${text.slice(0, 600)}…  (${num(text.length)} chars)` : text;
}

const rawJson = (value) =>
    el('pre', { class: 'raw-json', html: highlightJson(value) });

function disclosure(icon, label, body) {
    return el('details', { class: 'disclose' }, [
        el('summary', {}, [
            el('i', { class: `fa-solid ${icon} lead` }),
            label,
            el('i', { class: 'fa-solid fa-chevron-right chev' }),
        ]),
        el('div', { class: 'content' }, [body]),
    ]);
}
