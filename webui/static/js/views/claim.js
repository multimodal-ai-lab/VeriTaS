/* The claim detail view: claim, gold verdict, timeline and every evidence item. */

import { api } from '../api.js';
import { renderMultimodal } from '../media.js';
import {
    badge, date, decimal, el, highlightJson, humanize, num, PROXIMITY_STYLE,
    REASON_HELP, ROLE_STYLE, SOURCE_KIND_ICON, STATUS_STYLE, toast, truncate,
} from '../util.js';

export async function renderClaim(root, claimId) {
    root.append(el('div', { class: 'skeleton', style: { height: '160px', marginBottom: '16px' } }));
    root.append(el('div', { class: 'skeleton', style: { height: '320px' } }));

    const detail = await api.claim(claimId);
    root.replaceChildren();

    root.append(head(detail));
    root.append(claimPanel(detail));
    root.append(timelinePanel(detail));
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
        facts.append(el('dt', { text: label, title: title ?? null }));
        facts.append(el('dd', {}, [value]));
    };
    fact('t_c (claim)', el('span', { text: date(claim.t_c, { time: true }) }), 'claims.date');
    fact('t_f (fact-check)', el('span', { text: date(claim.t_f, { time: true }) }),
        'latest published time among the non-dismissed reviews');
    fact('Window', el('span', {
        text: claim.t_c && claim.t_f
            ? `${decimal((new Date(claim.t_f) - new Date(claim.t_c)) / 86400000, 1)} days`
            : '—',
    }), 't_f − t_c');
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
                    renderMultimodal(claim.content, { textClass: 'claim-quote', wide: true }),
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
            el('span', { class: 'hint', text: 'never modified by the reconstruction' }),
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

function reviewsPanel(reviews) {
    if (!reviews?.length) return null;
    const list = el('div', {});
    for (const review of reviews) {
        list.append(el('div', { style: { padding: '10px 0', borderTop: '1px solid var(--border)' } }, [
            el('div', { class: 'chips' }, [
                badge(review.raw_publisher_name ?? 'unknown publisher', { tone: 'plain', icon: 'fa-building-columns' }),
                review.raw_rating ? badge(review.raw_rating, { tone: 'accent', icon: 'fa-gavel' }) : null,
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
            title: 'Studied interval t_c < t_e ≤ t_f',
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
            title: `#${point.evidence_id} · ${date(point.available_since, { time: true })} · ${humanize(point.zone)}`
                + `${point.admissible ? '' : ' · inadmissible'}`,
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
                    legendEntry('--ok', 'available before the claim (t_e ≤ t_c)'),
                    legendEntry('--warn', 'only during fact-checking (t_c < t_e ≤ t_f)'),
                    legendEntry('--bad', 'after the fact-check (t_e > t_f)'),
                    el('span', { style: { opacity: .7 } }, ['faded = inadmissible']),
                ]),
            ]),
        ]),
    ]);
}

const legendEntry = (colour, label) =>
    el('span', {}, [el('i', { style: { background: `var(${colour})` } }), label]);

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
                el('h3', { text: `E_${result.condition}` }),
                el('span', { class: 'hint', text: `${humanize(result.ensemble_mode)} · ${num(result.n_evidence)} items` }),
            ]),
            el('div', { class: 'card-body' }, [
                el('dl', { class: 'kv' }, [
                    el('dt', { text: 'Verdict recovered' }),
                    el('dd', {}, [result.is_close === null
                        ? badge('no result', { tone: 'warn', icon: 'fa-question' })
                        : badge(result.is_close ? 'yes' : 'no',
                            { tone: result.is_close ? 'ok' : 'bad', icon: result.is_close ? 'fa-check' : 'fa-xmark' })]),
                    el('dt', { text: 'Max property diff' }),
                    el('dd', { text: `${decimal(result.max_property_diff, 3)} (θ = ${decimal(result.threshold, 2)})` }),
                    el('dt', { text: 'Property diffs' }),
                    el('dd', {
                        text: Object.entries(result.property_diffs ?? {})
                            .map(([key, value]) => `${humanize(key)} ${decimal(value, 3)}`).join(' · ') || '—',
                    }),
                    el('dt', { text: 'Ensemble' }),
                    el('dd', { text: (result.model_specifiers ?? []).join(', ') || '—' }),
                    result.error ? el('dt', { text: 'Error' }) : null,
                    result.error ? el('dd', { style: { color: 'var(--bad)' }, text: result.error }) : null,
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
            el('span', { class: 'hint', text: 'can the gold verdict be recovered from this evidence alone?' }),
        ]),
        el('div', { class: 'grid cols-2' }, cards),
    ]);
}

/* ---------------------------------------------------------------- evidence */

function evidencePanel(detail) {
    const evidence = detail.evidence ?? [];
    const section = el('section', { class: 'section fade' });

    const filters = { view: 'all' };
    const list = el('div', { class: 'evidence-list' });

    const pills = el('div', { class: 'pill-group' });
    const options = [
        ['all', 'All', evidence.length],
        ['admissible', 'Admissible', evidence.filter((item) => item.admissible === true).length],
        ['inadmissible', 'Discarded', evidence.filter((item) => item.admissible === false).length],
        ['multimodal', 'Multimodal', evidence.filter((item) => item.proposition?.is_multimodal).length],
    ];
    for (const [value, label, count] of options) {
        pills.append(el('button', {
            class: filters.view === value ? 'active' : '',
            onclick: (event) => {
                filters.view = value;
                [...pills.children].forEach((child) => child.classList.remove('active'));
                event.currentTarget.classList.add('active');
                draw();
            },
        }, [label, el('span', { style: { opacity: .55, marginLeft: '6px' }, text: String(count) })]));
    }

    const draw = () => {
        const visible = evidence.filter((item) => {
            if (filters.view === 'admissible') return item.admissible === true;
            if (filters.view === 'inadmissible') return item.admissible === false;
            if (filters.view === 'multimodal') return Boolean(item.proposition?.is_multimodal);
            return true;
        });
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

function evidenceCard(item, index) {
    const admissible = item.admissible;
    const roleStyle = ROLE_STYLE[item.role] ?? { tone: 'plain', icon: 'fa-circle' };
    const proximityStyle = PROXIMITY_STYLE[item.source?.proximity] ?? { tone: 'plain', icon: 'fa-circle' };

    const card = el('article', {
        class: 'evidence rise',
        id: `evidence-${item.id}`,
        style: { '--i': Math.min(index, 8) },
        'data-admissible': String(admissible),
    });

    card.append(el('div', { class: 'evidence-head' }, [
        admissible === true ? badge('admissible', { tone: 'ok', icon: 'fa-circle-check' })
            : admissible === false
                ? badge(item.inadmissibility_reason ?? 'discarded',
                    { tone: 'bad', icon: 'fa-circle-xmark', title: REASON_HELP[item.inadmissibility_reason] ?? '' })
                : badge('not filtered', { tone: 'warn', icon: 'fa-hourglass-half' }),
        badge(item.role ?? 'auxiliary', roleStyle),
        badge(item.source?.proximity ?? 'secondary', proximityStyle),
        item.proposition?.is_multimodal
            ? badge(`${item.proposition.n_media} media`, { tone: 'violet', icon: 'fa-photo-film' })
            : null,
        item.dismissed ? badge('dismissed', { tone: 'bad', icon: 'fa-ban', title: item.dismissed_reason ?? '' }) : null,
        el('span', { class: 'spacer' }),
        el('span', { class: 'eid', text: `#${item.id}` }),
        el('button', {
            class: 'icon-btn', title: 'Copy a link to this evidence item',
            style: { width: '28px', height: '28px' },
            onclick: () => {
                const url = `${window.location.origin}/#/claims/${item.claim_id}/evidence/${item.id}`;
                navigator.clipboard?.writeText(url).then(() => toast('Link copied'), () => toast(url));
            },
        }, [el('i', { class: 'fa-solid fa-link', style: { fontSize: '.72rem' } })]),
    ]));

    // 2. Proposition with inline, interactive references and the media below it.
    card.append(renderMultimodal(item.proposition, { textClass: 'proposition' }));

    card.append(el('div', { class: 'evidence-source' }, [
        el('i', { class: `fa-solid ${SOURCE_KIND_ICON[item.source?.kind] ?? 'fa-circle-question'}`,
            style: { color: 'var(--accent)' } }),
        el('span', { class: 'name', text: item.source?.name ?? 'unknown source' }),
        el('span', { class: 'badge plain', text: humanize(item.source?.kind) }),
        el('span', { class: 'spacer' }),
        item.source?.locator
            ? el('a', {
                href: item.source.locator, target: '_blank', rel: 'noreferrer noopener',
                class: 'locator', title: item.source.locator,
            }, [item.source.domain ?? truncate(item.source.locator, 44),
                el('i', { class: 'fa-solid fa-arrow-up-right-from-square', style: { marginLeft: '6px' } })])
            : el('span', { class: 'locator', text: 'no locator' }),
    ]));

    card.append(el('div', { class: 'evidence-facts' }, [
        el('span', { title: 't_e — when the source became publicly available' },
            [el('i', { class: 'fa-solid fa-calendar-day' }), `t_e ${date(item.available_since, { time: true })}`]),
        el('span', { title: 'Extraction confidence (Stage 1)' },
            [el('i', { class: 'fa-solid fa-wand-magic-sparkles' }),
                `confidence ${decimal(item.extraction?.confidence, 2)}`]),
        el('span', { title: 'Whether the source could be re-retrieved (§3.1)' },
            [el('i', { class: `fa-solid ${item.accessible ? 'fa-link' : 'fa-link-slash'}` }),
                item.accessible === null ? 'accessibility unknown' : item.accessible ? 'accessible' : 'inaccessible']),
        item.faithfulness
            ? el('span', { title: 'Does the source still support the proposition? (§3.2)' },
                [el('i', { class: 'fa-solid fa-check-double' }),
                    `faithfulness ${decimal(item.faithfulness.assessment, 2)}`])
            : null,
        el('span', { title: 'When the pipeline retrieved this source' },
            [el('i', { class: 'fa-solid fa-clock-rotate-left' }), date(item.accessed_at)]),
    ]));

    card.append(temporalRow(item.temporal_validation));

    /* ---- disclosures: every remaining stored detail ---- */

    if (item.extraction?.reasoning) {
        card.append(disclosure('fa-wand-magic-sparkles', 'Extraction reasoning',
            el('div', { class: 'prose reasoning', text: item.extraction.reasoning })));
    }
    if (item.faithfulness) {
        card.append(disclosure('fa-check-double',
            `Faithfulness · ${decimal(item.faithfulness.assessment, 2)}`,
            judgementBody(item.faithfulness)));
    }
    if (item.temporal_validation) {
        card.append(disclosure('fa-hourglass-half', 'Temporal validation',
            judgementBody(item.temporal_validation, temporalFlags(item.temporal_validation))));
    }
    if (item.source?.content?.text) {
        card.append(disclosure('fa-file-lines',
            `Retrieved source content (${num(item.source.content.text.length)} chars`
            + `${item.source.content.n_media ? `, ${item.source.content.n_media} media` : ''})`,
            renderMultimodal(item.source.content, { textClass: 'prose reasoning', wide: true })));
    }

    card.append(disclosure('fa-table-list', 'All stored fields', fieldTable(item.columns)));
    card.append(disclosure('fa-code', 'Raw evidence object (full_evidence)',
        rawJson(item.full_evidence)));

    return card;
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
        { label: 'later event', value: temporal.later_event, tone: temporal.later_event ? 'bad' : 'ok',
            title: 'Does the source report a post-claim event that changes the veracity?' },
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
