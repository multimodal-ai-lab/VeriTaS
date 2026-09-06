/* The statistics dashboard. */

import { api } from '../api.js';
import { barChart, contingencyMatrix, donutChart, histogramChart, summaryList } from '../charts.js';
import {
    countUp, decimal, el, humanize, num, percent, REASON_HELP,
} from '../util.js';

const STATUS_COLOURS = {
    accepted: '--ok', rejected: '--bad', filtered: '--info',
    extracted: '--violet', pending: '--warn',
};

export async function renderOverview(root) {
    root.append(header());
    const body = el('div', { id: 'stats-body' });
    root.append(body);
    body.append(skeleton());

    const stats = await api.stats();
    body.replaceChildren();

    body.append(tiles(stats));
    body.append(claimSection(stats.claims));
    body.append(evidenceSection(stats.evidence));
    body.append(temporalSection(stats.evidence));
    body.append(sufficiencySection(stats.sufficiency));
}

function header() {
    return el('div', { class: 'page-head rise' }, [
        el('div', {}, [
            el('div', { class: 'eyebrow', text: 'Gold Evidence Reconstruction' }),
            el('h1', { text: 'Overview' }),
            el('p', {
                class: 'lede',
                text: 'What the reconstruction pipeline has produced so far: which claims it '
                    + 'processed, which evidence survived filtering, and how the evidence is '
                    + 'distributed around the claim and fact-check times.',
            }),
        ]),
        el('div', { class: 'spacer' }),
        el('a', { class: 'btn primary', href: '#/claims' }, [
            el('i', { class: 'fa-solid fa-list-check' }), 'Browse claims',
        ]),
    ]);
}

const skeleton = () => el('div', { class: 'grid cols-4 skeleton-grid' },
    Array.from({ length: 8 }, () => el('div', { class: 'skeleton' })));

/* ------------------------------------------------------------------ tiles */

function tile({ label, icon, tone, value, format = num, foot }) {
    const valueNode = el('div', { class: 'value', text: '—' });
    const node = el('div', { class: 'stat rise', style: { '--tone': `var(--${tone})` } }, [
        el('div', { class: 'label' }, [el('i', { class: `fa-solid ${icon}` }), label]),
        valueNode,
        foot ? el('div', { class: 'foot', text: foot }) : null,
    ]);
    countUp(valueNode, value, format);
    return node;
}

function tiles(stats) {
    const { claims, evidence } = stats;
    return el('div', { class: 'grid cols-4' }, [
        tile({
            label: 'Claims processed', icon: 'fa-flag', tone: 'accent',
            value: claims.n_processed, format: (v) => num(Math.round(v)),
            foot: `${num(claims.n_with_evidence)} have stored evidence`,
        }),
        tile({
            label: 'Accepted instances', icon: 'fa-circle-check', tone: 'ok',
            value: claims.n_accepted, format: (v) => num(Math.round(v)),
            foot: `${percent(claims.acceptance_rate)} acceptance rate`,
        }),
        tile({
            label: 'Evidence items', icon: 'fa-layer-group', tone: 'violet',
            value: evidence.n_total, format: (v) => num(Math.round(v)),
            foot: `${num(evidence.n_distinct_sources)} distinct sources`,
        }),
        tile({
            label: 'Admissible', icon: 'fa-filter-circle-dollar', tone: 'info',
            value: evidence.n_admissible, format: (v) => num(Math.round(v)),
            foot: `${percent(evidence.admissible_rate)} of all candidates`,
        }),
        tile({
            label: 'Multimodal evidence', icon: 'fa-photo-film', tone: 'accent',
            value: evidence.n_multimodal, format: (v) => num(Math.round(v)),
            foot: `${percent(evidence.multimodal_rate)} reference media`,
        }),
        tile({
            label: 'In the studied window', icon: 'fa-hourglass-half', tone: 'warn',
            value: evidence.n_in_window, format: (v) => num(Math.round(v)),
            foot: 'admissible with t_c < t_e ≤ t_f',
        }),
        tile({
            label: 'Undated sources', icon: 'fa-calendar-xmark', tone: 'bad',
            value: evidence.n_undated, format: (v) => num(Math.round(v)),
            foot: 'no determinable publication time',
        }),
        tile({
            label: 'Evidence per claim', icon: 'fa-divide', tone: 'info',
            value: claims.evidence_per_claim?.mean ?? 0, format: (v) => decimal(v, 1),
            foot: `median ${decimal(claims.evidence_per_claim?.median, 1)}`,
        }),
    ]);
}

/* ------------------------------------------------------------- sections */

const section = (icon, heading, hint, children) =>
    el('section', { class: 'section fade' }, [
        el('h2', {}, [
            el('i', { class: `fa-solid ${icon}` }), heading,
            hint ? el('span', { class: 'hint', text: hint }) : null,
        ]),
        ...[].concat(children),
    ]);

const panel = (heading, hint, body) =>
    el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
            el('h3', { text: heading }),
            hint ? el('span', { class: 'hint', text: hint }) : null,
        ]),
        el('div', { class: 'card-body' }, [body]),
    ]);

function claimSection(claims) {
    const statuses = claims.statuses ?? [];
    return section('fa-flag', 'Claims', 'one row per instance the pipeline touched', [
        el('div', { class: 'grid cols-2' }, [
            panel('Reconstruction status', `${num(claims.n_processed)} claims`,
                donutChart(statuses, {
                    colours: STATUS_COLOURS,
                    centreLabel: percent(claims.acceptance_rate, 0),
                })),
            panel('Why instances were rejected', 'claims.gold_evidence_reason',
                barChart(claims.rejection_reasons, { tone: 'bad', help: REASON_HELP })),
        ]),
        el('div', { class: 'grid cols-2', style: { marginTop: '16px' } }, [
            panel('Evidence per claim', 'candidates / admissible',
                el('div', { class: 'grid cols-2' }, [
                    el('div', {}, [
                        el('div', { class: 'hint', style: { color: 'var(--muted)', marginBottom: '6px' }, text: 'candidates' }),
                        summaryList(claims.evidence_per_claim, { format: (v) => decimal(v, 1) }),
                    ]),
                    el('div', {}, [
                        el('div', { class: 'hint', style: { color: 'var(--muted)', marginBottom: '6px' }, text: 'admissible' }),
                        summaryList(claims.admissible_per_claim, { format: (v) => decimal(v, 1) }),
                    ]),
                ])),
            panel('Fact-checking duration', 't_f − t_c in days',
                summaryList(claims.fact_check_duration_days, { format: (v) => decimal(v, 1) })),
        ]),
        progressPanel(claims.progress),
    ]);
}

/** Processing throughput per day, stacked by the status claims ended in. */
function progressPanel(progress) {
    if (!progress?.length) return el('div');

    const byDay = new Map();
    for (const row of progress) {
        const day = String(row.day);
        if (!byDay.has(day)) byDay.set(day, {});
        byDay.get(day)[row.status ?? 'unknown'] = row.count;
    }
    const days = [...byDay.keys()].sort();
    const statuses = [...new Set(progress.map((row) => row.status ?? 'unknown'))];
    const max = Math.max(...days.map((day) => Object.values(byDay.get(day)).reduce((a, b) => a + b, 0)), 1);

    const width = Math.max(days.length * 14, 200);
    const height = 130;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('class', 'spark');

    days.forEach((day, index) => {
        let y = height;
        statuses.forEach((status) => {
            const count = byDay.get(day)[status] ?? 0;
            if (!count) return;
            const barHeight = (count / max) * (height - 4);
            y -= barHeight;
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', String(index * (width / days.length) + 1));
            rect.setAttribute('width', String(Math.max(width / days.length - 2, 1)));
            rect.setAttribute('y', String(y));
            rect.setAttribute('height', String(barHeight));
            rect.setAttribute('fill', `var(${STATUS_COLOURS[status] ?? '--muted'})`);
            rect.style.opacity = '0';
            rect.style.animation = `fade 400ms var(--ease) ${Math.min(index * 8, 500)}ms both`;
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            label.textContent = `${day} · ${humanize(status)}: ${num(count)}`;
            rect.append(label);
            svg.append(rect);
        });
    });

    const legend = el('div', { class: 'timeline-legend' }, statuses.map((status) =>
        el('span', {}, [
            el('i', { style: { background: `var(${STATUS_COLOURS[status] ?? '--muted'})` } }),
            humanize(status),
        ])));

    return el('div', { class: 'card', style: { marginTop: '16px' } }, [
        el('div', { class: 'card-head' }, [
            el('h3', { text: 'Processing throughput' }),
            el('span', { class: 'hint', text: `${days[0]} → ${days[days.length - 1]}` }),
        ]),
        el('div', { class: 'card-body' }, [svg, legend]),
    ]);
}

function evidenceSection(evidence) {
    return section('fa-layer-group', 'Evidence',
        'distributions over the admissible items unless noted', [
            el('div', { class: 'grid cols-2' }, [
                panel('Source kind', 'admissible only', barChart(evidence.source_kinds, { tone: 'accent' })),
                panel('Source proximity', 'admissible only',
                    donutChart(evidence.proximities, {
                        colours: { primary: '--ok', secondary: '--info', tertiary: '--muted' },
                    })),
                panel('Evidence role', 'admissible only',
                    donutChart(evidence.roles, {
                        colours: { essential: '--accent', auxiliary: '--info', background: '--muted' },
                    })),
                panel('Why candidates were discarded', 'all filtered items',
                    barChart(evidence.inadmissibility_reasons, { tone: 'bad', help: REASON_HELP })),
                panel('Most frequent source domains', 'all candidates',
                    barChart(evidence.top_domains, { tone: 'violet', limit: 12 })),
                panel('Faithfulness ratings', '−1 contradicts … +1 entails',
                    valueBars(evidence.faithfulness, (value) =>
                        (value >= 0.334 ? '--ok' : value <= -0.334 ? '--bad' : '--warn'))),
            ]),
        ]);
}

/** Bars for a `[{value, count, share}]` series, coloured by the value. */
function valueBars(series, tone) {
    const mapped = (series ?? []).map((entry) => ({
        label: entry.value === null ? 'unknown' : decimal(entry.value, 2),
        count: entry.count,
        share: entry.share,
        value: entry.value,
    }));
    return barChart(mapped, { tone: (entry) => tone(entry.value ?? 0), limit: 16 });
}

function temporalSection(evidence) {
    return section('fa-clock-rotate-left', 'Temporal position of the evidence',
        'the interval under study is t_c < t_e ≤ t_f', [
            el('div', { class: 'grid cols-2' }, [
                panel('t_e − t_c', 'days between claim and evidence',
                    el('div', {}, [
                        histogramChart(evidence.delta_to_claim.histogram, { tone: 'accent', zeroAt: 0 }),
                        summaryList(evidence.delta_to_claim.summary, { format: (v) => decimal(v, 1) }),
                    ])),
                panel('t_e − t_f', 'days between fact-check and evidence',
                    el('div', {}, [
                        histogramChart(evidence.delta_to_fact_check.histogram, { tone: 'violet', zeroAt: 0 }),
                        summaryList(evidence.delta_to_fact_check.summary, { format: (v) => decimal(v, 1) }),
                    ])),
            ]),
            el('div', { class: 'grid cols-3', style: { marginTop: '16px' } }, [
                tile({
                    label: 'Available before the claim', icon: 'fa-backward', tone: 'ok',
                    value: evidence.n_before_claim, format: (v) => num(Math.round(v)),
                    foot: 'admissible, t_e ≤ t_c',
                }),
                tile({
                    label: 'Only during fact-checking', icon: 'fa-hourglass-half', tone: 'warn',
                    value: evidence.n_in_window, format: (v) => num(Math.round(v)),
                    foot: `${percent(evidence.in_window_rate)} of admissible items`,
                }),
                tile({
                    label: 'Inaccessible sources', icon: 'fa-link-slash', tone: 'bad',
                    value: evidence.n_inaccessible, format: (v) => num(Math.round(v)),
                    foot: 'could not be re-retrieved',
                }),
            ]),
        ]);
}

function sufficiencySection(sufficiency) {
    const recover = sufficiency.recoverability ?? {};
    const modes = sufficiency.modes ?? [];

    const modeTable = el('table', { class: 'fields' });
    modeTable.append(el('tr', {}, [
        el('td', { class: 'key', text: 'mode / condition' }),
        el('td', { class: 'key', text: 'claims' }),
        el('td', { class: 'key', text: 'verdict recovered' }),
        el('td', { class: 'key', text: 'mean max property diff' }),
    ]));
    for (const mode of modes) {
        modeTable.append(el('tr', {}, [
            el('td', { class: 'val', text: `${humanize(mode.ensemble_mode)} · E_${mode.condition}` }),
            el('td', { class: 'val', text: num(mode.count) }),
            el('td', { class: 'val', text: `${num(mode.n_close)} (${percent(mode.close_rate, 0)})` }),
            el('td', {
                class: 'val',
                text: `${decimal(mode.mean_max_property_diff, 3)} (θ = ${decimal(mode.threshold, 2)})`,
            }),
        ]));
    }

    return section('fa-scale-unbalanced', 'Sufficiency validation',
        'can the gold verdict be recovered from the evidence alone?', [
            el('div', { class: 'grid cols-2' }, [
                panel('Recoverability contingency', `${num(recover.n_paired_claims)} paired claims`,
                    el('div', {}, [
                        contingencyMatrix(recover),
                        el('div', { class: 'timeline-legend', style: { marginTop: '12px' } }, [
                            el('span', { text: `E_claim: ${percent(recover.rate_E_claim)}` }),
                            el('span', { text: `E_factcheck: ${percent(recover.rate_E_factcheck)}` }),
                            el('span', { text: `gain: ${percent(recover.gain_from_fact_check_period)}` }),
                        ]),
                        el('p', {
                            style: { color: 'var(--muted)', fontSize: '.8rem', marginTop: '10px' },
                            text: 'A surplus in "only E_factcheck" indicates that evidence appearing '
                                + 'during the fact-checking period is needed to recover the gold verdict. '
                                + 'The significance test on those discordant pairs is reported by '
                                + 'scripts.gold_evidence.run_temporal_analysis.',
                        }),
                    ])),
                panel('Ensemble runs', 'gold_evidence_results', modeTable),
            ]),
        ]);
}
