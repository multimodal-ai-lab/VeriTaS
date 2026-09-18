/* The statistics dashboard. */

import { api } from '../api.js';
import {
    barChart, contingencyMatrix, donutChart, funnelChart, signedLogHistogram,
    stackedBars, summaryList,
} from '../charts.js';
import { E_c, E_f, math } from '../math.js';
import {
    countUp, decimal, el, humanize, infoTip, num, percent, REASON_HELP,
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
    body.append(funnelSection(stats.evidence));
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

function tile({ label, icon, tone, value, format = num, foot, footNodes }) {
    const valueNode = el('div', { class: 'value', text: '—' });
    const node = el('div', { class: 'stat rise', style: { '--tone': `var(--${tone})` } }, [
        el('div', { class: 'label' }, [el('i', { class: `fa-solid ${icon}` }), label]),
        valueNode,
        footNodes ? el('div', { class: 'foot' }, footNodes)
            : foot ? el('div', { class: 'foot', text: foot }) : null,
    ]);
    countUp(valueNode, value, format);
    return node;
}

const rounded = (value) => num(Math.round(value));

function tiles(stats) {
    const { claims, evidence } = stats;
    return el('div', { class: 'grid cols-4' }, [
        tile({
            label: 'Claims processed', icon: 'fa-flag', tone: 'accent',
            value: claims.n_processed, format: rounded,
            foot: `${num(claims.n_with_evidence)} have stored evidence`,
        }),
        tile({
            label: 'Accepted instances', icon: 'fa-circle-check', tone: 'ok',
            value: claims.n_accepted, format: rounded,
            foot: `${percent(claims.acceptance_rate)} acceptance rate`,
        }),
        tile({
            label: 'Evidence items', icon: 'fa-layer-group', tone: 'violet',
            value: evidence.n_total, format: rounded,
            foot: `${num(evidence.n_sources)} sources, `
                + `${num(evidence.n_distinct_sources)} of them distinct`,
        }),
        tile({
            label: 'Admissible', icon: 'fa-filter-circle-dollar', tone: 'info',
            value: evidence.n_admissible, format: rounded,
            footNodes: [`${percent(evidence.admissible_rate)} of all candidates — this is `, E_f()],
        }),
        tile({
            label: 'Multimodal evidence', icon: 'fa-photo-film', tone: 'accent',
            value: evidence.n_multimodal, format: rounded,
            foot: `${num(evidence.n_with_images)} with images, ${num(evidence.n_with_videos)} with videos`,
        }),
        tile({
            label: 'In the studied window', icon: 'fa-hourglass-half', tone: 'warn',
            value: evidence.n_in_window, format: rounded,
            footNodes: ['admissible with ', math('t_c < t_e \\le t_f', 't_c < t_e ≤ t_f')],
        }),
        tile({
            label: 'Undated sources', icon: 'fa-calendar-xmark', tone: 'bad',
            value: evidence.n_undated, format: rounded,
            footNodes: ['no determinable ', math('t_e')],
        }),
        tile({
            label: 'Evidence per claim', icon: 'fa-divide', tone: 'info',
            value: claims.evidence_per_claim?.mean ?? 0, format: (value) => decimal(value, 1),
            foot: `median ${decimal(claims.evidence_per_claim?.median, 1)}`,
        }),
    ]);
}

/* ------------------------------------------------------------- sections */

const section = (icon, heading, hint, children) =>
    el('section', { class: 'section fade' }, [
        el('h2', {}, [
            el('i', { class: `fa-solid ${icon}` }),
            ...[].concat(heading),
            hint ? el('span', { class: 'hint' }, [].concat(hint)) : null,
        ]),
        ...[].concat(children),
    ]);

const panel = (heading, hint, body) =>
    el('div', { class: 'card' }, [
        el('div', { class: 'card-head' }, [
            el('h3', {}, [].concat(heading)),
            hint ? el('span', { class: 'hint' }, [].concat(hint)) : null,
        ]),
        el('div', { class: 'card-body' }, [body]),
    ]);

function claimSection(claims) {
    const statuses = claims.statuses ?? [];
    return section('fa-flag', 'Claims',
        infoTip('One row per claim the reconstruction has touched. Claims it never reached have no status and are not counted here.'), [
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
                        el('div', { class: 'sub-label', text: 'candidates' }),
                        summaryList(claims.evidence_per_claim, { format: (v) => decimal(v, 1) }),
                    ]),
                    el('div', {}, [
                        el('div', { class: 'sub-label', text: 'admissible' }),
                        summaryList(claims.admissible_per_claim, { format: (v) => decimal(v, 1) }),
                    ]),
                ])),
            panel('Fact-checking duration',
                [math('t_f - t_c', 't_f − t_c'), ' in days'],
                summaryList(claims.fact_check_duration_days, { format: (v) => decimal(v, 1) })),
        ]),
        progressPanel(claims.progress),
    ]);
}

/** Processing throughput per day, as one horizontal stacked bar per day. */
function progressPanel(progress) {
    if (!progress?.length) return el('div');

    const byDay = new Map();
    for (const row of progress) {
        const day = String(row.day);
        if (!byDay.has(day)) byDay.set(day, {});
        byDay.get(day)[row.status ?? 'unknown'] = row.count;
    }
    // Most recent first: the last few days of a run are what one usually checks.
    const days = [...byDay.keys()].sort().reverse();
    const statuses = [...new Set(progress.map((row) => row.status ?? 'unknown'))];

    const rows = days.map((day) => ({
        label: day,
        title: `${day}: ${statuses
            .map((status) => `${humanize(status)} ${num(byDay.get(day)[status] ?? 0)}`)
            .join(', ')}`,
        segments: statuses.map((status) => ({ key: status, count: byDay.get(day)[status] ?? 0 })),
    }));

    return el('div', { class: 'card', style: { marginTop: '16px' } }, [
        el('div', { class: 'card-head' }, [
            el('h3', { text: 'Processing throughput' }),
            el('span', { class: 'hint', text: `${days.length} day(s), newest first` }),
        ]),
        el('div', { class: 'card-body' }, [
            el('div', { class: 'scroll-y' },
                [stackedBars(rows, { colours: STATUS_COLOURS })]),
            el('div', { class: 'timeline-legend' }, statuses.map((status) =>
                el('span', {}, [
                    el('i', { style: { background: `var(${STATUS_COLOURS[status] ?? '--neutral'})` } }),
                    humanize(status),
                ]))),
        ]),
    ]);
}

/* ------------------------------------------------------------------ funnel */

/** Where the evidence is lost between Stage 1 and the two evidence sets. */
function funnelSection(evidence) {
    const stages = evidence.funnel ?? [];
    const admissible = stages.find((stage) => stage.discrepancy !== undefined);

    return section('fa-filter', 'Reconstruction funnel',
        infoTip('Every admissibility criterion, in the order the pipeline applies them '
            + '(veritas/gold_evidence/admissibility.py).'), [
            el('div', { class: 'card' }, [
                el('div', { class: 'card-head' }, [
                    el('h3', { text: 'Evidence surviving each stage' }),
                    infoTip('An item leaves the funnel at the first criterion it fails, so the steps are exclusive and the counts add up.'),
                ]),
                el('div', { class: 'card-body' }, [
                    funnelChart(stages, {
                        help: { ...REASON_HELP, after_claim: 'Only available after the claim was made.' },
                        labels: {
                            'Available before t_f': labelWith('Available before ', math('t_f')),
                            'Admissible (E_f)': labelWith('Admissible — ', E_f()),
                            'Already available at the claim (E_c)':
                                labelWith('Already available at the claim — ', E_c()),
                        },
                    }),
                    admissible?.discrepancy
                        ? el('p', { class: 'note warn-note' }, [
                            el('i', { class: 'fa-solid fa-triangle-exclamation' }),
                            ` The recorded reasons account for ${num(Math.abs(admissible.discrepancy))} `
                            + 'item(s) more than the stored admissible flag; the funnel shows the '
                            + 'stored flag.',
                        ])
                        : null,
                ]),
            ]),
        ]);
}

/** A funnel label made of plain text plus a math span. */
function labelWith(...parts) {
    return el('span', {}, parts);
}

/* ---------------------------------------------------------------- evidence */

function evidenceSection(evidence) {
    return section('fa-layer-group', 'Evidence',
        infoTip('Distributions over the admissible items unless a panel says otherwise.'), [
            el('div', { class: 'grid cols-2' }, [
                panel('Source kind', 'admissible only',
                    barChart(evidence.source_kinds, { tone: 'accent' })),
                panel('Source proximity', 'admissible only',
                    donutChart(evidence.proximities, {
                        colours: { primary: '--ok', secondary: '--info', tertiary: '--neutral' },
                    })),
                panel('Evidence role', 'admissible only',
                    donutChart(evidence.roles, {
                        colours: { essential: '--accent', auxiliary: '--info', background: '--neutral' },
                    })),
                panel('Why candidates were discarded', 'all filtered items',
                    barChart(evidence.inadmissibility_reasons, { tone: 'bad', help: REASON_HELP })),
                panel('Most frequent source domains', 'all candidates',
                    barChart(evidence.top_domains, { tone: 'violet', limit: 12 })),
                panel('Faithfulness ratings', '−1 contradicts … +1 entails',
                    valueBars(evidence.faithfulness, (value) =>
                        (value >= 0.334 ? '--ok' : value <= -0.334 ? '--bad' : '--neutral'))),
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

/* ---------------------------------------------------------------- temporal */

function temporalSection(evidence) {
    return section('fa-clock-rotate-left', 'Temporal position of the evidence',
        ['the interval under study is ', math('t_c < t_e \\le t_f', 't_c < t_e ≤ t_f')], [
            el('div', { class: 'grid cols-1' }, [
                histogramPanel(
                    [math('t_e - t_c', 't_e − t_c')],
                    infoTip('Days between the claim and the evidence becoming available. '
                        + 'Bars left of zero were already available when the claim was made.'),
                    evidence.delta_to_claim),
                histogramPanel(
                    [math('t_e - t_f', 't_e − t_f')],
                    infoTip('Days between the fact-check and the evidence. Bars right of '
                        + 'zero postdate the fact-check and are inadmissible.'),
                    evidence.delta_to_fact_check),
            ]),
            el('div', { class: 'grid cols-3', style: { marginTop: '16px' } }, [
                tile({
                    label: 'Available before the claim', icon: 'fa-backward', tone: 'ok',
                    value: evidence.n_before_claim, format: rounded,
                    footNodes: ['admissible, ', math('t_e \\le t_c', 't_e ≤ t_c')],
                }),
                tile({
                    label: 'Only during fact-checking', icon: 'fa-hourglass-half', tone: 'warn',
                    value: evidence.n_in_window, format: rounded,
                    foot: `${percent(evidence.in_window_rate)} of admissible items`,
                }),
                tile({
                    label: 'Inaccessible sources', icon: 'fa-link-slash', tone: 'bad',
                    value: evidence.n_inaccessible, format: rounded,
                    foot: 'could not be re-retrieved',
                }),
            ]),
        ]);
}

function histogramPanel(heading, hint, delta) {
    return panel(heading, hint, el('div', { class: 'hist-panel' }, [
        signedLogHistogram(delta.histogram),
        el('div', { class: 'hist-summary' }, [
            summaryList(delta.summary, { format: (value) => decimal(value, 1) }),
        ]),
    ]));
}

/* ------------------------------------------------------------- sufficiency */

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
            el('td', { class: 'val' }, [
                `${humanize(mode.ensemble_mode)} · `,
                mode.condition === 'claim' ? E_c() : E_f(),
            ]),
            el('td', { class: 'val', text: num(mode.count) }),
            el('td', { class: 'val', text: `${num(mode.n_close)} (${percent(mode.close_rate, 0)})` }),
            el('td', { class: 'val' }, [
                `${decimal(mode.mean_max_property_diff, 3)} (`,
                math('theta'), ` = ${decimal(mode.threshold, 2)})`,
            ]),
        ]));
    }

    return section('fa-scale-unbalanced', 'Sufficiency validation',
        infoTip('Whether an ensemble, shown only the claim and the evidence of one '
            + 'condition, predicts a verdict close enough to the gold verdict.'), [
            el('div', { class: 'grid cols-2' }, [
                panel('Recoverability contingency', `${num(recover.n_paired_claims)} paired claims`,
                    el('div', {}, [
                        contingencyMatrix(recover, {
                            header: {
                                f_yes: labelWith(E_f(), ' recovers'),
                                f_no: labelWith(E_f(), ' fails'),
                                c_yes: labelWith(E_c(), ' recovers'),
                                c_no: labelWith(E_c(), ' fails'),
                            },
                        }),
                        el('div', { class: 'timeline-legend', style: { marginTop: '12px' } }, [
                            el('span', {}, [E_c(), `: ${percent(recover.rate_E_c)}`]),
                            el('span', {}, [E_f(), `: ${percent(recover.rate_E_f)}`]),
                            el('span', { text: `gain: ${percent(recover.gain_from_fact_check_period)}` }),
                        ]),
                        el('p', { class: 'note' }, [
                            'A surplus in "only ', E_f(), '" indicates that evidence appearing '
                            + 'during the fact-checking period is needed to recover the gold verdict. '
                            + 'The significance test on those discordant pairs is reported by ',
                            el('code', { text: 'scripts.gold_evidence.run_temporal_analysis' }), '.',
                        ]),
                    ])),
                panel('Ensemble runs', 'gold_evidence_results', modeTable),
            ]),
        ]);
}
