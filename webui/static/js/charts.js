/* Hand-rolled SVG/CSS charts.
 *
 * Deliberately dependency-free: the dashboard needs a handful of shapes (bars,
 * donut, log histogram, stacked bars, funnel, 2x2 matrix) and shipping a
 * charting library for those would be more code than this file. Every chart
 * animates in once and states its values as text as well as as geometry.
 *
 * Colours come from the tokens in style.css, which mirror the VeriTaS plotting
 * palette in `scripts/stats/common.py`.
 */

import { el, humanize, num, percent } from './util.js';

/** Colours cycled through by the donut, in the order of the design palette. */
const PALETTE = ['--accent', '--ok', '--warn', '--bad', '--info', '--violet', '--neutral'];

const toneVar = (tone) => (tone?.startsWith('--') ? `var(${tone})` : `var(--${tone || 'accent'})`);

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Creates an SVG element with attributes and optional text content. */
function svgEl(tag, attrs = {}, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined) continue;
        node.setAttribute(key, String(value));
    }
    if (text !== null) node.textContent = text;
    return node;
}

function title(text) {
    const node = document.createElementNS(SVG_NS, 'title');
    node.textContent = text;
    return node;
}

const swatch = (tone) =>
    el('span', { class: 'swatch', style: { background: toneVar(tone) } });

/**
 * Horizontal bars for a `[{label, count, share}]` series.
 * `tone` may be a colour name or a function `(entry) => colourName`.
 */
export function barChart(series, { tone = 'accent', limit = 12, help = {}, showShare = true } = {}) {
    const entries = (series ?? []).slice(0, limit);
    if (!entries.length) return emptyNote('No data yet');

    const max = Math.max(...entries.map((entry) => entry.count), 1);
    const container = el('div', { class: 'bars' });

    entries.forEach((entry, index) => {
        const colour = toneVar(typeof tone === 'function' ? tone(entry) : tone);
        container.append(el('div', {
            class: 'bar-row',
            style: { '--i': index },
            title: help[entry.label] || `${humanize(entry.label)}: ${num(entry.count)}`,
        }, [
            el('span', { class: 'lbl', text: humanize(entry.label) }),
            el('span', { class: 'track' }, [
                el('span', {
                    class: 'fill',
                    style: { width: `${(entry.count / max) * 100}%`, '--tone': colour, '--i': index },
                }),
            ]),
            el('span', {
                class: 'num',
                text: showShare && entry.share !== null && entry.share !== undefined
                    ? `${num(entry.count)} · ${percent(entry.share, 0)}`
                    : num(entry.count),
            }),
        ]));
    });
    return container;
}

/** Donut for a `[{label, count, share}]` series, with a legend beside it. */
export function donutChart(series, { colours = null, centreLabel = '' } = {}) {
    const entries = (series ?? []).filter((entry) => entry.count > 0);
    if (!entries.length) return emptyNote('No data yet');

    const total = entries.reduce((sum, entry) => sum + entry.count, 0);
    const radius = 60;
    const circumference = 2 * Math.PI * radius;

    const svg = svgEl('svg', { viewBox: '0 0 150 150', class: 'donut' });

    let offset = 0;
    const legend = el('div', { class: 'donut-legend' });

    entries.forEach((entry, index) => {
        const colour = toneVar((colours && colours[entry.label]) || PALETTE[index % PALETTE.length]);
        const fraction = entry.count / total;
        const arc = svgEl('circle', {
            cx: 75, cy: 75, r: radius, stroke: colour,
            'stroke-dasharray': `${fraction * circumference} ${circumference}`,
            'stroke-dashoffset': -offset * circumference,
            transform: 'rotate(-90 75 75)',
        });
        arc.style.opacity = '0';
        arc.style.animation = `fade 500ms var(--ease) ${index * 90}ms both`;
        arc.append(title(`${humanize(entry.label)}: ${num(entry.count)} (${percent(fraction, 1)})`));
        svg.append(arc);
        offset += fraction;

        legend.append(el('div', { class: 'item' }, [
            el('span', { class: 'swatch', style: { background: colour } }),
            el('span', { class: 'name', text: humanize(entry.label) }),
            el('span', { class: 'val', text: `${num(entry.count)} · ${percent(fraction, 0)}` }),
        ]));
    });

    if (centreLabel) {
        svg.append(svgEl('text', {
            x: 75, y: 80, 'text-anchor': 'middle', fill: 'var(--text)',
            'font-size': 19, 'font-weight': 650,
        }, centreLabel));
    }

    return el('div', { class: 'donut-wrap' }, [svg, legend]);
}

/**
 * Column histogram of signed day differences on a symmetric logarithmic axis.
 *
 * The backend supplies the bins in log space, split so that zero is always a bin
 * *edge*: every bar lies entirely before or entirely after the reference time,
 * so no bar has to be read as "partly before, partly after". Both axes carry
 * tick labels - the x-axis at the decades the data spans, the y-axis at round
 * counts.
 */
export function signedLogHistogram(histogram, {
    negativeTone = '--ok', positiveTone = '--bad', height = 176, unit = 'd',
} = {}) {
    const bins = histogram?.bins ?? [];
    if (!bins.length || !histogram.n) return emptyNote('No dated evidence yet');

    const width = 760;
    const padLeft = 48;
    const padRight = 12;
    const padTop = 12;
    const padBottom = 38;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const limit = histogram.limit || 1;
    const yTicks = countTicks(histogram.max_count);
    const yMax = Math.max(yTicks[yTicks.length - 1], 1);

    const x = (logValue) => padLeft + ((logValue + limit) / (2 * limit)) * plotWidth;
    const y = (count) => padTop + plotHeight - (count / yMax) * plotHeight;

    const svg = svgEl('svg', {
        viewBox: `0 0 ${width} ${height}`,
        class: 'histogram',
        preserveAspectRatio: 'xMidYMid meet',
        role: 'img',
        'aria-label': `Histogram of ${num(histogram.n)} day differences on a logarithmic axis`,
    });

    // --- y-axis and grid ---------------------------------------------------
    for (const tick of yTicks) {
        svg.append(svgEl('line', {
            x1: padLeft, x2: width - padRight, y1: y(tick), y2: y(tick),
            class: tick === 0 ? 'axis' : 'grid',
        }));
        svg.append(svgEl('text', {
            x: padLeft - 7, y: y(tick) + 3.5, class: 'tick', 'text-anchor': 'end',
        }, num(tick)));
    }
    svg.append(svgEl('text', {
        x: 11, y: padTop + plotHeight / 2, class: 'axis-title', 'text-anchor': 'middle',
        transform: `rotate(-90 11 ${padTop + plotHeight / 2})`,
    }, 'evidence items'));

    // --- bars --------------------------------------------------------------
    bins.forEach((bin, index) => {
        if (!bin.count) return;
        const left = x(bin.log_start);
        const right = x(bin.log_end);
        const rect = svgEl('rect', {
            x: left + 0.4,
            width: Math.max(right - left - 0.8, 0.8),
            y: y(bin.count),
            height: Math.max(y(0) - y(bin.count), 0.8),
            class: 'col',
            fill: toneVar(bin.side === 'negative' ? negativeTone : positiveTone),
        });
        rect.style.transformOrigin = `0 ${y(0)}px`;
        rect.style.animation = `grow-y 560ms var(--ease) ${Math.min(index * 9, 400)}ms both`;
        rect.append(title(`${formatDays(bin.start)} to ${formatDays(bin.end)} ${unit}`
            + ` — ${num(bin.count)} item${bin.count === 1 ? '' : 's'}`));
        svg.append(rect);
    });

    // --- the zero divider, which no bin crosses ----------------------------
    svg.append(svgEl('line', { x1: x(0), x2: x(0), y1: padTop - 4, y2: y(0) + 5, class: 'zero' }));

    // --- x-axis ------------------------------------------------------------
    for (const tick of histogram.ticks ?? []) {
        svg.append(svgEl('line', {
            x1: x(tick.log), x2: x(tick.log), y1: y(0), y2: y(0) + 4, class: 'axis',
        }));
        svg.append(svgEl('text', {
            x: x(tick.log), y: y(0) + 15, class: 'tick', 'text-anchor': 'middle',
        }, tick.label));
    }
    svg.append(svgEl('text', {
        x: padLeft + plotWidth / 2, y: height - 4, class: 'axis-title', 'text-anchor': 'middle',
    }, 'days, symmetric log scale'));

    return el('div', { class: 'chart' }, [
        svg,
        el('div', { class: 'chart-note' }, [
            el('span', {}, [swatch(negativeTone), `earlier: ${num(histogram.n_negative)}`]),
            el('span', { text: `n = ${num(histogram.n)}` }),
            el('span', {}, [swatch(positiveTone), `later: ${num(histogram.n_positive)}`]),
        ]),
    ]);
}

/** Round y-axis ticks from 0 up to at least `maxCount`. Mirrors `stats.count_ticks`. */
export function countTicks(maxCount, target = 4) {
    if (!maxCount || maxCount <= 0) return [0];
    const raw = maxCount / target;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const multiple = [1, 2, 2.5, 5, 10].find((candidate) => raw <= candidate * magnitude) ?? 10;
    const step = Math.max(Math.round(multiple * magnitude), 1);
    const ticks = [];
    for (let value = 0; value < maxCount + step; value += step) ticks.push(value);
    return ticks;
}

/** Day counts are shown at the precision they carry: sub-day values need decimals. */
function formatDays(value) {
    const magnitude = Math.abs(value);
    if (magnitude < 1) return value.toFixed(2);
    if (magnitude < 10) return value.toFixed(1);
    return Math.round(value).toLocaleString('en-US');
}

/**
 * Horizontal stacked bars: one row per item, segments stacked left to right.
 * `rows` are `{label, title, segments: [{key, count}]}`; `colours` maps a
 * segment key to a colour token.
 */
export function stackedBars(rows, { colours = {}, max = null, suffix = '' } = {}) {
    const entries = rows ?? [];
    if (!entries.length) return emptyNote('No data yet');

    const total = (row) => row.segments.reduce((sum, segment) => sum + segment.count, 0);
    const scale = max ?? Math.max(...entries.map(total), 1);

    const list = el('div', { class: 'stack-rows' });
    entries.forEach((row, index) => {
        const track = el('div', { class: 'stack-track' });
        for (const segment of row.segments) {
            if (!segment.count) continue;
            track.append(el('span', {
                class: 'stack-seg',
                style: {
                    width: `${(segment.count / scale) * 100}%`,
                    background: toneVar(colours[segment.key] ?? '--neutral'),
                    '--i': index,
                },
                title: `${row.label} · ${humanize(segment.key)}: ${num(segment.count)}`,
            }));
        }
        list.append(el('div', { class: 'stack-row', style: { '--i': index } }, [
            el('span', { class: 'lbl', title: row.title ?? row.label, text: row.label }),
            track,
            el('span', { class: 'num', text: `${num(total(row))}${suffix}` }),
        ]));
    });
    return list;
}

/**
 * The reconstruction funnel: how many evidence items survive each stage.
 * `stages` come from `stats.evidence_funnel`; `labels` may supply a rich node
 * (e.g. one containing MathJax) for a stage label.
 */
export function funnelChart(stages, { help = {}, labels = {} } = {}) {
    const entries = stages ?? [];
    if (!entries.length) return emptyNote('No evidence yet');

    const start = Math.max(entries[0].count, 1);
    const list = el('div', { class: 'funnel' });

    entries.forEach((stage, index) => {
        const isFirst = index === 0;
        const isLast = index === entries.length - 1;
        const isAdmissible = index === entries.length - 2;
        const tone = isFirst ? '--neutral' : isLast ? '--accent' : isAdmissible ? '--ok' : '--info';

        const label = el('span', { class: 'lbl' });
        label.append(labels[stage.label] ?? document.createTextNode(stage.label));

        list.append(el('div', {
            class: `funnel-stage${isLast || isAdmissible ? ' terminal' : ''}`,
            style: { '--i': index },
        }, [
            label,
            el('div', { class: 'funnel-track' }, [
                el('span', {
                    class: 'funnel-fill',
                    style: { width: `${Math.max((stage.count / start) * 100, 0.5)}%`,
                        background: toneVar(tone), '--i': index },
                    title: `${num(stage.count)} of ${num(start)} candidates (${percent(stage.share, 1)})`,
                }),
                stage.dropped
                    ? el('span', {
                        class: 'funnel-loss',
                        style: { width: `${(stage.dropped / start) * 100}%` },
                        title: `${humanize(stage.reason)}: ${num(stage.dropped)} removed here`
                            + (help[stage.reason] ? `\n${help[stage.reason]}` : ''),
                    })
                    : null,
            ]),
            el('span', { class: 'num' }, [
                el('strong', { text: num(stage.count) }),
                el('small', { text: percent(stage.share, 0) }),
            ]),
            el('span', {
                class: `drop${stage.dropped ? '' : ' none'}`,
                title: stage.reason
                    ? (help[stage.reason] ?? humanize(stage.reason))
                    : 'Nothing is removed at this step',
                text: stage.dropped ? `−${num(stage.dropped)}` : '—',
            }),
        ]));
    });
    return list;
}

/** The `E_c` x `E_f` recoverability contingency table. */
export function contingencyMatrix(recoverability, { header = {} } = {}) {
    const cells = recoverability?.contingency ?? {};
    const cell = (value, highlight) =>
        el('div', { class: `cell ${highlight ? 'hi' : 'lo'}`, text: num(value ?? 0) });
    const head = (key, fallback) =>
        el('div', { class: 'h' }, [header[key] ?? document.createTextNode(fallback)]);

    return el('div', { class: 'matrix' }, [
        el('div', { class: 'h', text: '' }),
        head('f_yes', 'E_f recovers'),
        head('f_no', 'E_f fails'),

        head('c_yes', 'E_c recovers'),
        cell(cells.both, true),
        cell(cells.only_E_c, false),

        head('c_no', 'E_c fails'),
        cell(cells.only_E_f, true),
        cell(cells.neither, false),
    ]);
}

/** Five-number summary rendered as a compact definition list. */
export function summaryList(summary, { format = (value) => num(value) } = {}) {
    const list = el('dl', { class: 'kv' });
    list.append(el('dt', { text: 'n' }), el('dd', { text: num(summary?.n) }));
    list.append(el('dt', { text: 'mean ± sd' }),
        el('dd', { text: `${format(summary?.mean)} ± ${format(summary?.std)}` }));
    list.append(el('dt', { text: 'median' }), el('dd', { text: format(summary?.median) }));
    list.append(el('dt', { text: 'p25 – p75' }),
        el('dd', { text: `${format(summary?.p25)} – ${format(summary?.p75)}` }));
    list.append(el('dt', { text: 'min – max' }),
        el('dd', { text: `${format(summary?.min)} – ${format(summary?.max)}` }));
    return list;
}

export const emptyNote = (message) =>
    el('div', { class: 'empty', style: { padding: '28px 16px' } }, [
        el('i', { class: 'fa-solid fa-chart-simple' }),
        el('span', { text: message }),
    ]);
