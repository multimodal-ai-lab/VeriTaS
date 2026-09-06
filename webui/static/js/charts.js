/* Hand-rolled SVG/CSS charts.
 *
 * Deliberately dependency-free: the dashboard needs four shapes (bars, donut,
 * histogram, 2x2 matrix) and shipping a charting library for those would be
 * more code than this file. Every chart animates in once and is keyboard- and
 * screen-reader-neutral (values are also present as text).
 */

import { el, humanize, num, percent } from './util.js';

/** Colours cycled through by the donut, in the order of the design palette. */
const PALETTE = ['--accent', '--ok', '--warn', '--bad', '--info', '--violet', '--muted'];

const toneVar = (tone) => (tone?.startsWith('--') ? `var(${tone})` : `var(--${tone || 'accent'})`);

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

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 150 150');
    svg.setAttribute('class', 'donut');

    let offset = 0;
    const legend = el('div', { class: 'donut-legend' });

    entries.forEach((entry, index) => {
        const colour = toneVar((colours && colours[entry.label]) || PALETTE[index % PALETTE.length]);
        const fraction = entry.count / total;
        const arc = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        arc.setAttribute('cx', '75');
        arc.setAttribute('cy', '75');
        arc.setAttribute('r', String(radius));
        arc.setAttribute('stroke', colour);
        arc.setAttribute('stroke-dasharray', `${fraction * circumference} ${circumference}`);
        arc.setAttribute('stroke-dashoffset', String(-offset * circumference));
        arc.setAttribute('transform', 'rotate(-90 75 75)');
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
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', '75');
        text.setAttribute('y', '80');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', 'var(--text)');
        text.setAttribute('font-size', '19');
        text.setAttribute('font-weight', '650');
        text.textContent = centreLabel;
        svg.append(text);
    }

    return el('div', { class: 'donut-wrap' }, [svg, legend]);
}

/**
 * Column histogram for `{bins: [{start, end, count}], n}`.
 * `zeroAt` draws a dashed reference line at that x value (t_c or t_f).
 */
export function histogramChart(histogram, { tone = 'accent', zeroAt = 0, unit = 'd' } = {}) {
    const bins = histogram?.bins ?? [];
    if (!bins.length) return emptyNote('No dated evidence yet');

    const width = 640;
    const height = 120;
    const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
    const low = histogram.low;
    const high = histogram.high;
    const span = high - low || 1;
    const x = (value) => ((value - low) / span) * width;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height + 18}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('class', 'spark');
    svg.style.setProperty('--tone', toneVar(tone));

    bins.forEach((bin, index) => {
        if (!bin.count) return;
        const barHeight = (bin.count / maxCount) * height;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const left = x(bin.start);
        rect.setAttribute('x', String(left + 0.5));
        rect.setAttribute('width', String(Math.max(x(bin.end) - left - 1, 1)));
        rect.setAttribute('y', String(height - barHeight));
        rect.setAttribute('height', String(barHeight));
        rect.setAttribute('class', 'col');
        rect.style.transformOrigin = `0 ${height}px`;
        rect.style.animation = `grow-y 600ms var(--ease) ${index * 12}ms both`;
        rect.append(title(`${bin.start.toFixed(1)}…${bin.end.toFixed(1)} ${unit}: ${num(bin.count)}`));
        svg.append(rect);
    });

    const axis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    axis.setAttribute('x1', '0'); axis.setAttribute('x2', String(width));
    axis.setAttribute('y1', String(height)); axis.setAttribute('y2', String(height));
    axis.setAttribute('class', 'axis');
    svg.append(axis);

    if (zeroAt !== null && zeroAt >= low && zeroAt <= high) {
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        marker.setAttribute('x1', String(x(zeroAt))); marker.setAttribute('x2', String(x(zeroAt)));
        marker.setAttribute('y1', '0'); marker.setAttribute('y2', String(height));
        marker.setAttribute('class', 'zero');
        svg.append(marker);
    }

    const wrap = el('div', {}, [svg]);
    wrap.append(el('div', {
        class: 'timeline-legend',
        style: { justifyContent: 'space-between' },
    }, [
        el('span', { text: `${low.toFixed(0)} ${unit}` }),
        el('span', { text: `n = ${num(histogram.n)}` }),
        el('span', { text: `${high.toFixed(0)} ${unit}` }),
    ]));
    return wrap;
}

/** The `E_claim` x `E_factcheck` recoverability contingency table. */
export function contingencyMatrix(recoverability) {
    const cells = recoverability?.contingency ?? {};
    const cell = (value, highlight) =>
        el('div', { class: `cell ${highlight ? 'hi' : 'lo'}`, text: num(value ?? 0) });

    return el('div', { class: 'matrix' }, [
        el('div', { class: 'h', text: '' }),
        el('div', { class: 'h', text: 'E_factcheck recovers' }),
        el('div', { class: 'h', text: 'E_factcheck fails' }),

        el('div', { class: 'h', text: 'E_claim recovers' }),
        cell(cells.both, true),
        cell(cells.only_E_claim, false),

        el('div', { class: 'h', text: 'E_claim fails' }),
        cell(cells.only_E_factcheck, true),
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

function title(text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    node.textContent = text;
    return node;
}

export const emptyNote = (message) =>
    el('div', { class: 'empty', style: { padding: '28px 16px' } }, [
        el('i', { class: 'fa-solid fa-chart-simple' }),
        el('span', { text: message }),
    ]);
