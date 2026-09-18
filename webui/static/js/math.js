/* MathJax integration.
 *
 * Notation the pipeline's own documentation uses - t_c, t_f, t_e, E_c, E_f, the
 * closeness threshold theta - is written as real mathematics rather than as
 * ASCII approximations.
 *
 * Every math span is created with its *plain-text* form as its content and the
 * TeX in `data-tex`. If MathJax loads, `typeset()` swaps in the TeX and renders
 * it; if the CDN is unreachable the page still reads `t_c`, not `\(t_c\)`.
 */

import { el } from './util.js';

/** The terms used across the UI, so they stay spelled the same way everywhere. */
export const TERMS = {
    t_c: { tex: 't_c', text: 't_c', title: 'Claim time: when the claim was made (claims.date)' },
    t_f: { tex: 't_f', text: 't_f', title: 'Fact-check time: the latest publication time among the claim\'s reviews' },
    t_e: { tex: 't_e', text: 't_e', title: 'Evidence time: when the source became publicly available' },
    E_c: { tex: 'E_c', text: 'E_c', title: 'Evidence available at the claim time (t_e ≤ t_c)' },
    E_f: { tex: 'E_f', text: 'E_f', title: 'Evidence available at the fact-check time (t_e ≤ t_f)' },
    theta: { tex: '\\theta', text: 'θ', title: 'Closeness threshold' },
};

/** An inline math span, e.g. `math('t_c')` or `math('t_e \\le t_c', 't_e ≤ t_c')`. */
export function math(tex, text = null, options = {}) {
    const term = TERMS[tex];
    const source = term ? term.tex : tex;
    const fallback = text ?? (term ? term.text : tex);
    return el('span', {
        class: 'math',
        'data-tex': source,
        title: options.title ?? term?.title ?? null,
        text: fallback,
    });
}

/** Shorthands for the two evidence sets, which appear on nearly every view. */
export const E_c = (options) => math('E_c', null, options);
export const E_f = (options) => math('E_f', null, options);

/** The studied interval, used as a caption in several places. */
export const window_ = () => math('t_c < t_e \\le t_f', 't_c < t_e ≤ t_f');

/**
 * Typesets every not-yet-rendered math span inside `root`.
 * Safe to call before MathJax has finished loading: the call is retried once the
 * startup promise resolves, and it is a no-op if the script never arrives.
 */
export function typeset(root = document.body) {
    const pending = [...root.querySelectorAll('.math[data-tex]:not([data-typeset])')];
    if (!pending.length) return Promise.resolve();

    const mathJax = window.MathJax;
    if (!mathJax?.typesetPromise) {
        // Not loaded (yet). Try again once, after the script has had a chance to
        // arrive; the plain-text fallback stays on screen until then.
        if (!typeset._retrying) {
            typeset._retrying = true;
            mathJax?.startup?.promise
                ?.then(() => { typeset._retrying = false; typeset(root); })
                ?? setTimeout(() => { typeset._retrying = false; typeset(root); }, 1200);
        }
        return Promise.resolve();
    }

    for (const node of pending) {
        node.textContent = `\\(${node.dataset.tex}\\)`;
        node.dataset.typeset = '';
    }
    return mathJax.typesetPromise(pending).catch(() => {
        // Rendering failed (bad TeX, or the fonts did not load): restore the
        // readable fallback rather than leaving delimiters on screen.
        for (const node of pending) {
            node.textContent = node.dataset.tex.replace(/\\\\|\\le|\\theta/g,
                (token) => ({ '\\le': '≤', '\\theta': 'θ' }[token] ?? ''));
        }
    });
}
