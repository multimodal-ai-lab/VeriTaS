/* Rendering of multimodal text.
 *
 * A proposition (or a claim, or a scraped source) is delivered by the API as an
 * ordered list of `text` and `media` segments. We keep every media reference
 * visible inline as a chip and render the referenced media right beneath the
 * text. Chip and figure are linked both ways: hovering either one highlights
 * the other, clicking a chip brings its medium into view.
 */

import { api } from './api.js';
import { bytes, el, truncate } from './util.js';

const KIND_ICON = { image: 'fa-image', video: 'fa-film', audio: 'fa-volume-high' };

/**
 * Renders a multimodal payload.
 *
 * @param {object} payload  `{text, segments, media}` as returned by the API.
 * @param {object} options
 *   - `textClass`  class of the text container
 *   - `wide`       render the figures large (used for claims and sources)
 *   - `emptyText`  what to show when there is no text at all
 * @returns {HTMLElement} a container holding the text and, if any, the gallery.
 */
export function renderMultimodal(payload, options = {}) {
    const { textClass = 'rich-text', wide = false, emptyText = '—' } = options;
    const container = el('div', { class: 'multimodal' });

    const segments = payload?.segments ?? [];
    const textNode = el('div', { class: textClass });

    if (!segments.length) {
        textNode.append(el('span', { class: 'seg-text muted', text: payload?.text || emptyText }));
    }
    for (const part of segments) {
        if (part.type === 'text') {
            textNode.append(el('span', { class: 'seg-text', text: part.text }));
        } else {
            textNode.append(referenceChip(part, payload));
        }
    }
    container.append(textNode);

    const media = payload?.media ?? [];
    if (media.length) {
        const gallery = el('div', { class: 'media-gallery' });
        media.forEach((item, index) => gallery.append(mediaFigure(item, { wide, index })));
        container.append(gallery);
        link(container);
    }
    return container;
}

/** The inline `<image:42>` chip that stays in the text. */
function referenceChip(part, payload) {
    const known = (payload?.media ?? []).find((item) => item.reference === part.reference);
    const missing = known ? !known.exists : true;
    return el('span', {
        class: `ref${missing ? ' missing' : ''}`,
        'data-ref': part.reference,
        role: 'button',
        tabindex: '0',
        title: missing
            ? `${part.reference} — file not found in the ezMM registry`
            : `${part.reference} — click to jump to the medium`,
    }, [
        el('i', { class: `fa-solid ${missing ? 'fa-triangle-exclamation' : KIND_ICON[part.kind] ?? 'fa-paperclip'}` }),
        `${part.kind}:${part.id}`,
    ]);
}

/** One figure in the gallery below the text. */
function mediaFigure(item, { wide = false, index = 0 } = {}) {
    const figure = el('figure', {
        class: `media-figure${wide ? ' wide' : ''}`,
        'data-ref': item.reference,
        style: { '--i': index },
    });

    const frame = el('div', { class: 'frame' });
    if (!item.exists) {
        frame.append(el('div', { class: 'missing-note' }, [
            el('i', { class: 'fa-solid fa-image-slash fa-lg' }),
            el('span', { text: 'file not in registry' }),
        ]));
    } else if (item.kind === 'image') {
        frame.append(el('img', {
            src: api.mediaUrl(item.kind, item.id),
            alt: item.reference,
            loading: 'lazy',
            decoding: 'async',
            onclick: () => openLightbox(item),
        }));
    } else if (item.kind === 'video') {
        frame.append(el('video', {
            src: api.mediaUrl(item.kind, item.id),
            controls: true,
            preload: 'metadata',
            playsinline: true,
        }));
    } else {
        frame.append(el('audio', { src: api.mediaUrl(item.kind, item.id), controls: true, style: { width: '100%' } }));
    }
    figure.append(frame);

    figure.append(el('figcaption', {}, [
        el('span', { text: `${item.kind}:${item.id}` }),
        item.source_url
            ? el('a', {
                href: item.source_url, target: '_blank', rel: 'noreferrer noopener',
                title: item.source_url,
            }, [el('i', { class: 'fa-solid fa-arrow-up-right-from-square' })])
            : el('span', { class: 'muted', text: item.size ? bytes(item.size) : '' }),
    ]));
    return figure;
}

/** Wires the hover/click highlighting between chips and figures, scoped to
 *  `container` so two propositions on the same page never affect each other. */
function link(container) {
    const targets = (reference, selector) =>
        [...container.querySelectorAll(`${selector}[data-ref="${cssEscape(reference)}"]`)];

    const setHot = (reference, on) => {
        for (const node of [...targets(reference, '.ref'), ...targets(reference, '.media-figure')]) {
            node.classList.toggle('hot', on);
        }
    };

    for (const node of container.querySelectorAll('.ref, .media-figure')) {
        const reference = node.dataset.ref;
        node.addEventListener('mouseenter', () => setHot(reference, true));
        node.addEventListener('mouseleave', () => setHot(reference, false));
        node.addEventListener('focus', () => setHot(reference, true));
        node.addEventListener('blur', () => setHot(reference, false));
    }

    for (const chip of container.querySelectorAll('.ref')) {
        const jump = () => {
            const [figure] = targets(chip.dataset.ref, '.media-figure');
            if (!figure) return;
            figure.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
            figure.classList.add('hot');
            setTimeout(() => figure.classList.remove('hot'), 1400);
        };
        chip.addEventListener('click', jump);
        chip.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); jump(); }
        });
    }
}

/** `CSS.escape` is not available everywhere; references are simple enough. */
const cssEscape = (value) => String(value).replace(/["\\]/g, '\\$&');

/* ----------------------------------------------------------------- lightbox */

function openLightbox(item) {
    const close = () => {
        overlay.remove();
        document.removeEventListener('keydown', onKey);
    };
    const onKey = (event) => { if (event.key === 'Escape') close(); };

    const overlay = el('div', {
        class: 'lightbox',
        onclick: (event) => { if (event.target === overlay) close(); },
    }, [
        el('i', { class: 'fa-solid fa-xmark close', onclick: close, title: 'Close (Esc)' }),
        el('img', { src: api.mediaUrl(item.kind, item.id), alt: item.reference }),
        el('div', { class: 'caption' }, [
            el('span', { text: item.reference }),
            item.size ? el('span', { text: bytes(item.size) }) : null,
            item.source_url
                ? el('a', { href: item.source_url, target: '_blank', rel: 'noreferrer noopener' },
                    [truncate(item.source_url, 70)])
                : null,
        ]),
    ]);

    document.body.append(overlay);
    document.addEventListener('keydown', onKey);
}
