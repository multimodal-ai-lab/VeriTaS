/* A small Markdown renderer for retrieved source content.
 *
 * scrapeMM stores a source as Markdown, so showing it as preformatted text
 * wastes its structure - headings, lists and links all read as noise. This
 * renders the subset that scraped articles actually use.
 *
 * Two properties matter more than completeness:
 *
 * 1. *Safety.* The content is scraped from the open web, i.e. attacker-supplied.
 *    Everything here is built with `createElement` and `textContent`; no string
 *    ever reaches `innerHTML`, so no markup in the source can execute. Link
 *    targets are restricted to http(s) and mailto.
 * 2. *Media references survive.* `<image:42>` must stay an interactive chip
 *    pointing at the medium rendered beneath, so inline parsing hands those back
 *    to the caller instead of treating them as text.
 */

import { el } from './util.js';

const REF_REGEX = /<(image|video|audio):(\d+)>/;

//: Only these schemes may become a link. Anything else is rendered as plain text.
const SAFE_SCHEME = /^(https?:|mailto:)/i;

/**
 * Renders Markdown into a container element.
 *
 * @param {string} text        the Markdown source
 * @param {object} options
 *   - `renderRef(kind, id, reference)` builds the node for a media reference;
 *     when omitted, the reference is left as literal text.
 * @returns {HTMLElement} a `<div class="markdown">` holding the rendered blocks.
 */
export function renderMarkdown(text, { renderRef = null } = {}) {
    const root = el('div', { class: 'markdown' });
    for (const block of blocks(String(text ?? ''))) {
        const node = renderBlock(block, renderRef);
        if (node) root.append(node);
    }
    return root;
}

/* ------------------------------------------------------------------ blocks */

/** Splits the source into block descriptors, keeping fenced code verbatim. */
function blocks(text) {
    const lines = text.replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let index = 0;

    while (index < lines.length) {
        const line = lines[index];

        if (!line.trim()) { index += 1; continue; }

        const fence = line.match(/^\s*(```+|~~~+)\s*([\w+-]*)\s*$/);
        if (fence) {
            const marker = fence[1][0];
            const body = [];
            index += 1;
            while (index < lines.length && !new RegExp(`^\\s*${marker}{3,}\\s*$`).test(lines[index])) {
                body.push(lines[index]);
                index += 1;
            }
            index += 1;  // the closing fence, or the end of the input
            out.push({ type: 'code', text: body.join('\n'), language: fence[2] });
            continue;
        }

        const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
        if (heading) {
            out.push({ type: 'heading', level: heading[1].length, text: heading[2] });
            index += 1;
            continue;
        }

        if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
            out.push({ type: 'rule' });
            index += 1;
            continue;
        }

        if (/^\s{0,3}>/.test(line)) {
            const body = [];
            while (index < lines.length && /^\s{0,3}>/.test(lines[index])) {
                body.push(lines[index].replace(/^\s{0,3}>\s?/, ''));
                index += 1;
            }
            out.push({ type: 'quote', text: body.join('\n') });
            continue;
        }

        const bullet = line.match(/^(\s*)([-*+]|\d+[.)])\s+/);
        if (bullet) {
            const ordered = /\d/.test(bullet[2]);
            const items = [];
            while (index < lines.length) {
                const match = lines[index].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
                if (!match || /\d/.test(match[2]) !== ordered) break;
                const parts = [match[3]];
                index += 1;
                // Continuation lines belong to the item they are indented under.
                while (index < lines.length
                       && lines[index].trim()
                       && !/^(\s*)([-*+]|\d+[.)])\s+/.test(lines[index])
                       && /^\s{2,}/.test(lines[index])) {
                    parts.push(lines[index].trim());
                    index += 1;
                }
                items.push(parts.join(' '));
            }
            out.push({ type: 'list', ordered, items });
            continue;
        }

        const paragraph = [];
        while (index < lines.length && lines[index].trim()
               && !/^\s{0,3}(#{1,6}\s|>|```|~~~)/.test(lines[index])
               && !/^(\s*)([-*+]|\d+[.)])\s+/.test(lines[index])) {
            paragraph.push(lines[index].trim());
            index += 1;
        }
        if (paragraph.length) out.push({ type: 'paragraph', text: paragraph.join('\n') });
        else index += 1;  // defensive: never spin on a line nothing consumed
    }
    return out;
}

function renderBlock(block, renderRef) {
    switch (block.type) {
        case 'heading': {
            // Source headings sit inside a card, so start two levels down.
            const level = Math.min(block.level + 2, 6);
            return el(`h${level}`, {}, inline(block.text, renderRef));
        }
        case 'rule':
            return el('hr');
        case 'code':
            return el('pre', {}, [el('code', { text: block.text })]);
        case 'quote':
            return el('blockquote', {},
                [...renderMarkdown(block.text, { renderRef }).childNodes]);
        case 'list': {
            const list = el(block.ordered ? 'ol' : 'ul');
            for (const item of block.items) list.append(el('li', {}, inline(item, renderRef)));
            return list;
        }
        case 'paragraph':
        default:
            return el('p', {}, inline(block.text, renderRef));
    }
}

/* ------------------------------------------------------------------ inline */

/**
 * Parses inline Markdown into an array of nodes and strings.
 * Handled: media references, images, links, code spans, bold, italic.
 */
export function inline(text, renderRef = null) {
    const out = [];
    let rest = String(text ?? '');

    while (rest) {
        const token = nextToken(rest);
        if (!token) { out.push(rest); break; }

        if (token.index > 0) out.push(rest.slice(0, token.index));
        out.push(token.build(renderRef));
        rest = rest.slice(token.index + token.length);
    }
    return out.filter((part) => part !== '');
}

/** Finds the earliest inline construct in `text`. */
function nextToken(text) {
    const candidates = [];

    const reference = REF_REGEX.exec(text);
    if (reference) {
        candidates.push({
            index: reference.index,
            length: reference[0].length,
            build: (renderRef) => renderRef
                ? renderRef(reference[1], Number(reference[2]), reference[0])
                : reference[0],
        });
    }

    const code = /`([^`]+)`/.exec(text);
    if (code) {
        candidates.push({
            index: code.index, length: code[0].length,
            build: () => el('code', { text: code[1] }),
        });
    }

    const image = /!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/.exec(text);
    if (image) {
        candidates.push({
            index: image.index, length: image[0].length,
            // Remote images are not loaded: the page must not fetch from the
            // scraped host. The alt text and the link are what carry meaning.
            build: () => link(image[2], image[1] || image[2], 'md-image'),
        });
    }

    const anchor = /\[([^\]]+)\]\(([^)\s]+)[^)]*\)/.exec(text);
    if (anchor) {
        candidates.push({
            index: anchor.index, length: anchor[0].length,
            build: () => link(anchor[2], anchor[1]),
        });
    }

    const strong = /\*\*([^*]+)\*\*|__([^_]+)__/.exec(text);
    if (strong) {
        candidates.push({
            index: strong.index, length: strong[0].length,
            build: (renderRef) => el('strong', {}, inline(strong[1] ?? strong[2], renderRef)),
        });
    }

    const emphasis = /(?<![*\w])\*([^*\n]+)\*(?!\*)|(?<![_\w])_([^_\n]+)_(?!_)/.exec(text);
    if (emphasis) {
        candidates.push({
            index: emphasis.index, length: emphasis[0].length,
            build: (renderRef) => el('em', {}, inline(emphasis[1] ?? emphasis[2], renderRef)),
        });
    }

    const bare = /https?:\/\/[^\s<>()]+/.exec(text);
    if (bare) {
        candidates.push({
            index: bare.index, length: bare[0].length,
            build: () => link(bare[0], bare[0]),
        });
    }

    if (!candidates.length) return null;
    candidates.sort((a, b) => a.index - b.index || b.length - a.length);
    return candidates[0];
}

/** A link, but only to a scheme that cannot execute script. */
function link(href, text, className = '') {
    if (!SAFE_SCHEME.test(href)) return text;
    return el('a', {
        href,
        class: className || null,
        target: '_blank',
        rel: 'noreferrer noopener nofollow',
        title: href,
        text,
    });
}
