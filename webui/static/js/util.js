/* Small DOM and formatting helpers shared by all views. */

/** Creates an element. `attrs.class`, `attrs.html`, `attrs.text` and `on*` are special. */
export function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'html') node.innerHTML = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'style' && typeof value === 'object') setStyle(node, value);
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of [].concat(children)) {
        if (child === null || child === undefined || child === false) continue;
        node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
}

/** Applies a style object. Custom properties (`--tone`, `--i`) must go through
 *  `setProperty`; assigning them to `node.style` silently does nothing. */
function setStyle(node, styles) {
    for (const [property, value] of Object.entries(styles)) {
        if (value === null || value === undefined) continue;
        if (property.startsWith('--')) node.style.setProperty(property, String(value));
        else node.style[property] = value;
    }
}

export const escapeHtml = (value) =>
    String(value ?? '').replace(/[&<>"']/g, (character) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[character]));

/* ------------------------------------------------------------------ format */

const NUMBER = new Intl.NumberFormat('en-US');

export const num = (value) => (value === null || value === undefined ? '—' : NUMBER.format(value));

export function decimal(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return Number(value).toFixed(digits);
}

export function percent(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return `${(value * 100).toFixed(digits)}%`;
}

export function days(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    const rounded = Number(value).toFixed(digits);
    return `${rounded > 0 ? '+' : ''}${rounded} d`;
}

export function date(value, { time = false } = {}) {
    if (!value) return '—';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    const options = time
        ? { year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }
        : { year: 'numeric', month: 'short', day: '2-digit' };
    return parsed.toLocaleDateString('en-GB', options);
}

export function relative(value) {
    if (!value) return '—';
    const then = new Date(value).getTime();
    if (Number.isNaN(then)) return '—';
    const seconds = (then - Date.now()) / 1000;
    const steps = [[60, 'second'], [60, 'minute'], [24, 'hour'], [7, 'day'], [4.348, 'week'], [12, 'month'], [Infinity, 'year']];
    let amount = seconds;
    for (const [size, unit] of steps) {
        if (Math.abs(amount) < size) {
            return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(Math.round(amount), unit);
        }
        amount /= size;
    }
    return '—';
}

export const bytes = (value) => {
    if (!value && value !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = value;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return `${size < 10 && index > 0 ? size.toFixed(1) : Math.round(size)} ${units[index]}`;
};

/** `some_enum_value` -> `Some enum value`.
 *  Anything that is not a lower-case slug is left exactly as it is, so numbers
 *  (`-0.33`), domains (`bbc.co.uk`) and model names survive untouched. */
export const humanize = (value) => {
    if (value === null || value === undefined || value === '') return '—';
    const text = String(value);
    if (!/^[a-z][a-z0-9_]*$/.test(text)) return text;
    if (!text.includes('_') && text.length <= 3) return text;  // language codes
    const spaced = text.replace(/_+/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

export const truncate = (value, length = 160) => {
    const text = String(value ?? '');
    return text.length > length ? `${text.slice(0, length - 1)}…` : text;
};

/* ------------------------------------------------------------ vocabularies */

/** Tone (colour class) and icon per claim status. */
export const STATUS_STYLE = {
    accepted: { tone: 'ok', icon: 'fa-circle-check' },
    rejected: { tone: 'bad', icon: 'fa-circle-xmark' },
    filtered: { tone: 'info', icon: 'fa-filter' },
    extracted: { tone: 'violet', icon: 'fa-wand-magic-sparkles' },
    pending: { tone: 'warn', icon: 'fa-hourglass-half' },
};

export const ROLE_STYLE = {
    essential: { tone: 'accent', icon: 'fa-star' },
    auxiliary: { tone: 'info', icon: 'fa-star-half-stroke' },
    background: { tone: 'plain', icon: 'fa-layer-group' },
};

export const PROXIMITY_STYLE = {
    primary: { tone: 'ok', icon: 'fa-bullseye' },
    secondary: { tone: 'info', icon: 'fa-diagram-project' },
    tertiary: { tone: 'plain', icon: 'fa-book-atlas' },
};

/** Admissibility is a tri-state; these are the names the browser filters on. */
export const ADMISSIBILITY_STYLE = {
    admissible: { tone: 'ok', icon: 'fa-circle-check' },
    inadmissible: { tone: 'bad', icon: 'fa-circle-xmark' },
    unfiltered: { tone: 'warn', icon: 'fa-hourglass-half' },
};

/** Where an item sits relative to the studied interval `t_c < t_e <= t_f`. */
export const ZONE_STYLE = {
    before_claim: { tone: 'ok', icon: 'fa-backward' },
    in_window: { tone: 'warn', icon: 'fa-clock' },
    after_fact_check: { tone: 'bad', icon: 'fa-forward' },
    unvalidated: { tone: 'plain', icon: 'fa-circle-question' },
};

export const ZONE_HELP = {
    before_claim: 'Available before the claim, t_e <= t_c.',
    in_window: 'Inside the studied interval, t_c < t_e <= t_f.',
    after_fact_check: 'Published after the fact-check, t_e > t_f (§3.3).',
    unvalidated: 'The temporal check has not run for this item.',
};

export const SOURCE_KIND_ICON = {
    social_media_post: 'fa-hashtag',
    news_article: 'fa-newspaper',
    fact_check: 'fa-magnifying-glass-chart',
    scientific_manuscript: 'fa-flask',
    official_statement: 'fa-bullhorn',
    government_record: 'fa-landmark',
    database: 'fa-database',
    encyclopedia: 'fa-book',
    offline: 'fa-plug-circle-xmark',
    tool: 'fa-screwdriver-wrench',
    other: 'fa-circle-question',
};

/** The `media` filter vocabulary. Mirrors `queries.MEDIA_FILTERS`. */
export const MEDIA_OPTIONS = [
    ['', 'Any modality'],
    ['any', 'With any media'],
    ['image', 'With images'],
    ['video', 'With videos'],
    ['none', 'Text only'],
];

/** A select over `MEDIA_OPTIONS`, bound to `state[key]`. */
export function mediaSelect(state, key, apply) {
    return el('select', {
        'aria-label': 'Filter by modality',
        onchange: (event) => apply({ [key]: event.target.value }),
    }, MEDIA_OPTIONS.map(([value, label]) =>
        el('option', { value, selected: state[key] === value, text: label })));
}

/** Explanations shown as tooltips, taken from the pipeline's own docstrings. */
export const REASON_HELP = {
    not_filtered: 'Stage 2 did not complete for this item.',
    low_extraction_confidence: 'Extraction confidence below the configured minimum.',
    inaccessible: 'The source could not be re-retrieved (§3.1).',
    undated_source: 'No publication time could be determined (see undated_policy).',
    unfaithful: 'The retrieved source no longer supports the proposition (§3.2).',
    after_fact_check: 'The source postdates the fact-check, t_e > t_f (§3.3).',
    fact_check_source: 'The source is itself a professional fact-check and leaks the verdict (§3.3).',
    later_event: 'The proposition rests on a change of the world that happened only after the claim (§3.3).',
    no_gold_verdict: 'The claim has no current gold verdict.',
    no_claim_time: 'The claim has no date t_c.',
    no_fact_check_time: 'No review provides a publication time t_f.',
    nothing_extracted: 'Stage 1 returned neither evidence nor a verdict rationale.',
    essential_evidence_lost: 'An essential evidence item lost every one of its sources in Stage 2.',
    sufficiency_validation_failed: 'The ensemble did not return a usable verdict.',
    insufficient_evidence: 'The predicted verdict was not close enough to the gold verdict.',
};

export const badge = (label, { tone = '', icon = '', title = '' } = {}) =>
    el('span', { class: `badge ${tone}`.trim(), title: title || null },
        [icon ? el('i', { class: `fa-solid ${icon}` }) : null, humanize(label)]);

/** Animates a number from 0 to its target when it first appears. */
export function countUp(node, target, format = num) {
    if (target === null || target === undefined || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        node.textContent = format(target);
        return;
    }
    const duration = 620;
    const start = performance.now();
    const step = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - (1 - progress) ** 3;
        node.textContent = format(target * eased);
        if (progress < 1) requestAnimationFrame(step);
        else node.textContent = format(target);
    };
    requestAnimationFrame(step);
}

/* ------------------------------------------------------------------- misc */

export const qs = (selector, root = document) => root.querySelector(selector);
export const qsa = (selector, root = document) => [...root.querySelectorAll(selector)];

/**
 * Copies text to the clipboard.
 *
 * `navigator.clipboard` only exists in a secure context, and the viewer is
 * normally served over plain HTTP on an internal host - so the modern API is
 * simply absent there. Fall back to a hidden textarea, and as a last resort show
 * the text so it can be copied by hand.
 */
export async function copyText(text, { label = 'Link copied' } = {}) {
    try {
        if (window.isSecureContext && navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            toast(label);
            return true;
        }
    } catch { /* fall through to the legacy path */ }

    try {
        const area = el('textarea', {
            value: text, readonly: true,
            style: { position: 'fixed', top: '-1000px', opacity: '0' },
        });
        document.body.append(area);
        area.select();
        area.setSelectionRange(0, text.length);
        const copied = document.execCommand('copy');
        area.remove();
        if (copied) {
            toast(label);
            return true;
        }
    } catch { /* fall through to showing the text */ }

    // Both paths can be blocked outright (an embedded frame without the
    // clipboard permission). Showing the link is then the only thing left that
    // still lets the reader get at it.
    toast(text, { duration: 12000, selectable: true });
    return false;
}


/**
 * An info icon that explains the thing next to it on hover.
 *
 * Explanations are worth having but not worth the horizontal space: a heading
 * followed by a sentence of prose reads as two competing titles. Behind an icon
 * the sentence is one hover away and the heading stays a heading.
 */
export function infoTip(text, { placement = '' } = {}) {
    return el('i', {
        class: `fa-solid fa-circle-info info-tip tip ${placement}`.trim(),
        'data-tip': text,
        tabindex: '0',
        role: 'note',
        'aria-label': text,
    });
}


/** Truncates `text` to `length`, exposing the full value as a hover tooltip. */
export function truncated(text, length, { tone = '' } = {}) {
    const full = String(text ?? '');
    if (full.length <= length) return el('span', { class: tone, text: full });
    return el('span', {
        class: `tip ${tone}`.trim(),
        'data-tip': full,
        tabindex: '0',
        text: `${full.slice(0, length).trimEnd()}…`,
    });
}


export function toast(message, { duration = 2600, selectable = false } = {}) {
    document.querySelector('.toast')?.remove();
    const node = el('div', {
        class: `toast${selectable ? ' selectable' : ''}`,
        text: message,
    });
    document.body.append(node);
    setTimeout(() => node.remove(), duration);
    return node;
}

/** Syntax-highlights a JSON value for the raw inspector.
 *  Built recursively rather than by regex, so strings containing quotes,
 *  braces or HTML cannot break the markup. */
export function highlightJson(value, indent = 0) {
    const pad = '  '.repeat(indent);
    const padInner = '  '.repeat(indent + 1);

    if (value === null || value === undefined) return '<span class="z">null</span>';
    if (typeof value === 'boolean') return `<span class="b">${value}</span>`;
    if (typeof value === 'number') return `<span class="n">${value}</span>`;
    if (typeof value === 'string') {
        return `<span class="s">${escapeHtml(JSON.stringify(value))}</span>`;
    }
    if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        const items = value.map((item) => `${padInner}${highlightJson(item, indent + 1)}`);
        return `[\n${items.join(',\n')}\n${pad}]`;
    }
    const entries = Object.entries(value);
    if (entries.length === 0) return '{}';
    const lines = entries.map(([key, item]) =>
        `${padInner}<span class="k">${escapeHtml(JSON.stringify(key))}</span>: ${highlightJson(item, indent + 1)}`);
    return `{\n${lines.join(',\n')}\n${pad}}`;
}
