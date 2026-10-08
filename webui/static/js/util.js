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

/** A timestamp from the API as a `Date`.
 *
 *  The database stores timestamps without a time zone, in UTC, and the API passes
 *  them on as they are (`2026-09-30T12:22:13`). `new Date()` would read such a
 *  string as *local* time and show the server's clock unconverted, so a value
 *  without a zone designator is read as UTC here - and then shown in the
 *  viewer's own time zone. Date-only strings are UTC in JavaScript anyway. */
export function parseTimestamp(value) {
    if (value === null || value === undefined || value === '') return null;
    if (value instanceof Date) return value;
    let text = String(value).trim();
    if (/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(text)) {
        text = `${text.replace(' ', 'T')}Z`;
    }
    const parsed = new Date(text);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
}

/** Whether a timestamp is exactly midnight UTC, i.e. most likely a calendar date
 *  that was stored as a timestamp (claim dates often are). Such a value is shown
 *  as that date, not shifted into the viewer's time zone - where it could land on
 *  the previous day - and without a meaningless "00:00". */
const isUtcMidnight = (parsed) =>
    parsed.getUTCHours() === 0 && parsed.getUTCMinutes() === 0
    && parsed.getUTCSeconds() === 0 && parsed.getUTCMilliseconds() === 0;

export function date(value, { time = false } = {}) {
    if (!value) return '—';
    const parsed = parseTimestamp(value);
    if (!parsed) return String(value);
    const day = { year: 'numeric', month: 'short', day: '2-digit' };
    if (isUtcMidnight(parsed)) {
        return parsed.toLocaleDateString('en-GB', { ...day, timeZone: 'UTC' });
    }
    // The viewer's local time zone (the browser's default).
    return time
        ? parsed.toLocaleString('en-GB', { ...day, hour: '2-digit', minute: '2-digit' })
        : parsed.toLocaleDateString('en-GB', day);
}

export function relative(value) {
    if (!value) return '—';
    const then = parseTimestamp(value)?.getTime() ?? NaN;
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
    const spaced = text.replace(/_+/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

const LANGUAGE_NAMES = (() => {
    try {
        return new Intl.DisplayNames(['en'], { type: 'language' });
    } catch {
        return null;  // A browser without Intl.DisplayNames shows the codes.
    }
})();

/** An ISO language code as its English name: `en` -> `English`, `pt-BR` ->
 *  `Brazilian Portuguese`. Anything the browser does not know stays as it is. */
export function languageName(code) {
    if (!code) return '—';
    try {
        const name = LANGUAGE_NAMES?.of(String(code));
        return name && name.toLowerCase() !== String(code).toLowerCase() ? name : String(code);
    } catch {
        return String(code);  // Not a well-formed language tag.
    }
}

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
    deferred: { tone: 'warn', icon: 'fa-clock' },
    pending: { tone: 'plain', icon: 'fa-hourglass-half' },
};

/** Tone, icon and definition per evidence role. The definitions mirror the
 *  extraction prompt; the `title` becomes the badge's tooltip. */
export const ROLE_STYLE = {
    key: { tone: 'accent', icon: 'fa-star',
        title: 'Establishes a central factual premise underlying the gold verdict. Removing it likely breaks the verdict.' },
    auxiliary: { tone: 'info', icon: 'fa-star-half-stroke',
        title: 'Corroborates, qualifies, or strengthens the main justification without being its principal evidential basis. Removing it would not break the verdict.' },
    background: { tone: 'plain', icon: 'fa-layer-group',
        title: 'Context for understanding the claim or its circumstances, without directly contributing to the justification of the gold verdict.' },
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
    locator_missing: 'The article cites the source without a locator, so there is nothing to retrieve (§3.1).',
    inaccessible: 'The source could not be re-retrieved (§3.1).',
    undated_source: 'No publication time could be determined (see undated_policy).',
    unfaithful: 'The retrieved source no longer supports the proposition (§3.2).',
    after_fact_check: 'The source postdates the fact-check, t_e > t_f (§3.3).',
    fact_check_source: 'The source is itself a professional fact-check and leaks the verdict (§3.3).',
    later_event: 'The proposition rests on a change of the world that happened only after the claim (§3.3).',
    no_gold_verdict: 'The claim has no current gold verdict.',
    no_claim_time: 'The claim has no date t_c.',
    no_fact_check_time: 'No review provides a publication time t_f.',
    no_fact_check_article: 'None of the reviews of the claim has a readable fact-checking article to extract evidence from.',
    extraction_failed: 'Evidence extraction failed for every article; the claim is retried on the next run.',
    nothing_extracted: 'No longer assigned: Stage 1 returned neither evidence nor a verdict rationale.',
    key_evidence_lost: 'No longer assigned: a key evidence item lost every one of its sources in Stage 2.',
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
