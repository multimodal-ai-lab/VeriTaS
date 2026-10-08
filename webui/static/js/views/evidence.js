/* The evidence browser: filter, search and page through evidence items across
 * all claims.
 *
 * The claim detail view shows one claim's items in their context; this view is
 * the other way round - it makes the evidence itself the unit of inspection, so
 * questions like "which primary social-media sources were discarded as
 * unfaithful?" can be answered without opening one claim after another.
 */

import { api } from '../api.js';
import { renderMultimodal } from '../media.js';
import { math } from '../math.js';
import {
    ADMISSIBILITY_STYLE, badge, date, decimal, el, humanize, languageName, mediaSelect, num,
    PROXIMITY_STYLE, REASON_HELP, ROLE_STYLE, SOURCE_KIND_ICON, truncate, ZONE_HELP,
    ZONE_STYLE,
} from '../util.js';

const SORT_LABELS = {
    id: 'Evidence ID',
    claim: 'Claim ID',
    available_since: 'Availability (t_e)',
    confidence: 'Extraction confidence',
    faithfulness: 'Faithfulness',
    updated: 'Last updated',
};

/** Filters held as a list, so several values can be selected at once. */
const MULTI = ['admissibility', 'zone', 'kind', 'proximity', 'role', 'language'];

/** Filters held as a single value. */
const SINGLE = {
    q: '', reason: '', domain: '', media: '', accessible: '', later_event: '',
    claim_id: '', min_confidence: '', min_faithfulness: '', max_faithfulness: '',
};

const DEFAULTS = { sort: 'id', order: 'desc', limit: 25, offset: 0 };

/** Renders the browser. `params` is the parsed query part of the route hash. */
export async function renderEvidence(root, params, { navigate }) {
    const state = { ...SINGLE, ...DEFAULTS };
    for (const key of MULTI) state[key] = params.getAll(key);
    for (const key of Object.keys(SINGLE)) state[key] = params.get(key) ?? '';
    state.sort = params.get('sort') ?? DEFAULTS.sort;
    state.order = params.get('order') ?? DEFAULTS.order;
    state.limit = Number(params.get('limit') ?? DEFAULTS.limit);
    state.offset = Number(params.get('offset') ?? DEFAULTS.offset);

    const apply = (changes, { resetPage = true } = {}) => {
        const next = { ...state, ...changes };
        if (resetPage) next.offset = 0;
        const search = new URLSearchParams();
        for (const [key, value] of Object.entries(next)) {
            if (value === '' || value === null || value === undefined) continue;
            if (Array.isArray(value)) value.forEach((entry) => search.append(key, entry));
            else if (!(key === 'offset' && !value)) search.set(key, value);
        }
        navigate(`#/evidence?${search.toString()}`);
    };

    /** Toggles one value of a multi-valued filter. */
    const toggle = (key, value) => apply({
        [key]: state[key].includes(value)
            ? state[key].filter((entry) => entry !== value)
            : [...state[key], value],
    });

    root.append(el('div', { class: 'page-head rise' }, [
        el('div', {}, [
            el('div', { class: 'eyebrow', text: 'Gold Evidence Reconstruction' }),
            el('h1', { text: 'Evidence' }),
            el('p', {
                class: 'lede',
                text: 'Every reconstructed evidence item, across all claims. Filter by what '
                    + 'the pipeline recorded about it — admissibility, source kind, proximity, '
                    + 'role, temporal position, language — and search the propositions.',
            }),
        ]),
    ]));

    const toolbar = el('div', { class: 'toolbar rise' });
    const filterbar = el('div', { class: 'toolbar filters rise' });
    root.append(toolbar, filterbar);

    const results = el('div', { id: 'evidence-results' });
    root.append(results);
    results.append(el('div', { class: 'evidence-list' },
        Array.from({ length: 5 }, () => el('div', { class: 'skeleton', style: { height: '168px' } }))));

    const [options, page] = await Promise.all([
        api.evidenceOptions(),
        api.evidenceList(requestParams(state)),
    ]);

    toolbar.replaceChildren(...buildToolbar(state, options, apply, toggle));
    filterbar.replaceChildren(...buildFilterBar(state, options, apply));
    results.replaceChildren(buildResults(page, apply));
}

/** State to query parameters. The API names the list filters in the singular. */
function requestParams(state) {
    return {
        admissibility: state.admissibility,
        zone: state.zone,
        kind: state.kind,
        proximity: state.proximity,
        role: state.role,
        language: state.language,
        q: state.q || null,
        reason: state.reason || null,
        domain: state.domain || null,
        media: state.media || null,
        accessible: state.accessible || null,
        later_event: state.later_event || null,
        claim_id: state.claim_id || null,
        min_confidence: state.min_confidence || null,
        min_faithfulness: state.min_faithfulness || null,
        max_faithfulness: state.max_faithfulness || null,
        sort: state.sort,
        order: state.order,
        limit: state.limit,
        offset: state.offset,
    };
}

/** Everything a user has narrowed down to, so it can be shown and reset. */
function activeFilters(state) {
    const active = [];
    for (const key of MULTI) active.push(...state[key].map(() => key));
    for (const key of Object.keys(SINGLE)) if (state[key] !== '') active.push(key);
    return active;
}

const cleared = () => Object.fromEntries([
    ...MULTI.map((key) => [key, []]),
    ...Object.entries(SINGLE),
]);

/* ----------------------------------------------------------------- toolbar */

function buildToolbar(state, options, apply, toggle) {
    const search = el('input', {
        type: 'search',
        placeholder: 'Search proposition text…',
        value: state.q,
        'aria-label': 'Search proposition text',
    });
    let timer;
    search.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => apply({ q: search.value.trim() }), 350);
    });
    search.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') { clearTimeout(timer); apply({ q: search.value.trim() }); }
    });

    const sortSelect = el('select', {
        'aria-label': 'Sort by',
        onchange: (event) => apply({ sort: event.target.value }),
    }, Object.entries(SORT_LABELS).map(([value, label]) =>
        el('option', { value, selected: state.sort === value, text: label })));

    const orderButton = el('button', {
        class: 'icon-btn',
        title: state.order === 'desc' ? 'Descending' : 'Ascending',
        onclick: () => apply({ order: state.order === 'desc' ? 'asc' : 'desc' }, { resetPage: false }),
    }, [el('i', { class: `fa-solid fa-arrow-${state.order === 'desc' ? 'down' : 'up'}-wide-short` })]);

    return [
        el('div', { class: 'search' }, [el('i', { class: 'fa-solid fa-magnifying-glass' }), search]),
        pills('admissibility', state, options.admissibility, ADMISSIBILITY_STYLE, toggle, apply),
        pills('zone', state, options.zones, ZONE_STYLE, toggle, apply, ZONE_HELP),
        sortSelect,
        orderButton,
    ];
}

function buildFilterBar(state, options, apply) {
    const controls = [
        facet(state, 'kind', 'Any source kind', options.source_kinds, apply),
        facet(state, 'proximity', 'Any proximity', options.proximities, apply),
        facet(state, 'role', 'Any role', options.roles, apply),
        facet(state, 'language', 'Any language', options.languages, apply),
        choice(state, 'domain', 'Any domain', options.domains, apply),
        choice(state, 'reason', 'Any inadmissibility reason',
            options.inadmissibility_reasons, apply, REASON_HELP),
        mediaSelect(state, 'media', apply),
        tristate(state, 'accessible', ['Accessible or not', 'Accessible', 'Inaccessible'], apply),
        tristate(state, 'later_event',
            ['Later event or not', 'Reports a later event', 'No later event'], apply),
    ];

    const active = activeFilters(state);
    if (active.length) {
        controls.push(el('button', {
            class: 'btn ghost',
            onclick: () => apply(cleared()),
        }, [el('i', { class: 'fa-solid fa-rotate-left' }),
            `Reset ${active.length} filter${active.length === 1 ? '' : 's'}`]));
    }
    return controls;
}

/** A multi-select pill group over a small, fixed vocabulary. */
function pills(key, state, entries, styles, toggle, apply, help = {}) {
    const group = el('div', { class: 'pill-group' });
    group.append(el('button', {
        class: state[key].length ? '' : 'active',
        onclick: () => apply({ [key]: [] }),
        text: 'All',
    }));
    for (const entry of entries ?? []) {
        if (!entry.count) continue;  // Nothing to select; keep the bar short.
        const active = state[key].includes(entry.label);
        group.append(el('button', {
            class: active ? 'active' : '',
            title: help[entry.label] ?? `${num(entry.count)} items`,
            onclick: () => toggle(key, entry.label),
        }, [
            el('i', { class: `fa-solid ${styles[entry.label]?.icon ?? 'fa-circle'}` }),
            ` ${humanize(entry.label)}`,
            el('span', { style: { opacity: .55, marginLeft: '5px' }, text: num(entry.count) }),
        ]));
    }
    return group;
}

/** A select over a multi-valued filter. It sets one value; a URL carrying
 *  several keeps them all, and the select then shows how many are in force. */
function facet(state, key, placeholder, entries, apply) {
    const selected = state[key];
    const select = el('select', {
        'aria-label': placeholder,
        onchange: (event) => apply({ [key]: event.target.value ? [event.target.value] : [] }),
    }, [el('option', {
        value: '',
        selected: selected.length === 0,
        text: selected.length > 1 ? `${placeholder} (${selected.length} selected)` : placeholder,
    })]);
    for (const entry of entries ?? []) {
        select.append(el('option', {
            value: entry.label,
            selected: selected.length === 1 && selected[0] === entry.label,
            text: `${key === 'language' ? languageName(entry.label) : humanize(entry.label)} `
                + `(${num(entry.count)})`,
        }));
    }
    return select;
}

/** A select over a single-valued filter. */
function choice(state, key, placeholder, entries, apply, help = {}) {
    const select = el('select', {
        'aria-label': placeholder,
        onchange: (event) => apply({ [key]: event.target.value }),
    }, [el('option', { value: '', text: placeholder })]);
    for (const entry of entries ?? []) {
        select.append(el('option', {
            value: entry.label,
            selected: state[key] === entry.label,
            title: help[entry.label] ?? null,
            text: `${humanize(entry.label)} (${num(entry.count)})`,
        }));
    }
    return select;
}

/** A select over a boolean filter: unset, true, false. */
function tristate(state, key, [any, yes, no], apply) {
    return el('select', {
        'aria-label': any,
        onchange: (event) => apply({ [key]: event.target.value }),
    }, [
        el('option', { value: '', text: any }),
        el('option', { value: 'true', selected: state[key] === 'true', text: yes }),
        el('option', { value: 'false', selected: state[key] === 'false', text: no }),
    ]);
}

/* ----------------------------------------------------------------- results */

function buildResults(page, apply) {
    if (!page.evidence.length) {
        return el('div', { class: 'empty fade' }, [
            el('i', { class: 'fa-solid fa-inbox' }),
            el('div', { text: 'No evidence item matches these filters.' }),
            el('button', { class: 'btn', onclick: () => apply(cleared()) },
                [el('i', { class: 'fa-solid fa-rotate-left' }), 'Reset filters']),
        ]);
    }

    const list = el('div', { class: 'evidence-list' },
        page.evidence.map((item, index) => evidenceCard(item, index)));

    const from = page.offset + 1;
    const to = Math.min(page.offset + page.limit, page.total);
    const pagination = el('div', { class: 'pagination fade' }, [
        el('button', {
            class: 'btn', disabled: page.offset <= 0,
            onclick: () => apply({ offset: Math.max(page.offset - page.limit, 0) }, { resetPage: false }),
        }, [el('i', { class: 'fa-solid fa-chevron-left' }), 'Previous']),
        el('span', { text: `${num(from)}–${num(to)} of ${num(page.total)}` }),
        el('button', {
            class: 'btn', disabled: to >= page.total,
            onclick: () => apply({ offset: page.offset + page.limit }, { resetPage: false }),
        }, ['Next', el('i', { class: 'fa-solid fa-chevron-right' })]),
    ]);

    return el('div', {}, [list, pagination]);
}

/** One *source* as a card, with the proposition it reports.
 *
 *  The browser lists sources because that is what every Stage-2 judgement is about;
 *  the evidence item they belong to is one click away, on the claim detail view. */
function evidenceCard(item, index) {
    const roleStyle = ROLE_STYLE[item.role] ?? { tone: 'plain', icon: 'fa-circle' };
    const proximityStyle = PROXIMITY_STYLE[item.source?.proximity] ?? { tone: 'plain', icon: 'fa-circle' };
    const zoneStyle = ZONE_STYLE[item.zone] ?? ZONE_STYLE.unvalidated;
    const detail = `#/claims/${item.claim_id}/evidence/${item.evidence_id}`;

    const card = el('article', {
        class: 'evidence evidence-card rise',
        style: { '--i': Math.min(index, 8) },
        'data-admissible': String(item.admissible),
    });

    card.append(el('div', { class: 'evidence-head' }, [
        item.admissible === true ? badge('admissible', ADMISSIBILITY_STYLE.admissible)
            : item.admissible === false
                ? badge(item.inadmissibility_reason ?? 'discarded',
                    { ...ADMISSIBILITY_STYLE.inadmissible,
                        title: REASON_HELP[item.inadmissibility_reason] ?? '' })
                : badge('not filtered', ADMISSIBILITY_STYLE.unfiltered),
        badge(item.role ?? 'auxiliary', roleStyle),
        // One proposition can have several sources; the item survives as long as
        // one of them does, so a discarded source need not be a loss.
        (item.n_sources ?? 1) > 1 ? badge(`1 of ${item.n_sources} sources`, {
            tone: 'plain', icon: 'fa-clone',
            title: 'This proposition is reported by several sources.',
        }) : null,
        item.later_event?.change_detected
            ? badge('later event', { tone: 'bad', icon: 'fa-hourglass-half',
                title: REASON_HELP.later_event ?? '' })
            : null,
        item.admissible === false && item.evidence_admissible
            ? badge('item survives', {
                tone: 'ok', icon: 'fa-life-ring',
                title: 'Another source still reports this proposition, so the '
                    + 'evidence item was not lost with this source.',
            })
            : null,
        badge(item.source?.proximity ?? 'secondary', proximityStyle),
        item.zone ? badge(item.zone, { ...zoneStyle, title: ZONE_HELP[item.zone] ?? '' }) : null,
        item.dismissed ? badge('dismissed', { tone: 'bad', icon: 'fa-ban', title: item.dismissed_reason ?? '' }) : null,
        el('span', { class: 'spacer' }),
        el('span', { class: 'eid', title: 'Source ID', text: `#${item.id}` }),
        el('a', {
            class: 'icon-btn', href: detail, title: 'Open in the claim detail view',
            style: { width: '28px', height: '28px' },
        }, [el('i', { class: 'fa-solid fa-arrow-right', style: { fontSize: '.72rem' } })]),
    ]));

    card.append(renderMultimodal(item.proposition, { textClass: 'proposition' }));

    card.append(el('div', { class: 'evidence-source' }, [
        el('i', {
            class: `fa-solid ${SOURCE_KIND_ICON[item.source?.kind] ?? 'fa-circle-question'}`,
            style: { color: 'var(--accent)' },
        }),
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
            [el('i', { class: 'fa-solid fa-calendar-day' }),
                math('t_e'), ` ${date(item.available_since)}`]),
        el('span', { title: 'Extraction confidence (Stage 1)' },
            [el('i', { class: 'fa-solid fa-wand-magic-sparkles' }),
                `confidence ${decimal(item.extraction?.confidence, 2)}`]),
        el('span', { title: 'Whether the source could be re-retrieved (§3.1)' },
            [el('i', { class: `fa-solid ${item.accessible ? 'fa-link' : 'fa-link-slash'}` }),
                item.accessible === null ? 'accessibility unknown' : item.accessible ? 'accessible' : 'inaccessible']),
        item.faithfulness
            ? el('span', {
                title: item.faithfulness.justification
                    || 'Does the source still support the proposition? (§3.2)',
            }, [el('i', { class: 'fa-solid fa-check-double' }),
                `faithfulness ${decimal(item.faithfulness.assessment, 2)}`])
            : null,
        item.later_event?.change_detected
            ? el('span', { title: item.later_event.justification ?? '' },
                [el('i', { class: 'fa-solid fa-triangle-exclamation' }), 'later event'])
            : null,
    ]));

    card.append(el('a', { class: 'evidence-claim', href: detail }, [
        el('span', { class: 'label' }, [
            el('i', { class: 'fa-solid fa-quote-left' }),
            `Claim #${item.claim_id}`,
            item.claim?.language
                ? el('span', { class: 'badge plain', title: item.claim.language,
                    text: languageName(item.claim.language) })
                : null,
            el('span', { class: 'when', text: date(item.claim?.t_c) }),
        ]),
        el('span', { class: 'text', text: truncate(stripRefs(item.claim?.data), 200) }),
    ]));

    return card;
}

/** Media references would clutter the one-line claim preview. */
const stripRefs = (text) => String(text ?? '').replace(/<(image|video|audio):\d+>/g, '🖼').trim();
