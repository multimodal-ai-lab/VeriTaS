/* The claim browser: filter, search and page through processed claims. */

import { api } from '../api.js';
import {
    badge, date, el, humanize, num, REASON_HELP, relative, STATUS_STYLE, truncate,
} from '../util.js';

const SORT_LABELS = {
    updated: 'Last processed',
    id: 'Claim ID',
    date: 'Claim date (t_c)',
    status: 'Status',
};

/** Renders the browser. `params` is the parsed query part of the route hash. */
export async function renderClaims(root, params, { navigate }) {
    const state = {
        status: params.getAll('status'),
        q: params.get('q') ?? '',
        reason: params.get('reason') ?? '',
        has_media: params.get('has_media') ?? '',
        released: params.get('released') ?? '',
        sort: params.get('sort') ?? 'updated',
        order: params.get('order') ?? 'desc',
        limit: Number(params.get('limit') ?? 25),
        offset: Number(params.get('offset') ?? 0),
    };

    const apply = (changes, { resetPage = true } = {}) => {
        const next = { ...state, ...changes };
        if (resetPage) next.offset = 0;
        const search = new URLSearchParams();
        for (const [key, value] of Object.entries(next)) {
            if (value === '' || value === null || value === undefined) continue;
            if (Array.isArray(value)) value.forEach((entry) => search.append(key, entry));
            else if (!(key === 'offset' && !value)) search.set(key, value);
        }
        navigate(`#/claims?${search.toString()}`);
    };

    root.append(el('div', { class: 'page-head rise' }, [
        el('div', {}, [
            el('div', { class: 'eyebrow', text: 'Gold Evidence Reconstruction' }),
            el('h1', { text: 'Claims' }),
            el('p', {
                class: 'lede',
                text: 'Every claim the reconstruction has processed, with the state it reached '
                    + 'and how much of its evidence survived filtering.',
            }),
        ]),
    ]));

    const toolbar = el('div', { class: 'toolbar rise' });
    root.append(toolbar);

    const results = el('div', { id: 'claim-results' });
    root.append(results);
    results.append(el('div', { class: 'claim-list' },
        Array.from({ length: 6 }, () => el('div', { class: 'skeleton', style: { height: '124px' } }))));

    const [filters, page] = await Promise.all([
        api.filters(),
        api.claims({
            status: state.status,
            q: state.q || null,
            reason: state.reason || null,
            has_media: state.has_media || null,
            released: state.released || null,
            sort: state.sort,
            order: state.order,
            limit: state.limit,
            offset: state.offset,
        }),
    ]);

    toolbar.replaceChildren(...buildToolbar(state, filters, apply));
    results.replaceChildren(buildResults(page, apply));
}

function buildToolbar(state, filters, apply) {
    const search = el('input', {
        type: 'search',
        placeholder: 'Search claim text…',
        value: state.q,
        'aria-label': 'Search claim text',
    });
    let timer;
    search.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => apply({ q: search.value.trim() }), 350);
    });
    search.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') { clearTimeout(timer); apply({ q: search.value.trim() }); }
    });

    const statusPills = el('div', { class: 'pill-group' });
    const isAll = state.status.length === 0;
    statusPills.append(el('button', {
        class: isAll ? 'active' : '',
        onclick: () => apply({ status: [] }),
        text: 'All',
    }));
    for (const entry of filters.statuses ?? []) {
        const active = state.status.includes(entry.label);
        statusPills.append(el('button', {
            class: active ? 'active' : '',
            title: `${num(entry.count)} claims`,
            onclick: () => apply({
                status: active
                    ? state.status.filter((value) => value !== entry.label)
                    : [...state.status, entry.label],
            }),
        }, [
            el('i', { class: `fa-solid ${STATUS_STYLE[entry.label]?.icon ?? 'fa-circle'}` }),
            ` ${humanize(entry.label)}`,
            el('span', { style: { opacity: .55, marginLeft: '5px' }, text: num(entry.count) }),
        ]));
    }

    const reasonSelect = el('select', {
        'aria-label': 'Rejection reason',
        onchange: (event) => apply({ reason: event.target.value }),
    }, [el('option', { value: '', text: 'Any rejection reason' })]);
    for (const entry of filters.rejection_reasons ?? []) {
        reasonSelect.append(el('option', {
            value: entry.label,
            selected: state.reason === entry.label,
            text: `${humanize(entry.label)} (${num(entry.count)})`,
        }));
    }

    const mediaSelect = el('select', {
        'aria-label': 'Media filter',
        onchange: (event) => apply({ has_media: event.target.value }),
    }, [
        el('option', { value: '', text: 'Any modality' }),
        el('option', { value: 'true', selected: state.has_media === 'true', text: 'With media' }),
        el('option', { value: 'false', selected: state.has_media === 'false', text: 'Text only' }),
    ]);

    const releasedSelect = el('select', {
        'aria-label': 'Release filter',
        onchange: (event) => apply({ released: event.target.value }),
    }, [
        el('option', { value: '', text: 'Released or not' }),
        el('option', { value: 'true', selected: state.released === 'true', text: 'Released only' }),
        el('option', { value: 'false', selected: state.released === 'false', text: 'Unreleased only' }),
    ]);

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
        statusPills,
        reasonSelect,
        mediaSelect,
        releasedSelect,
        sortSelect,
        orderButton,
    ];
}

function buildResults(page, apply) {
    if (!page.claims.length) {
        return el('div', { class: 'empty fade' }, [
            el('i', { class: 'fa-solid fa-inbox' }),
            el('div', { text: 'No claim matches these filters.' }),
            el('button', { class: 'btn', onclick: () => apply({ status: [], q: '', reason: '', has_media: '', released: '' }) },
                [el('i', { class: 'fa-solid fa-rotate-left' }), 'Reset filters']),
        ]);
    }

    const list = el('div', { class: 'claim-list' },
        page.claims.map((claim, index) => claimCard(claim, index)));

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

function claimCard(claim, index) {
    const style = STATUS_STYLE[claim.status] ?? { tone: 'plain', icon: 'fa-circle' };
    const total = Math.max(claim.n_evidence, 1);

    return el('a', {
        class: 'claim-card rise',
        style: { '--i': index },
        href: `#/claims/${claim.id}`,
    }, [
        el('div', { class: 'row' }, [
            badge(claim.status ?? 'unprocessed', style),
            claim.reason
                ? badge(claim.reason, { tone: 'plain', icon: 'fa-circle-info', title: REASON_HELP[claim.reason] ?? '' })
                : null,
            claim.n_multimodal > 0
                ? badge(`${claim.n_multimodal} multimodal`, { tone: 'violet', icon: 'fa-photo-film' })
                : null,
            claim.released ? badge('released', { tone: 'info', icon: 'fa-box-open' }) : null,
            claim.is_rectified ? badge('rectified', { tone: 'plain', icon: 'fa-pen-nib' }) : null,
            el('span', { class: 'id', style: { marginLeft: 'auto' }, text: `#${claim.id}` }),
        ]),

        el('p', { class: 'text', text: truncate(stripRefs(claim.data), 340) }),

        el('div', { class: 'meta' }, [
            el('span', { title: 'Claim date t_c' },
                [el('i', { class: 'fa-solid fa-calendar-day' }), date(claim.t_c)]),
            el('span', { title: 'Latest fact-check publication t_f' },
                [el('i', { class: 'fa-solid fa-flag-checkered' }), date(claim.t_f)]),
            el('span', { title: 'Candidates / admissible' },
                [el('i', { class: 'fa-solid fa-layer-group' }),
                    `${num(claim.n_admissible)} of ${num(claim.n_evidence)} admissible`]),
            claim.language ? el('span', {}, [el('i', { class: 'fa-solid fa-language' }), claim.language]) : null,
            el('span', { style: { marginLeft: 'auto' }, title: claim.updated_at ?? '' },
                [el('i', { class: 'fa-solid fa-clock-rotate-left' }), relative(claim.updated_at)]),
        ]),

        claim.n_evidence > 0
            ? el('div', { class: 'evidence-bar' }, [
                el('span', { class: 'adm', style: { width: `${(claim.n_admissible / total) * 100}%` } }),
                el('span', { class: 'inadm', style: { width: `${(claim.n_inadmissible / total) * 100}%` } }),
                el('span', { class: 'pend', style: { width: `${(claim.n_unfiltered / total) * 100}%` } }),
            ])
            : null,
    ]);
}

/** Media references would clutter the one-line preview; the detail view keeps them. */
const stripRefs = (text) => String(text ?? '').replace(/<(image|video|audio):\d+>/g, '🖼').trim();
