/* Hash router and application shell.
 *
 * Routes:
 *   #/                        overview / statistics
 *   #/claims?filters          claim browser
 *   #/claims/:id              claim detail
 *   #/claims/:id/evidence/:id claim detail, scrolled to one evidence item
 */

import { api, ApiError } from './api.js';
import { el, qs, qsa } from './util.js';
import { renderOverview } from './views/overview.js';
import { renderClaims } from './views/claims.js';
import { renderClaim } from './views/claim.js';

const view = qs('#view');

/* ------------------------------------------------------------------- theme */

const THEME_KEY = 'veritas-webui-theme';

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const icon = qs('#theme-toggle i');
    icon.className = `fa-solid fa-${theme === 'dark' ? 'sun' : 'moon'}`;
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* private mode */ }
    const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    applyTheme(stored ?? preferred);
    qs('#theme-toggle').addEventListener('click', () =>
        applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
}

/* ------------------------------------------------------------------ health */

async function pollHealth() {
    const node = qs('#health');
    try {
        const health = await api.health();
        const connected = health.database?.connected;
        node.className = `health ${connected ? 'ok' : 'down'}`;
        qs('.text', node).textContent = connected
            ? `${health.database.name}`
            : 'database unreachable';
        node.title = connected
            ? `Connected to ${health.database.host}:${health.database.port}/${health.database.name}\n`
              + `Media registry: ${health.media.registry_path} `
              + `(${health.media.registry_found ? 'found' : 'NOT found'})`
            : 'The backend cannot reach the database.';
        qs('#version').textContent = `v${health.version}`;
        if (!health.media?.registry_found) {
            node.classList.add('down');
            node.title += '\nMedia will not render: the ezMM registry was not found.';
        }
    } catch {
        node.className = 'health down';
        qs('.text', node).textContent = 'offline';
    }
}

/* ------------------------------------------------------------------ router */

function parseRoute() {
    const hash = window.location.hash.replace(/^#/, '') || '/';
    const [path, search] = hash.split('?');
    const parts = path.split('/').filter(Boolean);
    return { parts, params: new URLSearchParams(search ?? '') };
}

function markNav(route) {
    for (const link of qsa('#nav a')) {
        link.classList.toggle('active', link.dataset.route === route);
    }
}

const navigate = (hash) => { window.location.hash = hash; };

async function render() {
    const { parts, params } = parseRoute();
    view.replaceChildren();
    window.scrollTo(0, 0);

    try {
        if (parts[0] === 'claims' && parts[1]) {
            markNav('claims');
            await renderClaim(view, Number(parts[1]));
        } else if (parts[0] === 'claims') {
            markNav('claims');
            await renderClaims(view, params, { navigate });
        } else {
            markNav('overview');
            await renderOverview(view);
        }
    } catch (error) {
        view.replaceChildren(errorPanel(error));
    }
}

function errorPanel(error) {
    const isApi = error instanceof ApiError;
    return el('div', { class: 'error fade' }, [
        el('i', { class: 'fa-solid fa-triangle-exclamation' }),
        el('h2', { text: isApi && error.status === 404 ? 'Not found' : 'Something went wrong' }),
        el('code', { text: String(error.message ?? error) }),
        el('div', { class: 'chips', style: { marginTop: '10px' } }, [
            el('button', { class: 'btn', onclick: () => render() },
                [el('i', { class: 'fa-solid fa-rotate-right' }), 'Retry']),
            el('a', { class: 'btn', href: '#/' },
                [el('i', { class: 'fa-solid fa-house' }), 'Overview']),
        ]),
    ]);
}

/* -------------------------------------------------------------------- boot */

initTheme();
window.addEventListener('hashchange', render);
render();
pollHealth();
setInterval(pollHealth, 30_000);
