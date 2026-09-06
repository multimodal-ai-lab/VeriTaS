/* Thin wrapper around the JSON API. */

class ApiError extends Error {
    constructor(message, status) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

async function request(path, params) {
    const url = new URL(path, window.location.origin);
    for (const [key, value] of Object.entries(params ?? {})) {
        if (value === null || value === undefined || value === '') continue;
        if (Array.isArray(value)) value.forEach((entry) => url.searchParams.append(key, entry));
        else url.searchParams.set(key, value);
    }

    let response;
    try {
        response = await fetch(url, { headers: { Accept: 'application/json' } });
    } catch (error) {
        throw new ApiError(`Cannot reach the server (${error.message}).`, 0);
    }

    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const body = await response.json();
            if (body?.detail) detail = body.detail;
        } catch { /* the body was not JSON; keep the status line */ }
        throw new ApiError(detail, response.status);
    }
    return response.json();
}

export const api = {
    health: () => request('/api/health'),
    stats: () => request('/api/stats'),
    filters: () => request('/api/filters'),
    claims: (params) => request('/api/claims', params),
    claim: (id) => request(`/api/claims/${id}`),
    evidence: (id) => request(`/api/evidence/${id}`),
    mediaUrl: (kind, id) => `/api/media/${kind}/${id}`,
};

export { ApiError };
