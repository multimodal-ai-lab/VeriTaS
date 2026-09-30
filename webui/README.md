# Gold Evidence Web UI

A read-only viewer for the results of the
[Gold Evidence Reconstruction](../veritas/gold_evidence/README.md): browse the
claims the pipeline has processed, inspect every stored detail of each
reconstructed evidence item and source (including their media), and see aggregate
statistics.

It is a separate service. It never imports `veritas`, never writes to the
database — every connection runs with `default_transaction_read_only = on` — and
never calls an LLM or scrapes anything.

## Running it

```bash
docker compose up webui
```

Then open <http://localhost:8080>.

The image is built from the repository root. The service reads its database
credentials from the mounted `config.yaml`; every value can be overridden with an
environment variable, most conveniently through a `.env` file next to
`docker-compose.yaml`:

```dotenv
# Where the pipeline's PostgreSQL runs, as seen from inside the container.
VERITAS_DB_HOST=host.docker.internal
VERITAS_DB_PORT=5432
# Optional: only needed if they should differ from config.yaml.
VERITAS_DB_NAME=veritas_db_3
VERITAS_DB_USER=postgres
VERITAS_DB_PASSWORD=veritas

# The ezMM item registry on the host - the directory holding item_registry.db
# and the image/, video/ and audio/ folders. Required for media rendering.
EZMM_HOST_PATH=/srv/datasets/ezmm

WEBUI_PORT=8080
```

If the pipeline's database runs as another compose service, set
`VERITAS_DB_HOST` to that service's name instead.

### On a podman host

Where `docker` is a shim over rootless podman, `docker compose build` fails at
the last step (`sending tarball` -> `connection reset by peer`): buildx cannot
use podman as a BuildKit backend, so it exports the finished image as a tarball
and pushes it through the podman socket, which drops the upload. Build with
podman itself instead and let compose only run the container:

```bash
podman build -t veritas-webui:latest -f webui/Dockerfile .
docker compose -f docker-compose.yaml -f docker-compose.podman.yaml up -d --no-build webui
```

Forgetting one of the `-f` flags silently falls back to the bridge setup, which
cannot reach a database on the host's loopback - the log then shows a connection
refused for whatever `VERITAS_DB_HOST` names. Put the pair in `.env` so every
`docker compose` invocation picks both up and a plain `docker compose up -d
webui` is enough:

```dotenv
COMPOSE_FILE=docker-compose.yaml:docker-compose.podman.yaml
```

With the override active the container shares the host's network namespace, so
there is no port mapping to apply - the server binds `WEBUI_PORT` itself, and the
UI is on that port of the host. The variable therefore selects the port in both
setups.

Leave `VERITAS_DB_HOST` unset, though: each compose file supplies the right
default for its own networking (`127.0.0.1` here, `host.docker.internal` on
Docker), and pinning it in `.env` overrides both.

The override puts the container into the host's network namespace. Without it
the service cannot reach a PostgreSQL that listens on the host's `127.0.0.1`
(no bridge network can), and `host.docker.internal` does not resolve, because
that name is a Docker invention. It also avoids CNI, whose bridge setup is a
second source of trouble on podman 3: if compose has already created a network,
`podman network ls` may show it with a `VERSION` the installed CNI plugins
reject, and the container then fails to start with `CNI network ... not found`.
Correct the `cniVersion` in `~/.config/cni/net.d/<network>.conflist` to the
version the built-in `podman` network uses, or remove the network.

Without Docker:

```bash
pip install -r webui/requirements.txt
uvicorn webui.main:app --reload --port 8080
```

The health indicator in the top bar shows whether the database and the media
registry were found; hover it for the resolved paths.

### Public access over TLS

The UI can be served publicly over HTTPS by an optional [Caddy](https://caddyserver.com)
front (`webui-tls` in `docker-compose.yaml`), which obtains and renews the
certificate from Let's Encrypt on its own. It is off by default and switched on in
`.env`:

```dotenv
COMPOSE_PROFILES=tls
WEBUI_DOMAIN=veritas.example.org
# Keep plain HTTP on WEBUI_PORT local, so the UI is only reachable via HTTPS.
WEBUI_BIND=127.0.0.1
```

Then `docker compose up -d` starts both services, and the UI is at
`https://veritas.example.org`. Requirements:

- `WEBUI_DOMAIN` resolves (A/AAAA record) to this host's public address.
- Ports 80 and 443 are reachable from the internet - 80 for Let's Encrypt's HTTP
  challenge and the redirect to HTTPS - and not used by another web server.
- On a podman host (`COMPOSE_FILE=docker-compose.yaml:docker-compose.podman.yaml`)
  the front shares the host's network like the UI. Rootless podman can bind 80 and
  443 only after `sudo sysctl net.ipv4.ip_unprivileged_port_start=80` (persist it
  in `/etc/sysctl.d/`). There, `WEBUI_BIND=127.0.0.1` makes the UI itself bind
  the loopback only.

Certificates live in the `caddy_data` volume, so a restart does not request new
ones. `WEBUI_UPSTREAM` overrides where the front finds the UI (default `webui:<port>`
on Docker, `127.0.0.1:<port>` on a podman host).

**The UI has no login.** Serving it publicly makes every stored claim, evidence item,
source and model reasoning trace readable by anyone who knows the address - it is
read-only, but not private. Restrict access (e.g. Caddy `basic_auth`, or an IP
allow-list in the firewall) if the data must not be public yet.

## What it shows

**Overview** — claim statuses and rejection reasons, evidence admissibility and
why candidates were discarded, source kind / proximity / role distributions, the
most frequent source domains, faithfulness ratings, the distributions of
`t_e − t_c` and `t_e − t_f` (on a symmetric logarithmic axis), the
reconstruction funnel, and the `E_c` vs `E_f` recoverability
contingency. These are live summaries of the current database state; the numbers
for a write-up come from `scripts.gold_evidence.run_temporal_analysis`, which
remains the single source of truth for the reported statistics.

**Claims** — filter by reconstruction status, rejection reason, language,
modality (any media / images / videos / text only) and release state, search the
claim text, and sort by ID, claim date,
status or when the pipeline last touched the claim.

**Evidence** — the same browser one level down, with the *source* rather than the
claim as the unit. An evidence item is one proposition and the sources that report
it, and every Stage-2 judgement — accessibility, `t_e`, faithfulness, the temporal
checks — belongs to a source, so that is what a row is. Filter by admissibility
(and, for discarded sources, by the reason), by where the source falls relative to
the studied interval `t_c < t_e ≤ t_f`, by source kind, proximity, role, domain,
claim language, modality, accessibility and whether it reports a later event;
search the propositions; sort by ID, claim, `t_e`, extraction confidence or
faithfulness. A card carrying a discarded source says whether its item survived
anyway — it does as long as one of its sources did — and links into the claim
detail view, where the whole item is shown.

**Claim detail** — the claim itself (with its media), the reference times
`t_c` and `t_f`, the gold verdict, the professional fact-checks behind it, a
timeline placing every dated source relative to the studied interval
`t_c < t_e ≤ t_f`, the sufficiency-validation results per condition, the verdict
rationale, and one card per evidence item.

Each evidence card shows the proposition, its role and whether the item survived,
followed by one block per source: that source's admissibility, faithfulness and
temporal validation up front, and every remaining stored value — the model
reasoning traces, the retrieved source content, all flat columns and the complete
`full_source` blob — behind disclosures. The item's own extraction reasoning,
columns and `full_evidence` blob sit on the card itself. Nothing that is stored is
hidden.

## Multimodal propositions

A proposition may contain ezMM references such as `<image:105202>`. The UI keeps
those references visible inline as chips and renders the referenced media
directly beneath the text. Hovering a chip highlights its medium and vice versa;
clicking a chip scrolls to the medium, clicking an image opens it full size.
Videos stream with range requests, so seeking works.

Media files are resolved through the ezMM registry (`item_registry.db`). Paths
recorded there were written by the machine that ran the pipeline; when such a
path does not exist inside the container, the UI falls back to the canonical
`<registry root>/<kind>/<id>.<ext>` layout. References whose file cannot be found
are shown as a red chip rather than silently dropped.

## How a claim is loaded

Opening a claim used to transfer every scraped source document and JSONB blob the
claim holds - a claim with twenty items and several sources each runs to
megabytes, all of it behind collapsed cards. The claim endpoint therefore returns
a light column set (`EVIDENCE_ITEM_SUMMARY_COLUMNS` / `SOURCE_SUMMARY_COLUMNS`):
enough for a collapsed card, nothing more. Expanding an item fetches
`/api/evidence/items/{id}`, which carries the reasoning traces, the retrieved
source content and the raw objects.

Media resolution is `stat`-only for the same reason: the store keeps hundreds of
thousands of files in one directory per kind, so globbing it once per reference
dominated the response. Misses are cached, so a reference the store does not hold
is probed once rather than on every render.

## Retrieved source content

A source's stored content is Markdown, and it is rendered as such rather than
shown as preformatted text. `static/js/markdown.js` handles the subset scraped
articles use - headings, lists, quotes, code, links, emphasis - and builds the
result with `createElement`/`textContent` only. No string from a scraped page
ever reaches `innerHTML`, so markup in a source cannot execute, and link targets
are restricted to `http(s)` and `mailto`. Media references are handed back to the
caller mid-parse, so they stay interactive chips pointing at the media rendered
with them.

## Notation

The UI uses the same symbols as the pipeline's documentation, typeset with
MathJax: `t_c` (claim time), `t_f` (fact-check time), `t_e` (evidence time),
`E_c` and `E_f` for the two evidence sets, and the closeness threshold. Every
math span carries a plain-text fallback, so the page stays readable when the
MathJax CDN is unreachable.

The value colours - positive, negative, neutral, plus the TUDa orange and blue -
are the plotting palette from `scripts/stats/common.py`, so a chart here and a
figure in a write-up use the same hues. They live as CSS tokens in
`static/css/style.css`; the `*-ink` variants are the same hues adjusted for text
contrast.

The `t_e - t_c` / `t_e - t_f` histograms bin on `sign(v) * log10(1 + |v|)`, with
zero as a bin *edge*: no bar spans the reference time, so a bar is always
entirely before or entirely after it. A difference of exactly zero counts as
"before", matching the admissibility rule.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Database and media-registry status |
| `GET /api/stats` | Everything the overview renders |
| `GET /api/filters` | Available statuses and rejection reasons, with counts |
| `GET /api/claims` | Paginated, filtered claim list |
| `GET /api/claims/{id}` | Claim, verdict, reviews, evidence, results, timeline |
| `GET /api/evidence` | Paginated, filtered list of evidence sources across all claims |
| `GET /api/evidence/options` | Available evidence facets, with counts |
| `GET /api/evidence/{id}` | One evidence source, fully expanded, with its item |
| `GET /api/evidence/items/{id}` | One evidence item with every stored field of it and its sources |
| `GET /api/media/{kind}/{id}` | The media file (supports range requests) |
| `GET /api/media/{kind}/{id}/meta` | Media metadata without the payload |

Interactive documentation is served at `/docs`.

## Tests

```bash
pytest tests/webui
```

They cover the media-reference parsing, the aggregation helpers (including the
symmetric-log histogram and the funnel), both SQL filter
builders and the row-to-payload serialization. Like the other tests in this
repository they are pure: no database, no network.

## Note on `config.yaml`

`config.yaml` holds API keys as well as database credentials. It is mounted
read-only and the UI reads only its `database` and `ezmm_path` entries, but the
file is still inside the container. If that is not acceptable in your deployment,
drop the volume mount and pass `VERITAS_DB_*` and `EZMM_PATH` as environment
variables instead — the service works without `config.yaml`.
