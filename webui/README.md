# Gold Evidence Web UI

A read-only viewer for the results of the
[Gold Evidence Reconstruction](../veritas/gold_evidence/README.md): browse the
claims the pipeline has processed, inspect every stored detail of each
reconstructed evidence item (including its media), and see aggregate statistics.

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

Without Docker:

```bash
pip install -r webui/requirements.txt
uvicorn webui.main:app --reload --port 8080
```

The health indicator in the top bar shows whether the database and the media
registry were found; hover it for the resolved paths.

## What it shows

**Overview** — claim statuses and rejection reasons, evidence admissibility and
why candidates were discarded, source kind / proximity / role distributions, the
most frequent source domains, faithfulness ratings, the distributions of
`t_e − t_c` and `t_e − t_f`, and the `E_claim` vs `E_factcheck` recoverability
contingency. These are live summaries of the current database state; the numbers
for a write-up come from `scripts.gold_evidence.run_temporal_analysis`, which
remains the single source of truth for the reported statistics.

**Claims** — filter by reconstruction status, rejection reason, modality and
release state, search the claim text, and sort by ID, claim date, status or when
the pipeline last touched the claim.

**Claim detail** — the claim itself (with its media), the reference times
`t_c` and `t_f`, the gold verdict, the professional fact-checks behind it, a
timeline placing every dated evidence item relative to the studied interval
`t_c < t_e ≤ t_f`, the sufficiency-validation results per condition, and one
card per evidence item.

Each evidence card shows the proposition, source, admissibility, faithfulness and
temporal validation up front, and every remaining stored value — extraction
reasoning, both model reasoning traces, the retrieved source content, all flat
columns and the complete `full_evidence` blob — behind disclosures. Nothing that
is stored is hidden.

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

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Database and media-registry status |
| `GET /api/stats` | Everything the overview renders |
| `GET /api/filters` | Available statuses and rejection reasons, with counts |
| `GET /api/claims` | Paginated, filtered claim list |
| `GET /api/claims/{id}` | Claim, verdict, reviews, evidence, results, timeline |
| `GET /api/evidence/{id}` | One evidence item, fully expanded |
| `GET /api/media/{kind}/{id}` | The media file (supports range requests) |
| `GET /api/media/{kind}/{id}/meta` | Media metadata without the payload |

Interactive documentation is served at `/docs`.

## Tests

```bash
pytest tests/webui
```

They cover the media-reference parsing, the aggregation helpers, the SQL filter
builder and the row-to-payload serialization. Like the other tests in this
repository they are pure: no database, no network.

## Note on `config.yaml`

`config.yaml` holds API keys as well as database credentials. It is mounted
read-only and the UI reads only its `database` and `ezmm_path` entries, but the
file is still inside the container. If that is not acceptable in your deployment,
drop the volume mount and pass `VERITAS_DB_*` and `EZMM_PATH` as environment
variables instead — the service works without `config.yaml`.
