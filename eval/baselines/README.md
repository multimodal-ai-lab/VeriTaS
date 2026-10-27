# Baselines

The fact-checking baseline for the VeriTaS benchmark, with support for OpenAI, Gemini, Anthropic, self-hosted models (via vLLM), and OpenRouter.

There is exactly one baseline, with one prompt. Every provider runs it the same way:

1. The model gets one prompt (`common/prompts.py`, no system prompt) with the claim and the day it was published
   (rendered like the VeriTaS pipeline renders claims), plus any images/videos. It also says what day it is
   ("Today is October 19, 2026."): the day the prompt is built, which the trace's prompt records.
2. It gathers evidence with two tools:
   - `web_search(query)`: Google results (via the scrapeMM server's Serper search), **restricted to content published before the claim's fact-check** (see "Cutoff semantics"). Returns titles, URLs, dates and snippets.
   - `fetch_url(url)`: the main content of a web page, without navigation, headers, footers, cookie banners
     and the like (via the scrapeMM server, multimodal output, `strip=True`): its text, cut at
     **25,000 characters**, with its **images and videos in their places**, as scrapeMM delivers the page (an
     ezMM `MultimodalSequence`). How many images and videos are shown is limited per page and per fact-check
     (see [Media limits](#media-limits)). A note takes the place of the others (`[image not shown]`), of ones
     shown before and of images that cannot be read (SVG, truncated files). Videos over 50 MB are downloaded at
     a lower resolution. A fetch that takes longer than 3 minutes is given up
     ("Could not retrieve <url>: the page took too long to load."): fetches have hung on the scrapeMM server.
     It only fetches **URLs the model has seen**: in `web_search` results or in pages it fetched. They are
     compared without scheme, `www.`, trailing slash and fragment, with the host in lower case and
     percent-encoding decoded, as the model may write them so. Any other URL, even on a site the model has seen,
     is rejected: otherwise the model could get around the cutoff, e.g. by fetching a search engine's results
     page, an archived copy (`web.archive.org/...`) or a fact-check it remembers.
   - **The fact-checks of all IFCN signatories and EFCSN members are blocked**, but not the rest of their sites.
     At its start, `run_benchmark.py` reads the `fact_check_scopes` of every publisher in the database (table
     `publishers`) whose IFCN status is `active` or `in_renewal`, or whose EFCSN status is `active`. A scope is a
     host, or a host and a path, under which the publisher publishes fact-checks (`leadstories.com/hoax-alert`,
     `verifica.efe.com`), or a whole site (`fullfact.org`, `voxukraine.org`). A publisher without scopes (66 of
     the 201) gives its `domains` instead, which block their hosts as a whole (`france24.com`, `kompas.com`), but
     not their subdomains (`observers.france24.com`). Some domains in the database are only public suffixes
     (`com.br`, `qc.ca`, `or.id`, `org.tw`, `com.tn`) and block nothing. `run_config.json` records all of them
     as `signatory_scopes` (359 in October 2026). Each claim adds its own fact-checks (`review_urls`, or `review_url`,
     in the dataset) and the scopes of their publishers. A URL is blocked if its host is one of these, `www.`
     aside, and its path is at or under one of their paths, whatever its scheme, query, fragment and case.
     Subdomains have scopes of their own. The pages under a claim's fact-check (`…/amp`) are blocked with it,
     and so is any URL with its slug in the path, on any site: the slug is the last path segment with a letter
     in it, if it has at least 12 characters. Sites serve an article under several paths and on subdomains
     (AAP republishes `aap.com.au/factcheck/<slug>` as `aapnews.aap.com.au/news/<slug>`), and archives and
     other sites copy it.
     - `web_search` drops the results that are blocked, and those whose title or snippet shows a blocked URL,
       with or without scheme. That includes URLs that Google broke with spaces
       (`https://maldita. es/desinfo/…`), and URLs cut short if what shows is the start of a blocked URL or scope
       (`https://leadstories.com/hoa …`). So are the results whose title or snippet has the slug of the claim's
       fact-check, in a URL or not, also broken with spaces by Google (`2026-09-15- maryam-…`) or cut short at
       either end to a piece of at least 20 characters (`… detained-for-laundering-11-million.html.`). These are
       mostly the fact-checkers' own posts, also on Instagram and Facebook, which give a fact-check's headline
       with its URL or slug. To still show ten results, `web_search` requests further pages of
       Google results, up to `MAX_SEARCH_PAGES` (3). It stops once it has ten, or at a page that brings no result
       it has not seen. A page with fewer than ten results is not always the last, and later pages repeat some
       results. The log line counts the dropped results and the pages (`-> 10 results (4 blocked, 2 pages)`), and
       a line before it names each dropped result and the rule that dropped it (`dropped https://fullfact.org/…, a
       URL in the fact-check scope fullfact.org`, `…, whose title or snippet has '…', a piece of the slug … of the
       claim's fact-check`).
     - `fetch_url` answers for a blocked URL as for any URL the model has not seen, so a model that guesses a
       fact-check's URL does not learn that it exists. The log names the rule (`-> blocked, a URL in the
       fact-check scope …`).
     - Fetched pages lose their links to blocked URLs. A markdown link goes with its text and title, which may
       give the verdict away as a headline does, and with the images in its text: a teaser of an article is one
       link with its image, date and headline. If the text holds brackets or is longer than 1000 characters, where
       it begins is uncertain, and only the link's target and title go. Bare URLs go too, also those in HTML
       links, whose text stays. The log line counts the removed links (`5 blocked links removed`), and one line
       per rule lists their URLs (`removed the links to 3 URLs in the fact-check scope fullfact.org: …`).

     Without the blocking, the models reached the claim's own fact-check in up to a tenth of the Q3 claims
     (Gemini in 8%, with an MSE of 0.03 on those against 0.21). Google's date filter lets fact-checks through at
     times, and fact-checkers' pages link to them in their current versions: in tag pages, roundups and lists of
     latest articles. Replayed on Gemini 3.8 Flash's Q3 traces, the blocking drops 15.2% of the search results
     (in 538 of the 1000 claims), refuses 528 of 2,500 fetches, and removes 13,913 links from 584 of 2,262
     fetched pages. Not blocked:
     - copies and archives of other fact-checks than the claim's own on other sites
       (`web.archive.org/web/…/<url>`);
     - fact-checks of publishers that are neither IFCN signatories nor EFCSN members, unless they reviewed the
       claim;
     - fact-checks of signatories without scopes on subdomains or on domains missing from the database (Taiwan
       FactCheck Center and Agence Science-Presse have only a public suffix there);
     - the text around a removed link, e.g. a post's headline.
3. It answers in JSON:
   ```json
   {
     "integrity": "compromised" | "intact" | "unknown",
     "certainty": "certain" | "rather certain" | "rather uncertain" | null,
     "justification": "<1-4 sentences with the evidence URLs as Markdown hyperlinks>"
   }
   ```
   The answer is the last JSON object in the response with an `integrity` field (in a ```` ```json ```` block or not).
   Integrity and certainty make the verdict in the 7-class label scheme, e.g. `intact` + `rather certain` →
   `Intact (rather certain)`; `unknown` makes `Unknown`. A final response without a valid answer (no JSON, or no
   valid integrity and certainty, e.g. because it was cut off or blocked) is not a verdict: the claim gets the
   status `no_answer` and runs again on `--resume`.

Per claim, the model may call `web_search` up to 40 times and `fetch_url` up to 40 times (`DEFAULT_MAX_SEARCHES`,
`DEFAULT_MAX_FETCHES` in `common/tools.py`), in at most 50 model calls (`MAX_TURNS` in `providers/base.py`). The
last model call is made without tools, to force a verdict, and the model is told so: "You have used up your whole
quota of tool calls. You must provide an answer given the information you were able to gather so far." Switching
tools off alone is not enough for every model: gemini-3.8-flash called tools anyway, and then gave no answer.

### How the models see media

The prompt and every tool result are ezMM sequences: text, and ezMM items (images, videos) in their places. They
are kept as lists, not as `MultimodalSequence`s: those take any `<image:7>` in text (a page's, an error message's)
for a reference to an item, and fail if there is no such item.
The claim's images and videos follow its text, inside the quotes (the dataset gives no position for them in it):
`"<claim text> <image:1>` [image] `"` `Published on: ...`. Each provider gets a sequence in order, with each image
and video right after its reference (`<image:40>`, `<video:1>`), so that the model can name it:

| Provider | Images | Videos | Tool results with media |
|----------|--------|--------|-------------------------|
| `openai`, `anthropic` | In place | 5 frames each, in place | In the tool result, interleaved |
| `gemini` | In place | Natively, uploaded with the File API and deleted when the claim is done | After the function responses, in the same turn: a function response holds JSON, and the model ignores uploaded videos in its parts |
| `selfhosted`, `openrouter` | In place | 5 frames each, in place | In a user message after the tool messages, which take only text |

The claim's images are sent as they are, unless larger than 2000 px: re-encoding would blur the traces of editing a
fact-checker looks for. A page's images and all frames of page videos are JPEGs of at most 1024 px. Page videos
longer than 2 minutes reach Gemini as frames too: native video costs tokens by the second, in every model call.

### Media limits

Every model call sends the whole conversation again, with the media of all pages fetched so far. The limits keep it
within the APIs' request limits (Anthropic: 100 images per request, and at most 2000 px per side once there are more
than 20) and the context window. They are fields of `MediaLimits` (`common/media.py`), passed to `UnifiedFactChecker`
and the providers as `media_limits`, and options of `run_benchmark.py`:

| Option | Default | Meaning |
|--------|---------|---------|
| `--page-images` | 10 | Images shown per fetched page, the first ones in page order |
| `--page-videos` | 2 | Videos shown per fetched page, the first ones in page order |
| `--total-images` | 40 | Images of fetched pages shown in a whole fact-check |
| `--total-videos` | 4 | Videos of fetched pages shown in a whole fact-check |
| `--request-images` | none | Images in one request at most, the claim's included and a video counting as `--video-frames` images: page images and videos beyond them are not shown. For hosts that take fewer than the limits above add up to, e.g. 64 on OpenRouter's Near AI |
| `--video-frames` | 5 | Frames per video for models without native video (all but Gemini) |
| `--native-video-seconds` | 120 | Page videos longer than this reach models with native video as frames too. The claim's videos are always native |
| `--page-image-side` | 1024 | Longest side, in pixels, of the page images and of the frames of page videos |
| `--claim-image-side` | 2000 | Longest side, in pixels, of the claim's images and frames: larger ones are scaled down, the others stay as they are |

With the defaults, a model without native video gets at most 40 page images and 20 frames of page videos, plus the
claim's media (in Q3: at most 4 images and 2 videos). `run_config.json` records the limits of a run.

### Where the media go

scrapeMM downloads every image and video of a fetched page (not only those shown) and registers each in ezMM's store,
an SQLite database. `run_benchmark.py` keeps that store in a temporary folder on local disk (in `$TMPDIR`, else
`/tmp`) instead of the repository's `.ezmm/`, and deletes it at the end: on `/mnt/vast`, a network file system, each
registration waited for a database lock, which made fetches take minutes, and Slurm jobs on several nodes sharing the
database could corrupt it. A claim's page media are deleted when its fact-check ends, unless a claim running in
parallel uses the same file. Nothing is kept: the trace records each medium's source URL instead.

### Reasoning effort

Every provider gets the same reasoning effort (`--reasoning-effort`, default `medium`), in its API's own setting:

| Provider | Setting |
|----------|---------|
| `openai` | `reasoning.effort` |
| `anthropic` | `output_config.effort` |
| `gemini` | `thinking_config.thinking_level` (upper case, e.g. `MEDIUM`). Not for Gemini 2 models, which reject it: they think as much as they decide to |
| `selfhosted` | `reasoning_effort` |
| `openrouter` | `reasoning.effort` |

Which values a model takes depends on its vendor and on the model itself. Reasoning is always on, so there is
no `none`. A model that does not take the value fails every request with a 400 error.

### Retries

Rate limits and transient server errors (HTTP 429, 500, 502, 503, 504 and Anthropic's 529 "overloaded") are
retried up to 5 times per model call, with exponential backoff (2 s, 4 s, 8 s, ...). The OpenAI and Anthropic SDKs
retry them twice more themselves. Other errors fail the claim at once; `--resume` runs it again. A failed claim keeps
what it got to: its trace, evidence URLs and token usage are written like those of a successful one, with the status
`error` and the error message.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset
You need to run the export script in the VeriTaS repo to generate the claims dataset.

A dataset export (a `claims.json` holding a list of claims, with a `readme.txt`, `images/` and `videos/`) is
converted into the format `run_benchmark.py` reads with
```bash
python eval/baselines/scripts/convert_export.py <export_dir>
```
which writes `<export_dir>/claims_benchmark.json`. Pass that file as `--dataset`.

## Configuration

Run everything from the repository root: the configuration is read from `config.yaml` there.

**scrapeMM server** (required, used by both tools):
```yaml
scrapemm:
  url: http://<scrapemm-server>:<port>
  api_key: <key>
```
The Serper API key for `web_search` is configured on the scrapeMM server (Secrets in its web UI), not here. Without `config.yaml`, the scrapeMM client falls back to the `SCRAPEMM_API_URL` / `SCRAPEMM_API_KEY` environment variables.

**Model providers.** API keys can be configured in three ways (in order of priority):

1. **Explicit parameters**: Pass an `api_keys` dict when creating `UnifiedFactChecker`
2. **`config.yaml`** (recommended):
   ```yaml
   api_secrets:
     openai: "sk-..."
     google: "AIza..."
     anthropic: "sk-ant-..."
     openrouter: "sk-or-..."

   selfhosted:          # vLLM's OpenAI-compatible endpoint
     url: http://<vllm-host>:<port>/v1
     key: <key>
   ```
3. **Environment variables**: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_BASE_URL` / `VLLM_API_KEY`

**VeriTaS database** (required): at its start, `run_benchmark.py` reads the fact-check scopes of all IFCN
signatories and EFCSN members, and looks up the claims' reviews by their URLs, for their publishing dates (if the
dataset has no `review_date`) and their publishers' fact-check scopes. The connection is read-only, so the
database refuses any change.
```yaml
database:
  database: veritas_db_3
  user: <user>
  password: <password>
  host: <host>
  port: 5432
```

## Providers

| Provider | Default Model | Native Video |
|----------|---------------|--------------|
| `openai` | gpt-5.2 | No (extracts frames) |
| `gemini` | gemini-2.5-flash | Yes |
| `anthropic` | claude-sonnet-4-6 | No (extracts frames) |
| `selfhosted` | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | No (extracts frames) |
| `openrouter` | z-ai/glm-5.3-flash | No (extracts frames) |

Any model with function calling can be used via `--model`. For `openrouter`, this is any OpenRouter model ID that supports tools and reasoning, and `--openrouter-host` is required: every request goes to that one endpoint (`provider.only`), named by its `tag` in `https://openrouter.ai/api/v1/models/<model>/endpoints`, e.g. `deepinfra/fp8`. Otherwise OpenRouter would spread the requests over the model's hosts, which differ in quantization and context length. Pick a host whose `supports_tool_choice` has both `auto` and `none` (the last turn sends `none`). Requests also set `provider.require_parameters`, so a host that lacks tools or reasoning fails every request instead of silently running without them.

### Cutoff semantics

`web_search` only returns content published before the claim's fact-check: before the day its first review was
published (`review_date` in the dataset, which `convert_export.py` writes, or else `reviews.published` in the
database). Claims without either fall back to the claim date. The prompt still gives the claim date.

The review's own day is **excluded**, not just capped. Search APIs filter at day
granularity with an inclusive end date, so a cutoff of the claim day itself would
admit content published hours *after* the claim — and fact-checks of a viral claim
often appear the same day. `web_search` therefore passes the claim day as scrapeMM's
`SerperQuery(before=...)`, which leaves out that day and everything after it: a claim
dated `2024-02-05T12:22` searches with `cd_max:2/4/2024`.

The tradeoff is that genuine same-day pre-claim evidence is also dropped. The APIs
expose no intra-day resolution, so this is the only way to guarantee no leakage, and
losing evidence is the safe direction for a benchmark. Results produced before this
change used an inclusive same-day cutoff and are not comparable at the boundary.

Only Google's organic results are passed to the model: the knowledge graph, answer box
and similar blocks are not date-filtered. Note that Google dates a page by when it was
published or first indexed, and that `fetch_url` retrieves a page's *current* version,
so the cutoff cannot rule out leakage completely.

## Python API

### Single Provider

```python
from eval.baselines import UnifiedFactChecker

fc = UnifiedFactChecker(provider="gemini")

# Or with a custom model
fc = UnifiedFactChecker(provider="openai", model="gpt-5.2")

# Fact-check a claim
result = fc.check_claim(
    claim="The Earth is flat",
    image_paths=["photo.jpg"],  # optional
    video_paths=["video.mp4"],  # optional (native for Gemini)
    claim_date="2024-02-05T12:00:00",  # web_search only returns content from before this day
)

print(result.verdict)  # One of the 7 labels, e.g. "Compromised (certain)"
print(result.justification)  # The justification of the JSON answer
print(result.retrieved_urls)  # URLs the model fetched with fetch_url
```

`check_claim` raises `NoAnswerError` if the model's final response holds no valid answer, and `FactCheckError` if
the fact-check fails, e.g. on an API error. Both hold the result so far in `.result`, with its trace and URLs.
Media limits are set with `media_limits`, e.g. `UnifiedFactChecker(provider="gemini", media_limits=MediaLimits(total_images=20))`
(`from eval.baselines import MediaLimits`).

### Multiple Providers

```python
from eval.baselines import UnifiedFactChecker

# Default providers: openai, gemini, anthropic
fc = UnifiedFactChecker()

# Or specific providers
fc = UnifiedFactChecker(providers=["openai", "gemini", "selfhosted"])

# Check with one of them
result = fc.check_claim("The Earth is flat", claim_date="2024-02-05", provider="gemini")
```

### Direct Provider Access

```python
from eval.baselines import GeminiFactChecker

fc = GeminiFactChecker(model="gemini-2.5-flash")
result = fc.check_claim("Some claim", video_paths=["video.mp4"])
```

## CLI Usage

Run the benchmark on the VeriTaS dataset (from the repository root):

```bash
# Single provider
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --limit 10

# Multiple providers
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --providers openai gemini anthropic --limit 10

# Custom model
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --model gemini-2.5-flash --limit 10

# Parallel workers
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --workers 4 --limit 20

# Self-hosted model via vLLM
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider selfhosted --limit 10

# Use specific scrapeMM backends, in order (default: auto, the server picks per domain)
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --scrapemm-methods firecrawl decodo

# Resume previous run
python eval/baselines/scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --resume eval/baselines/results/gpt-5.2_2024-12-22_14-30
```

### CLI Options

| Option | Type / Values | Description |
|--------|----------------|-------------|
| `--dataset` | `PATH` (required) | Path to `claims.json`. |
| `--output` | `PATH` | Output directory, which must not contain results yet. If omitted, an auto-generated directory under `results/` is used. |
| `--resume` | `PATH` | Resume from an existing run directory: claims without a successful result are (re)run. |
| `--provider` | `openai` / `gemini` / `anthropic` / `selfhosted` / `openrouter` | Run a single provider. |
| `--providers` | one or more provider names | Run multiple providers in one invocation. |
| `--model` | model name | Model override for single-provider runs (`--provider`). |
| `--limit` | integer | Maximum number of claims to process. |
| `--workers` | integer (default: `1`) | Number of parallel workers. |
| `--scrapemm-methods` | subset of `integrations` / `browser` / `firecrawl` / `decodo`, or `auto` (default: `auto`) | scrapeMM backends `fetch_url` uses, in order. `auto` lets the server pick per domain. |
| `--reasoning-effort` | `minimal` / `low` / `medium` / `high` / `xhigh` / `max` (default: `medium`) | How much the model reasons (see [Reasoning effort](#reasoning-effort)). |
| `--openrouter-host` | endpoint tag, e.g. `deepinfra/fp8` | The one OpenRouter endpoint every request goes to (see [Providers](#providers)). Required for `openrouter`. |
| `--page-images`, `--total-images`, ... | see [Media limits](#media-limits) | Which images and videos the models get, and in what form. |

Notes:
- If neither `--provider` nor `--providers` is specified, `openai`, `gemini` and `anthropic` are used.
- `--output` and `--resume` are mutually exclusive. On resume, the failed rows of the resumed providers are
  removed from `results.csv` and `results_detailed.jsonl`, and their claims run again. Every claim thus keeps
  one row per provider. The CSV's header is brought up to date as well (rows of earlier runs get empty new columns). The resume directory may not exist yet, so a job can always pass the same `--resume`.
- `--model` is applied only when `--provider` is used (single-provider mode).
- The former options `--custom-search`, `--no-search`, `--scrape-mode`, `--label-scheme`, `--seven-bin-prediction-mode` and `--mode` were removed: their settings are now fixed (custom tools, 7-class labels, integrity only).

## Output

Results are saved to `results/<model>_YYYY-MM-DD_HH-MM/`:

```
results/gpt-5.2_2024-12-22_14-30/
├── results.csv                 # Per-claim results (incl. justification and evidence URLs as JSON arrays)
├── results_detailed.jsonl      # Full details with the final response, evidence URLs and the full trace
├── summary.json                # Metrics (accuracy, F1, MSE/MAE, confusion matrix), 7-bin and coarsened 3-bin, the run statistics, and at the end every blocked URL
├── confusion_matrix_*.png      # 7-bin, coarsened 3-bin and 3-bin-vs-7-bin plots per provider
├── property_metrics_table.*    # Integrity metrics table (md/csv/pdf)
├── results_for_leaderboard.json  # Verdicts and justifications for the leaderboard
├── fact_check_exposure.json    # How often the claims' own fact-checks reached the model
├── run_statistics.json         # Turns, tool calls, time and tokens per fact-check, and what the tools blocked
└── run_config.json             # Run configuration
```

`results_for_leaderboard.json` holds one entry per claim, sorted by claim ID:

```json
[
  {"claim_id": 12345, "verdict": 1, "reasoning": "..."},
  {"claim_id": 12346, "verdict": -0.33, "reasoning": "..."},
  {"claim_id": 12347, "verdict": 0, "reasoning": "..."}
]
```

The verdict is the label's value, rounded to 2 decimals: `1` = Intact (certain), `0.67` = Intact (rather certain),
`0.33` = Intact (rather uncertain), `0` = Unknown, `-0.33` = Compromised (rather uncertain), `-0.67` = Compromised
(rather certain), `-1` = Compromised (certain). The reasoning is the justification of the model's JSON answer, or
its whole final response when the answer has none, cut off at 500 characters (ending in `...`). Claims without a
successful result (status `error` or `no_answer`) are left out. A run with
several providers writes one file per provider, `results_for_leaderboard_<provider>.json`. The file is written at
the end of every run, so `--resume` on a finished run (re)writes it as well.

`fact_check_exposure.json` checks the traces for the claims' own fact-checks, which the tools block (see above).
It finds a fact-check by its URL and by its slug, the last path segment with a letter in it (if it has at least 12
characters), by the tools' own rules. So it counts what got through them, e.g. a piece of the slug shorter than
20 characters is no slug to it either, and what the model wrote by itself. Per provider, it counts the claims in which the tools showed the
fact-check anyway (`shown_by_tools`: as a search result, a retrieved page, a link on a fetched page, or its slug, a
piece of it or its URL in a search preview or fetched page) and those in which only the model had it
(`only_by_model`: it cited it, tried to fetch it, searched for its slug or wrote its slug, as many fact-checkers
build the slug from the headline). `claims_by_id` gives the ways
of each such claim, each with an excerpt of where it first shows: the URL, the search result's title or snippet, or
the text around the slug or URL (150 characters on each side). It is also written at the end of every run. For the results of an earlier run:
`python eval/baselines/scripts/fact_check_exposure.py <results dir> --dataset <claims file>`.

`run_statistics.json` gives, per provider, the mean and median of the turns (model calls), of the calls to each tool
and to all, of the time per fact-check (`duration_seconds`, wall-clock with retries, while claims run in
parallel) and of the tokens (`tokens`: input, output with reasoning, and cache reads and writes, both part of the
input; writes as Anthropic, and OpenRouter for models whose writes are billed, report them), all of the successful
fact-checks; the tokens also in all (`total`), the failed fact-checks' included, as they were billed too. `cost_usd`
is what the API billed, cache discounts included, the same way; only OpenRouter says it (`claims` is 0 for the
others). Runs from before 2026-10-21 recorded no cache tokens. Under `blocked`, it lists by claim ID every `web_search` result the tools
dropped and every `fetch_url` call they refused, with the rule that blocked it
(`{"tool": "web_search", "query": ..., "url": ..., "rule": "a URL in the fact-check scope fullfact.org"}`). Each
claim's line of `results_detailed.jsonl` holds the same list (`result.blocked`) and its `duration_seconds`; runs from
before 2026-10-23 recorded neither. It is also written at the end of every run, and `summary.json` has the same statistics per provider (`run_statistics`), but for the list of what was blocked by claim. Instead, `summary.json` ends with every URL the tools blocked, by provider, once each and sorted, with the rule that first blocked it (`blocked_urls`: `{"gemini": {"https://fullfact.org/…": "a URL in the fact-check scope fullfact.org", …}}`). For the results of an earlier run:
`python eval/baselines/scripts/run_statistics.py <results dir>`.

MSE and MAE compare the predicted label's value (-1, -2/3, ..., 1) with the claim's continuous
ground-truth integrity score. They are only reported for the 7 bins the model predicts on: coarsening the
prediction to 3 bins would score values the model never gave.

For every claim, both `results.csv` and `results_detailed.jsonl` record four lists of evidence URLs:

| Field | Contents |
|-------|----------|
| `encountered_urls` | URLs of all `web_search` results the model saw |
| `retrieved_urls` | URLs the model successfully fetched with `fetch_url` |
| `linked_urls` | URLs in the fetched page content the model saw. Relative links (`/about`, `//cdn.example.org/a.jpg`) are resolved against the page's URL |
| `quoted_urls` | URLs cited in the final justification (may include URLs that were never retrieved) |

In the CSV each list is a JSON array, e.g. `json.loads(row["retrieved_urls"])`.

### Trace

`results_detailed.jsonl` also records each claim's full `trace`: everything that happened, in
chronological order. Every entry has a `type` and, but for the prompt, the `turn` (model call) it
belongs to:

| `type` | Fields | Contents |
|--------|--------|----------|
| `prompt` | `text` | The prompt (there is no system prompt) |
| `reasoning` | `text` | The model's reasoning, as far as its API returns it |
| `text` | `text` | Text the model wrote, e.g. the final JSON answer |
| `tool_call` | `id`, `name`, `args` | A `web_search` or `fetch_url` call (`args` is `null` if it was not valid JSON) |
| `message` | `text` | A message to the model besides the prompt and the tool results: before its last call, that it must answer now |
| `stop` | `reason` | Why the model call ended, as its API says: e.g. `completed` / `max_output_tokens` (OpenAI), `tool_use` / `end_turn` / `max_tokens` (Anthropic), `STOP` / `MAX_TOKENS` / `SAFETY` (Gemini), `tool_calls` / `stop` / `length` (chat completions) |
| `tool_result` | `id`, `name`, `text`, `media` | What the tool returned to the model, e.g. the fetched page content. `media` lists the page's images and videos shown to the model, with their `reference`, `kind` and `source_url` (where scrapeMM got them) |

No API returns a model's raw chain of thought. What `reasoning` holds depends on the provider:

| Provider | Reasoning | How it is requested |
|----------|-----------|---------------------|
| `openai` | A summary of the reasoning | `reasoning={"effort": ..., "summary": "auto"}` (reasoning models only) |
| `anthropic` | A summary of the thinking | `thinking={"type": "adaptive", "display": "summarized"}`: Claude 5 models always think, their thinking is just empty by default. On Claude 4.6 models, this turns thinking on |
| `gemini` | Summaries of the thoughts | `thinking_config=ThinkingConfig(include_thoughts=True)` |
| `selfhosted` | The reasoning itself | Returned by vLLM for reasoning models served with a reasoning parser (`reasoning_content` / `reasoning`) |
| `openrouter` | The reasoning as the model's upstream returns it, raw or summarized | `reasoning={"effort": ...}`. OpenRouter returns the reasoning in `message.reasoning` |

The tool results make the trace large: up to about 150,000 characters per claim.

## FactCheckResult

The `check_claim` methods return a `FactCheckResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | str \| None | One of the 7 labels, e.g. "Intact (rather certain)" or "Unknown". None if the model gave no valid answer |
| `reasoning` | str | Full final response of the model |
| `justification` | str | The `justification` of the JSON answer ("" if absent) |
| `citations` | list[str] | Search result URLs, fetched URLs and URLs cited in the response |
| `encountered_urls` | list[str] | URLs of the `web_search` results |
| `retrieved_urls` | list[str] | URLs successfully fetched with `fetch_url` |
| `linked_urls` | list[str] | URLs in the content of the fetched pages |
| `quoted_urls` | list[str] | URLs cited in the justification |
| `trace` | list[dict] | Everything that happened, in order (see [Trace](#trace)) |
| `model` | str | Model used |
| `provider` | str | Provider name |
| `usage` | dict | Token usage summed over the model calls: `input_tokens` (the whole prompts, cached tokens included), `output_tokens`, and of the input tokens `cache_read_tokens` (read from the API's prompt cache) and `cache_write_tokens` (written to it, as Anthropic, and OpenRouter for models whose writes are billed, report it); for OpenRouter also `cost`, what it billed in USD, cache discounts included |

## Architecture

```
baselines/
├── common/                 # Shared utilities
│   ├── claims.py           # Claim extraction/parsing
│   ├── media.py            # Image/video handling
│   ├── metrics.py          # Classification/regression metrics (incl. 7-to-3-bin coarsening)
│   ├── parsing.py          # Parsing of the JSON answer, URL extraction
│   ├── prompts.py          # The baseline prompt
│   ├── tools.py            # web_search / fetch_url via scrapeMM, with temporal filtering
│   └── types.py            # Label schemes, FactCheckResult, constants
├── providers/              # Provider implementations (message format per API)
│   ├── base.py             # The tool-calling loop all providers share
│   ├── openai.py           # OpenAI (Responses API)
│   ├── gemini.py           # Gemini (native video)
│   ├── anthropic.py        # Anthropic
│   ├── chat_completions.py # Message format of the OpenAI-compatible chat completions APIs
│   ├── selfhosted.py       # Self-hosted models via vLLM
│   └── openrouter.py       # OpenRouter
├── factchecker.py          # UnifiedFactChecker orchestration
└── scripts/                # Benchmarking/analysis scripts
    ├── run_benchmark.py
    ├── analyze_rectification.py
    ├── compare_experiments_3bin_vs_7bin.py
    ├── compute_quarterly_metrics.py
    ├── compare_quarterly.py
    ├── compute_moving_average.py
    └── ...
```
