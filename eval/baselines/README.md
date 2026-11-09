# Baselines

How to run the VeriTaS fact-checking baseline on a dataset export from the VeriTaS Website with OpenAI, Gemini, Anthropic,
self-hosted (vLLM) or OpenRouter models.

## What you need

- A VeriTaS Release (Can be downloaded here https://veritas.mai.informatik.tu-darmstadt.de/releases)
- The fact-check scopes file (`fact_checker_scopes.json`) (Can be downloaded here https://veritas.mai.informatik.tu-darmstadt.de/challenge/signatory-blocking). It lists the sites and
  sections where fact-checkers publish their fact-checks, which the model's tools block so that it cannot look
  up the answer.
- Access to a scrapeMM server, which runs the model's web searches and page fetches.
- An API key for the model provider you want to run.

## Setup

```bash
pip install -r requirements.txt
```

Run everything from the repository root, where the configuration is read from `config.yaml`:

```yaml
scrapemm:
  url: http://<scrapemm-server>:<port>
  api_key: <key>

api_secrets:           # Only the providers you run
  openai: "sk-..."
  google: "AIza..."
  anthropic: "sk-ant-..."
  openrouter: "sk-or-..."

selfhosted:            # Only for --provider selfhosted: vLLM's OpenAI-compatible endpoint
  url: http://<vllm-host>:<port>/v1
  key: <key>
```

Instead of `config.yaml`, the keys can come from the environment variables
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_BASE_URL` / `VLLM_API_KEY`
and `SCRAPEMM_API_URL` / `SCRAPEMM_API_KEY`.

## Dataset

If you downloaded the dataset from the VeriTaS Website, you need to convert it once to the format `run_benchmark` uses.

```bash
python eval/baselines/scripts/convert_export.py <export_dir>
```

This writes `<export_dir>/claims_benchmark.json`, the file to pass as `--dataset`.

## Run

```bash
python -m eval.baselines.scripts.run_benchmark \
    --dataset <export_dir>/claims_benchmark.json \
    --blocked-scopes fact_checker_scopes.json \
    --provider openai --model gpt-6-luna \
    --workers 30 \
    --resume eval/baselines/results/gpt-6-luna
```

More examples:

```bash
# Try a few claims first
python -m eval.baselines.scripts.run_benchmark --dataset <claims> --blocked-scopes <scopes> \
    --provider gemini --model gemini-3.8-flash --limit 10

# OpenRouter: --openrouter-host is required
python -m eval.baselines.scripts.run_benchmark --dataset <claims> --blocked-scopes <scopes> --provider openrouter \
    --model deepseek/deepseek-v4.1-flash --openrouter-host inference-net/fp8 --workers 30 --resume <run_dir>

# Self-hosted model behind the vLLM endpoint in config.yaml
python -m eval.baselines.scripts.run_benchmark --dataset <claims> --blocked-scopes <scopes> \
    --provider selfhosted --model <served-model> --resume <run_dir>
```

### Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--dataset` | required | The converted claims file |
| `--blocked-scopes` | required | The fact-check scopes file |
| `--provider` | | `openai`, `gemini`, `anthropic`, `selfhosted` or `openrouter` |
| `--providers` | `openai gemini anthropic` | Several providers in one run, each with its default model |
| `--model` | the provider's default | The model, with `--provider` only |
| `--output` | `eval/baselines/results/<model>_<YYYY-MM-DD_HH-MM>/` | A new results folder |
| `--resume` | | A results folder to continue: claims without a successful result run (again). The folder need not exist yet. Not together with `--output` |
| `--limit` | all | Number of claims |
| `--workers` | 1 | Claims run in parallel |
| `--reasoning-effort` | `medium` | `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. A model that does not take the value fails every request with a 400 error |
| `--openrouter-host` | | The one OpenRouter endpoint all requests go to, by its `tag` in `https://openrouter.ai/api/v1/models/<model>/endpoints` (e.g. `inference-net/fp8`). Pick a host whose `supports_tool_choice` has both `auto` and `none` |
| `--scrapemm-methods` | `auto` | scrapeMM backends for `fetch_url`, in order: a subset of `integrations`, `browser`, `firecrawl`, `decodo` |

### Resuming

Pass the same `--resume` folder every time, e.g. in a Slurm script you resubmit after a timeout: the first run
creates the folder, later runs retry only the claims that failed or are missing. The metrics, `summary.json` and
the other files are rewritten at the end of every run.

The results are in the run's folder: `results.csv` (one row per claim), `results_detailed.jsonl` (with the full
traces), `summary.json` (metrics) and `run_config.json` (the run's settings).
