# Baselines

Unified fact-checking baselines for the VeriTaS benchmark with support for OpenAI, Gemini, Perplexity, and Llama (self-hosted via vLLM).

## Installation

```bash
pip install -r requirements.txt
```

## Dataset
You need to run the export script in the VeriTaS repo to generate the claims dataset:

## Configuration

API keys can be configured in three ways (in order of priority):

1. **Veritas config** (recommended): Add keys to `config/globals.yaml`:
   ```yaml
   openai: "sk-..."
   google: "AIza..."
   perplexity: "pplx-..."  # optional
   ```

2. **Environment variables**:
   - `OPENAI_API_KEY` - for OpenAI provider
   - `GOOGLE_API_KEY` - for Gemini provider
   - `PERPLEXITY_API_KEY` - for Perplexity provider

3. **Explicit parameters**: Pass `api_keys` dict when creating `UnifiedFactChecker`

If you're using Veritas, the baselines will automatically use the API keys from `config/globals.yaml`.

## Providers

| Provider | Default Model | Native Video | Search Capability | Temporal Filtering |
|----------|---------------|--------------|-------------------|-------------------|
| OpenAI | gpt-5.2 | No (extracts frames) | Built-in + custom search | Built-in: prompt-only; custom: pre-date filtered |
| Gemini | gemini-2.5-flash | Yes | Built-in + custom search | Built-in: prompt-only; custom: pre-date filtered |
| Perplexity | sonar-pro | No (extracts frames) | Built-in (and used in custom mode too) | API-enforced (`search_before_date_filter`) |
| Llama (vLLM) | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | No (extracts frames) | Custom search only | Custom: pre-date filtered |

### Recommended Search Mode

Use `--custom-search` for benchmarking and serious evaluation. Native provider search tools do not offer sufficient filtering controls for strict temporal benchmarking. Custom search provides:
- deterministic pre-date filtering
- controllable retrieval pipeline
- optional page-content scraping for richer evidence

### ⚠️ Important: Temporal Data Leakage

If you run OpenAI/Gemini with built-in native search (without `--custom-search`), date constraints are prompt-level only (soft constraint). For temporal integrity, prefer `--custom-search` (or Perplexity built-in filtering).

### Model Options

**OpenAI**: `gpt-5.2`, `gpt-4.1`

**Gemini**: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`

**Perplexity**: `sonar`, `sonar-pro`, `sonar-reasoning-pro`

**Llama (self-hosted via vLLM)**: any model name your endpoint serves (default: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`)

## Python API

### Single Provider

```python
from eval.baselines import UnifiedFactChecker

# Initialize with one provider (recommended: custom_search=True for strict filtering)
fc = UnifiedFactChecker(provider="openai", custom_search=True)

# Or with custom model
fc = UnifiedFactChecker(provider="gemini", model="gemini-2.5-flash")

# Fact-check a claim
result = fc.check_claim(
    claim="The Earth is flat",
    image_paths=["photo.jpg"],  # optional
    video_paths=["video.mp4"],  # optional (native for Gemini)
    claim_date="2024-02-05T12:00:00",  # optional (prevents data leakage)
)

print(result.verdict)  # "Intact", "Compromised", or "Unknown"
print(result.reasoning)  # Full explanation
print(result.citations)  # List of source URLs
```

### Multiple Providers

```python
from eval.baselines import UnifiedFactChecker

# Initialize all providers
fc = UnifiedFactChecker()

# Or specific providers
fc = UnifiedFactChecker(providers=["openai", "gemini", "llama"], custom_search=True)

# Check with all providers
results = fc.check_claim_all_providers("The Earth is flat")
for provider, result in results.items():
    print(f"{provider}: {result.verdict}")

# Check with consensus (requires min 2 providers to agree)
consensus, all_results = fc.check_claim_with_consensus(
    "The Earth is flat",
    min_agreement=2,
)
if consensus:
    print(f"Consensus: {consensus.verdict}")
else:
    print("No consensus reached")
```

### Direct Provider Access

```python
from eval.baselines import OpenAIFactChecker, GeminiFactChecker, PerplexityFactChecker

# Use providers directly
fc = GeminiFactChecker(model="gemini-2.5-flash")
result = fc.check_claim("Some claim", video_paths=["video.mp4"])

# Llama direct provider is available from baselines.providers
from eval.baselines.providers import LlamaFactChecker

llama_fc = LlamaFactChecker()
```

## CLI Usage

Run the benchmark on the VeriTaS dataset:

```bash
# Module invocation (recommended from repo root)
python -m baselines.scripts.run_benchmark --dataset /path/to/claims.json --provider openai --custom-search --limit 10

# Script invocation
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --custom-search --limit 10

# Single provider (recommended: custom search)
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --custom-search --limit 10

# Multiple providers
python scripts/run_benchmark.py --dataset /path/to/claims.json --providers openai gemini perplexity --custom-search --limit 10

# Custom model
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --model gemini-2.5-flash --limit 10

# Parallel workers
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --custom-search --workers 4 --limit 20

# 7-class labels with two-step direction+certainty prediction
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --label-scheme 7 --seven-bin-prediction-mode two_step

# Staged property-loop mode
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider gemini --mode staged --label-scheme 7

# Disable web search (parametric knowledge only)
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --no-search

# Llama via self-hosted vLLM (custom search path)
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider llama --custom-search --limit 10

# Resume previous run
python scripts/run_benchmark.py --dataset /path/to/claims.json --provider openai --resume results/gpt-5.2_2024-12-22_14-30
```

### CLI Options

| Option | Type / Values | Description |
|--------|----------------|-------------|
| `--dataset` | `PATH` (required) | Path to `claims.json`. |
| `--output` | `PATH` | Output directory. If omitted, an auto-generated directory under `results/` is used. |
| `--resume` | `PATH` | Resume from an existing run directory. |
| `--provider` | `openai` / `gemini` / `perplexity` / `llama` | Run a single provider. |
| `--providers` | one or more provider names | Run multiple providers in one invocation. |
| `--model` | model name | Model override for single-provider runs (`--provider`). |
| `--limit` | integer | Maximum number of claims to process. |
| `--workers` | integer (default: `1`) | Number of parallel workers. |
| `--custom-search` | flag | Use custom search pipeline with date filtering and content retrieval. |
| `--no-search` | flag | Disable web search and use parametric model knowledge only. |
| `--label-scheme` | `3` or `7` (default: `3`) | Label granularity: 3-class or 7-class with certainty levels. |
| `--mode` | `integrity` or `staged` (default: `integrity`) | Integrity single-pass baseline vs staged property-loop benchmark. |
| `--seven-bin-prediction-mode` | `direct` or `two_step` (default: `direct`) | For 7-class labels: direct combined verdict vs direction+certainty format. |

Notes:
- If neither `--provider` nor `--providers` is specified, all providers are used.
- `--model` is applied only when `--provider` is used (single-provider mode).
- `--seven-bin-prediction-mode` is relevant only when `--label-scheme 7`.

## Output

Results are saved to `results/<model>_YYYY-MM-DD_HH-MM/`:

```
results/gpt-5.2_2024-12-22_14-30/
├── results.csv              # Per-claim results
├── results_detailed.jsonl   # Full details with reasoning
├── summary.json             # Metrics (accuracy, F1, confusion matrix)
├── confusion_matrix_*.png   # Visualization per provider
└── run_config.json          # Run configuration
```

## FactCheckResult

The `check_claim` methods return a `FactCheckResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | str | "Intact", "Compromised", or "Unknown" |
| `reasoning` | str | Full explanation from the model |
| `citations` | list[str] | Source URLs used |
| `model` | str | Model used |
| `provider` | str | Provider name |
| `usage` | dict | Token usage statistics |

## Architecture

```
baselines/
├── common/                # Shared utilities
│   ├── claims.py          # Claim extraction/parsing
│   ├── media.py           # Image/video handling
│   ├── metrics.py         # Classification/regression metrics
│   ├── prompts.py         # Prompt templates (3-class / 7-class / two-step)
│   ├── search.py          # Custom retrieval + temporal filtering
│   └── types.py           # Label schemes, FactCheckResult, constants
├── providers/             # Provider implementations
│   ├── base.py            # Abstract base class + verdict extraction
│   ├── openai.py          # OpenAI with native search
│   ├── openai_custom.py   # OpenAI + custom search tool
│   ├── gemini.py          # Gemini with native search
│   ├── gemini_custom.py   # Gemini + custom search tool
│   ├── perplexity.py      # Perplexity integration
│   └── llama.py           # Self-hosted Llama (vLLM) + custom search
├── factchecker.py         # UnifiedFactChecker orchestration
└── scripts/               # Benchmarking/analysis scripts
    ├── run_benchmark.py
    ├── analyze_rectification.py
    ├── compare_experiments_3bin_vs_7bin.py
    ├── compute_quarterly_metrics.py
    ├── compare_quarterly.py
    ├── compute_moving_average.py
    └── ...
```
