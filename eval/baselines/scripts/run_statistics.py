#!/usr/bin/env python3
"""How a benchmark run went, per provider: the fact-checks' turns, tool calls and time, and what
the tools blocked.

    turns              model calls per fact-check, mean and median
    tool_calls         calls per fact-check of each tool and of all, mean and median
    duration_seconds   wall-clock time per fact-check, retries included, mean and median. Claims
                       run in parallel and share the scrapeMM server and the APIs' rate limits
    tokens             input, output (reasoning included), cache read and cache write tokens (both
                       part of the input; writes as Anthropic, and OpenRouter for models whose
                       writes are billed, report them): per fact-check, mean and median, and in
                       all, the failed fact-checks' included as they were billed too
    cost_usd           what the API billed, cache discounts included, the same way; only OpenRouter
                       says it
    blocked            the web_search results dropped and the fetch_url calls refused because
                       they are or show a blocked URL, each with the rule that blocked it (see
                       ToolSession.blocked), by claim ID

Turns, tool calls, time and tokens per fact-check are of the successful fact-checks; what was
blocked, of all that got to use the tools. Runs from before 2026-10-23 recorded neither the time
nor what was blocked, and runs from before 2026-10-21 not the cache tokens: their claims are left
out of these ("claims" says how many there are).

run_benchmark.py writes the result to run_statistics.json in its output directory. For the
results of a run:
    python eval/baselines/scripts/run_statistics.py dev/q3_baselines/gpt-6-astra/results
"""

import argparse
import json
import statistics
from pathlib import Path

TOOLS = ("web_search", "fetch_url")
TOKENS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _mean_and_median(values: list[float], digits: int = 2) -> dict:
    if not values:
        return {"mean": None, "median": None}
    return {"mean": round(statistics.mean(values), digits), "median": round(statistics.median(values), digits)}


def _usage(records: list[dict], successful: list[dict], kind: str, digits: int = 2) -> dict:
    """Usage of `kind` per successful fact-check (mean, median) and in all, failed ones included."""
    def used(selection):
        return [r["result"]["usage"][kind] for r in selection
                if r.get("result") and kind in (r["result"].get("usage") or {})]
    return {**_mean_and_median(used(successful), digits), "total": round(sum(used(records)), digits),
            "claims": len(used(records))}


def run_statistics(jsonl_path: Path) -> dict:
    records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.setdefault(record["provider"], []).append(record)

    summary = {}
    for provider, provider_records in records.items():
        successful = [r for r in provider_records if r["status"] == "success" and r.get("result")]
        traces = [r["result"].get("trace") or [] for r in successful]
        calls = [[entry["name"] for entry in trace if entry["type"] == "tool_call"] for trace in traces]
        tools = list(TOOLS) + sorted({name for names in calls for name in names} - set(TOOLS))
        durations = [r["duration_seconds"] for r in successful if r.get("duration_seconds") is not None]
        recorded = [r for r in provider_records if r.get("result") and "blocked" in r["result"]]
        blocked = [entry for r in recorded for entry in r["result"]["blocked"]]
        summary[provider] = {
            "claims": len(provider_records),
            "successful_claims": len(successful),
            "turns": _mean_and_median([max((e.get("turn", -1) for e in trace), default=-1) + 1 for trace in traces]),
            "tool_calls": {
                **{tool: _mean_and_median([names.count(tool) for names in calls]) for tool in tools},
                "all": _mean_and_median([len(names) for names in calls]),
            },
            "duration_seconds": {**_mean_and_median(durations), "claims": len(durations)},
            "tokens": {kind: _usage(provider_records, successful, kind) for kind in TOKENS},
            "cost_usd": _usage(provider_records, successful, "cost", digits=6),
            "blocked": {
                "claims": len(recorded),
                "web_search_results": sum(entry["tool"] == "web_search" for entry in blocked),
                "fetch_url": sum(entry["tool"] == "fetch_url" for entry in blocked),
                "by_claim": {str(r["claim_id"]): r["result"]["blocked"] for r in recorded if r["result"]["blocked"]},
            },
        }
    return summary


def write_run_statistics(jsonl_path: Path, output_dir: Path) -> dict:
    if not jsonl_path.exists():
        return {}
    summary = run_statistics(jsonl_path)
    path = output_dir / "run_statistics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for provider, stats in summary.items():
        calls, duration, blocked = stats["tool_calls"], stats["duration_seconds"], stats["blocked"]
        tokens = {kind: stats["tokens"][kind]["total"] for kind in TOKENS}
        cost = stats["cost_usd"]
        cost_text = f"cost in all: ${cost['total']:.4f} (${cost['mean']} per fact-check); " if cost["claims"] else ""
        print(f"Run statistics ({provider}): turns {stats['turns']['mean']} (median {stats['turns']['median']}), "
              f"web_search {calls['web_search']['mean']}, fetch_url {calls['fetch_url']['mean']} per fact-check, "
              f"{duration['mean']} s (median {duration['median']} s); tokens in all: {tokens['input_tokens']:,} input "
              f"({tokens['cache_read_tokens']:,} read from and {tokens['cache_write_tokens']:,} written to the cache), "
              f"{tokens['output_tokens']:,} output; {cost_text}blocked: "
              f"{blocked['web_search_results']} search results and {blocked['fetch_url']} fetches")
    print(f"Run statistics saved to: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Output directory of a run, with results_detailed.jsonl")
    args = parser.parse_args()
    write_run_statistics(args.results_dir / "results_detailed.jsonl", args.results_dir)


if __name__ == "__main__":
    main()
