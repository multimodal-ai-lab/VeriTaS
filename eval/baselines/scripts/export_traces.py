#!/usr/bin/env python3
"""Export the traces of benchmark runs: a folder per run, named after the run's directory (the one
holding its results folder, e.g. claude-fable-5-1), with one JSON file per claim holding the claim
and the fact-check's result with its trace (no ground truth, final response, usage or citations),
and a readme.txt that explains the fields.

A claim's file is <claim_id>.json, or <claim_id>_<provider>.json if the run has several
providers. Of a claim recorded more than once, its last record counts, as in results.csv. E.g.:
    python eval/baselines/scripts/export_traces.py \\
        dev/q3_baselines/{claude-fable-5-1,mimo-v2.6-pro}/results --output dev/q3_baselines/traces
"""

import argparse
import json
from pathlib import Path

CLAIM_FIELDS = ("claim_id", "claim_text", "claim_date", "review_date")
RESULT_FIELDS_LEFT_OUT = ("reasoning", "provider", "usage", "citations")

README = """\
Traces of the Q3 benchmark runs of the VeriTaS fact-checking baseline, one folder per model with one
JSON file per claim (<model>/<claim_id>.json).

claim_id       The claim's ID in the VeriTaS database.
claim_text     The claim's text.
claim_date     When the claim was published.
review_date    When the claim's fact-check was published.
result         What the fact-check:
  verdict           The model's answer as a label of the 7-class scheme, from "Intact (certain)"
                    to "Compromised (certain)", or "Unknown".
  model             The model that fact-checked the claim.
  justification     The justification of the model's JSON answer.
  encountered_urls  The URLs of all web_search results the model was shown.
  retrieved_urls    The pages fetch_url retrieved.
  linked_urls       The links in the content of those pages.
  quoted_urls       The URLs the justification cites.
  blocked           Each search result dropped and each fetch refused, with the tool, the query,
                    the URL and the rule that blocked it.
  trace            The fact-checks whole trace containing the initial prompt, tool calls, the 
                   model's reasoning and the final answer.
    prompt       The initial prompt.
    reasoning    The model's thinking in that turn, as the API returns it (in full, summarized,
                 or empty, depending on the model).
    text         Text the model wrote: before its tool calls, or its final answer (with the
                 JSON answer the verdict and justification come from).
    tool_call    A tool call: its id, the tool's name (web_search or fetch_url) and its args.
    stop         Why the model stopped: to call tools, or with its answer ("stop", "end_turn",
                 "completed"; "max_tokens" or "length" if it was cut off).
    tool_result  The tool's reply to the call with the same id.
"""


def _exported_result(result: dict) -> dict:
    exported = {key: value for key, value in result.items() if key not in RESULT_FIELDS_LEFT_OUT}
    exported.setdefault("blocked", None)  # Not recorded by runs from before 2026-10-23
    exported["trace"] = exported.pop("trace")  # Last, after "blocked"
    return exported


def export_traces(jsonl_path: Path, output_dir: Path) -> int:
    records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue  # A line a crash cut short
            if record is not None:
                records[(record.get("provider"), record["claim_id"])] = record

    several_providers = len({provider for provider, _ in records}) > 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for (provider, claim_id), record in records.items():
        result = record.get("result")
        export = {field: record.get(field) for field in CLAIM_FIELDS}
        export["result"] = _exported_result(result) if result else None
        name = f"{claim_id}_{provider}.json" if several_providers else f"{claim_id}.json"
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
    return len(records)


def export_runs(jsonl_paths: list[Path], output_dir: Path) -> dict[str, int]:
    counts = {}
    for jsonl_path in jsonl_paths:
        run = jsonl_path.parent.parent.name
        if run in counts:
            raise ValueError(f"Two runs would share the folder {run}")
        counts[run] = export_traces(jsonl_path, output_dir / run)
    (output_dir / "readme.txt").write_text(README, encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+",
                        help="results_detailed.jsonl of each run, or the run's results directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the traces")
    args = parser.parse_args()
    jsonl_paths = [path / "results_detailed.jsonl" if path.is_dir() else path for path in args.results]
    for run, n in export_runs(jsonl_paths, args.output).items():
        print(f"{run}: {n} traces")
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
