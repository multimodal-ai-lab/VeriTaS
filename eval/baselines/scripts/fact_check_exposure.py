#!/usr/bin/env python3
"""How often the claims' own fact-checks reached the model in a benchmark run.

The tools block each claim's fact-checks (`review_urls`, or `review_url`), also under other
paths with their slug, and the fact-check scopes of their publishers and of all IFCN signatories
and EFCSN members, and remove their links from what they show (see ToolSession). This checks the
run's traces for what of the claim's own fact-checks got through anyway, by the same rules (a
ToolSession that blocks only them), in what the tools showed the model:
    search_result   a search result is the fact-check
    retrieved       fetch_url retrieved it
    linked          a fetched page links to it
    search_preview  a search result's title or snippet has its slug, a piece of it or its URL
    fetched_page    a fetched page's text has its slug or URL
and for what the model did by itself, which needs no leak, as many fact-checkers build the slug
from the headline, which often restates the claim:
    cited           the justification cites it
    fetch_attempt   the model called fetch_url with its URL
    searched        a search query has its slug
    mentioned       the model's reasoning or answer has its slug
URLs count whatever their scheme, "www.", parameters and fragment, as do the pages under them
and other URLs with the fact-check's slug in their path (aliases, archives, copies). Replies that only repeat a URL the model wrote ("Error: Cannot fetch <url>. ...") are left out.
Each way of a claim comes with an excerpt of where it first shows: the URL, the search result's
title or snippet, or the text around the slug or URL.

run_benchmark.py writes the result to fact_check_exposure.json in its output directory. For the
results of a run:
    python eval/baselines/scripts/fact_check_exposure.py dev/q3_baselines/gemini-3.8-flash/results \
        --dataset dev/q3_baselines/dataset/claims_benchmark.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.common.parsing import remove_links  # noqa: E402
from baselines.common.tools import ToolSession  # noqa: E402

SHOWN = ("search_result", "retrieved", "linked", "search_preview", "fetched_page")
OWN = ("cited", "fetch_attempt", "searched", "mentioned")

EXCERPT_CHARS = 150


def _excerpt(text: str, start: int, end: int) -> str:
    """`text` from EXCERPT_CHARS characters before `start` to as many after `end`, on one line."""
    excerpt = " ".join(text[max(0, start - EXCERPT_CHARS):end + EXCERPT_CHARS].split())
    return ("…" if start > EXCERPT_CHARS else "") + excerpt + ("…" if end + EXCERPT_CHARS < len(text) else "")


def _first_change(text: str, changed: str) -> int:
    return next((i for i, (a, b) in enumerate(zip(text, changed)) if a != b), min(len(text), len(changed)))


def claim_exposure(record: dict, review_urls: list[str]) -> dict[str, str]:
    """The ways (see above) the claim's fact-checks at `review_urls` show in `record`, its line of
    results_detailed.jsonl, each with an excerpt of where it first does, Google's breaks of the
    slug taken out."""
    session = ToolSession(claim_date=record.get("claim_date"), blocked_urls=review_urls)
    result = record.get("result") or {}
    ways = {}

    def add(way: str, text: str, start: int | None = None, end: int | None = None):
        if way not in ways:
            ways[way] = text if start is None else _excerpt(text, start, end)

    def add_slug(way: str, text: str) -> bool:
        if found := session._find_blocked_slug(text):
            add(way, found[0], *found[1].span())
        return bool(found)

    for way, key in (("search_result", "encountered_urls"), ("retrieved", "retrieved_urls"), ("linked", "linked_urls")):
        if url := next((url for url in result.get(key) or [] if session.is_blocked(url)), None):
            add(way, url)
    if url := next((url for url in result.get("quoted_urls") or [] if session.is_blocked(url)), None):
        justification = result.get("justification") or ""
        position = justification.find(url)
        if position < 0:
            add("cited", url)
        else:
            add("cited", justification, position, position + len(url))

    trace = result.get("trace") or []
    calls = {entry["id"]: entry for entry in trace if entry["type"] == "tool_call"}
    for entry in trace:
        text = entry.get("text") or ""
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        if entry["type"] == "tool_call":
            if entry["name"] == "fetch_url" and session.is_blocked(str(args.get("url") or "")):
                add("fetch_attempt", str(args["url"]))
            elif entry["name"] == "web_search":
                add_slug("searched", str(args.get("query") or ""))
        elif entry["type"] in ("reasoning", "text"):
            add_slug("mentioned", text)
        elif entry["type"] == "tool_result" and not text.startswith(("Error:", "Could not retrieve")):
            if entry["name"] == "web_search":
                previews = [line for line in text.split("\n") if line.startswith("Snippet: ") or re.match(r"\[\d+\] ", line)]
                if preview := next((preview for preview in previews if session._blocked_in_preview(preview)), None):
                    add("search_preview", " ".join(preview.split()))
            elif entry["name"] == "fetch_url":
                call_args = calls.get(entry["id"], {}).get("args")
                page_url = str(call_args.get("url") or "") if isinstance(call_args, dict) else ""
                if not add_slug("fetched_page", text):
                    redacted, n_removed = remove_links(text, session.is_blocked, base_url=page_url)
                    if n_removed:
                        position = _first_change(text, redacted)
                        add("fetched_page", text, position, position + 1)
    return {way: ways[way] for way in SHOWN + OWN if way in ways}


def exposure_summary(jsonl_path: Path, claims: list[dict]) -> dict:
    """Per provider: how many claims the tools showed their fact-check and how many only the model
    had it, the claims per way, and each claim's ways with excerpts (see claim_exposure)."""
    review_urls = {int(claim["id"]): claim.get("review_urls") or [claim["review_url"]]
                   for claim in claims if claim.get("review_urls") or claim.get("review_url")}
    summary = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            stats = summary.setdefault(record["provider"], {
                "claims": 0, "claims_with_review_url": 0, "shown_by_tools": 0, "only_by_model": 0,
                "ways": dict.fromkeys(SHOWN + OWN, 0), "claims_by_id": {},
            })
            stats["claims"] += 1
            claim_review_urls = review_urls.get(int(record["claim_id"]))
            if not claim_review_urls:
                continue
            stats["claims_with_review_url"] += 1
            ways = claim_exposure(record, claim_review_urls)
            if not ways:
                continue
            if set(ways) & set(SHOWN):
                stats["shown_by_tools"] += 1
            else:
                stats["only_by_model"] += 1
            for way in ways:
                stats["ways"][way] += 1
            stats["claims_by_id"][str(record["claim_id"])] = ways
    return summary


def write_fact_check_exposure(jsonl_path: Path, claims: list[dict], output_dir: Path) -> dict:
    if not jsonl_path.exists():
        return {}
    summary = exposure_summary(jsonl_path, claims)
    path = output_dir / "fact_check_exposure.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for provider, stats in summary.items():
        print(f"Fact-checks of the claims ({provider}): shown by the tools in {stats['shown_by_tools']} "
              f"of {stats['claims_with_review_url']} claims, had by the model only in {stats['only_by_model']}")
    print(f"Fact-check exposure saved to: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Output directory of a run, with results_detailed.jsonl")
    parser.add_argument("--dataset", type=Path, required=True, help="The claims file of the run, with the review_url of each claim")
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8") as f:
        claims = json.load(f)["claims"]
    write_fact_check_exposure(args.results_dir / "results_detailed.jsonl", claims, args.results_dir)


if __name__ == "__main__":
    main()
