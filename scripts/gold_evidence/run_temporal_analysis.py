"""Temporal analysis of the reconstructed gold evidence (Spec §5).

Reads what the reconstruction stored in the DB and exports claim-level and
evidence-level results plus aggregate tables. Optionally (re-)runs the
sufficiency validator for the two conditions

    E_claim      = {e | t_e <= t_c}
    E_factcheck  = {e | t_e <= t_f}

so that the central question - whether evidence appearing *during* the
professional fact-checking period is necessary to reconstruct the gold verdict -
can be answered on the current evidence table.

Examples:
    python -m scripts.gold_evidence.run_temporal_analysis
    python -m scripts.gold_evidence.run_temporal_analysis --revalidate --limit 300
    python -m scripts.gold_evidence.run_temporal_analysis --out exports/gold_evidence/run_a
"""

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime

from veritas import logger
from veritas.db import db
from veritas.gold_evidence import (
    CONDITIONS,
    ensemble_mode as default_ensemble_mode,
    proximity_threshold as default_threshold,
)
from veritas.gold_evidence.admissibility import select
from veritas.gold_evidence.analysis import aggregate, build_claim_record, build_evidence_record
from veritas.gold_evidence.filtering import get_reference_times
from veritas.gold_evidence.sufficiency import ENSEMBLE_MODES, validate_sufficiency
from veritas.util import run_with_semaphore

DEFAULT_OUT_ROOT = os.path.join("exports", "gold_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="Output directory (default: exports/gold_evidence/<timestamp>)")
    parser.add_argument("--mode", choices=ENSEMBLE_MODES, default=default_ensemble_mode,
                        help="Which ensemble mode's results to analyze")
    parser.add_argument("--threshold", type=float, default=default_threshold,
                        help="Proximity threshold used when --revalidate is set")
    parser.add_argument("--limit", type=int, default=None,
                        help="Analyze at most this many claims")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-run the sufficiency validator for both conditions")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def collect(args) -> tuple[list, list[dict]]:
    """Builds the claim-level and evidence-level rows."""
    claims = await db.get_claims_for_gold_evidence(
        limit=args.limit, statuses=None, released_first=True)
    claims = [c for c in claims if c.gold_evidence_status]
    logger.info(f"Analyzing {len(claims)} claims with reconstruction results.")

    evidence_by_claim = await db.get_evidence_for_claims([c.id for c in claims])

    claim_records, evidence_records = [], []

    async def _process(claim):
        evidence = evidence_by_claim.get(claim.id, [])
        t_c, t_f = await get_reference_times(claim)
        gold = await claim.current_verdict

        stored = {r["condition"]: r for r in
                  await db.get_gold_evidence_results(claim.id, ensemble_mode=args.mode)}

        if args.revalidate and gold is not None:
            for condition in CONDITIONS:
                subset = select(evidence, condition)
                result = await validate_sufficiency(claim, subset, gold, condition=condition,
                                                    mode=args.mode, threshold=args.threshold)
                await db.save_gold_evidence_result(result.to_db_dict())
                stored[condition] = result.to_db_dict()

        claim_records.append(build_claim_record(
            claim=claim, gold=gold, evidence=evidence, t_c=t_c, t_f=t_f, results=stored))
        for item in evidence:
            evidence_records.append(build_evidence_record(item, claim_id=claim.id,
                                                          t_c=t_c, t_f=t_f))

    await run_with_semaphore([_process(c) for c in claims], limit=args.concurrency,
                             show_progress=True, progress_description="Collecting results")
    claim_records.sort(key=lambda r: r.claim_id)
    evidence_records.sort(key=lambda r: (r["claim_id"], r["evidence_id"] or 0))
    return claim_records, evidence_records


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def render_markdown(aggregates: dict, mode: str) -> str:
    """A compact, paper-ready summary of the aggregate tables."""
    recoverability = aggregates["recoverability"]
    contingency = recoverability["contingency"]

    lines = [
        "# Gold Evidence Reconstruction - Temporal Analysis",
        "",
        f"Ensemble mode: `{mode}`",
        "",
        "## Coverage",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Claims analyzed | {aggregates['n_claims']} |",
        f"| Evidence candidates | {aggregates['n_evidence_candidates']} |",
        f"| Admissible evidence items | {aggregates['n_evidence_admissible']} |",
        f"| Items in the interval t_c < t_e <= t_f | {aggregates['n_evidence_in_window']} |",
        f"| Share of claims with such evidence | {_pct(aggregates['share_claims_with_window_evidence'])} |",
        f"| Share of admissible evidence in the interval | {_pct(aggregates['share_evidence_in_window'])} |",
        f"| Share of evidence candidates rejected | {_pct(aggregates['share_evidence_rejected'])} |",
        "",
        "## Time differences (days)",
        "",
        "| Distribution | n | mean | sd | median | IQR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("t_e - t_c", "distribution_t_e_minus_t_c"),
                       ("t_e - t_f", "distribution_t_e_minus_t_f"),
                       ("t_f - t_c", "distribution_t_f_minus_t_c")):
        d = aggregates[key]
        if d.get("n"):
            lines.append(f"| {label} | {d['n']} | {d['mean']:.1f} | {d['std']:.1f} | "
                         f"{d['median']:.1f} | [{d['q1']:.1f}, {d['q3']:.1f}] |")
        else:
            lines.append(f"| {label} | 0 | - | - | - | - |")

    lines += [
        "",
        "## Evidence composition (admissible items)",
        "",
        _table("Source type", aggregates["source_kind_distribution"]),
        "",
        _table("Source proximity", aggregates["source_proximity_distribution"]),
        "",
        _table("Evidence role", aggregates["role_distribution"]),
        "",
        _table("Modality", {k: v for k, v in aggregates["modality_composition"].items()
                            if isinstance(v, int)}),
        "",
        "## Rejections",
        "",
        _table("Evidence rejection reason", aggregates["rejection_reasons"]),
        "",
        _table("Instance status", aggregates["instance_statuses"]),
        "",
        _table("Instance rejection reason", aggregates["instance_rejection_reasons"]),
        "",
        "## Gold verdict recoverability",
        "",
        f"Paired claims: {recoverability['n_paired_claims']}",
        "",
        "| | E_factcheck recoverable | E_factcheck not recoverable |",
        "| --- | ---: | ---: |",
        f"| **E_claim recoverable** | {contingency['both']} | {contingency['only_E_claim']} |",
        f"| **E_claim not recoverable** | {contingency['only_E_factcheck']} | {contingency['neither']} |",
        "",
        f"- Recoverable from `E_claim`: {recoverability['recoverable_from_E_claim']} "
        f"({_pct(recoverability['rate_E_claim'])})",
        f"- Recoverable from `E_factcheck`: {recoverability['recoverable_from_E_factcheck']} "
        f"({_pct(recoverability['rate_E_factcheck'])})",
        f"- Gain attributable to the fact-checking period: "
        f"{_pct(recoverability['gain_from_fact_check_period'])}",
        f"- McNemar exact two-sided p: "
        f"{_fmt(recoverability['mcnemar_exact_p'])}",
        "",
    ]
    return "\n".join(lines)


def _table(title: str, counts: dict) -> str:
    if not counts:
        return f"*No {title.lower()} data.*"
    total = sum(counts.values()) or 1
    rows = [f"| {title} | n | share |", "| --- | ---: | ---: |"]
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        rows.append(f"| {key} | {value} | {value / total:.1%} |")
    return "\n".join(rows)


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"


def _fmt(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else "n/a"


async def main() -> None:
    args = parse_args()
    logger.setLevel(args.log_level)
    await db.connect_maybe_initialize(max_connections=4)

    out_dir = args.out or os.path.join(
        DEFAULT_OUT_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    try:
        claim_records, evidence_records = await collect(args)
        claim_dicts = [record.to_dict() for record in claim_records]
        aggregates = aggregate(claim_records, evidence_records)

        write_csv(os.path.join(out_dir, "claims.csv"), claim_dicts)
        write_json(os.path.join(out_dir, "claims.json"), claim_dicts)
        write_csv(os.path.join(out_dir, "evidence.csv"), evidence_records)
        write_json(os.path.join(out_dir, "evidence.json"), evidence_records)
        write_json(os.path.join(out_dir, "aggregates.json"), aggregates)

        markdown = render_markdown(aggregates, args.mode)
        with open(os.path.join(out_dir, "aggregates.md"), "w", encoding="utf-8") as f:
            f.write(markdown)

        print(markdown)
        print(f"\nWrote results to {out_dir}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
