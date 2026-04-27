#!/usr/bin/env python3
import asyncio
from collections import defaultdict

from tabulate import tabulate

from veritas.common.annotation.rating import Rating
from veritas.common.verdict import Verdict
from veritas.db import db
from veritas.db.annotation_db import annotation_db
from veritas.metric import calculate_metrics
from scripts.stats.human_evaluation.common import build_human_verdict, format_metrics


async def main() -> None:
    await db.connect_maybe_initialize()
    await annotation_db.connect_maybe_initialize()

    print("Fetching data...")
    all_verdicts = await db.get_verdicts()
    auto_verdicts: dict[int, Verdict] = {v.claim_id: v for v in all_verdicts}

    # Fetch all non-dismissed claims to filter out dismissed claims
    query = "SELECT id FROM claims WHERE NOT dismissed"
    rows = await db._fetch(query)
    valid_claim_ids: set[int] = {r['id'] for r in rows}

    human_ratings = await annotation_db.get_completed_ratings()
    human_media_ratings = await annotation_db.get_completed_media_ratings()

    # Fetch users, completed counts, and dismissed counts
    users = await annotation_db.get_all_users()
    user_map: dict[int, str] = {u['id']: u['email'] for u in users}
    completed_counts: dict[int, int] = await annotation_db.get_completed_counts()
    dismissed_counts: dict[int, int] = await annotation_db.get_dismissed_counts()
    excluded_counts: dict[int, int] = await annotation_db.get_excluded_counts()

    # ── Group annotations by (annotator_id, claim_id) ──
    # claim_ratings_by_annotator: annotator_id -> claim_id -> prop -> Rating
    claim_ratings_by_annotator: dict[int, dict[int, dict[str, Rating]]] = defaultdict(lambda: defaultdict(dict))
    # media_ratings_by_annotator: annotator_id -> claim_id -> media_id -> prop -> Rating
    media_ratings_by_annotator: dict[int, dict[int, dict[int, dict[str, Rating]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))

    for entry in human_ratings:
        claim_id: int = entry['claim_id']
        if claim_id not in auto_verdicts or claim_id not in valid_claim_ids:
            continue
        for prop, rating in entry['ratings'].items():
            annotator_id: int = int(rating.rater.split(':')[-1])
            claim_ratings_by_annotator[annotator_id][claim_id][prop] = rating

    for entry in human_media_ratings:
        claim_id: int = entry['claim_id']
        media_id: int = entry['media_id']
        if claim_id not in auto_verdicts or claim_id not in valid_claim_ids:
            continue
        for prop, rating in entry['ratings'].items():
            annotator_id: int = int(rating.rater.split(':')[-1])
            media_ratings_by_annotator[annotator_id][claim_id][media_id][prop] = rating

    # ── Build Verdict pairs per annotator ──
    # annotator_id -> (list[human_verdict], list[auto_verdict])
    annotator_verdict_pairs: dict[int, tuple[list[Verdict], list[Verdict]]] = {}

    all_annotator_ids: set[int] = set(claim_ratings_by_annotator.keys()) | set(media_ratings_by_annotator.keys())

    for annotator_id in all_annotator_ids:
        claim_ids_for_annotator = set(claim_ratings_by_annotator.get(annotator_id, {}).keys()) | \
                                  set(media_ratings_by_annotator.get(annotator_id, {}).keys())
        human_verdicts: list[Verdict] = []
        target_verdicts: list[Verdict] = []
        for cid in sorted(claim_ids_for_annotator):
            auto_v = auto_verdicts.get(cid)
            if auto_v is None:
                continue
            cr = claim_ratings_by_annotator.get(annotator_id, {}).get(cid, {})
            mr = media_ratings_by_annotator.get(annotator_id, {}).get(cid, {})
            human_v = build_human_verdict(cid, auto_v.review_ids, cr, mr, auto_v)
            human_verdicts.append(human_v)
            target_verdicts.append(auto_v)
        if human_verdicts:
            annotator_verdict_pairs[annotator_id] = (human_verdicts, target_verdicts)

    # ── Calculate metrics per annotator ──
    results: list[list] = []
    sorted_annotator_ids: list[int] = sorted(annotator_verdict_pairs.keys())

    for annotator_id in sorted_annotator_ids:
        human_vs, auto_vs = annotator_verdict_pairs[annotator_id]
        metrics = calculate_metrics(human_vs, auto_vs)

        email = user_map.get(annotator_id, "Unknown")
        n_completed = completed_counts.get(annotator_id, 0)
        n_dismissed = dismissed_counts.get(annotator_id, 0)
        n_excluded = excluded_counts.get(annotator_id, 0)
        total_n = n_completed + n_dismissed
        share_dismissed = n_dismissed / total_n if total_n > 0 else 0.0

        results.append([
            annotator_id, email, total_n, n_completed, n_dismissed, f"{share_dismissed:.1%}",
            n_excluded,
            *format_metrics(metrics, "total"),
            *format_metrics(metrics, "integrity"),
        ])

    print("\nQuality of Annotations w.r.t. VeriTaS Automatic Scores:")
    headers = ["ID", "Email", "N (total)", "N (completed)", "Dism.", "Dism. %", "Excluded",
               "MSE", "MAE", "3-bin Acc", "7-bin Acc",
               "Int. MSE", "Int. MAE", "Int. 3-bin Acc", "Int. 7-bin Acc"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # ── IAA: build verdict pairs for each pair of annotators sharing claims ──
    # Group all human verdicts by claim_id for IAA
    # claim_id -> list of (annotator_id, human_verdict)
    claim_annotator_verdicts: dict[int, list[tuple[int, Verdict]]] = defaultdict(list)
    for annotator_id, (human_vs, auto_vs) in annotator_verdict_pairs.items():
        for hv in human_vs:
            claim_annotator_verdicts[hv.claim_id].append((annotator_id, hv))

    # Global IAA pairs
    iaa_preds: list[Verdict] = []
    iaa_targets: list[Verdict] = []
    # Per-annotator IAA pairs
    annotator_iaa_pairs: dict[int, tuple[list[Verdict], list[Verdict]]] = defaultdict(lambda: ([], []))

    for claim_id, annotator_verdicts in claim_annotator_verdicts.items():
        if len(annotator_verdicts) < 2:
            continue
        for i in range(len(annotator_verdicts)):
            for j in range(i + 1, len(annotator_verdicts)):
                aid_i, v_i = annotator_verdicts[i]
                aid_j, v_j = annotator_verdicts[j]
                # Global IAA (symmetric)
                iaa_preds.append(v_i)
                iaa_targets.append(v_j)
                iaa_preds.append(v_j)
                iaa_targets.append(v_i)
                # Per-annotator IAA
                annotator_iaa_pairs[aid_i][0].append(v_i)
                annotator_iaa_pairs[aid_i][1].append(v_j)
                annotator_iaa_pairs[aid_j][0].append(v_j)
                annotator_iaa_pairs[aid_j][1].append(v_i)

    # Global IAA table
    if iaa_preds:
        iaa_metrics = calculate_metrics(iaa_preds, iaa_targets)
        iaa_results = [[
            len(iaa_preds) // 2,
            *format_metrics(iaa_metrics, "total"),
        ]]
        print("\nInter-Annotator Agreement (all pairs):")
        iaa_headers = ["N (pairs)", "MSE", "MAE", "3-bin Acc", "7-bin Acc"]
        print(tabulate(iaa_results, headers=iaa_headers, tablefmt="grid"))
    else:
        print("\nNo multiple evaluations available for IAA calculation.")

    # Per-annotator IAA table
    iaa_per_annotator: list[list] = []
    for annotator_id in sorted_annotator_ids:
        email = user_map.get(annotator_id, "Unknown")
        if annotator_id in annotator_iaa_pairs and annotator_iaa_pairs[annotator_id][0]:
            preds, targets = annotator_iaa_pairs[annotator_id]
            iaa_m = calculate_metrics(preds, targets)
            n_pairs = len(preds)
            iaa_per_annotator.append([
                annotator_id, email, n_pairs,
                *format_metrics(iaa_m, "total"),
            ])
        else:
            iaa_per_annotator.append([
                annotator_id, email, 0, "nan", "nan", "nan", "nan"
            ])

    print("\nInter-Annotator Agreement per Annotator:")
    iaa_per_headers = ["ID", "Email", "N (pairs)", "MSE", "MAE", "3-bin Acc", "7-bin Acc"]
    print(tabulate(iaa_per_annotator, headers=iaa_per_headers, tablefmt="grid"))


if __name__ == "__main__":
    asyncio.run(main())
