#!/usr/bin/env python3
"""Inspect an individual annotator: quality metrics, IAA, and most disagreeing annotations."""
import asyncio
from collections import defaultdict

from tabulate import tabulate

from veritas.common.annotation.rating import Rating
from veritas.common.verdict import Verdict
from veritas.db import db
from veritas.db.annotation_db import annotation_db
from veritas.metric import calculate_metrics
from scripts.stats.human_evaluation.common import build_human_verdict, format_metrics


async def main(target_id: int) -> None:
    await db.connect_maybe_initialize()
    await annotation_db.connect_maybe_initialize()

    print("Fetching data...")
    all_verdicts = await db.get_verdicts()
    auto_verdicts: dict[int, Verdict] = {v.claim_id: v for v in all_verdicts}

    # Non-dismissed claims
    rows = await db._fetch("SELECT id FROM claims WHERE NOT dismissed")
    valid_claim_ids: set[int] = {r['id'] for r in rows}

    human_ratings = await annotation_db.get_completed_ratings()
    human_media_ratings = await annotation_db.get_completed_media_ratings()

    users = await annotation_db.get_all_users()
    user_map: dict[int, str] = {u['id']: u['email'] for u in users}
    completed_counts: dict[int, int] = await annotation_db.get_completed_counts()
    dismissed_counts: dict[int, int] = await annotation_db.get_dismissed_counts()
    excluded_counts: dict[int, int] = await annotation_db.get_excluded_counts()

    if target_id not in user_map:
        print(f"Annotator ID {target_id} not found.")
        return

    # ── Group annotations by (annotator_id, claim_id) ──
    claim_ratings_by_annotator: dict[int, dict[int, dict[str, Rating]]] = defaultdict(lambda: defaultdict(dict))
    media_ratings_by_annotator: dict[int, dict[int, dict[int, dict[str, Rating]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    # Track annotation_ids for the target annotator: claim_id -> annotation_id
    target_annotation_ids: dict[int, int] = {}

    for entry in human_ratings:
        claim_id: int = entry['claim_id']
        if claim_id not in auto_verdicts or claim_id not in valid_claim_ids:
            continue
        for prop, rating in entry['ratings'].items():
            annotator_id: int = int(rating.rater.split(':')[-1])
            claim_ratings_by_annotator[annotator_id][claim_id][prop] = rating
            if annotator_id == target_id:
                target_annotation_ids[claim_id] = entry['annotation_id']

    for entry in human_media_ratings:
        claim_id: int = entry['claim_id']
        media_id: int = entry['media_id']
        if claim_id not in auto_verdicts or claim_id not in valid_claim_ids:
            continue
        for prop, rating in entry['ratings'].items():
            annotator_id: int = int(rating.rater.split(':')[-1])
            media_ratings_by_annotator[annotator_id][claim_id][media_id][prop] = rating

    # ── Build Verdict pairs per annotator ──
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

    # ── 1. Quality of Annotations w.r.t. VeriTaS Automatic Scores ──
    email: str = user_map.get(target_id, "Unknown")
    n_completed: int = completed_counts.get(target_id, 0)
    n_dismissed: int = dismissed_counts.get(target_id, 0)
    n_excluded: int = excluded_counts.get(target_id, 0)
    total_n: int = n_completed + n_dismissed
    share_dismissed: float = n_dismissed / total_n if total_n > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"Annotator {target_id} — {email}")
    print(f"{'=' * 70}")

    if target_id in annotator_verdict_pairs:
        human_vs, auto_vs = annotator_verdict_pairs[target_id]
        metrics = calculate_metrics(human_vs, auto_vs)
    else:
        metrics = {}

    summary = [
        [target_id, email, total_n, n_completed, n_dismissed, f"{share_dismissed:.1%}",
         n_excluded, *format_metrics(metrics, "total", score_fmt=".4f", pct_fmt=".2%")]
    ]
    headers = ["ID", "Email", "N (total)", "N (completed)", "Dism.", "Dism. %", "Excluded",
               "MSE", "MAE", "3-bin Acc", "7-bin Acc"]
    print("\nQuality of Annotations w.r.t. VeriTaS Automatic Scores:")
    print(tabulate(summary, headers=headers, tablefmt="grid"))

    # ── 2. Inter-Annotator Agreement for this annotator ──
    # Group human verdicts by claim_id
    claim_annotator_verdicts: dict[int, list[tuple[int, Verdict]]] = defaultdict(list)
    for annotator_id, (human_vs, _) in annotator_verdict_pairs.items():
        for hv in human_vs:
            claim_annotator_verdicts[hv.claim_id].append((annotator_id, hv))

    iaa_preds: list[Verdict] = []
    iaa_targets: list[Verdict] = []

    for claim_id, annotator_verdicts in claim_annotator_verdicts.items():
        if len(annotator_verdicts) < 2:
            continue
        for i in range(len(annotator_verdicts)):
            for j in range(i + 1, len(annotator_verdicts)):
                aid_i, v_i = annotator_verdicts[i]
                aid_j, v_j = annotator_verdicts[j]
                if aid_i == target_id:
                    iaa_preds.append(v_i)
                    iaa_targets.append(v_j)
                if aid_j == target_id:
                    iaa_preds.append(v_j)
                    iaa_targets.append(v_i)

    print("\nInter-Annotator Agreement:")
    if iaa_preds:
        iaa_metrics = calculate_metrics(iaa_preds, iaa_targets)
        iaa_table = [[
            len(iaa_preds),
            *format_metrics(iaa_metrics, "total", score_fmt=".4f", pct_fmt=".2%"),
        ]]
        print(tabulate(iaa_table, headers=["N (pairs)", "MSE", "MAE", "3-bin Acc", "7-bin Acc"], tablefmt="grid"))
    else:
        print("  No shared claims with other annotators (nan).")

    # ── 3. Top 10 most disagreeing annotations ──
    # 3a. Disagreements with VeriTaS automatic scores
    # Collect per-property disagreements for the target annotator
    auto_disagreements: list[tuple[float, int, str, str, float, float, str | None]] = []
    if target_id in annotator_verdict_pairs:
        human_vs, auto_vs = annotator_verdict_pairs[target_id]
        for hv, av in zip(human_vs, auto_vs):
            ann_id = target_annotation_ids.get(hv.claim_id, 0)
            # Claim-level properties
            for prop in ["veracity", "context_coverage"]:
                h_rating = getattr(hv, prop, None)
                a_rating = getattr(av, prop, None)
                if h_rating is not None and a_rating is not None:
                    diff = abs(h_rating.score - a_rating.score)
                    explanation = h_rating.explanation
                    auto_disagreements.append((diff, ann_id, f"claim {hv.claim_id}", prop, a_rating.score, h_rating.score, explanation))
            # Media-level properties
            a_media_by_ref = {mv.reference: mv for mv in av.media_verdicts}
            for h_mv in hv.media_verdicts:
                a_mv = a_media_by_ref.get(h_mv.reference)
                if a_mv is None:
                    continue
                ref_id = int(h_mv.reference.split(":")[-1].rstrip(">"))
                for prop in ["authenticity", "contextualization"]:
                    h_rating = getattr(h_mv, prop, None)
                    a_rating = getattr(a_mv, prop, None)
                    if h_rating is not None and a_rating is not None:
                        diff = abs(h_rating.score - a_rating.score)
                        explanation = h_rating.explanation
                        auto_disagreements.append((diff, ann_id, f"claim {hv.claim_id} / media {ref_id}", prop, a_rating.score, h_rating.score, explanation))

    auto_disagreements.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 10 Most Disagreeing Annotations (vs. VeriTaS Automatic Scores):")
    if auto_disagreements:
        rows_auto = []
        for diff, ann_id, label, prop, auto_score, human_score, explanation in auto_disagreements[:10]:
            expl_short = (explanation[:80] + "...") if explanation and len(explanation) > 80 else (explanation or "")
            rows_auto.append([ann_id, label, prop, f"{auto_score:.4f}", f"{human_score:.4f}", f"{diff:.4f}", expl_short])
        print(tabulate(rows_auto, headers=["Ann. ID", "Target", "Property", "Auto Score", "Human Score", "|Diff|", "Explanation"], tablefmt="grid"))
    else:
        print("  No annotations available.")

    # 3b. Disagreements with other annotators
    iaa_disagreements: list[tuple[float, int, str, str, str, float, float, str | None]] = []
    for claim_id, annotator_verdicts in claim_annotator_verdicts.items():
        # Find target's verdict
        target_verdict: Verdict | None = None
        for aid, v in annotator_verdicts:
            if aid == target_id:
                target_verdict = v
                break
        if target_verdict is None:
            continue
        ann_id = target_annotation_ids.get(claim_id, 0)
        for aid, other_v in annotator_verdicts:
            if aid == target_id:
                continue
            other_email = user_map.get(aid, str(aid))
            # Claim-level
            for prop in ["veracity", "context_coverage"]:
                t_rating = getattr(target_verdict, prop, None)
                o_rating = getattr(other_v, prop, None)
                if t_rating is not None and o_rating is not None:
                    diff = abs(t_rating.score - o_rating.score)
                    explanation = t_rating.explanation
                    iaa_disagreements.append((diff, ann_id, f"claim {claim_id}", prop, other_email, t_rating.score, o_rating.score, explanation))
            # Media-level
            o_media_by_ref = {mv.reference: mv for mv in other_v.media_verdicts}
            for t_mv in target_verdict.media_verdicts:
                o_mv = o_media_by_ref.get(t_mv.reference)
                if o_mv is None:
                    continue
                ref_id = int(t_mv.reference.split(":")[-1].rstrip(">"))
                for prop in ["authenticity", "contextualization"]:
                    t_rating = getattr(t_mv, prop, None)
                    o_rating = getattr(o_mv, prop, None)
                    if t_rating is not None and o_rating is not None:
                        diff = abs(t_rating.score - o_rating.score)
                        explanation = t_rating.explanation
                        iaa_disagreements.append((diff, ann_id, f"claim {claim_id} / media {ref_id}", prop, other_email, t_rating.score, o_rating.score, explanation))

    iaa_disagreements.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 10 Most Disagreeing Annotations (vs. Other Annotators):")
    if iaa_disagreements:
        rows_iaa = []
        for diff, ann_id, label, prop, other_email, my_score, other_score, explanation in iaa_disagreements[:10]:
            expl_short = (explanation[:80] + "...") if explanation and len(explanation) > 80 else (explanation or "")
            rows_iaa.append([ann_id, label, prop, other_email, f"{my_score:.4f}", f"{other_score:.4f}", f"{diff:.4f}", expl_short])
        print(tabulate(rows_iaa, headers=["Ann. ID", "Target", "Property", "Other Annotator", "My Score", "Other Score", "|Diff|", "Explanation"], tablefmt="grid"))
    else:
        print("  No shared claims with other annotators.")


if __name__ == "__main__":
    asyncio.run(main(target_id=39))
