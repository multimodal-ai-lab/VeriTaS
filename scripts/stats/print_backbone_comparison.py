#!/usr/bin/env python3
"""Compare claims and verdicts produced by two different backbones.

The *vanilla* backbone results live in ``veritas_db_3`` (the default DB) and
serve as ground truth.  The *test* backbone results (Gemini 3, Q4 2025 only)
live in ``veritas_db_q4_2025``.

The script:
1. Finds matching reviews (by ID) across both databases.
2. Collects the claims belonging to those reviews and pairs them up.
3. Prints the total number of matching claim pairs.
4. Prints a random sample of 10 claim pairs for side-by-side comparison.
5. Computes the four metrics (MSE, MAE, 3-bin Acc, 7-bin Acc) on the paired
   verdicts, using the vanilla verdicts as ground truth.
"""

import asyncio
import random

from tabulate import tabulate

from veritas import database
from veritas.db.veritas_db import db, row_to_verdict, VeritasDB
from veritas.common.verdict import Verdict
from veritas.metric import calculate_metrics

VANILLA_DB = "veritas_db_3"
TEST_DB = "veritas_db_q4_2025"

SAMPLE_SIZE = 10


async def main() -> None:
    # ── Connect to both databases ──
    vanilla_db = db
    _, db_user, db_password, db_host, db_port = database.values()
    test_db = VeritasDB(database=TEST_DB, user=db_user, password=db_password, host=db_host, port=db_port)

    vanilla_db.db_name = VANILLA_DB
    test_db.db_name = TEST_DB

    await vanilla_db.connect()
    await test_db.connect()

    print(f"Connected to vanilla DB ({VANILLA_DB}) and test DB ({TEST_DB}).\n")

    # ── Fetch reviews from both DBs, index by review ID ──
    vanilla_reviews = await vanilla_db._fetch(
        "SELECT id, claim_id FROM reviews WHERE dismissed = FALSE AND claim_id IS NOT NULL"
    )
    test_reviews = await test_db._fetch(
        "SELECT id, claim_id FROM reviews WHERE dismissed = FALSE AND claim_id IS NOT NULL"
    )

    # Map review ID -> claim_id for each DB
    vanilla_id_to_claim: dict[int, int] = {}
    for row in vanilla_reviews:
        vanilla_id_to_claim[row["id"]] = row["claim_id"]

    test_id_to_claim: dict[int, int] = {}
    for row in test_reviews:
        test_id_to_claim[row["id"]] = row["claim_id"]

    # ── Match claims via shared review IDs ──
    matching_review_ids = set(vanilla_id_to_claim.keys()) & set(test_id_to_claim.keys())
    print(f"Matching reviews (both with at least one claim): {len(matching_review_ids)}")

    # Collect (vanilla_claim_id, test_claim_id) pairs
    claim_pairs: list[tuple[int, int]] = []
    for rid in matching_review_ids:
        claim_pairs.append((vanilla_id_to_claim[rid], test_id_to_claim[rid]))

    # Deduplicate pairs (a claim may be linked to multiple reviews)
    claim_pairs = list(set(claim_pairs))
    print(f"Total matching claim pairs: {len(claim_pairs)}\n")

    if not claim_pairs:
        print("No matching claim pairs found.")
        await vanilla_db.close()
        await test_db.close()
        return

    # ── Fetch claims and verdicts for all paired claim IDs ──
    vanilla_claim_ids = list({p[0] for p in claim_pairs})
    test_claim_ids = list({p[1] for p in claim_pairs})

    vanilla_claims_rows = await vanilla_db._fetch(
        "SELECT id, data FROM claims WHERE id = ANY($1)", vanilla_claim_ids
    )
    test_claims_rows = await test_db._fetch(
        "SELECT id, data FROM claims WHERE id = ANY($1)", test_claim_ids
    )

    vanilla_claim_text: dict[int, str] = {row["id"]: row["data"] for row in vanilla_claims_rows}
    test_claim_text: dict[int, str] = {row["id"]: row["data"] for row in test_claims_rows}

    vanilla_verdict_rows = await vanilla_db._fetch(
        "SELECT id, full_verdict, claim_id FROM verdicts WHERE claim_id = ANY($1) AND is_current = TRUE",
        vanilla_claim_ids,
    )
    test_verdict_rows = await test_db._fetch(
        "SELECT id, full_verdict, claim_id FROM verdicts WHERE claim_id = ANY($1) AND is_current = TRUE",
        test_claim_ids,
    )

    vanilla_verdicts: dict[int, Verdict] = {}
    for row in vanilla_verdict_rows:
        vanilla_verdicts[row["claim_id"]] = row_to_verdict(row)

    test_verdicts: dict[int, Verdict] = {}
    for row in test_verdict_rows:
        test_verdicts[row["claim_id"]] = row_to_verdict(row)

    # Keep only pairs where both sides have a verdict
    valid_pairs = [
        (v_cid, t_cid)
        for v_cid, t_cid in claim_pairs
        if v_cid in vanilla_verdicts and t_cid in test_verdicts
    ]
    print(f"Claim pairs with verdicts on both sides: {len(valid_pairs)}\n")

    if not valid_pairs:
        print("No claim pairs with verdicts on both sides.")
        await vanilla_db.close()
        await test_db.close()
        return

    # ── Print random sample of 10 claim pairs ──
    sample = random.sample(valid_pairs, min(SAMPLE_SIZE, len(valid_pairs)))

    print(f"{'=' * 100}")
    print(f"Random sample of {len(sample)} claim pairs (side-by-side comparison)")
    print(f"{'=' * 100}\n")

    for i, (v_cid, t_cid) in enumerate(sample, 1):
        v_verdict = vanilla_verdicts[v_cid]
        t_verdict = test_verdicts[t_cid]

        v_text = vanilla_claim_text.get(v_cid, "(no text)")
        t_text = test_claim_text.get(t_cid, "(no text)")

        print(f"--- Pair {i} (vanilla claim {v_cid} vs test claim {t_cid}) ---")
        print(f"  Vanilla claim: {v_text[:200]}{'...' if len(v_text) > 200 else ''}")
        print(f"  Test claim:    {t_text[:200]}{'...' if len(t_text) > 200 else ''}")
        print()

        # Verdict comparison table
        rows = []
        for prop in ["veracity", "context_coverage"]:
            v_rating = getattr(v_verdict, prop, None)
            t_rating = getattr(t_verdict, prop, None)
            rows.append([
                prop,
                f"{v_rating.score:.2f}" if v_rating else "N/A",
                f"{t_rating.score:.2f}" if t_rating else "N/A",
            ])

        # Media properties
        for v_mv in v_verdict.media_verdicts:
            ref = v_mv.reference
            t_mv = next((m for m in t_verdict.media_verdicts if m.reference == ref), None)
            rows.append([
                f"authenticity ({ref})",
                f"{v_mv.authenticity.score:.2f}",
                f"{t_mv.authenticity.score:.2f}" if t_mv else "N/A",
            ])
            rows.append([
                f"contextualization ({ref})",
                f"{v_mv.contextualization.score:.2f}",
                f"{t_mv.contextualization.score:.2f}" if t_mv else "N/A",
            ])

        try:
            v_integrity = v_verdict.integrity
            v_int_str = f"{v_integrity.score:.2f}"
        except (ValueError, AttributeError):
            v_int_str = "N/A"
        try:
            t_integrity = t_verdict.integrity
            t_int_str = f"{t_integrity.score:.2f}"
        except (ValueError, AttributeError):
            t_int_str = "N/A"
        rows.append(["integrity", v_int_str, t_int_str])

        print(tabulate(rows, headers=["Property", "Vanilla", "Test"], tablefmt="grid"))
        print()

    # ── Dismissal rates ──
    vanilla_total = await vanilla_db._fetchval(
        "SELECT COUNT(*) FROM claims WHERE id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1))",
        list(matching_review_ids),
    )
    vanilla_dismissed = await vanilla_db._fetchval(
        "SELECT COUNT(*) FROM claims WHERE dismissed = TRUE AND id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1))",
        list(matching_review_ids),
    )
    test_total = await test_db._fetchval(
        "SELECT COUNT(*) FROM claims WHERE id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1))",
        list(matching_review_ids),
    )
    test_dismissed = await test_db._fetchval(
        "SELECT COUNT(*) FROM claims WHERE dismissed = TRUE AND id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1))",
        list(matching_review_ids),
    )

    v_rate = vanilla_dismissed / vanilla_total if vanilla_total else 0
    t_rate = test_dismissed / test_total if test_total else 0

    print(f"{'=' * 100}")
    print("Dismissal rates (among claims linked to matching reviews)")
    print(f"{'=' * 100}\n")
    print(tabulate(
        [["Vanilla", vanilla_total, vanilla_dismissed, f"{v_rate:.1%}"],
         ["Test", test_total, test_dismissed, f"{t_rate:.1%}"]],
        headers=["DB", "Total claims", "Dismissed", "Rate"],
        tablefmt="grid",
    ))
    print()

    # Top 5 dismissal reasons
    vanilla_reasons = await vanilla_db._fetch(
        "SELECT dismissed_reason, COUNT(*) as cnt FROM claims WHERE dismissed = TRUE AND id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1)) GROUP BY dismissed_reason ORDER BY cnt DESC LIMIT 5",
        list(matching_review_ids),
    )
    test_reasons = await test_db._fetch(
        "SELECT dismissed_reason, COUNT(*) as cnt FROM claims WHERE dismissed = TRUE AND id IN (SELECT DISTINCT claim_id FROM reviews WHERE claim_id IS NOT NULL AND id = ANY($1)) GROUP BY dismissed_reason ORDER BY cnt DESC LIMIT 5",
        list(matching_review_ids),
    )

    print("Top 5 dismissal reasons (Vanilla):")
    print(tabulate(
        [[row["dismissed_reason"] or "(no reason)", row["cnt"]] for row in vanilla_reasons],
        headers=["Reason", "Count"],
        tablefmt="grid",
    ))
    print()

    print("Top 5 dismissal reasons (Test):")
    print(tabulate(
        [[row["dismissed_reason"] or "(no reason)", row["cnt"]] for row in test_reasons],
        headers=["Reason", "Count"],
        tablefmt="grid",
    ))
    print()

    # ── Claim pairs where one is dismissed and the other isn't ──
    # Fetch ALL reviews (incl. dismissed) with a claim_id to find dismissal mismatches
    vanilla_all_reviews = await vanilla_db._fetch(
        "SELECT id, claim_id FROM reviews WHERE claim_id IS NOT NULL"
    )
    test_all_reviews = await test_db._fetch(
        "SELECT id, claim_id FROM reviews WHERE claim_id IS NOT NULL"
    )
    vanilla_all_id_to_claim = {row["id"]: row["claim_id"] for row in vanilla_all_reviews}
    test_all_id_to_claim = {row["id"]: row["claim_id"] for row in test_all_reviews}
    all_matching_rids = set(vanilla_all_id_to_claim.keys()) & set(test_all_id_to_claim.keys())

    # Fetch dismissed status for all matched claims
    all_v_claim_ids = list({vanilla_all_id_to_claim[r] for r in all_matching_rids})
    all_t_claim_ids = list({test_all_id_to_claim[r] for r in all_matching_rids})

    VALIDATION_FIELDS = [
        "is_ambiguous", "media_expose_verdict", "text_exposes_verdict",
        "missing_referenced_media", "is_inconsistent", "is_unshareable",
    ]
    validation_select = ", ".join(VALIDATION_FIELDS)

    v_dismissed_rows = await vanilla_db._fetch(
        f"SELECT id, dismissed, dismissed_reason, data, {validation_select} FROM claims WHERE id = ANY($1)", all_v_claim_ids
    )
    t_dismissed_rows = await test_db._fetch(
        f"SELECT id, dismissed, dismissed_reason, data, {validation_select} FROM claims WHERE id = ANY($1)", all_t_claim_ids
    )
    v_claim_info = {row["id"]: row for row in v_dismissed_rows}
    t_claim_info = {row["id"]: row for row in t_dismissed_rows}

    # Find pairs where exactly one side is dismissed
    dismissal_mismatch_pairs: list[tuple[int, int]] = []
    for rid in all_matching_rids:
        v_cid = vanilla_all_id_to_claim[rid]
        t_cid = test_all_id_to_claim[rid]
        v_info = v_claim_info.get(v_cid)
        t_info = t_claim_info.get(t_cid)
        if v_info and t_info:
            v_dis = v_info["dismissed"]
            t_dis = t_info["dismissed"]
            if v_dis != t_dis:
                dismissal_mismatch_pairs.append((v_cid, t_cid))
    dismissal_mismatch_pairs = list(set(dismissal_mismatch_pairs))

    print(f"{'=' * 100}")
    print(f"Claim pairs where one is dismissed and the other isn't: {len(dismissal_mismatch_pairs)}")
    print(f"{'=' * 100}\n")

    if dismissal_mismatch_pairs:
        dm_sample = random.sample(dismissal_mismatch_pairs, min(5, len(dismissal_mismatch_pairs)))
        for i, (v_cid, t_cid) in enumerate(dm_sample, 1):
            v_info = v_claim_info[v_cid]
            t_info = t_claim_info[t_cid]
            v_dis = v_info["dismissed"]
            t_dis = t_info["dismissed"]
            dismissed_side = "Vanilla" if v_dis else "Test"
            reason = (v_info["dismissed_reason"] if v_dis else t_info["dismissed_reason"]) or "(no reason)"
            print(f"--- Mismatch pair {i} (vanilla claim {v_cid} vs test claim {t_cid}) ---")
            print(f"  Vanilla claim: {str(v_info['data'])[:200]}")
            print(f"  Test claim:    {str(t_info['data'])[:200]}")
            print(f"  Dismissed side: {dismissed_side}")
            print(f"  Dismissal reason: {reason}")
            if reason == "Claim failed validation.":
                dismissed_info = v_info if v_dis else t_info
                failed_checks = [f for f in VALIDATION_FIELDS if dismissed_info.get(f)]
                if failed_checks:
                    print(f"  Failed validation checks: {', '.join(failed_checks)}")
            print()

    # ── Top 5 claim pairs with strongest verdict MSE difference ──
    pair_mse_list: list[tuple[int, int, float]] = []
    for v_cid, t_cid in valid_pairs:
        v_v = vanilla_verdicts[v_cid]
        t_v = test_verdicts[t_cid]
        scores = []
        for prop in ["veracity", "context_coverage"]:
            v_r = getattr(v_v, prop, None)
            t_r = getattr(t_v, prop, None)
            if v_r and t_r:
                scores.append((v_r.score - t_r.score) ** 2)
        for v_mv in v_v.media_verdicts:
            t_mv = next((m for m in t_v.media_verdicts if m.reference == v_mv.reference), None)
            if t_mv:
                scores.append((v_mv.authenticity.score - t_mv.authenticity.score) ** 2)
                scores.append((v_mv.contextualization.score - t_mv.contextualization.score) ** 2)
        if scores:
            pair_mse_list.append((v_cid, t_cid, sum(scores) / len(scores)))

    pair_mse_list.sort(key=lambda x: x[2], reverse=True)
    top5_mse = pair_mse_list[:5]

    print(f"{'=' * 100}")
    print("Top 5 claim pairs with strongest verdict difference (by MSE)")
    print(f"{'=' * 100}\n")

    for i, (v_cid, t_cid, pair_mse_val) in enumerate(top5_mse, 1):
        v_verdict = vanilla_verdicts[v_cid]
        t_verdict = test_verdicts[t_cid]
        v_text = vanilla_claim_text.get(v_cid, "(no text)")
        t_text = test_claim_text.get(t_cid, "(no text)")

        print(f"--- #{i} MSE={pair_mse_val:.4f} (vanilla claim {v_cid} vs test claim {t_cid}) ---")
        print(f"  Vanilla claim: {v_text[:200]}{'...' if len(v_text) > 200 else ''}")
        print(f"  Test claim:    {t_text[:200]}{'...' if len(t_text) > 200 else ''}")
        print()

        rows = []
        for prop in ["veracity", "context_coverage"]:
            v_rating = getattr(v_verdict, prop, None)
            t_rating = getattr(t_verdict, prop, None)
            rows.append([
                prop,
                f"{v_rating.score:.2f}" if v_rating else "N/A",
                f"{t_rating.score:.2f}" if t_rating else "N/A",
            ])
        for v_mv in v_verdict.media_verdicts:
            ref = v_mv.reference
            t_mv = next((m for m in t_verdict.media_verdicts if m.reference == ref), None)
            rows.append([
                f"authenticity ({ref})",
                f"{v_mv.authenticity.score:.2f}",
                f"{t_mv.authenticity.score:.2f}" if t_mv else "N/A",
            ])
            rows.append([
                f"contextualization ({ref})",
                f"{v_mv.contextualization.score:.2f}",
                f"{t_mv.contextualization.score:.2f}" if t_mv else "N/A",
            ])
        try:
            v_int_str = f"{v_verdict.integrity.score:.2f}"
        except (ValueError, AttributeError):
            v_int_str = "N/A"
        try:
            t_int_str = f"{t_verdict.integrity.score:.2f}"
        except (ValueError, AttributeError):
            t_int_str = "N/A"
        rows.append(["integrity", v_int_str, t_int_str])
        print(tabulate(rows, headers=["Property", "Vanilla", "Test"], tablefmt="grid"))
        print()

    # ── Compute metrics ──
    vanilla_verdict_list: list[Verdict] = []
    test_verdict_list: list[Verdict] = []
    for v_cid, t_cid in valid_pairs:
        vanilla_verdict_list.append(vanilla_verdicts[v_cid])
        test_verdict_list.append(test_verdicts[t_cid])

    metrics = calculate_metrics(test_verdict_list, vanilla_verdict_list)

    print(f"{'=' * 100}")
    print("Metrics (test backbone vs vanilla ground truth)")
    print(f"{'=' * 100}\n")

    headers = ["Property", "MSE", "MAE", "3-bin Acc", "7-bin Acc"]
    rows = []
    for prop in ["total", "veracity", "context_coverage", "authenticity", "contextualization", "integrity"]:
        m = metrics.get(prop)
        if m is None:
            rows.append([prop, "N/A", "N/A", "N/A", "N/A"])
        else:
            rows.append([
                prop,
                f"{m['mse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['acc_3bin']:.4f}",
                f"{m['acc_7bin']:.4f}",
            ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    await vanilla_db.close()
    await test_db.close()


if __name__ == "__main__":
    asyncio.run(main())
