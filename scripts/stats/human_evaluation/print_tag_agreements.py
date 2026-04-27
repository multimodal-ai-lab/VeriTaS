from __future__ import annotations

import asyncio

from ezmm import MultimodalSequence
from veritas.common.annotation.rating import Rating, RatingAggregated
from veritas.common.verdict import MediumVerdict, Verdict
from veritas.db.annotation_db import annotation_db
from veritas.db.veritas_db import db

CLAIM_LEVEL_PROPERTIES = ["veracity", "context_coverage"]
MEDIA_PROPERTIES = ["authenticity", "contextualization"]


async def fetch_verdicts(no_agreement_filtering: bool = False) -> list[tuple[int, Verdict, Verdict]]:
    """Fetch (claim_id, human_verdict, automated_verdict) triples from DBs."""
    # 1. Load human ratings
    print("Fetching human ratings...")
    claim_ratings = await annotation_db.get_completed_ratings()
    media_ratings = await annotation_db.get_completed_media_ratings()

    # 2. Load automated verdicts
    print("Fetching automated verdicts...")
    verdicts = await db.get_verdicts()
    verdict_by_claim = {v.claim_id: v for v in verdicts}

    # 3. Group human ratings by claim
    human_claim_groups: dict[int, dict[str, list[Rating]]] = {}
    for r in claim_ratings:
        cid = r['claim_id']
        if cid not in human_claim_groups:
            human_claim_groups[cid] = {p: [] for p in CLAIM_LEVEL_PROPERTIES}
        for prop, rating in r['ratings'].items():
            if prop in human_claim_groups[cid]:
                human_claim_groups[cid][prop].append(rating)

    human_media_groups: dict[int, dict[int, dict[str, list[Rating] | str]]] = {}
    for r in media_ratings:
        cid = r['claim_id']
        mid = r['media_id']
        if cid not in human_media_groups:
            human_media_groups[cid] = {}
        if mid not in human_media_groups[cid]:
            human_media_groups[cid][mid] = {p: [] for p in MEDIA_PROPERTIES}
            human_media_groups[cid][mid]["claim_data"] = r["claim_data"]
        for prop, rating in r['ratings'].items():
            if prop in human_media_groups[cid][mid]:
                human_media_groups[cid][mid][prop].append(rating)

    # 4. Create Verdict objects
    results: list[tuple[int, Verdict, Verdict]] = []
    all_claim_ids = (set(human_claim_groups.keys()) | set(human_media_groups.keys())) & set(verdict_by_claim.keys())

    for cid in all_claim_ids:
        auto_verdict = verdict_by_claim[cid]
        claim = await auto_verdict.claim
        if claim.dismissed:
            if not no_agreement_filtering or not claim.dismissed_reason == "Verdict agreement too low.":
                continue

        h_props = {}
        if cid in human_claim_groups:
            for prop, ratings in human_claim_groups[cid].items():
                if ratings:
                    h_props[prop] = RatingAggregated(individual_ratings=ratings, rater="human")

        h_media_verdicts = []
        if cid in human_media_groups:
            for mid, group_data in human_media_groups[cid].items():
                h_media_props = {}
                for prop in MEDIA_PROPERTIES:
                    ratings = group_data.get(prop)
                    if ratings:
                        h_media_props[prop] = RatingAggregated(individual_ratings=ratings, rater="human")

                if "authenticity" in h_media_props and "contextualization" in h_media_props:
                    claim_data = group_data["claim_data"]
                    media_items = MultimodalSequence(claim_data).unique_items()
                    reference = None
                    for item in media_items:
                        if item.id == mid:
                            reference = item.reference
                            break

                    h_media_verdicts.append(MediumVerdict(
                        reference=reference,
                        authenticity=h_media_props["authenticity"],
                        contextualization=h_media_props["contextualization"]
                    ))

        try:
            human_verdict = Verdict(
                claim_id=cid,
                review_ids=set(),
                veracity=h_props.get("veracity"),
                context_coverage=h_props.get("context_coverage"),
                media_verdicts=h_media_verdicts
            )
            results.append((cid, human_verdict, auto_verdict))
        except Exception:
            continue

    return results


def filter_verdicts(
        data_points: list[tuple[int, Verdict, Verdict]],
        exclude_single_human_ratings: bool = True,
        exclude_human_disagreement: bool = True
) -> list[tuple[int, Verdict, Verdict]]:
    filtered = []
    for cid, hv, av in data_points:
        if exclude_single_human_ratings and hv.n_ratings < 2:
            continue
        if exclude_human_disagreement and not hv.sufficient_agreement:
            continue
        filtered.append((cid, hv, av))
    return filtered


async def main():
    await db.connect_maybe_initialize()
    await annotation_db.connect_maybe_initialize()

    data_points = await fetch_verdicts()
    filtered_data = filter_verdicts(data_points)

    total_media_items = 0
    matching_tags_count = 0

    # We want to compare authenticity tags for each media item
    # Tags to track individually: "AI-generated", "Manipulated", "Forged"
    tags_to_track = ["AI-generated", "Manipulated", "Forged"]
    tag_matches = {tag: 0 for tag in tags_to_track}

    for cid, hv, av in filtered_data:
        h_media = {mv.medium.id: mv for mv in hv.media_verdicts}
        a_media = {mv.medium.id: mv for mv in av.media_verdicts}

        for mid in h_media.keys() & a_media.keys():
            h_mv = h_media[mid]
            a_mv = a_media[mid]

            h_tags = set(h_mv.authenticity.tags)
            a_tags = set(a_mv.authenticity.tags)

            total_media_items += 1
            if h_tags == a_tags:
                matching_tags_count += 1

            for tag in tags_to_track:
                if (tag in h_tags) == (tag in a_tags):
                    tag_matches[tag] += 1

    print(f"\nAnalyzed {len(filtered_data)} claims with {total_media_items} media items.")
    if total_media_items > 0:
        accuracy = matching_tags_count / total_media_items
        print(f"Global authenticity tag agreement: {accuracy:.2%} ({matching_tags_count}/{total_media_items})")
        for tag in tags_to_track:
            tag_acc = tag_matches[tag] / total_media_items
            print(f"  - {tag}: {tag_acc:.2%} ({tag_matches[tag]}/{total_media_items})")
    else:
        print("No media items found for comparison.")


if __name__ == "__main__":
    asyncio.run(main())
