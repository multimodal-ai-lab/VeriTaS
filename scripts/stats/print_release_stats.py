import asyncio

import numpy as np
from ezmm import MultimodalSequence
from langcodes import Language

from veritas.common.annotation.rating import map_3_bins, Category3Bin, map_7_bins, Category7Bin
from veritas.db import db


async def print_release_stats():
    await db.connect_maybe_initialize()

    query = """
            SELECT c.id, c.data, c.language, c.appearance_ids, c.review_ids, v.integrity
            FROM claims c
                     LEFT JOIN verdicts v ON c.id = v.claim_id AND v.is_current = TRUE
            WHERE c.released_quarter = TRUE
               OR c.released_longitudinal = TRUE \
            """
    rows = await db._fetch(query)

    # Fetch publisher mapping for reviews
    publisher_query = "SELECT id, publisher_id FROM reviews WHERE id = ANY($1)"
    all_review_ids = set()
    for row in rows:
        if row["review_ids"]:
            all_review_ids.update(row["review_ids"])

    review_to_publisher = {}
    if all_review_ids:
        publisher_rows = await db._fetch(publisher_query, list(all_review_ids))
        review_to_publisher = {r["id"]: r["publisher_id"] for r in publisher_rows}

    # Fetch appearance mapping for platforms
    appearance_query = "SELECT id, url FROM appearances WHERE id = ANY($1)"
    all_appearance_ids = set()
    for row in rows:
        if row["appearance_ids"]:
            all_appearance_ids.update(row["appearance_ids"])

    appearance_to_url = {}
    if all_appearance_ids:
        appearance_rows = await db._fetch(appearance_query, list(all_appearance_ids))
        appearance_to_url = {r["id"]: r["url"] for r in appearance_rows}

    from veritas.util import get_domain
    from veritas.common.platforms import PLATFORMS
    from collections import Counter

    def get_platform(url: str | None) -> str | None:
        if not url:
            return None
        domain = get_domain(url)
        return PLATFORMS.get(domain, domain)

    lang_counter = Counter()
    pub_counter = Counter()
    platform_counter = Counter()
    word_counts = []
    total_appearances = 0

    counts = {
        "compromised": 0,
        "unknown": 0,
        "intact": 0,
        "total": 0,
    }
    fine_grained_counts = Counter()
    media_stats = {
        "with_images": 0,
        "with_videos": 0,
        "with_media": 0,
        "without_media": 0,
        "images_only": 0,
        "videos_only": 0,
        "both": 0,
        "count_0": 0,
        "count_1": 0,
        "count_2": 0,
        "count_3": 0,
        "count_4": 0,
        "total_images": 0,
        "total_videos": 0
    }

    for row in rows:
        counts["total"] += 1

        # Category
        integrity = row["integrity"]
        if integrity is None:
            counts["unknown"] += 1
            fine_grained_counts[Category7Bin.NEUTRAL] += 1
        else:
            cat = map_3_bins(integrity)
            if cat == Category3Bin.NEGATIVE:
                counts["compromised"] += 1
            elif cat == Category3Bin.POSITIVE:
                counts["intact"] += 1
            else:
                counts["unknown"] += 1

            cat_7 = map_7_bins(integrity)
            fine_grained_counts[cat_7] += 1

        # Languages
        if row["language"]:
            lang_counter[row["language"]] += 1

        # Publishers
        for rid in (row["review_ids"] or []):
            pub_id = review_to_publisher.get(rid)
            if pub_id:
                pub_counter[pub_id] += 1

        # Platforms
        claim_platforms = set()
        appearance_ids = row["appearance_ids"] or []
        total_appearances += len(appearance_ids)
        for aid in appearance_ids:
            url = appearance_to_url.get(aid)
            platform = get_platform(url)
            if platform:
                claim_platforms.add(platform)

        for platform in claim_platforms:
            platform_counter[platform] += 1

        # Word count
        n_words = len(row["data"].split())
        word_counts.append(n_words)

        # Media
        mm_seq = MultimodalSequence(row["data"])
        n_images = len(mm_seq.images)
        n_videos = len(mm_seq.videos)

        media_stats["total_images"] += n_images
        media_stats["total_videos"] += n_videos

        if n_images > 0:
            media_stats["with_images"] += 1
        if n_videos > 0:
            media_stats["with_videos"] += 1

        if n_images > 0 and n_videos == 0:
            media_stats["images_only"] += 1
        elif n_videos > 0 and n_images == 0:
            media_stats["videos_only"] += 1
        elif n_images > 0 and n_videos > 0:
            media_stats["both"] += 1

        total_media = n_images + n_videos
        if total_media == 0:
            media_stats["count_0"] += 1
        elif total_media == 1:
            media_stats["count_1"] += 1
        elif total_media == 2:
            media_stats["count_2"] += 1
        elif total_media == 3:
            media_stats["count_3"] += 1
        else:
            media_stats["count_4"] += 1

        if n_images > 0 or n_videos > 0:
            media_stats["with_media"] += 1
        else:
            media_stats["without_media"] += 1

    # Print results
    print(f"{'Category':<15} | {'Count':>12} | {'Share':>8}")
    print("-" * 41)
    for key, label in [("compromised", "Compromised"), ("unknown", "Unknown"), ("intact", "Intact")]:
        count = counts[key]
        share = (count / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"{label:<15} | {count:>12,} | {share:>7.1f}%")
    print("-" * 41)
    print(f"{'Total':<15} | {counts['total']:>12,} | {'100.0%':>8}")
    print("\n")

    print(f"{'7-Bin Category':<25} | {'Count':>12} | {'Share':>8}")
    print("-" * 51)
    for cat in [
        Category7Bin.NEG_CERTAIN,
        Category7Bin.NEG_RATHER_CERTAIN,
        Category7Bin.NEG_RATHER_UNCERTAIN,
        Category7Bin.NEUTRAL,
        Category7Bin.POS_RATHER_UNCERTAIN,
        Category7Bin.POS_RATHER_CERTAIN,
        Category7Bin.POS_CERTAIN
    ]:
        label = cat.name.replace("_", " ").lower().capitalize()
        count = fine_grained_counts[cat]
        share = (count / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"{label:<25} | {count:>12,} | {share:>7.1f}%")
    print("-" * 51)
    print("\n")

    print(f"{'Media Summary':<25} | {'Count':>12} | {'Share':>8}")
    print("-" * 51)
    for key, label in [("with_images", "Claims with images"), ("with_videos", "Claims with videos"),
                       ("with_media", "Claims with media"), ("without_media", "Claims without media"),
                       ("images_only", "  images only"), ("videos_only", "  videos only"),
                       ("both", "  both")]:
        count = media_stats[key]
        share = (count / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"{label:<25} | {count:>12,} | {share:>7.1f}%")

    print("-" * 51)
    for key, label in [("count_0", "0 media"), ("count_1", "1 medium"), ("count_2", "2 media"),
                       ("count_3", "3 media"), ("count_4", "4+ media")]:
        count = media_stats[key]
        share = (count / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"{label:<25} | {count:>12,} | {share:>7.1f}%")

    print("-" * 51)
    print(f"{'Total images':<25} | {media_stats['total_images']:>12,}")
    print(f"{'Total videos':<25} | {media_stats['total_videos']:>12,}")
    print("\n")

    # Summary of unique items
    n_langs = len(lang_counter)
    n_langs_50 = sum(1 for _lang, _count in lang_counter.items() if _count >= 50)
    n_pubs = len(pub_counter)
    n_platforms = len(platform_counter)

    # Sort languages by occurrence descending
    sorted_langs = [Language.get(lang).display_name() for lang, count in lang_counter.most_common()]
    langs_str = ", ".join(sorted_langs)

    avg_words = np.mean(word_counts) if word_counts else 0
    std_words = np.std(word_counts) if word_counts else 0

    print(f"{'Description':<35} | {'Count':>10}")
    print("-" * 48)
    print(f"{'Unique languages':<35} | {n_langs:>10,}")
    print(f"{'Languages (>=50 occurrences)':<35} | {n_langs_50:>10,}")
    print(f"{'Unique publishers':<35} | {n_pubs:>10,}")
    print(f"{'Unique platforms':<35} | {n_platforms:>10,}")
    print(f"{'Total appearances':<35} | {total_appearances:>10,}")
    print(f"{'Average claim length (words)':<35} | {avg_words:>7.1f} ± {std_words:<4.1f}")

    print(f"\nLanguages: {langs_str}")


if __name__ == "__main__":
    asyncio.run(print_release_stats())
