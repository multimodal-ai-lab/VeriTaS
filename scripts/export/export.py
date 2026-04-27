"""
Export data from the VeriTaS Database into benchmark format for release.

Exports claims with aggregated ensemble decisions (no individual model predictions) and
    media files, organized by quarter, 1000 claims per quarter, and a longitudinal split
    containing 100 samples per quarter, spanning the entire time period.
Output: One zip file per quarter containing JSON data and media files, and one zip file
    for the longitudinal split.

Features:
- Omits dismissed claims.
- Maintains exact balance between intact and compromised claims.
- Intact claims which are NOT rectified claims take precedence over rectified claims.
- Many claims have a rectified variant. If one variant is included, the
    other variant must NOT be included in the returned sample.
- Claims with media take precedence over text-only claims as long as the final share
    of media-based claims does not exceed 80%.
- The share of NEI claims should match the share of all original (i.e. non-rectified)
    NEI claims.
- Claims with complete verdict (all LLM predictions present) take precedence over
    claims with incomplete verdicts.
- Random sampling using ORDER BY RANDOM() to avoid bias.
- Exports aggregated ensemble values (no individual model predictions)
- Includes claim text, date, and all media files
- The longitudinal split contains only claims that also occur in the quarter splits and,
    for each quarter, exhibits the same properties as each quarter.
- Excludes (near) duplicates both in the visual and in the textual embedding space.

Output Structure:
    Each quarter produces a zip file: veritas_{year}_q{quarter}.zip
    The longitudinal split produces a zip file:
    veritas_longitudinal_{start_year}_q{start_quarter}_{end_year}_q{end_quarter}.zip
    
    Contents:
    - meta.json: Metadata about this split
    {
     // Quarter split:
     "year": 2024,  
     "quarter": 1,
     
     // Longitudinal split:
     "start_year": 2024,  
     "start_quarter": 1,  
     "end_year": 2025,  
     "end_quarter": 4,
     
     "claim_counts": {
       "intact": 488,
       "nei": 24,
       "compromised": 488,
       "total": 1000
     },
     "media_counts": {
       "images": 415,
       "videos": 278
     }
    }
    - claims.json: Claim data with standardized verdicts
        {
          "claims": [
            {
              "id": 123,
              "text": "This is an outrageous statement.",
              "date": "2024-01-15T10:30:00",
              "language": "en",
              "review_url": "https://afp.com/review/aweghzrddgfy",
              "media": [
                {
                  "type": "image",
                  "id": 256,
                  "file_path": "images/256.jpg",
                  "authenticity": {
                    "score": 0.67,
                    "justification": "An explanation for the score."
                  },
                  "contextualization": {
                    "score": 0.33,
                    "justification": "An explanation for the score."
                  },
                {
                  "type": "video",
                  "id": 64,
                  "file_path": "videos/64.mp4",
                  "authenticity": {
                    "score": 0.0,
                    "justification": "An explanation for the score."
                  },
                  "contextualization": {
                    "score": 0.0,
                    "justification": "An explanation for the score."
                  }
                }
              ],
              "veracity": {
                "score": 0.83,
                "justification": "An explanation for the score."
              },
              "context_coverage": {
                "score": -0.92,
                "justification": "An explanation for the score."
              },
              "integrity": {
                "score": -0.92,
                "decisive_property": "context_coverage"
              }
            },
            {
              "id": 124,
              "text": "This is another outrageous statement, but without any media.",
              "date": "2024-01-16T10:30:00",
              "language": "en",
              "review_url": "https://dpa.com/review/alksuefhösld",
              "media": [],
              "veracity": {
                "score": -0.4,
                "justification": "An explanation for the score."
              },
              "context_coverage": null,
              "integrity": {
                "score": -0.4,
                "decisive_property": "veracity"
              }
            }
          ]
        }

    - images/ and videos/: Directories with all needed media files
"""

import asyncio
import json
import random
import shutil
import zipfile
from collections import defaultdict, Counter
from datetime import date
from pathlib import Path

from ezmm import MultimodalSequence, Image, Video, Item

from veritas.common import Claim
from veritas.common.annotation.rating import Category3Bin
from veritas.db import db
from veritas.util.util import get_quarter_date_range


# Applied to quarters with less than 80% media share
MAX_MEDIA_SHARE = 0.80


async def fetch_claims_for_quarter(
        start_date: date, end_date: date,
) -> list[Claim]:
    """
    Fetch all candidate claims for a specific quarter.

    Args:
        start_date: Start of quarter
        end_date: End of quarter (inclusive)
    """
    # Fetch all relevant claims with their current verdict's integrity score
    query = """
            SELECT *
            FROM claims
            WHERE (NOT dismissed OR dismissed_reason = 'In favor of rectified version.')
              AND date BETWEEN $1 AND $2
              AND verdict_ids != '{}'
            ORDER BY RANDOM(); \
            """

    rows = await db._fetch(query, start_date, end_date)
    return [Claim.model_validate(dict(row)) for row in rows]


async def sample_claims(
        claims: list[Claim],
        target_count: int = 1000,
        prioritize: bool = True,
) -> list[Claim]:
    """Chooses a balanced sample of claims from the given list, up to the target count.

    Sampling criteria:
    - The number of intact claims must meet exactly the number of compromised claims.
    - Intact claims which are NOT rectified claims take precedence over rectified claims.
    - Many claims have a variant with opposite verdict. If one variant is included, the other
      variant must NOT be included in the returned sample.
    - Claims with media take precedence over text-only claims as long as the final share of
      media-based claims does not exceed 80%.
    - The share of NEI claims should match the share of all original (i.e. non-rectified) NEI claims.
    - Claims with complete verdict (Verdict.is_complete()) take precedence over claims with
      incomplete verdicts.

    Args:
        claims: List of Claim objects to sample from
        target_count: Target number of claims to sample (default 1000)
        prioritize: Whether to prioritize claims as described above (default True)
    """
    # Ensure target_count is even
    assert target_count % 2 == 0

    # Calculate NEI share from original (non-rectified) claims
    original_claims = [c for c in claims if not c.is_rectified]
    if not original_claims:
        original_nei_share = 0.0
    else:
        original_nei_count = 0
        for c in original_claims:
            v = await c.current_verdict
            if v.integrity.as_3_bin() == Category3Bin.NEUTRAL:
                original_nei_count += 1
        original_nei_share = original_nei_count / len(original_claims)

    target_nei_count = round(target_count * original_nei_share / 2) * 2  # Ensure even number
    target_per_category = (target_count - target_nei_count) // 2  # Is even

    # Categorize claims and gather necessary info for sorting
    categorized_claims: dict[Category3Bin, list[dict]] = defaultdict(list)
    for claim in claims:
        verdict = await claim.current_verdict
        assert verdict, f"No current verdict for claim {claim.id}"
        variant = await claim.variant
        claim_mm_seq = MultimodalSequence(claim.data)
        media = claim_mm_seq.has_videos() or claim_mm_seq.has_images()
        categorized_claims[verdict.integrity.as_3_bin()].append({
            "claim": claim,
            "category": verdict.integrity.as_3_bin(),
            "verdict": verdict,
            "has_media": media,
            "is_complete": verdict.is_complete(4),
            "is_rectified": claim.is_rectified,
            "uses_article_media": claim.media_origin == "article",
            "has_rectified_variant": variant is not None and variant.is_rectified
                                     and not variant.dismissed and variant.verdict_ids,
        })

    # Flatten list of claims
    claim_pool = []
    for cat in categorized_claims:
        claim_pool.extend(categorized_claims[cat])

    # Randomize first to ensure equal-priority items are sampled randomly
    random.shuffle(claim_pool)

    # Priority sorting
    # Criteria: Variant-free > Non-article media > Media at all > Complete Verdict > Non-Rectified
    def sort_key(item):
        return (
            not item["has_rectified_variant"],
            not item["uses_article_media"],  # Prefer even text-only claims over article media
            item["has_media"],
            item["is_complete"],
            not item["is_rectified"]
        )

    # Sort by priority
    if prioritize:
        claim_pool.sort(key=sort_key, reverse=True)

    # Get counts of available claims in each category
    available_intact = len(categorized_claims[Category3Bin.POSITIVE])
    available_compromised = len(categorized_claims[Category3Bin.NEGATIVE])
    available_nei = len(categorized_claims[Category3Bin.NEUTRAL])

    # OUTDATED: Anticipate maximum target share of media for each category
    # n_intact_with_media = sum(1 for c in categorized_claims[Category3Bin.POSITIVE] if c["has_media"])
    # n_compromised_with_media = sum(1 for c in categorized_claims[Category3Bin.NEGATIVE] if c["has_media"])
    # max_media_share = min(MAX_MEDIA_SHARE,
    #                       n_intact_with_media / target_per_category +0.1,
    #                       n_compromised_with_media / target_per_category +0.1)  # Allow for 10% difference

    n_media_claims = sum(1 for c in claim_pool if c["has_media"])
    max_media_share = max(MAX_MEDIA_SHARE, n_media_claims / len(claim_pool))

    # Refined selection with all constraints
    selected_ids = set()
    selected_claims: list[Claim] = []
    selected_claims_cat: dict[Category3Bin, list[Claim]] = defaultdict(list)
    media_claim_counter = Counter()

    def pick(target_cat: Category3Bin) -> dict:
        """Get the highest-priority claim from the pool for the given category, if available.
        Removes it from the pool."""
        for i, candidate in enumerate(claim_pool):
            if candidate["category"] == target_cat:
                return claim_pool.pop(i)
        raise ValueError(f"No more claims available for category {target_cat.name}.")

    def is_suitable(candidate: dict) -> bool:
        """Checks whether the given claim is suitable for inclusion in the sample."""
        claim: Claim = candidate["claim"]

        # Avoid duplicates
        if claim.id in selected_ids:
            return False

        # Omit rectified/original variants
        if claim.variant_id in selected_ids:
            return False

        # Media cap check
        if candidate["has_media"] and candidate["category"] != Category3Bin.NEUTRAL:
            n_media_claims = media_claim_counter[candidate["category"]]
            if n_media_claims + 1 > target_per_category * max_media_share:
                return False

        return True

    def add(candidate: dict) -> None:
        """Saves the given claim to the sample for release."""
        claim: Claim = candidate["claim"]

        selected_claims.append(claim)
        selected_ids.add(claim.id)
        selected_claims_cat[candidate["category"]].append(claim)
        if candidate["has_media"]:
            media_claim_counter[candidate["category"]] += 1

    def choose(target_cat: Category3Bin) -> None:
        """Selects the next suitable highest-priority claim from the pool for the
        given category and saves it."""
        while True:
            candidate = pick(target_cat)
            if is_suitable(candidate):
                add(candidate)
                return

    # Fill up NEI claims first
    for _ in range(target_nei_count):
        choose(Category3Bin.NEUTRAL)

    # Pick intact and compromised claims alternatingly to avoid category bias due to variant exclusion
    while len(selected_claims) < target_count:
        choose(Category3Bin.POSITIVE)
        choose(Category3Bin.NEGATIVE)

    # Ensure we meet the overall target count
    assert len(selected_claims) == target_count, \
        (f"Failed to meet target count: {len(selected_claims)} != {target_count}. Available claims: "
         f"{available_intact} intact, {available_compromised} compromised, {available_nei} NEI")

    # Ensure the category distribution is balanced
    n_intact = len(selected_claims_cat[Category3Bin.POSITIVE])
    n_compromised = len(selected_claims_cat[Category3Bin.NEGATIVE])
    assert n_intact == n_compromised, (f"Intact and compromised claim counts are not balanced: "
                                       f"{n_intact} != {n_compromised}")

    # Re-shuffle the combined list so they aren't sorted by priority or integrity
    random.shuffle(selected_claims)

    return selected_claims


async def prepare_export_data(sampled_claims: list[Claim]) -> tuple[list[dict], dict[str, Item], dict]:
    """
    Prepare claim data for JSON export and collect all referenced media.

    Args:
        sampled_claims: List of Claim objects to prepare

    Returns:
        Tuple of (list of prepared claim dicts, dict of all media items, dict of counts)
    """
    prepared_claims = []
    integrity_counter = Counter()
    images_count = 0
    videos_count = 0
    all_media: dict[str, Item] = {}

    for claim in sampled_claims:
        verdict = await claim.current_verdict
        if not verdict:
            print(f"  Warning: No current verdict for claim {claim.id}")
            continue

        category = verdict.integrity.as_3_bin()
        integrity_counter[category] += 1

        # Extract aggregated values from media verdicts
        media_verdicts = []
        for medium_verdict in verdict.media_verdicts:
            medium = medium_verdict.medium
            aggregated_media = {
                "type": medium.kind,
                "id": medium.id,
                "file_path": f"{medium.kind}s/{medium.file_path.name}",
                "authenticity": {
                    "score": round(medium_verdict.authenticity.score, 2),
                    "justification": medium_verdict.authenticity.explanation,
                },
                "contextualization": {
                    "score": round(medium_verdict.contextualization.score, 2),
                    "justification": medium_verdict.contextualization.explanation,
                },
            }
            media_verdicts.append(aggregated_media)

        claim_mm_seq = MultimodalSequence(claim.data)

        review = (await claim.reviews)[0]

        # Build claim dict for JSON
        claim_dict = {
            "id": claim.id,
            "text": " ".join(i for i in claim_mm_seq if isinstance(i, str)).strip(),
            "date": claim.date.isoformat(),
            "language": claim.language,
            "review_url": str(review.url),
            "media": media_verdicts,
            "veracity": {
                "score": round(verdict.veracity.score, 2) if verdict.veracity else None,
                "justification": verdict.veracity.explanation if verdict.veracity else None,
            } if verdict.veracity else None,
            "context_coverage": {
                "score": round(verdict.context_coverage.score, 2) if verdict.context_coverage else None,
                "justification": verdict.context_coverage.explanation if verdict.context_coverage else None,
            } if verdict.context_coverage else None,
            "integrity": {
                "score": round(verdict.integrity.score, 2),
                "decisive_property": verdict.compromising_property_name,
            },
        }
        prepared_claims.append(claim_dict)

        # Collect media items
        media = claim_mm_seq.unique_items()
        for medium in media:
            all_media[f"{medium.kind}/{medium.file_path.name}"] = medium
            if isinstance(medium, Image):
                images_count += 1
            elif isinstance(medium, Video):
                videos_count += 1

    counts = {
        "claim_counts": {
            "intact": integrity_counter[Category3Bin.POSITIVE],
            "nei": integrity_counter[Category3Bin.NEUTRAL],
            "compromised": integrity_counter[Category3Bin.NEGATIVE],
            "total": len(prepared_claims),
        },
        "media_counts": {
            "images": images_count,
            "videos": videos_count,
        },
    }

    return prepared_claims, all_media, counts


def create_export_zip(
        output_path: Path,
        temp_dir: Path,
        meta_data: dict,
        prepared_claims: list[dict],
        all_media: dict[str, Item],
) -> Path:
    """
    Create the export zip file containing metadata, claims, and media.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Write metadata JSON
        meta_json_path = temp_dir / "meta.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        # Write claims JSON
        claims_json_path = temp_dir / "claims.json"
        with open(claims_json_path, "w", encoding="utf-8") as f:
            json.dump({"claims": prepared_claims}, f, indent=2, ensure_ascii=False)

        # Copy media files
        media_dir = temp_dir / "media"
        media_dir.mkdir(exist_ok=True)

        copied_count = 0
        missing_count = 0

        for item in all_media.values():
            media_type = "image" if isinstance(item, Image) else "video"
            filename = item.file_path.name
            src_path = item.file_path
            dst_dir = media_dir / f"{media_type}s"
            dst_dir.mkdir(exist_ok=True)
            dst_path = dst_dir / filename

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"  Warning: Media file not found: {src_path}")
                missing_count += 1

        if copied_count or missing_count:
            print(f"  Copied {copied_count} media files ({missing_count} missing)")

        # Create zip file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(meta_json_path, "meta.json")
            zipf.write(claims_json_path, "claims.json")
            for item in all_media.values():
                media_type = "image" if isinstance(item, Image) else "video"
                filename = item.file_path.name
                file_path = media_dir / f"{media_type}s" / filename
                if file_path.exists():
                    zipf.write(file_path, f"{media_type}s/{filename}")

        return output_path
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def export_quarter(
        year: int,
        quarter: int,
        output_dir: Path,
        target_count: int = 1000,
        resample: bool = True,
) -> tuple[Path | None, list[Claim]]:
    """
    Export data for a specific quarter.

    Args:
        year: Year
        quarter: Quarter (1-4)
        output_dir: Output directory for zip files
        target_count: Target number of claims per quarter
        resample: Whether to resample claims or just load released claims

    Returns:
        Tuple of (Path to created zip file, list of sampled Claim objects), or (None, []) if no data
    """
    start_date, end_date = get_quarter_date_range(year, quarter)
    print(f"\nExporting Q{quarter} {year} ({start_date} to {end_date})...")

    if resample:
        # Fetch candidate claims
        claims = await fetch_claims_for_quarter(start_date, end_date)
        if not claims:
            print(f"  No claims found for Q{quarter} {year}")
            return None, []

        print(f"  Found {len(claims)} candidate claims")

        sampled_claims = await sample_claims(claims, target_count)
        print(f"  Sampled {len(sampled_claims)} claims")
    else:
        # Fetch already released claims
        query = "SELECT * FROM claims WHERE released_quarter = TRUE AND date BETWEEN $1 AND $2"
        rows = await db._fetch(query, start_date, end_date)
        sampled_claims = [Claim.model_validate(dict(row)) for row in rows]
        if not sampled_claims:
            print(f"  No released claims found for Q{quarter} {year}")
            return None, []

        print(f"  Loaded {len(sampled_claims)} already released claims")

    # Prepare claim data
    prepared_claims, all_media, counts = await prepare_export_data(sampled_claims)

    print(
        f"  Integrity distribution: {counts['claim_counts']['intact']} positive, {counts['claim_counts']['nei']} unknown (nei), {counts['claim_counts']['compromised']} negative")
    print(f"  Media files: {len(all_media)}")

    # Create zip file
    zip_filename = f"veritas_{year}_q{quarter}.zip"
    zip_path = output_dir / zip_filename
    temp_dir = output_dir / f"temp_{year}_q{quarter}"

    meta_data = {
        "year": year,
        "quarter": quarter,
        **counts
    }

    create_export_zip(zip_path, temp_dir, meta_data, prepared_claims, all_media)

    print(f"  Created: {zip_path}")
    return zip_path, sampled_claims


async def export_longitudinal(
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
        claims_per_quarter: dict[tuple[int, int], list[Claim]],
        output_dir: Path,
        target_count_per_quarter: int = 100,
        resample: bool = True,
) -> tuple[Path | None, list[Claim]]:
    """
    Export a longitudinal split across all processed quarters.

    Args:
        start_year: Start year of the split
        start_quarter: Start quarter of the split
        end_year: End year of the split
        end_quarter: End quarter of the split
        claims_per_quarter: Dictionary mapping (year, quarter) to its list of sampled claims
        output_dir: Output directory for zip files
        target_count_per_quarter: Target number of claims per quarter for this split
        resample: Whether to resample claims or just load released claims

    Returns:
        Tuple of (Path to created zip file, list of sampled claims for the longitudinal split), or (None, []) if no data
    """
    print(f"\nExporting longitudinal split Q{start_quarter} {start_year} to Q{end_quarter} {end_year}...")

    # Create zip filename and path
    zip_filename = f"veritas_longitudinal_{start_year}_q{start_quarter}_{end_year}_q{end_quarter}.zip"
    zip_path = output_dir / zip_filename

    if not resample:
        # Load released longitudinal claims
        start_date, _ = get_quarter_date_range(start_year, start_quarter)
        _, end_date = get_quarter_date_range(end_year, end_quarter)
        query = "SELECT * FROM claims WHERE released_longitudinal = TRUE AND date BETWEEN $1 AND $2"
        rows = await db._fetch(query, start_date, end_date)
        longitudinal_claims = [Claim.model_validate(dict(row)) for row in rows]

        if not longitudinal_claims:
            print(f"  No released longitudinal claims found for given range.")
            return None, []

        print(f"  Loaded {len(longitudinal_claims)} already released longitudinal claims")
    else:
        existing_claims = []
        if zip_path.exists():
            try:
                confirm = input(f"  Longitudinal split {zip_filename} already exists. Append to it? (y/n): ").lower()
            except EOFError:
                print("  Non-interactive environment, skipping append.")
                confirm = 'n'

            if confirm == 'y':
                print("  Appending to existing split...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    with z.open('claims.json') as f:
                        existing_data = json.load(f)
                        existing_ids = [c['id'] for c in existing_data['claims']]
                        print(f"  Found {len(existing_ids)} existing claims in ZIP")

                        if existing_ids:
                            query = "SELECT * FROM claims WHERE id = ANY($1)"
                            rows = await db._fetch(query, existing_ids)
                            existing_claims = [Claim.model_validate(dict(row)) for row in rows]

                            if len(existing_claims) < len(existing_ids):
                                missing = len(existing_ids) - len(existing_claims)
                                print(f"  Warning: Could not find {missing} claims in database, they will be skipped.")
            else:
                print("  Overwriting existing split...")

        longitudinal_claims = existing_claims.copy()
        existing_ids_set = {c.id for c in existing_claims}

        for (year, quarter), claims in claims_per_quarter.items():
            if not claims:
                continue

            # Filter out claims that are already in the existing split
            new_claims = [c for c in claims if c.id not in existing_ids_set]
            if not new_claims:
                continue

            # Sample from the claims
            # Do not order by priority here because we want to get a representative sample
            sampled = await sample_claims(new_claims, target_count_per_quarter, prioritize=False)
            longitudinal_claims.extend(sampled)

    if not longitudinal_claims:
        print("  No claims for longitudinal split.")
        return None, []

    if existing_claims if resample else False:
        print(
            f"  Total claims for longitudinal split: {len(longitudinal_claims)} ({len(existing_claims)} existing + {len(longitudinal_claims) - len(existing_claims)} new)")
    else:
        if resample:
            print(f"  Sampled {len(longitudinal_claims)} claims for longitudinal split")
        else:
            print(f"  Loaded {len(longitudinal_claims)} claims for longitudinal split")

    # Sort claims by date
    longitudinal_claims.sort(key=lambda c: c.date.isoformat() if c.date else "")

    # Prepare claim data
    prepared_claims, all_media, counts = await prepare_export_data(longitudinal_claims)

    temp_dir = output_dir / "temp_longitudinal"

    meta_data = {
        "start_year": start_year,
        "start_quarter": start_quarter,
        "end_year": end_year,
        "end_quarter": end_quarter,
        **counts
    }

    create_export_zip(zip_path, temp_dir, meta_data, prepared_claims, all_media)

    print(f"  Created: {zip_path}")
    return zip_path, longitudinal_claims


async def rebuild_longitudinal(
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
        output_dir: str | Path = "exports/",
        target_count: int = 100,
) -> Path | None:
    """
    Rebuild the longitudinal split by loading already released claims from the database.

    Args:
        start_year: Start year
        start_quarter: Start quarter (1-4)
        end_year: End year
        end_quarter: End quarter (1-4)
        output_dir: Output directory for zip files
        target_count: Target number of claims per quarter for this split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rebuilding longitudinal split from Q{start_quarter} {start_year} to Q{end_quarter} {end_year}")

    await db.connect_maybe_initialize()

    try:
        # Generate quarters
        quarters = []
        cy, cq = start_year, start_quarter
        while (cy < end_year) or (cy == end_year and cq <= end_quarter):
            quarters.append((cy, cq))
            if cq == 4:
                cy += 1
                cq = 1
            else:
                cq += 1

        claims_per_quarter = {}
        for year, quarter in quarters:
            start_date, end_date = get_quarter_date_range(year, quarter)
            query = """
                    SELECT *
                    FROM claims
                    WHERE (released_quarter = TRUE OR released_longitudinal = TRUE)
                      AND date BETWEEN $1 AND $2
                    ORDER BY date;
                    """
            rows = await db._fetch(query, start_date, end_date)
            quarter_claims = [Claim.model_validate(dict(row)) for row in rows]
            if quarter_claims:
                claims_per_quarter[(year, quarter)] = quarter_claims
                print(f"  Found {len(quarter_claims)} released claims for Q{quarter} {year}")

        if not claims_per_quarter:
            print("  No released claims found in the specified range.")
            return None

        # Reuse export_longitudinal
        # Note: export_longitudinal usually samples from the provided claims.
        # Since we want to rebuild it from *all* released claims of that quarter,
        # we pass them in. export_longitudinal will then sample target_count_per_quarter from them.
        zip_path, longitudinal_claims = await export_longitudinal(
            start_year, start_quarter, end_year, end_quarter,
            claims_per_quarter, output_dir, target_count
        )

        if zip_path and longitudinal_claims:
            # Ensure they are marked as released for longitudinal split
            claim_ids = [c.id for c in longitudinal_claims]
            if claim_ids:
                query = "UPDATE claims SET released_longitudinal = TRUE WHERE id = ANY($1)"
                await db._execute(query, claim_ids)
                print(f"  Marked {len(claim_ids)} claims as released for longitudinal split.")

        return zip_path

    finally:
        await db.close()


async def export_benchmark(
        resample: bool,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
        output_dir: str | Path = "exports/",
        target_count: int = 1000,
) -> None:
    """
    Export benchmark data for a range of quarters.

    Args:
        resample: Whether to resample claims for each quarter or just load the released claims from the DB
        start_year: Start year
        start_quarter: Start quarter (1-4)
        end_year: End year
        end_quarter: End quarter (1-4)
        output_dir: Output directory for zip files
        target_count: Target number of claims per quarter (default 1000)
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting benchmark data from Q{start_quarter} {start_year} to Q{end_quarter} {end_year}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Target claims per quarter: {target_count}")

    await db.connect_maybe_initialize()

    try:
        # Generate quarters
        quarters = []
        cy, cq = start_year, start_quarter
        while (cy < end_year) or (cy == end_year and cq <= end_quarter):
            quarters.append((cy, cq))
            if cq == 4:
                cy += 1
                cq = 1
            else:
                cq += 1

        print(f"\nProcessing {len(quarters)} quarters...")

        exported_files = []
        claims_per_quarter = {}

        if resample:
            # Check for already released claims
            released_claims_count = 0
            for year, quarter in quarters:
                start_date, end_date = get_quarter_date_range(year, quarter)
                query = "SELECT COUNT(*) FROM claims WHERE (released_quarter = TRUE OR released_longitudinal = TRUE) AND date BETWEEN $1 AND $2"
                released_claims_count += await db._fetchval(query, start_date, end_date)

            if released_claims_count > 0:
                print(f"\nWarning: {released_claims_count} claims in the requested quarters have already been released.")
                try:
                    confirm = input("Do you want to clear the 'released' status and proceed? (y/n): ").lower()
                except EOFError:
                    # Handle non-interactive environments by assuming 'no' or 'yes'
                    # For Junie/CI, we might want to default to 'no' to be safe,
                    # or 'yes' if we want to proceed.
                    print("Non-interactive environment detected. Aborting.")
                    return

                if confirm == 'y':
                    for year, quarter in quarters:
                        start_date, end_date = get_quarter_date_range(year, quarter)
                        query = "UPDATE claims SET released_quarter = FALSE, released_longitudinal = FALSE WHERE date BETWEEN $1 AND $2"
                        await db._execute(query, start_date, end_date)
                    print("Released status cleared.")
                else:
                    print("Export aborted.")
                    return

        # Export each quarter
        for year, quarter in quarters:
            zip_path, sampled_claims = await export_quarter(
                year, quarter, output_dir, target_count, resample=resample
            )
            if zip_path:
                exported_files.append(zip_path)
                claims_per_quarter[(year, quarter)] = sampled_claims

                if resample:
                    # Mark claims as released
                    claim_ids = [c.id for c in sampled_claims]
                    if claim_ids:
                        query = "UPDATE claims SET released_quarter = TRUE WHERE id = ANY($1)"
                        await db._execute(query, claim_ids)
                    print(f"  Marked {len(sampled_claims)} claims as released.")

        # Export longitudinal split
        zip_path, longitudinal_claims = await export_longitudinal(
            start_year, start_quarter, end_year, end_quarter,
            claims_per_quarter, output_dir, resample=resample
        )
        if zip_path:
            exported_files.append(zip_path)

            if resample:
                # Mark claims as released for longitudinal split
                claim_ids = [c.id for c in longitudinal_claims]
                if claim_ids:
                    query = "UPDATE claims SET released_longitudinal = TRUE WHERE id = ANY($1)"
                    await db._execute(query, claim_ids)
                    print(f"  Marked {len(claim_ids)} claims as released for longitudinal split.")

        # Print summary
        print("\n" + "=" * 60)
        print("Export complete!")
        print(f"Exported {len(exported_files)} files:")
        for path in exported_files:
            print(f"  - {path.name}")
        print("=" * 60)

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(
        export_benchmark(
            resample=True,
            start_year=2026,
            start_quarter=1,
            end_year=2026,
            end_quarter=1,
            output_dir="exports/",
            target_count=1000,
        )
    )
