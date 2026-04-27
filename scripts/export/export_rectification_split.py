"""
Export a rectification analysis split from the VeriTaS Database.

Creates a dataset with three groups:
1. All intact claims that are NOT rectified
2. 300 random intact AND rectified claims
3. 300 random compromised AND NOT rectified claims

Each claim includes a `is_rectified` flag in addition to the standard export format.

Output: A zip file containing claims.json, meta.json, and media files in the same
format as other benchmark splits.
"""

import argparse
import asyncio
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from ezmm import MultimodalSequence, Image, Video, Item

from veritas.common import Claim
from veritas.common.annotation.rating import Category3Bin
from veritas.db import db


async def fetch_all_claims() -> list[Claim]:
    """Fetch all non-dismissed claims with verdicts."""
    query = """
        SELECT *
        FROM claims
        WHERE (NOT dismissed OR dismissed_reason = 'In favor of rectified version.')
          AND verdict_ids != '{}'
        ORDER BY RANDOM();
    """
    rows = await db._fetch(query)
    return [Claim.model_validate(dict(row)) for row in rows]


async def categorize_claims(claims: list[Claim]) -> dict[str, list[Claim]]:
    """Categorize claims into the three groups for the rectification split.

    Groups:
        intact_not_rectified: Intact claims that are NOT rectified (all of them)
        intact_rectified: Intact claims that ARE rectified (sample 300)
        compromised_not_rectified: Compromised claims that are NOT rectified (sample 300)
    """
    intact_not_rectified = []
    intact_rectified = []
    compromised_not_rectified = []

    for claim in claims:
        verdict = await claim.current_verdict
        if not verdict:
            continue

        category = verdict.integrity.as_3_bin()

        if category == Category3Bin.POSITIVE and not claim.is_rectified:
            intact_not_rectified.append(claim)
        elif category == Category3Bin.POSITIVE and claim.is_rectified:
            intact_rectified.append(claim)
        elif category == Category3Bin.NEGATIVE and not claim.is_rectified:
            compromised_not_rectified.append(claim)

    return {
        "intact_not_rectified": intact_not_rectified,
        "intact_rectified": intact_rectified,
        "compromised_not_rectified": compromised_not_rectified,
    }


async def prepare_export_data(
        sampled_claims: list[Claim],
) -> tuple[list[dict], dict[str, Item], dict]:
    """
    Prepare claim data for JSON export and collect all referenced media.

    Same format as the standard export, but with an added `is_rectified` flag.

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

        # Build claim dict for JSON (same as standard format + is_rectified flag)
        claim_dict = {
            "id": claim.id,
            "text": " ".join(i for i in claim_mm_seq if isinstance(i, str)).strip(),
            "date": claim.date.isoformat(),
            "language": claim.language,
            "is_rectified": claim.is_rectified,
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
    """Create the export zip file containing metadata, claims, and media."""
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


async def export_rectification_split(
        output_dir: str | Path = "exports/",
        sample_size: int = 300,
) -> Path | None:
    """
    Export the rectification analysis split.

    Groups:
        1. All intact claims that are NOT rectified
        2. `sample_size` random intact AND rectified claims
        3. `sample_size` random compromised AND NOT rectified claims

    Args:
        output_dir: Output directory for zip files
        sample_size: Number of claims to sample for groups 2 and 3 (default 300)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting rectification analysis split...")

    await db.connect_maybe_initialize()

    try:
        # Fetch all claims
        claims = await fetch_all_claims()
        print(f"  Found {len(claims)} total claims")

        # Categorize
        groups = await categorize_claims(claims)

        print(f"  Intact & not rectified: {len(groups['intact_not_rectified'])}")
        print(f"  Intact & rectified:     {len(groups['intact_rectified'])}")
        print(f"  Compromised & not rectified: {len(groups['compromised_not_rectified'])}")

        # Group 1: all intact, not rectified
        group1 = groups["intact_not_rectified"]

        # Group 2: sample 300 intact, rectified
        available_rectified = groups["intact_rectified"]
        if len(available_rectified) < sample_size:
            print(f"  Warning: Only {len(available_rectified)} intact rectified claims available "
                  f"(requested {sample_size}). Using all of them.")
            group2 = available_rectified
        else:
            group2 = random.sample(available_rectified, sample_size)

        # Group 3: sample 300 compromised, not rectified
        available_compromised = groups["compromised_not_rectified"]
        if len(available_compromised) < sample_size:
            print(f"  Warning: Only {len(available_compromised)} compromised non-rectified claims available "
                  f"(requested {sample_size}). Using all of them.")
            group3 = available_compromised
        else:
            group3 = random.sample(available_compromised, sample_size)

        # Combine all groups
        all_sampled = group1 + group2 + group3
        random.shuffle(all_sampled)

        print(f"\n  Final split: {len(group1)} intact non-rectified + "
              f"{len(group2)} intact rectified + {len(group3)} compromised non-rectified "
              f"= {len(all_sampled)} total")

        # Prepare export data
        prepared_claims, all_media, counts = await prepare_export_data(all_sampled)

        # Add group-specific counts to metadata
        meta_data = {
            "split": "rectification_analysis",
            "group_counts": {
                "intact_not_rectified": len(group1),
                "intact_rectified": len(group2),
                "compromised_not_rectified": len(group3),
            },
            **counts,
        }

        print(f"  Integrity distribution: {counts['claim_counts']['intact']} intact, "
              f"{counts['claim_counts']['nei']} NEI, "
              f"{counts['claim_counts']['compromised']} compromised")
        print(f"  Media files: {len(all_media)}")

        # Create zip file
        zip_filename = "veritas_rectification_split.zip"
        zip_path = output_dir / zip_filename
        temp_dir = output_dir / "temp_rectification"

        create_export_zip(zip_path, temp_dir, meta_data, prepared_claims, all_media)

        print(f"\n  Created: {zip_path}")
        return zip_path

    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export rectification analysis split")
    parser.add_argument("--output-dir", type=str, default="exports/",
                        help="Output directory for the zip file (default: exports/)")
    parser.add_argument("--sample-size", type=int, default=300,
                        help="Number of claims to sample for groups 2 and 3 (default: 300)")
    args = parser.parse_args()

    asyncio.run(export_rectification_split(
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    ))
