"""
Export media files and their metadata from the VeriTaS database.

Exports all media (images and videos) referenced by claims, along with verdict metadata. 
Results can be filtered by:
- Media authenticity score range
- Authenticity property tags 
  - ai-generated
  - manipulated
  - forged
- Claim date range

Output Structure:
    veritas_media_export.zip
    ├── media.json        # Metadata for all exported media
    ├── images/
    │   └── <id>.<ext>
    └── videos/
        └── <id>.<ext>

        media.json format:
        [
            {
                "media_id": 256,
                "type": "image",
                "file_name": "256.jpg",
                "claims": [
                    {
                        "claim_id": 123,
                        "claim_date": "2024-01-15",
                        "appearance_urls": ["https://example.com/post/123"],
                        "authenticity": {
                            "score": -0.67,
                            "explanation": "...",
                            "tags": ["ai-generated"]
                        }
                    }
                ]
            },
            ...
        ]

Example:
python scripts/export_media.py --date-from 2024-01-01 --date-to 2024-12-31 --tags ai-generated --media-type video --output media_exports/veritas_media_export.zip
"""

import argparse
import asyncio
import json
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path

from ezmm import MultimodalSequence, Image, Video

from veritas import logger
from veritas.db import db


def build_export_readme(
    filters: dict,
    media_type: str,
    copy_files: bool,
) -> str:
    """Build README content that documents export options and active filters."""
    def _fmt_filter_value(value: object) -> str:
        if value is None:
            return "none"
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value) if value else "none"
        return str(value)

    lines = [
        "VeriTaS Media Export",
        "",
        f"Generated at (UTC): {datetime.utcnow().replace(microsecond=0).isoformat()}Z",
        "",
        "Output structure:",
        "- media.json: metadata grouped by unique medium",
        "- images/: image files referenced by media.json entries with type='image'",
        "- videos/: video files referenced by media.json entries with type='video'",
        "",
        "media.json schema:",
        "- Top level is a list of unique media items.",
        "- Each media item has: media_id, type, file_name, claims.",
        "- claims is a list of claim-specific entries where this medium appears.",
        "- Each claim entry has: claim_id, claim_date, appearance_urls, authenticity.",
        "",
        "Example media.json item:",
        "{",
        "  \"media_id\": 256,",
        "  \"type\": \"image\",",
        "  \"file_name\": \"256.jpg\",",
        "  \"claims\": [",
        "    {",
        "      \"claim_id\": 123,",
        "      \"claim_date\": \"2024-01-15\",",
        "      \"appearance_urls\": [\"https://example.com/post/123\"],",
        "      \"authenticity\": {",
        "        \"score\": -0.67,",
        "        \"explanation\": \"...\",",
        "        \"tags\": [\"ai-generated\"]",
        "      }",
        "    }",
        "  ]",
        "}",
        "",
        "Export options:",
        f"- media_type: {media_type}",
        f"- copy_files: {copy_files}",
        "",
        "Applied filters:",
        f"- date_from: {_fmt_filter_value(filters.get('date_from'))}",
        f"- date_to: {_fmt_filter_value(filters.get('date_to'))}",
        f"- authenticity_min: {_fmt_filter_value(filters.get('authenticity_min'))}",
        f"- authenticity_max: {_fmt_filter_value(filters.get('authenticity_max'))}",
        f"- property_tags: {_fmt_filter_value(filters.get('property_tags'))}",
    ]
    return "\n".join(lines) + "\n"


def build_media_query(
    filters: dict,
) -> tuple[str, list]:
    """
    Build a SQL query that fetches claims with media and their verdict data,
    applying optional filters for authenticity, property tags, and date.

    Returns:
        Tuple of (query_string, params_list)
    """
    base_select = """
        SELECT
            c.id AS claim_id,
            c.data,
            c.date,
            c.appearance_ids,
            v.media AS media_jsonb
        FROM claims c
        JOIN LATERAL (
            SELECT media FROM verdicts
            WHERE claim_id = c.id AND is_current = TRUE
            LIMIT 1
        ) v ON TRUE
    """

    conditions = [
        # Only claims with media references
        "c.data ~ '<(image|video):[0-9]+>'",
        # Only claims with media verdicts
        "v.media IS NOT NULL",
        "jsonb_array_length(v.media) > 0",
    ]
    params = []
    param_idx = 1

    # Date filters
    if filters.get("date_from"):
        conditions.append(f"c.date >= ${param_idx}")
        params.append(filters["date_from"])
        param_idx += 1
    if filters.get("date_to"):
        conditions.append(f"c.date <= ${param_idx}")
        params.append(filters["date_to"])
        param_idx += 1

    # Authenticity score range (average across individual ratings per media item)
    if filters.get("authenticity_min") is not None or filters.get("authenticity_max") is not None:
        auth_avg_expr = """
            (
                SELECT AVG((rating->>'score')::float)
                FROM jsonb_array_elements(mv->'authenticity'->'individual_ratings') AS rating
            )
        """.strip()

        auth_conditions = []
        if filters.get("authenticity_min") is not None:
            auth_conditions.append(f"{auth_avg_expr} >= ${param_idx}")
            params.append(filters["authenticity_min"])
            param_idx += 1
        if filters.get("authenticity_max") is not None:
            auth_conditions.append(f"{auth_avg_expr} <= ${param_idx}")
            params.append(filters["authenticity_max"])
            param_idx += 1

        auth_condition = " AND ".join(auth_conditions)
        conditions.append(f"""
            EXISTS (
                SELECT 1 FROM jsonb_array_elements(v.media) AS mv
                WHERE mv->'authenticity' IS NOT NULL
                  AND {auth_condition}
            )
        """)

    # Property tags filter (tags present in at least half of individual ratings)
    if filters.get("property_tags"):
        normalized_tags = [tag.lower() for tag in filters["property_tags"] if isinstance(tag, str) and tag.strip()]
        if normalized_tags:
            tags_param = f"${param_idx}"
            params.append(normalized_tags)
            param_idx += 1
            conditions.append(f"""
                EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(v.media) AS media_item
                    WHERE jsonb_array_length(media_item->'authenticity'->'individual_ratings') > 0
                    AND (
                        SELECT COUNT(*)
                        FROM jsonb_array_elements(media_item->'authenticity'->'individual_ratings') AS rating
                        WHERE rating->'tags' IS NOT NULL
                        AND EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements_text(rating->'tags') AS tag
                            WHERE LOWER(tag) = ANY({tags_param}::text[])
                        )
                    ) >= (jsonb_array_length(media_item->'authenticity'->'individual_ratings') / 2.0)
                )
            """)

    where_clause = "WHERE " + " AND ".join(conditions)
    query = f"""
        {base_select}
        {where_clause}
        ORDER BY c.date ASC NULLS LAST, c.id ASC
    """

    return query, params


def extract_media_metadata(
    claim_id: int,
    claim_date: date | datetime | None,
    claim_data: str,
    media_jsonb: list[dict],
    appearance_urls: list[str],
    filters: dict,
    media_type: str = "both",
) -> list[dict]:
    """
    Extract per-medium metadata from a claim's verdict JSONB, applying per-item
    filtering. A claim might have multiple media items, each with its own
    authenticity scores and tags.

    Returns a list of claim-level media metadata dicts that pass the filters.
    """
    # Build a lookup from media reference to verdict JSONB entry
    verdict_by_ref = {}
    for mv in media_jsonb:
        ref = mv.get("reference")
        if ref:
            verdict_by_ref[ref] = mv

    mm_seq = MultimodalSequence(claim_data)
    results = []

    for item in mm_seq.unique_items():
        item_type = "image" if isinstance(item, Image) else "video"
        if media_type != "both" and item_type != media_type:
            continue
        mv = verdict_by_ref.get(item.reference)
        if mv is None:
            continue

        auth_data = mv.get("authenticity")
        # Per-item authenticity score filter
        if auth_data and auth_data.get("individual_ratings"):
            individual = auth_data["individual_ratings"]
            avg_score = sum(r["score"] for r in individual) / len(individual)

            if filters.get("authenticity_min") is not None and avg_score < filters["authenticity_min"]:
                continue
            if filters.get("authenticity_max") is not None and avg_score > filters["authenticity_max"]:
                continue
        elif filters.get("authenticity_min") is not None or filters.get("authenticity_max") is not None:
            # No authenticity data but filter requires it — skip
            continue

        # Per-item property tags filter
        if filters.get("property_tags"):
            normalized_tags = {t.lower() for t in filters["property_tags"]}
            if auth_data and auth_data.get("individual_ratings"):
                individual = auth_data["individual_ratings"]
                n_ratings = len(individual)
                matching = sum(
                    1 for r in individual
                    if r.get("tags") and any(t.lower() in normalized_tags for t in r["tags"])
                )
                if matching < n_ratings / 2:
                    continue
            else:
                continue

        def format_rating(data: dict | None) -> dict | None:
            if not data or not data.get("individual_ratings"):
                return None
            individual = data["individual_ratings"]
            avg_score = sum(r["score"] for r in individual) / len(individual)
            # Collect tags by majority vote
            from collections import Counter
            tag_counts = Counter(t for r in individual for t in (r.get("tags") or []))
            majority_tags = [t for t, c in tag_counts.items() if c >= len(individual) / 2]
            return {
                "score": round(avg_score, 4),
                "explanation": data.get("explanation"),
                "tags": majority_tags,
            }

        results.append({
            "media_id": item.id,
            "type": item_type,
            "file_name": item.file_path.name,
            "claim_id": claim_id,
            "claim_date": claim_date.isoformat() if claim_date else None,
            "appearance_urls": appearance_urls,
            "authenticity": format_rating(auth_data),
        })

    return results


async def export_media(
    output_path: str | Path = "media_exports/veritas_media_export.zip",
    date_from: str | None = None,
    date_to: str | None = None,
    authenticity_min: float | None = None,
    authenticity_max: float | None = None,
    property_tags: list[str] | None = None,
    copy_files: bool = True,
    media_type: str = "both",
) -> None:
    """
    Export media with metadata into a zip file, optionally filtered.

    Args:
        output_path: Path for the output zip file
        date_from: Minimum claim date (YYYY-MM-DD)
        date_to: Maximum claim date (YYYY-MM-DD)
        authenticity_min: Minimum average authenticity score (-1 to 1)
        authenticity_max: Maximum average authenticity score (-1 to 1)
        property_tags: Only include media whose authenticity tags match (majority vote)
        copy_files: Whether to include the actual media files (default True)
        media_type: Export media type ("both", "image", or "video")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filters = {}
    if date_from:
        filters["date_from"] = datetime.strptime(date_from, "%Y-%m-%d").date()
    if date_to:
        filters["date_to"] = datetime.strptime(date_to, "%Y-%m-%d").date()
    if authenticity_min is not None:
        filters["authenticity_min"] = authenticity_min
    if authenticity_max is not None:
        filters["authenticity_max"] = authenticity_max
    if property_tags:
        filters["property_tags"] = property_tags

    logger.info("Filters: %s", {k: str(v) for k, v in filters.items()} if filters else "none")
    logger.info("Media type: %s", media_type)

    # Build and execute query
    query, params = build_media_query(filters)
    logger.info("Connecting to database...")
    await db.connect_maybe_initialize()

    try:
        logger.info("Querying claims with media verdicts...")
        rows = await db._fetch(query, *params)
        logger.info("Found %d claims with media matching filters.", len(rows))

        if not rows:
            logger.info("Nothing to export.")
            return

        # Batch-fetch all appearance URLs
        logger.info("Resolving appearance URLs...")
        all_appearance_ids = set()
        for row in rows:
            if row["appearance_ids"]:
                all_appearance_ids.update(row["appearance_ids"])

        appearance_url_map: dict[int, str] = {}
        if all_appearance_ids:
            app_rows = await db._fetch(
                "SELECT id, url FROM appearances WHERE id = ANY($1)",
                list(all_appearance_ids),
            )
            for app_row in app_rows:
                if app_row["url"]:
                    appearance_url_map[app_row["id"]] = str(app_row["url"])
        logger.info("Resolved %d appearance URLs.", len(appearance_url_map))

        logger.info("Extracting media metadata...")
        all_media_metadata = []
        media_files: dict[str, Path] = {}  # archive path -> source path
        unique_claim_ids = set()

        for row in rows:
            claim_id = row["claim_id"]
            claim_data = row["data"]
            claim_date = row["date"]
            media_jsonb = json.loads(row["media_jsonb"]) if isinstance(row["media_jsonb"], str) else row["media_jsonb"]

            if not media_jsonb:
                continue

            # Resolve appearance URLs for this claim
            appearance_urls = [
                appearance_url_map[aid]
                for aid in (row["appearance_ids"] or [])
                if aid in appearance_url_map
            ]

            mm_seq = MultimodalSequence(claim_data)
            items_by_filename = {item.file_path.name: item for item in mm_seq.unique_items()}

            entries = extract_media_metadata(
                claim_id,
                claim_date,
                claim_data,
                media_jsonb,
                appearance_urls,
                filters,
                media_type,
            )
            for entry in entries:
                all_media_metadata.append(entry)
                unique_claim_ids.add(claim_id)

                # Track the file for inclusion in the zip
                item = items_by_filename.get(entry["file_name"])
                if item:
                    subdir = "images" if isinstance(item, Image) else "videos"
                    media_files[f"{subdir}/{entry['file_name']}"] = item.file_path

        grouped_media: dict[tuple[int, str, str], dict] = {}
        for entry in all_media_metadata:
            media_key = (entry["media_id"], entry["type"], entry["file_name"])
            if media_key not in grouped_media:
                grouped_media[media_key] = {
                    "media_id": entry["media_id"],
                    "type": entry["type"],
                    "file_name": entry["file_name"],
                    "claims": [],
                }

            grouped_media[media_key]["claims"].append({
                "claim_id": entry["claim_id"],
                "claim_date": entry["claim_date"],
                "appearance_urls": entry["appearance_urls"],
                "authenticity": entry["authenticity"],
            })

        media_export_payload = list(grouped_media.values())

        n_images = sum(1 for m in media_export_payload if m["type"] == "image")
        n_videos = sum(1 for m in media_export_payload if m["type"] == "video")
        logger.info(
            "Collected %d media references -> %d unique media (%d images, %d videos) from %d claims.",
            len(all_media_metadata),
            len(media_export_payload),
            n_images,
            n_videos,
            len(unique_claim_ids),
        )

        # Write zip file
        logger.info("Writing zip archive to %s ...", output_path)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write a README with filters/options used for this export.
            zf.writestr("README.txt", build_export_readme(filters, media_type, copy_files))

            # Write metadata JSON
            media_json = json.dumps(media_export_payload, indent=2, ensure_ascii=False)
            zf.writestr("media.json", media_json)

            # Add media files
            if copy_files:
                added = 0
                missing = 0
                total = len(media_files)
                for i, (archive_path, src_path) in enumerate(media_files.items(), 1):
                    if src_path.exists():
                        zf.write(src_path, archive_path)
                        added += 1
                    else:
                        logger.warning("Missing media file: %s", src_path)
                        missing += 1
                    if i % 100 == 0 or i == total:
                        logger.info("  Adding files to archive: %d / %d", i, total)
                logger.info("Added %d media files to archive (%d missing).", added, missing)
            else:
                logger.info("Skipping media files (--no-copy), metadata only.")

        # Final summary
        zip_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("=" * 50)
        logger.info("Export complete.")
        logger.info("  Claims:  %d", len(unique_claim_ids))
        logger.info("  Images:  %d", n_images)
        logger.info("  Videos:  %d", n_videos)
        logger.info("  Total:   %d unique media items", len(media_export_payload))
        logger.info("  Archive: %s (%.1f MB)", output_path, zip_size_mb)
        logger.info("=" * 50)

    finally:
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Export media files and metadata from VeriTaS.")
    parser.add_argument("-o", "--output", default="exports/veritas_media_export.zip",
                        help="Output zip file path (default: exports/veritas_media_export.zip)")
    parser.add_argument("--date-from", help="Minimum claim date (YYYY-MM-DD)")
    parser.add_argument("--date-to", help="Maximum claim date (YYYY-MM-DD)")
    parser.add_argument("--authenticity-min", type=float, help="Minimum avg authenticity score (-1 to 1)")
    parser.add_argument("--authenticity-max", type=float, help="Maximum avg authenticity score (-1 to 1)")
    parser.add_argument("--tags", nargs="+", help="Filter by authenticity property tags (ai-generated, manipulated, forged)")
    parser.add_argument("--media-type", choices=["both", "image", "video"], default="both",
                        help="Export only images, videos, or both (default: both)")
    parser.add_argument("--no-copy", action="store_true", help="Only export metadata, skip copying media files")

    args = parser.parse_args()

    asyncio.run(export_media(
        output_path=args.output,
        date_from=args.date_from,
        date_to=args.date_to,
        authenticity_min=args.authenticity_min,
        authenticity_max=args.authenticity_max,
        property_tags=args.tags,
        copy_files=not args.no_copy,
        media_type=args.media_type,
    ))
