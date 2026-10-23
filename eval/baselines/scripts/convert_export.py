#!/usr/bin/env python3
"""Convert a VeriTaS dataset export into the claims file run_benchmark.py reads.

The export (see its readme.txt) holds claims.json, a list of claims with their aggregated
verdict, and the media in images/ and videos/. run_benchmark.py reads the release format
written by scripts/export/export.py: {"claims": [...]}, each claim with "id", "text", "date",
the ratings ("integrity", "veracity", "context_coverage") and its media, whose "file_path" is
relative to the claims file. So the converted file goes next to the export's claims.json. Each
claim also gets the URLs of its reviews ("review_url" the first, "review_urls" all) and the day
the first of them was published ("review_date"), before which web_search searches.

Scores are kept as exported, not rounded like in export.py: rounding can move a score into
another bin.

Usage:
    python eval/baselines/scripts/convert_export.py dev/q3_baselines/dataset
"""

import argparse
import json
from pathlib import Path


def convert_rating(rating: dict | None) -> dict | None:
    """A rating of the export as the release format has it: its score and justification."""
    if not rating:
        return None
    return {"score": rating["score"], "justification": rating.get("explanation")}


def convert_claim(claim: dict) -> dict:
    """A claim of the export in the release format."""
    verdict = claim.get("verdict") or {}
    media_verdicts = {mv["reference"]: mv for mv in verdict.get("media_verdicts") or []}
    reviews = claim.get("reviews") or []
    media = []
    for medium in claim.get("media") or []:
        medium_verdict = media_verdicts.get(medium["reference"]) or {}
        media.append({
            "type": medium["type"],
            "id": medium["id"],
            "file_path": f"{medium['type']}s/{medium['file_name']}",
            "authenticity": convert_rating(medium_verdict.get("authenticity")),
            "contextualization": convert_rating(medium_verdict.get("contextualization")),
        })
    return {
        "id": claim["claim_id"],
        "text": claim["text"],
        "date": claim["date"],
        "review_url": reviews[0]["url"] if reviews else None,
        "review_urls": [review["url"] for review in reviews],
        # The day the first review was published, before which web_search searches
        "review_date": min((review["published"] for review in reviews if review.get("published")), default=None),
        "media": media,
        "veracity": convert_rating(verdict.get("veracity")),
        "context_coverage": convert_rating(verdict.get("context_coverage")),
        "integrity": convert_rating(verdict.get("integrity")),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert a VeriTaS dataset export for run_benchmark.py")
    parser.add_argument("export_dir", type=Path, help="Directory of the unpacked export (with claims.json)")
    parser.add_argument("--output-name", default="claims_benchmark.json",
                        help="Name of the converted file, written into export_dir (default: claims_benchmark.json)")
    args = parser.parse_args()

    if args.output_name == "claims.json":
        parser.error("claims.json is the export itself; choose another name")
    with open(args.export_dir / "claims.json", encoding="utf-8") as f:
        claims = [convert_claim(claim) for claim in json.load(f)]

    output_path = args.export_dir / args.output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"claims": claims}, f, indent=2, ensure_ascii=False)

    media = [medium for claim in claims for medium in claim["media"]]
    missing = [m["file_path"] for m in media if not (args.export_dir / m["file_path"]).exists()]
    print(f"Wrote {len(claims)} claims to {output_path}")
    print(f"  with integrity score: {sum(c['integrity'] is not None for c in claims)}")
    print(f"  without date: {sum(not c['date'] for c in claims)}, without review date: {sum(not c['review_date'] for c in claims)}")
    print(f"  media: {len(media)} ({sum(m['type'] == 'image' for m in media)} images, "
          f"{sum(m['type'] == 'video' for m in media)} videos), missing files: {len(missing)}")
    for path in missing[:10]:
        print(f"    missing: {path}")


if __name__ == "__main__":
    main()
