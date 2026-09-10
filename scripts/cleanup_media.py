"""
Index the media VeriTaS uses and quarantine the ezMM files that nobody refers to.

The ezMM store accumulates every medium the pipeline ever downloaded, including
those belonging to reviews that were later dismissed, appearances that were
re-scraped, and articles that never made it into a claim. This script finds
those leftovers and gets them out of the way.

It works in two steps:

1. **Index.** Every `<image:42>`-style reference in `claims.data`, in the
   appearances' scraped content, in the articles' pages and in the gold
   evidence's raw source content is collected into the `media` table, together
   with the IDs of the claims, appearances, articles and evidence items it
   occurs in. The scan runs inside PostgreSQL, so no scraped content is ever
   transferred.
2. **Quarantine.** Every file in the store that the index does not mention is
   *moved* into `<ezmm path>/to-delete`, mirroring its path below the store
   root. Nothing is ever deleted, and nothing outside the `image/`, `video/`
   and `audio/` directories is touched, so a cleanup can always be undone by
   moving the quarantined files back.

Reporting only, does not touch a single file:

    python -m scripts.cleanup_media

Actually move the orphaned files:

    python -m scripts.cleanup_media --apply

Useful options: `--ezmm-path` and `--to-delete-dir` to override the locations
from `config.yaml`, `--skip-reindex` to reuse the index of a previous run, and
`--report` to additionally write the statistics as JSON.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from veritas import ezmm_path as configured_ezmm_path
from veritas import logger
from veritas.db import db
from veritas.util.media_store import (
    KINDS,
    execute_plan,
    format_move_result,
    format_plan,
    format_size,
    plan_cleanup,
    resolve_to_delete_dir,
    scan_media_store,
)


async def cleanup_media(
    ezmm_path: Path | str | None = None,
    to_delete_dir: Path | str | None = None,
    kinds: tuple[str, ...] = KINDS,
    apply: bool = False,
    reindex: bool = True,
    report_path: Path | str | None = None,
    force: bool = False,
) -> dict:
    """Rebuilds the media index and quarantines the unreferenced ezMM files.

    Args:
        ezmm_path: Root of the ezMM store. Defaults to `ezmm_path` in `config.yaml`.
        to_delete_dir: Where orphaned files go. Defaults to `<ezmm_path>/to-delete`.
        kinds: The ezMM item kinds to consider.
        apply: Whether to actually move the files. Reports only if False.
        reindex: Whether to rebuild the index before comparing it to the store.
        report_path: Optional path for a JSON copy of the statistics.
        force: Proceed even if the index turns out to be empty.

    Returns:
        The statistics of this run, as they are written to the JSON report.
    """
    root = ezmm_path or configured_ezmm_path
    if not root:
        raise ValueError("No media store given: pass --ezmm-path or set `ezmm_path` "
                         "in config.yaml.")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"The ezMM store '{root}' does not exist.")
    # Fail before touching the database if the quarantine directory is unusable.
    resolved_to_delete = resolve_to_delete_dir(root, to_delete_dir, kinds)

    await db.connect_maybe_initialize()
    try:
        if reindex:
            logger.info("Rebuilding the media index from claims, appearances, "
                        "articles and evidence ...")
            n_used = await db.rebuild_media_index()
            logger.info("Indexed %d media in use.", n_used)
        else:
            logger.info("Reusing the existing media index.")

        index = await db.get_media_index()
        index_stats = await db.get_media_index_stats()
    finally:
        await db.close()

    n_indexed = sum(len(ids) for ids in index.values())
    logger.info("Scanning the media store at '%s' ...", root)
    scan = scan_media_store(root, kinds)
    logger.info("Found %d files (%s).", scan.n_files, format_size(scan.n_bytes))

    plan = plan_cleanup(scan, index, resolved_to_delete, kinds)
    print(format_plan(plan, index_stats))

    statistics = {
        "index": index_stats,
        "plan": plan.as_dict(),
        "applied": False,
    }

    if not apply:
        print("\nNothing was moved. Re-run with --apply to move the unreferenced "
              "files into the quarantine directory.")
    elif not plan.orphaned:
        print("\nNothing to move: every file in the store is referenced.")
    elif n_indexed == 0 and not force:
        # An empty index is indistinguishable from "nothing is in use" and would
        # quarantine the entire store. Far more likely: the wrong database.
        print("\nAborted: the media index is empty, which would move the whole "
              "store. Check the database configuration, or pass --force if the "
              "store really is entirely unused.")
    else:
        print(f"\nMoving {len(plan.orphaned):,} files to '{plan.to_delete_dir}' ...")
        result = execute_plan(plan, on_progress=_log_progress)
        print(format_move_result(result))
        statistics["applied"] = True
        statistics["result"] = result.as_dict()

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(statistics, indent=2, default=str),
                               encoding="utf-8")
        print(f"\nWrote the statistics to '{report_path}'.")

    return statistics


def _log_progress(position: int, total: int) -> None:
    if position % 500 == 0 or position == total:
        logger.info("  Moved %d / %d files.", position, total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Index the media VeriTaS uses and quarantine the unreferenced ezMM files.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually move the unreferenced files. Without it, "
                             "the script only reports what it would move.")
    parser.add_argument("--ezmm-path", default=None,
                        help="Root of the ezMM media store (default: `ezmm_path` "
                             "from config.yaml).")
    parser.add_argument("--to-delete-dir", default=None,
                        help="Where to move unreferenced files "
                             "(default: <ezmm-path>/to-delete).")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS),
                        help="The ezMM item kinds to clean up (default: all).")
    parser.add_argument("--skip-reindex", action="store_true",
                        help="Reuse the existing media index instead of rebuilding it.")
    parser.add_argument("--report", default=None,
                        help="Path of an additional JSON report of the statistics.")
    parser.add_argument("--force", action="store_true",
                        help="Move files even if the media index is empty.")

    args = parser.parse_args()

    asyncio.run(cleanup_media(
        ezmm_path=args.ezmm_path,
        to_delete_dir=args.to_delete_dir,
        kinds=tuple(args.kinds),
        apply=args.apply,
        reindex=not args.skip_reindex,
        report_path=args.report,
        force=args.force,
    ))
