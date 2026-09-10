"""The ezMM media store on disk: enumerating it and separating the files that
VeriTaS still refers to from the ones it does not.

ezMM lays its store out as `<root>/<kind>/<id>.<ext>` (see `Item._default_file_path`),
next to the registry `item_registry.db` and the staging directory `items/`. That
layout is all this module needs, so it stays free of database, ezMM and network
imports - the part that decides which files get moved is therefore fully
unit-testable.

Only the kind directories are ever walked. Everything else below the root is out
of reach by construction, including the staging directory, whose size is reported
but whose contents are never touched.

Nothing here ever deletes: orphaned files are *moved* into a quarantine
directory (`to-delete` by default), which keeps every cleanup reversible with a
plain `mv` and lets a human decide when the space is actually released.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: The item kinds ezMM knows. Mirrors `ezmm.common.items.KINDS`.
KINDS = ("image", "video", "audio")

#: Where orphaned media are moved to, relative to the store root.
TO_DELETE_DIR_NAME = "to-delete"

#: ezMM stages a download here under a timestamp name until an ID is assigned,
#: and `Item.relocate` *copies* it into `<kind>/` rather than moving it, so a
#: duplicate of every downloaded medium is left behind. Reported, never touched:
#: a file here may still be in flight, and an item that was never relocated has
#: its registered path pointing right at it.
STAGING_DIR_NAME = "items"

#: A media file is named after the ID of the item it belongs to. Derivatives
#: such as `42.thumb.jpg` are attributed to item 42 as well, so that they share
#: its fate instead of being left behind.
_MEDIA_FILE_REGEX = re.compile(r"^(\d+)(?:\..*)?$")


@dataclass(frozen=True)
class MediaFile:
    """One file in the store, attributed to the ezMM item it belongs to."""

    kind: str
    media_id: int
    path: Path
    size: int

    @property
    def reference(self) -> str:
        return f"<{self.kind}:{self.media_id}>"


@dataclass(frozen=True)
class DirectoryUsage:
    """How much space a directory occupies. Purely informational."""

    path: Path
    exists: bool = False
    n_files: int = 0
    n_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
        }


@dataclass(frozen=True)
class StoreScan:
    """Everything found below the store root."""

    root: Path
    files: tuple[MediaFile, ...] = ()
    #: Files inside a kind directory whose name does not identify an item. They
    #: are never touched - an unexplained file is not evidence of an orphan.
    unrecognized: tuple[Path, ...] = ()
    #: Kind directories that do not exist (yet).
    missing_kinds: tuple[str, ...] = ()
    #: Size of ezMM's staging directory. Reported so that its growth is visible,
    #: but out of the cleanup's reach.
    staging: DirectoryUsage | None = None

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_bytes(self) -> int:
        return sum(file.size for file in self.files)


@dataclass(frozen=True)
class KindStats:
    """Per-kind summary of a cleanup plan."""

    kind: str
    n_files: int = 0
    n_bytes: int = 0
    n_keep: int = 0
    keep_bytes: int = 0
    n_orphaned: int = 0
    orphaned_bytes: int = 0
    #: Media of this kind that VeriTaS refers to.
    n_indexed: int = 0
    #: Indexed media that have no file in the store - broken references.
    n_missing_files: int = 0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
            "n_keep": self.n_keep,
            "keep_bytes": self.keep_bytes,
            "n_orphaned": self.n_orphaned,
            "orphaned_bytes": self.orphaned_bytes,
            "n_indexed": self.n_indexed,
            "n_missing_files": self.n_missing_files,
        }


@dataclass(frozen=True)
class CleanupPlan:
    """Which files to keep, which to move, and what that adds up to."""

    root: Path
    to_delete_dir: Path
    keep: tuple[MediaFile, ...] = ()
    orphaned: tuple[MediaFile, ...] = ()
    unrecognized: tuple[Path, ...] = ()
    per_kind: dict[str, KindStats] = field(default_factory=dict)
    #: Referenced media of a kind the store has no directory for.
    unknown_kinds: tuple[str, ...] = ()
    #: Size of ezMM's staging directory, carried over from the scan.
    staging: DirectoryUsage | None = None

    @property
    def n_files(self) -> int:
        return len(self.keep) + len(self.orphaned)

    @property
    def n_bytes(self) -> int:
        return self.keep_bytes + self.orphaned_bytes

    @property
    def keep_bytes(self) -> int:
        return sum(file.size for file in self.keep)

    @property
    def orphaned_bytes(self) -> int:
        """The disk space the cleanup is expected to free."""
        return sum(file.size for file in self.orphaned)

    @property
    def n_indexed(self) -> int:
        return sum(stats.n_indexed for stats in self.per_kind.values())

    @property
    def n_missing_files(self) -> int:
        return sum(stats.n_missing_files for stats in self.per_kind.values())

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "to_delete_dir": str(self.to_delete_dir),
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
            "n_keep": len(self.keep),
            "keep_bytes": self.keep_bytes,
            "n_orphaned": len(self.orphaned),
            "orphaned_bytes": self.orphaned_bytes,
            "n_unrecognized": len(self.unrecognized),
            "n_indexed": self.n_indexed,
            "n_missing_files": self.n_missing_files,
            "unknown_kinds": list(self.unknown_kinds),
            "per_kind": {kind: stats.as_dict() for kind, stats in self.per_kind.items()},
            "staging": self.staging.as_dict() if self.staging else None,
        }


@dataclass(frozen=True)
class MoveResult:
    """Outcome of executing a plan."""

    moved: tuple[tuple[Path, Path], ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()
    moved_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "n_moved": len(self.moved),
            "moved_bytes": self.moved_bytes,
            "n_failed": len(self.failed),
            "failures": [{"path": str(path), "error": error} for path, error in self.failed],
        }


# -- Scanning ----------------------------------------------------------------


def parse_media_file_name(name: str) -> int | None:
    """The ezMM item ID a file name belongs to, or None if it identifies none."""
    match = _MEDIA_FILE_REGEX.match(name)
    return int(match.group(1)) if match else None


def measure_directory(path: Path | str) -> DirectoryUsage:
    """Counts the files below a directory and adds up their sizes."""
    path = Path(path)
    if not path.is_dir():
        return DirectoryUsage(path=path, exists=False)
    n_files = 0
    n_bytes = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            n_files += 1
            n_bytes += entry.stat().st_size
    return DirectoryUsage(path=path, exists=True, n_files=n_files, n_bytes=n_bytes)


def scan_media_store(root: Path | str, kinds: tuple[str, ...] = KINDS) -> StoreScan:
    """Walks `<root>/<kind>` for every kind and attributes each file to an item.

    Only the kind directories are visited, so anything else below the root -
    `item_registry.db`, the quarantine directory, stray backups - is out of
    reach of the cleanup by construction.
    """
    root = Path(root)
    files: list[MediaFile] = []
    unrecognized: list[Path] = []
    missing_kinds: list[str] = []

    for kind in kinds:
        directory = root / kind
        if not directory.is_dir():
            missing_kinds.append(kind)
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            media_id = parse_media_file_name(path.name)
            if media_id is None:
                unrecognized.append(path)
                continue
            files.append(MediaFile(kind=kind, media_id=media_id, path=path,
                                   size=path.stat().st_size))

    return StoreScan(root=root, files=tuple(files), unrecognized=tuple(unrecognized),
                     missing_kinds=tuple(missing_kinds),
                     staging=measure_directory(root / STAGING_DIR_NAME))


# -- Planning ----------------------------------------------------------------


def resolve_to_delete_dir(root: Path | str, to_delete_dir: Path | str | None = None,
                          kinds: tuple[str, ...] = KINDS) -> Path:
    """The quarantine directory, defaulting to `<root>/to-delete`.

    Raises if it would sit inside a kind directory: the next run would scan it
    and, finding the same orphans again, move them onto themselves. Every kind is
    checked, not just the ones this run happens to scan.
    """
    root = Path(root)
    resolved = Path(to_delete_dir) if to_delete_dir else root / TO_DELETE_DIR_NAME
    # The staging directory is off-limits as well: ezMM writes in-flight downloads
    # there, and quarantining into it would put orphans back into the store.
    for name in sorted(set(kinds) | set(KINDS) | {STAGING_DIR_NAME}):
        directory = root / name
        if resolved == directory or directory in resolved.parents:
            raise ValueError(
                f"The quarantine directory must not lie inside the media store's "
                f"'{name}' directory, but '{resolved}' does."
            )
    return resolved


def plan_cleanup(scan: StoreScan, index: dict[str, set[int]],
                 to_delete_dir: Path | str | None = None,
                 kinds: tuple[str, ...] = KINDS) -> CleanupPlan:
    """Splits the scanned files into the ones VeriTaS refers to and the orphans.

    `index` maps a kind to the IDs of the media that occur in VeriTaS, as
    returned by `VeritasDB.get_media_index`.
    """
    resolved_to_delete = resolve_to_delete_dir(scan.root, to_delete_dir, kinds)

    keep: list[MediaFile] = []
    orphaned: list[MediaFile] = []
    for file in scan.files:
        if file.media_id in index.get(file.kind, set()):
            keep.append(file)
        else:
            orphaned.append(file)

    per_kind: dict[str, KindStats] = {}
    for kind in kinds:
        kind_keep = [file for file in keep if file.kind == kind]
        kind_orphaned = [file for file in orphaned if file.kind == kind]
        indexed = index.get(kind, set())
        present = {file.media_id for file in scan.files if file.kind == kind}
        per_kind[kind] = KindStats(
            kind=kind,
            n_files=len(kind_keep) + len(kind_orphaned),
            n_bytes=sum(file.size for file in kind_keep) + sum(f.size for f in kind_orphaned),
            n_keep=len(kind_keep),
            keep_bytes=sum(file.size for file in kind_keep),
            n_orphaned=len(kind_orphaned),
            orphaned_bytes=sum(file.size for file in kind_orphaned),
            n_indexed=len(indexed),
            n_missing_files=len(indexed - present),
        )

    return CleanupPlan(
        root=scan.root,
        to_delete_dir=resolved_to_delete,
        keep=tuple(keep),
        orphaned=tuple(orphaned),
        unrecognized=scan.unrecognized,
        per_kind=per_kind,
        unknown_kinds=tuple(sorted(set(index) - set(kinds))),
        staging=scan.staging,
    )


def destination_for(file: MediaFile, plan: CleanupPlan) -> Path:
    """Where a file goes inside the quarantine directory. The path below the
    store root is preserved, so the move can be undone by moving it back."""
    try:
        relative = file.path.relative_to(plan.root)
    except ValueError:  # A file outside the root: keep at least its kind and name.
        relative = Path(file.kind) / file.path.name
    return plan.to_delete_dir / relative


def free_destination(destination: Path) -> Path:
    """`destination`, or the first unused `<stem>.<n><suffix>` beside it.

    An earlier run may already have quarantined a file of that name. Overwriting
    it would destroy data, which is the one thing this module must not do.
    """
    if not destination.exists():
        return destination
    for counter in range(1, 10_000):
        candidate = destination.with_name(f"{destination.stem}.{counter}{destination.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free name beside '{destination}'.")


# -- Execution ---------------------------------------------------------------


def execute_plan(plan: CleanupPlan, on_progress=None) -> MoveResult:
    """Moves every orphaned file into the quarantine directory.

    Failures are collected instead of raised, so one unreadable file cannot
    abort a cleanup that still has thousands of files to go.
    """
    moved: list[tuple[Path, Path]] = []
    failed: list[tuple[Path, str]] = []
    moved_bytes = 0

    for position, file in enumerate(plan.orphaned, start=1):
        destination = destination_for(file, plan)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = free_destination(destination)
            # `shutil.move` falls back to copy+remove when the quarantine
            # directory lives on another volume.
            shutil.move(str(file.path), str(destination))
        except OSError as error:
            failed.append((file.path, str(error)))
        else:
            moved.append((file.path, destination))
            moved_bytes += file.size
        if on_progress is not None:
            on_progress(position, len(plan.orphaned))

    return MoveResult(moved=tuple(moved), failed=tuple(failed), moved_bytes=moved_bytes)


# -- Reporting ---------------------------------------------------------------


def format_size(n_bytes: int) -> str:
    """A human-readable, binary-prefixed file size."""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def format_plan(plan: CleanupPlan, index_stats: dict | None = None) -> str:
    """The cleanup plan as a report for the console or a log file."""
    lines = [
        "=" * 72,
        "VeriTaS media cleanup",
        "=" * 72,
        f"Media store:  {plan.root}",
        f"To delete:    {plan.to_delete_dir}",
        "",
        "Media index (media occurring in a claim, appearance, article or evidence item)",
    ]

    if index_stats:
        sources = index_stats.get("sources", {})
        lines += [
            f"  Indexed media:        {index_stats.get('n_used', 0):>10,}"
            f"  (of {index_stats.get('n_media', 0):,} rows in `media`)",
            f"  With embedding:       {index_stats.get('n_with_embedding', 0):>10,}",
            f"  Referring claims:     {sources.get('n_claims', 0):>10,}",
            f"  Referring appearances:{sources.get('n_appearances', 0):>10,}",
            f"  Referring articles:   {sources.get('n_articles', 0):>10,}",
            f"  Referring evidence:   {sources.get('n_evidence', 0):>10,}",
        ]

    lines += [
        "",
        f"{'Kind':<8}{'Files':>10}{'Size':>14}{'Keep':>10}{'Kept size':>14}"
        f"{'Move':>10}{'Freed size':>14}",
        "-" * 80,
    ]
    for kind, stats in plan.per_kind.items():
        lines.append(
            f"{kind:<8}{stats.n_files:>10,}{format_size(stats.n_bytes):>14}"
            f"{stats.n_keep:>10,}{format_size(stats.keep_bytes):>14}"
            f"{stats.n_orphaned:>10,}{format_size(stats.orphaned_bytes):>14}"
        )
    lines.append("-" * 80)
    lines.append(
        f"{'total':<8}{plan.n_files:>10,}{format_size(plan.n_bytes):>14}"
        f"{len(plan.keep):>10,}{format_size(plan.keep_bytes):>14}"
        f"{len(plan.orphaned):>10,}{format_size(plan.orphaned_bytes):>14}"
    )

    share = len(plan.orphaned) / plan.n_files * 100 if plan.n_files else 0.0
    lines += [
        "",
        f"Unreferenced files:  {len(plan.orphaned):,} of {plan.n_files:,} ({share:.1f} %)",
        f"Space to be freed:   {format_size(plan.orphaned_bytes)}",
    ]

    if plan.n_missing_files:
        lines.append(f"Broken references:   {plan.n_missing_files:,} indexed media "
                     f"have no file in the store")
    if plan.unrecognized:
        lines.append(f"Left untouched:      {len(plan.unrecognized):,} files whose name "
                     f"does not identify an item")
    if plan.unknown_kinds:
        lines.append(f"Unhandled kinds:     {', '.join(plan.unknown_kinds)} "
                     f"(referenced, but not scanned)")

    if plan.staging and plan.staging.n_files:
        lines += [
            "",
            f"Out of scope: ezMM's staging directory '{plan.staging.path}' holds "
            f"{plan.staging.n_files:,} file{'' if plan.staging.n_files == 1 else 's'} "
            f"({format_size(plan.staging.n_bytes)}).",
            "  ezMM copies a download from there into the kind directory instead of "
            "moving it, so",
            "  these are largely duplicates of media that already exist under "
            "image/, video/ or audio/.",
            "  They are left untouched: a file there may still be in flight, and an "
            "item that was",
            "  never relocated has its registered path pointing at it.",
        ]

    return "\n".join(lines)


def format_move_result(result: MoveResult) -> str:
    """The outcome of an executed plan, as a report."""
    lines = [
        "",
        f"Moved:  {len(result.moved):,} files ({format_size(result.moved_bytes)})",
    ]
    if result.failed:
        lines.append(f"Failed: {len(result.failed):,} files")
        for path, error in result.failed[:20]:
            lines.append(f"  {path}: {error}")
        if len(result.failed) > 20:
            lines.append(f"  ... and {len(result.failed) - 20:,} more")
    return "\n".join(lines)
