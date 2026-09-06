"""Tests for the ezMM media store scan, the cleanup plan and its execution.

The one property that matters most is checked from several angles: the cleanup
never removes anything, and never touches a file it cannot attribute to an
unreferenced medium.
"""

from pathlib import Path

import pytest

from veritas.util.media_store import (
    KINDS,
    TO_DELETE_DIR_NAME,
    execute_plan,
    format_plan,
    format_size,
    free_destination,
    parse_media_file_name,
    plan_cleanup,
    resolve_to_delete_dir,
    scan_media_store,
)


def make_store(root: Path, files: dict[str, bytes | str]) -> Path:
    """Creates a media store: `files` maps a path below `root` to its content."""
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    return root


@pytest.fixture
def store(tmp_path) -> Path:
    """A store with three images, two videos, one audio file and some noise."""
    return make_store(tmp_path / "ezmm", {
        "image/1.jpg": "a" * 100,
        "image/2.png": "b" * 200,
        "image/3.jpg": "c" * 300,
        "video/10.mp4": "d" * 1000,
        "video/11.mp4": "e" * 2000,
        "audio/20.mp3": "f" * 50,
        "item_registry.db": "not a medium",
    })


# --- Scanning ---------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("1.jpg", 1),
    ("42.mp4", 42),
    ("42", 42),
    ("42.thumb.jpg", 42),  # A derivative shares the fate of its item.
    ("0.jpg", 0),
    ("item_registry.db", None),
    ("x42.jpg", None),
    ("42x.jpg", None),
    (".hidden", None),
    ("", None),
])
def test_parse_media_file_name(name, expected):
    assert parse_media_file_name(name) == expected


def test_scan_attributes_every_file_to_its_item(store):
    scan = scan_media_store(store)

    assert scan.n_files == 6
    assert scan.n_bytes == 100 + 200 + 300 + 1000 + 2000 + 50
    assert {(file.kind, file.media_id) for file in scan.files} == {
        ("image", 1), ("image", 2), ("image", 3),
        ("video", 10), ("video", 11), ("audio", 20),
    }


def test_scan_ignores_everything_outside_the_kind_directories(store):
    """The registry, the quarantine directory and any stray file below the root
    must be out of the cleanup's reach by construction."""
    (store / TO_DELETE_DIR_NAME / "image").mkdir(parents=True)
    (store / TO_DELETE_DIR_NAME / "image" / "99.jpg").write_bytes(b"quarantined")
    (store / "backup.tar").write_bytes(b"important")

    scanned = {file.path for file in scan_media_store(store).files}

    assert store / "item_registry.db" not in scanned
    assert store / "backup.tar" not in scanned
    assert store / TO_DELETE_DIR_NAME / "image" / "99.jpg" not in scanned


def test_scan_reports_unattributable_files_instead_of_claiming_them(store):
    (store / "image" / "notes.txt").write_bytes(b"who put this here")

    scan = scan_media_store(store)

    assert scan.unrecognized == (store / "image" / "notes.txt",)
    assert all(file.path.name != "notes.txt" for file in scan.files)


def test_scan_reports_missing_kind_directories(tmp_path):
    root = make_store(tmp_path / "ezmm", {"image/1.jpg": "a"})

    scan = scan_media_store(root)

    assert scan.missing_kinds == ("video", "audio")
    assert scan.n_files == 1


def test_scan_of_an_empty_store_is_empty(tmp_path):
    root = tmp_path / "ezmm"
    root.mkdir()

    scan = scan_media_store(root)

    assert scan.files == ()
    assert scan.n_bytes == 0


def test_scan_descends_into_subdirectories(tmp_path):
    root = make_store(tmp_path / "ezmm", {"image/nested/7.jpg": "x" * 10})

    scan = scan_media_store(root)

    assert [(file.kind, file.media_id) for file in scan.files] == [("image", 7)]


# --- Planning ---------------------------------------------------------------


def test_plan_keeps_referenced_and_orphans_the_rest(store):
    index = {"image": {1, 3}, "video": {10}}

    plan = plan_cleanup(scan_media_store(store), index)

    assert {(f.kind, f.media_id) for f in plan.keep} == {("image", 1), ("image", 3), ("video", 10)}
    assert {(f.kind, f.media_id) for f in plan.orphaned} == {("image", 2), ("video", 11), ("audio", 20)}
    assert plan.keep_bytes == 100 + 300 + 1000
    assert plan.orphaned_bytes == 200 + 2000 + 50


def test_plan_keeps_everything_when_all_media_are_referenced(store):
    index = {"image": {1, 2, 3}, "video": {10, 11}, "audio": {20}}

    plan = plan_cleanup(scan_media_store(store), index)

    assert plan.orphaned == ()
    assert plan.orphaned_bytes == 0
    assert len(plan.keep) == 6


def test_plan_does_not_confuse_ids_across_kinds(store):
    """`<image:10>` must not keep `video/10.mp4` alive."""
    plan = plan_cleanup(scan_media_store(store), {"image": {10}})

    assert {(f.kind, f.media_id) for f in plan.orphaned} >= {("video", 10)}
    assert plan.keep == ()


def test_plan_counts_indexed_media_without_a_file(store):
    """Broken references are worth reporting, but must not create work."""
    plan = plan_cleanup(scan_media_store(store), {"image": {1, 404, 405}})

    assert plan.per_kind["image"].n_indexed == 3
    assert plan.per_kind["image"].n_missing_files == 2
    assert plan.n_missing_files == 2


def test_plan_reports_referenced_kinds_it_does_not_scan(store):
    plan = plan_cleanup(scan_media_store(store, ("image",)), {"pdf": {1}}, kinds=("image",))

    assert plan.unknown_kinds == ("pdf",)


def test_plan_totals_add_up(store):
    plan = plan_cleanup(scan_media_store(store), {"image": {1}})

    assert plan.n_files == len(plan.keep) + len(plan.orphaned)
    assert plan.n_bytes == plan.keep_bytes + plan.orphaned_bytes
    assert plan.n_files == sum(stats.n_files for stats in plan.per_kind.values())
    assert plan.n_bytes == sum(stats.n_bytes for stats in plan.per_kind.values())


def test_plan_restricted_to_one_kind_leaves_the_others_alone(store):
    plan = plan_cleanup(scan_media_store(store, ("video",)), {}, kinds=("video",))

    assert {file.kind for file in plan.orphaned} == {"video"}


# --- The quarantine directory ----------------------------------------------


def test_quarantine_defaults_to_the_store_root(tmp_path):
    assert resolve_to_delete_dir(tmp_path) == tmp_path / TO_DELETE_DIR_NAME


def test_quarantine_can_live_on_another_volume(tmp_path):
    elsewhere = tmp_path / "elsewhere" / "trash"

    assert resolve_to_delete_dir(tmp_path / "ezmm", elsewhere) == elsewhere


@pytest.mark.parametrize("relative", ["image", "image/trash", "video/a/b"])
def test_quarantine_inside_a_kind_directory_is_rejected(tmp_path, relative):
    """It would be re-scanned on the next run and moved onto itself."""
    with pytest.raises(ValueError):
        resolve_to_delete_dir(tmp_path, tmp_path / relative)


def test_quarantine_is_checked_against_every_kind_not_just_the_scanned_ones(tmp_path):
    """A run restricted to images must still not quarantine inside `video/`:
    the next, unrestricted run would pick those files up again."""
    with pytest.raises(ValueError):
        resolve_to_delete_dir(tmp_path, tmp_path / "video" / "trash", kinds=("image",))


# --- Execution --------------------------------------------------------------


def test_execute_moves_orphans_and_leaves_the_rest_in_place(store):
    plan = plan_cleanup(scan_media_store(store), {"image": {1, 3}, "video": {10}})

    result = execute_plan(plan)

    assert len(result.moved) == 3
    assert result.moved_bytes == 200 + 2000 + 50
    assert not result.failed
    # Referenced media stayed.
    assert (store / "image" / "1.jpg").is_file()
    assert (store / "image" / "3.jpg").is_file()
    assert (store / "video" / "10.mp4").is_file()
    # Orphans left their original location...
    assert not (store / "image" / "2.png").exists()
    # ... and are in the quarantine directory, under the same relative path.
    assert (store / TO_DELETE_DIR_NAME / "image" / "2.png").is_file()
    assert (store / TO_DELETE_DIR_NAME / "video" / "11.mp4").is_file()
    assert (store / TO_DELETE_DIR_NAME / "audio" / "20.mp3").is_file()


def test_execute_never_loses_a_byte(store):
    plan = plan_cleanup(scan_media_store(store), {"image": {1}})
    before = {path.name: path.read_bytes()
              for path in store.rglob("*") if path.is_file()}

    execute_plan(plan)

    after = {path.name: path.read_bytes()
             for path in store.rglob("*") if path.is_file()}
    assert after == before


def test_execute_does_not_touch_unattributable_files(store):
    stray = store / "image" / "notes.txt"
    stray.write_bytes(b"who put this here")
    plan = plan_cleanup(scan_media_store(store), {})

    execute_plan(plan)

    assert stray.is_file()
    assert (store / "item_registry.db").is_file()


def test_execute_never_overwrites_an_earlier_quarantine(store):
    """A file re-downloaded under the same ID must not destroy the older copy."""
    quarantined = store / TO_DELETE_DIR_NAME / "image" / "2.png"
    quarantined.parent.mkdir(parents=True)
    quarantined.write_bytes(b"the older copy")
    plan = plan_cleanup(scan_media_store(store), {})

    execute_plan(plan)

    assert quarantined.read_bytes() == b"the older copy"
    assert (store / TO_DELETE_DIR_NAME / "image" / "2.1.png").read_bytes() == b"b" * 200


def test_execute_reports_failures_instead_of_aborting(store, monkeypatch):
    plan = plan_cleanup(scan_media_store(store), {})
    real_move = __import__("shutil").move

    def fail_on_the_first_video(source, destination):
        if source.endswith("10.mp4"):
            raise PermissionError("file is in use")
        return real_move(source, destination)

    monkeypatch.setattr("veritas.util.media_store.shutil.move", fail_on_the_first_video)
    result = execute_plan(plan)

    assert len(result.failed) == 1
    assert "in use" in result.failed[0][1]
    assert len(result.moved) == 5
    assert (store / "video" / "10.mp4").is_file()


def test_execute_reports_progress(store):
    plan = plan_cleanup(scan_media_store(store), {})
    seen = []

    execute_plan(plan, on_progress=lambda position, total: seen.append((position, total)))

    assert seen == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_a_second_run_finds_nothing_left_to_do(store):
    execute_plan(plan_cleanup(scan_media_store(store), {"image": {1}}))

    plan = plan_cleanup(scan_media_store(store), {"image": {1}})

    assert plan.orphaned == ()
    assert len(plan.keep) == 1


def test_moves_are_reversible(store):
    plan = plan_cleanup(scan_media_store(store), {})
    result = execute_plan(plan)

    for source, destination in result.moved:
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(source)

    assert scan_media_store(store).n_files == 6


def test_free_destination_picks_an_unused_name(tmp_path):
    taken = tmp_path / "2.png"
    taken.write_bytes(b"x")

    assert free_destination(tmp_path / "3.png") == tmp_path / "3.png"
    assert free_destination(taken) == tmp_path / "2.1.png"


# --- Reporting --------------------------------------------------------------


@pytest.mark.parametrize("n_bytes,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (1024, "1.00 KiB"),
    (1536, "1.50 KiB"),
    (1024 ** 2, "1.00 MiB"),
    (3 * 1024 ** 3, "3.00 GiB"),
    (2 * 1024 ** 4, "2.00 TiB"),
])
def test_format_size(n_bytes, expected):
    assert format_size(n_bytes) == expected


def test_format_plan_states_the_numbers_that_matter(store):
    plan = plan_cleanup(scan_media_store(store), {"image": {1, 3}, "video": {10}})

    report = format_plan(plan)

    assert str(store) in report
    assert TO_DELETE_DIR_NAME in report
    assert format_size(plan.orphaned_bytes) in report
    assert "3 of 6" in report  # unreferenced of total
    for kind in KINDS:
        assert kind in report


def test_plan_serializes_to_json_friendly_statistics(store):
    plan = plan_cleanup(scan_media_store(store), {"image": {1, 404}})

    statistics = plan.as_dict()

    assert statistics["n_keep"] == 1
    assert statistics["n_orphaned"] == 5
    assert statistics["orphaned_bytes"] == plan.orphaned_bytes
    assert statistics["per_kind"]["image"]["n_missing_files"] == 1
    assert isinstance(statistics["root"], str)
