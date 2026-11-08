import csv
import json
from datetime import datetime

import pytest
from ezmm.common.registry import item_registry

from eval.baselines.scripts import run_benchmark as rb

# The classes as run_benchmark knows them: it imports the package as "baselines", not "eval.baselines"
from baselines.common.reviews import Review, lookup_reviews, signatory_scopes  # noqa: E402
from baselines.common.types import NoAnswerError  # noqa: E402

MediaLimits = rb.MediaLimits


class FakeFactChecker:
    """Stands in for UnifiedFactChecker: fails on the claims in `failing`."""

    failing: set[str] = set()
    no_answer: set[str] = set()
    media_stores: list = []
    blocked: dict = {}

    def __init__(self, **kwargs):
        pass

    def check_claim(self, claim, image_paths=None, video_paths=None, claim_date=None, provider=None, claim_id=None,
                    blocked_urls=None, cutoff_date=None, blocked_scopes=None):
        self.media_stores.append(item_registry.path)
        self.blocked[claim_id] = (blocked_urls, cutoff_date, blocked_scopes)
        if claim in self.failing:
            raise RuntimeError("Error code: 502 - Cannot reach compute node")
        if claim in self.no_answer:
            partial = rb.FactCheckResult(verdict=None, reasoning="", provider=provider,
                                         encountered_urls=["https://example.com/a"],
                                         trace=[{"turn": 0, "type": "stop", "reason": "max_tokens"}])
            raise NoAnswerError("No valid answer in the final response (stop reason: max_tokens)", partial)
        return rb.FactCheckResult(verdict="Intact (certain)", reasoning="VERDICT: Intact (certain)",
                                  justification=f"{claim} is true.", model="fake-model", provider=provider)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "UnifiedFactChecker", FakeFactChecker)
    monkeypatch.setattr(rb, "lookup_reviews", lambda urls: {
        "https://factcheck.example.org/claim-a": Review(datetime(2024, 2, 10, 8), ["factcheck.example.org"])})
    monkeypatch.setattr(rb, "signatory_scopes", lambda: ["signatory.example.org", "other.example.org/fact-checks"])
    claims = [
        {"id": 1, "text": "claim A", "date": "2024-02-05T12:00:00", "integrity": {"score": 1.0},
         "review_url": "https://factcheck.example.org/claim-a"},
        {"id": 2, "text": "claim B", "date": "2024-02-05T12:00:00", "integrity": {"score": 0.9}},
    ]
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"claims": claims}))
    return path


def _rows(run_dir):
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    with open(run_dir / "results_detailed.jsonl", encoding="utf-8") as f:
        jsonl_rows = [json.loads(line) for line in f]
    return ({r["claim_id"]: r["status"] for r in csv_rows},
            {str(r["claim_id"]): r["status"] for r in jsonl_rows})


def test_output_dir_is_used_and_resume_retries_failed_claims(dataset, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(FakeFactChecker, "failing", {"claim B"})
    rb.run_benchmark(dataset_path=str(dataset), output_dir=str(run_dir), providers=["openai"])

    expected = {"1": "success", "2": "error"}
    assert _rows(run_dir) == (expected, expected)
    leaderboard = json.loads((run_dir / "results_for_leaderboard.json").read_text())
    assert leaderboard == [{"claim_id": 1, "verdict": 1, "reasoning": "claim A is true."}]

    # A fresh run must not mix its rows into an existing run
    with pytest.raises(ValueError, match="--resume"):
        rb.run_benchmark(dataset_path=str(dataset), output_dir=str(run_dir), providers=["openai"])

    monkeypatch.setattr(FakeFactChecker, "failing", set())
    rb.run_benchmark(dataset_path=str(dataset), resume_dir=str(run_dir), providers=["openai"])

    expected = {"1": "success", "2": "success"}
    assert _rows(run_dir) == (expected, expected)
    leaderboard = json.loads((run_dir / "results_for_leaderboard.json").read_text())
    assert [e["claim_id"] for e in leaderboard] == [1, 2]
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        assert all(row["linked_urls"] == "[]" for row in csv.DictReader(f))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["providers"]["openai"]["errors"] == 0
    run_config = json.loads((run_dir / "run_config.json").read_text())
    assert run_config["reasoning_effort"] == "medium"
    assert run_config["media_limits"]["total_images"] == MediaLimits().total_images
    # MSE and MAE only for the 7 bins: the model's actual prediction against the ground truth
    assert "mse" in summary["providers"]["openai"]["metrics"]
    assert "mse" not in summary["providers"]["openai"]["metrics_3class"]
    with open(run_dir / "results_detailed.jsonl", encoding="utf-8") as f:
        assert all(json.loads(line)["duration_seconds"] >= 0 for line in f)
    assert json.loads((run_dir / "run_statistics.json").read_text())["openai"]["successful_claims"] == 2
    statistics = summary["providers"]["openai"]["run_statistics"]
    assert statistics["successful_claims"] == 2 and "tokens" in statistics
    assert "by_claim" not in statistics["blocked"] and "web_search_results" in statistics["blocked"]


def test_the_summary_ends_with_every_blocked_url(tmp_path):
    csv_path = tmp_path / "results.csv"
    rb.init_csv(csv_path)
    result = rb.FactCheckResult(verdict="Intact (certain)", reasoning="", provider="openai")
    rb.append_to_csv(csv_path, rb.build_csv_record(1, "claim A", [], {"integrity": 1.0}, result, "openai", "success"))
    scope = "a URL in the fact-check scope fullfact.org"
    preview = "whose title or snippet shows fullfact.org/x, a URL in the fact-check scope fullfact.org"
    by_claim = {
        "1": [{"tool": "web_search", "query": "q", "url": "https://fullfact.org/x", "rule": scope},
              {"tool": "fetch_url", "url": "https://example.org/fact-check", "rule": "a URL with the slug ..."}],
        "2": [{"tool": "web_search", "query": "q", "url": "https://www.instagram.com/p/1/", "rule": preview},
              {"tool": "web_search", "query": "q2", "url": "https://fullfact.org/x", "rule": scope}],
    }
    statistics = {"openai": {"blocked": {"claims": 2, "web_search_results": 3, "fetch_url": 1, "by_claim": by_claim}}}

    rb.generate_summary_from_csv(csv_path, tmp_path, ["openai"], statistics)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert list(summary) == ["providers", "blocked_urls"]
    assert summary["blocked_urls"] == {"openai": {
        "https://example.org/fact-check": "a URL with the slug ...",
        "https://fullfact.org/x": scope,
        "https://www.instagram.com/p/1/": preview,
    }}
    assert list(summary["blocked_urls"]["openai"]) == sorted(summary["blocked_urls"]["openai"])


def test_a_failed_claim_keeps_its_trace_and_urls_and_no_answer_is_counted_apart(dataset, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(FakeFactChecker, "no_answer", {"claim B"})
    rb.run_benchmark(dataset_path=str(dataset), output_dir=str(run_dir), providers=["openai"])

    expected = {"1": "success", "2": "no_answer"}
    assert _rows(run_dir) == (expected, expected)
    with open(run_dir / "results_detailed.jsonl", encoding="utf-8") as f:
        record = [json.loads(line) for line in f][1]
    assert record["error_message"] == "No valid answer in the final response (stop reason: max_tokens)"
    assert record["result"]["trace"] == [{"turn": 0, "type": "stop", "reason": "max_tokens"}]
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[1]
    assert (row["verdict"], row["encountered_urls"]) == ("", '["https://example.com/a"]')
    summary = json.loads((run_dir / "summary.json").read_text())["providers"]["openai"]
    assert (summary["successful_runs"], summary["no_answer"], summary["errors"]) == (1, 1, 0)

    monkeypatch.setattr(FakeFactChecker, "no_answer", set())
    rb.run_benchmark(dataset_path=str(dataset), resume_dir=str(run_dir), providers=["openai"])
    assert _rows(run_dir) == ({"1": "success", "2": "success"},) * 2


def test_the_fetched_pages_media_go_to_a_temporary_store(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeFactChecker, "media_stores", [])
    before = item_registry.path
    rb.run_benchmark(dataset_path=str(dataset), output_dir=str(tmp_path / "run"), providers=["openai"])

    store, = set(FakeFactChecker.media_stores)
    assert store != before and store.name.startswith("ezmm_") and not store.exists()
    assert item_registry.path == before


def test_the_claims_own_fact_check_is_blocked(dataset, tmp_path, monkeypatch):
    """With its publisher's fact-checks and those of all IFCN signatories and EFCSN members;
    web_search searches before the review's publishing date."""
    monkeypatch.setattr(FakeFactChecker, "blocked", {})
    rb.run_benchmark(dataset_path=str(dataset), output_dir=str(tmp_path / "run"), providers=["openai"])
    signatories = ["signatory.example.org", "other.example.org/fact-checks"]
    assert FakeFactChecker.blocked == {
        1: (["https://factcheck.example.org/claim-a"], "2024-02-10T08:00:00", [*signatories, "factcheck.example.org"]),
        2: ([], None, signatories),
    }
    records = [json.loads(line) for line in (tmp_path / "run" / "results_detailed.jsonl").read_text().splitlines()]
    assert {(r["claim_id"], r["review_date"]) for r in records} == {(1, "2024-02-10T08:00:00"), (2, None)}
    run_config = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert (run_config["signatory_scopes"], run_config["max_search_pages"]) == (signatories, rb.MAX_SEARCH_PAGES)


def test_media_limits_are_command_line_options(monkeypatch, dataset):
    runs = []
    monkeypatch.setattr(rb, "run_benchmark", lambda **kwargs: runs.append(kwargs))
    monkeypatch.setattr("sys.argv", ["run_benchmark.py", "--dataset", str(dataset), "--total-images", "12",
                                     "--native-video-seconds", "30.5"])
    rb.main()
    assert runs[0]["media_limits"] == MediaLimits(total_images=12, native_video_seconds=30.5)


def test_prepare_resume_keeps_successes_and_other_providers(tmp_path):
    csv_path, jsonl_path = tmp_path / "results.csv", tmp_path / "results_detailed.jsonl"
    rb.init_csv(csv_path)
    records = [
        {"claim_id": 1, "provider": "openai", "status": "success"},
        {"claim_id": 2, "provider": "openai", "status": "error"},
        {"claim_id": 2, "provider": "gemini", "status": "error"},
    ]
    for record in records:
        rb.append_to_csv(csv_path, {column: record.get(column) for column in rb.CSV_COLUMNS})
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records) + '{"claim_id": 3, "provi')

    assert rb.prepare_resume(csv_path, jsonl_path, ["openai"]) == 1

    with open(csv_path, newline="", encoding="utf-8") as f:
        kept = [(r["claim_id"], r["provider"]) for r in csv.DictReader(f)]
    assert kept == [("1", "openai"), ("2", "gemini")]
    # The line a crash cut short is gone too
    kept_jsonl = [(r["claim_id"], r["provider"]) for r in map(json.loads, jsonl_path.read_text().splitlines())]
    assert kept_jsonl == [(1, "openai"), (2, "gemini")]


def test_prepare_resume_updates_an_old_header(tmp_path):
    """Rows appended after resuming have the current columns, so the header must have them too."""
    csv_path = tmp_path / "results.csv"
    old_columns = [c for c in rb.CSV_COLUMNS if c != "linked_urls"] + ["retired_column"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({"claim_id": 1, "provider": "openai", "status": "success", "retired_column": "kept"})

    rb.prepare_resume(csv_path, tmp_path / "results_detailed.jsonl", ["openai"])
    rb.append_to_csv(csv_path, {"claim_id": 2, "provider": "openai", "status": "success", "linked_urls": "[]"})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == rb.CSV_COLUMNS + ["retired_column"]
    assert [(r["claim_id"], r["linked_urls"], r["retired_column"]) for r in rows] == [
        ("1", "", "kept"), ("2", "[]", None)]


def test_write_leaderboard_results(tmp_path):
    def record(claim_id, provider, verdict=None, justification="", status="success"):
        result = {"verdict": verdict, "justification": justification, "reasoning": f"response {claim_id}"}
        return {"claim_id": claim_id, "provider": provider, "status": status,
                "result": result if status == "success" else None}

    records = [
        record(3, "openai", "Compromised (rather uncertain)", "Doubtful, see [X](https://x.org)."),
        record(1, "openai", "Intact (rather certain)", "Likely true."),
        record(2, "openai", "Unknown"),  # No valid JSON answer: no justification
        record(4, "openai", status="error"),
        record(1, "gemini", "Compromised (certain)", "Not true."),  # Too short: the final response instead
        record(2, "gemini", "Intact (certain)", "x" * 501),
        # The leaderboard counts as JavaScript does: the emoji counts twice
        record(3, "gemini", "Intact (certain)", "x" * 498 + "🔻y"),  # 500 characters, 501 for JavaScript
        record(4, "gemini", "Intact (certain)", "x" * 496 + "🔻" + "y" * 10),  # Not cut in half
        record(5, "gemini", "Intact (certain)", "x" * 497 + "🔻y"),  # 500 for JavaScript
    ]
    jsonl_path = tmp_path / "results_detailed.jsonl"
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    rb.write_leaderboard_results(jsonl_path, tmp_path, ["openai", "gemini"])

    assert json.loads((tmp_path / "results_for_leaderboard_openai.json").read_text()) == [
        {"claim_id": 1, "verdict": 0.67, "reasoning": "Likely true."},
        {"claim_id": 2, "verdict": 0, "reasoning": "response 2"},
        {"claim_id": 3, "verdict": -0.33, "reasoning": "Doubtful, see [X](https://x.org)."},
    ]
    assert json.loads((tmp_path / "results_for_leaderboard_gemini.json").read_text()) == [
        {"claim_id": 1, "verdict": -1, "reasoning": "response 1"},
        {"claim_id": 2, "verdict": 1, "reasoning": "x" * 497 + "..."},
        {"claim_id": 3, "verdict": 1, "reasoning": "x" * 497 + "..."},
        {"claim_id": 4, "verdict": 1, "reasoning": "x" * 496 + "..."},
        {"claim_id": 5, "verdict": 1, "reasoning": "x" * 497 + "🔻y"},
    ]


def test_write_run_statistics(tmp_path):
    def record(claim_id, tools, usage, status="success", duration=None, blocked=None, provider="openai", cost=None):
        trace = [{"type": "prompt", "text": "..."}]
        for turn, names in enumerate(tools):
            trace += [{"turn": turn, "type": "tool_call", "id": f"{turn}-{i}", "name": name, "args": {}}
                      for i, name in enumerate(names)]
        trace.append({"turn": len(tools), "type": "text", "text": "answer"})
        result = {"verdict": "Intact (certain)", "trace": trace,
                  "usage": dict(zip(("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"), usage))}
        if cost is not None:
            result["usage"]["cost"] = cost
        if blocked is not None:
            result["blocked"] = blocked
        line = {"claim_id": claim_id, "provider": provider, "status": status, "result": result}
        return line | ({"duration_seconds": duration} if duration is not None else {})

    dropped = {"tool": "web_search", "query": "q", "url": "https://fullfact.org/x",
               "rule": "a URL in the fact-check scope fullfact.org"}
    refused = {"tool": "fetch_url", "url": "https://fullfact.org/y", "rule": "a URL in the fact-check scope fullfact.org"}
    records = [
        record(1, [["web_search", "web_search"], ["fetch_url"]], (100, 10, 60, 0), duration=30.0, blocked=[dropped, refused]),
        record(2, [["web_search"]], (50, 5, 20, 0), duration=10.0, blocked=[]),
        record(3, [], (30, 3, 0, 0), duration=20.0, blocked=[]),
        record(4, [["web_search"]] * 5, (200, 20, 100, 10), status="no_answer", duration=99.0, blocked=[dropped]),
        # From before durations, blocks and cache tokens were recorded
        record(5, [["web_search", "web_search", "web_search"]], (40, 4)),
        record(6, [["web_search"]], (80, 8, 40, 0), duration=15.0, blocked=[], provider="openrouter", cost=0.0012),
        record(7, [], (20, 2, 0, 0), status="no_answer", duration=5.0, blocked=[], provider="openrouter", cost=0.0003),
    ]
    jsonl_path = tmp_path / "results_detailed.jsonl"
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    rb.write_run_statistics(jsonl_path, tmp_path)

    all_stats = json.loads((tmp_path / "run_statistics.json").read_text())
    stats = all_stats["openai"]
    assert (stats["claims"], stats["successful_claims"]) == (5, 4)
    # Of the successful fact-checks: turns are model calls, the last one answering
    assert stats["turns"] == {"mean": 2.0, "median": 2.0}
    assert stats["tool_calls"] == {"web_search": {"mean": 1.5, "median": 1.5}, "fetch_url": {"mean": 0.25, "median": 0.0},
                                   "all": {"mean": 1.75, "median": 2.0}}
    assert stats["duration_seconds"] == {"mean": 20.0, "median": 20.0, "claims": 3}
    # Per successful fact-check, and in all, the failed ones included
    assert stats["tokens"] == {
        "input_tokens": {"mean": 55.0, "median": 45.0, "total": 420, "claims": 5},
        "output_tokens": {"mean": 5.5, "median": 4.5, "total": 42, "claims": 5},
        "cache_read_tokens": {"mean": 26.67, "median": 20.0, "total": 180, "claims": 4},
        "cache_write_tokens": {"mean": 0.0, "median": 0.0, "total": 10, "claims": 4},
    }
    assert stats["cost_usd"] == {"mean": None, "median": None, "total": 0, "claims": 0}
    assert all_stats["openrouter"]["cost_usd"] == {"mean": 0.0012, "median": 0.0012, "total": 0.0015, "claims": 2}
    # Of all fact-checks that recorded them
    assert stats["blocked"] == {"claims": 4, "web_search_results": 2, "fetch_url": 1,
                                "by_claim": {"1": [dropped, refused], "4": [dropped]}}


def test_write_fact_check_exposure(tmp_path):
    fact_check = "https://leadstories.com/hoax-alert/2026/07/fact-check-the-claim-is-false.html"

    def record(claim_id, trace=(), **urls):
        return {"claim_id": claim_id, "provider": "gemini", "status": "success", "claim_date": "2026-07-10",
                "result": {"trace": list(trace), **urls}}

    def search(query, text):
        return [{"turn": 0, "type": "tool_call", "id": "s", "name": "web_search", "args": {"query": query}},
                {"turn": 0, "type": "tool_result", "id": "s", "name": "web_search", "text": text}]

    def fetch(url, text):
        return [{"turn": 1, "type": "tool_call", "id": "f", "name": "fetch_url", "args": {"url": url}},
                {"turn": 1, "type": "tool_result", "id": "f", "name": "fetch_url", "text": text}]

    records = [
        record(1, search("the claim", "**Search results for: the claim**\n\n[1] Post\nURL: https://x.org/1\n"
                                      "Snippet: Fact Check: The claim ... fact-check-the-claim-is ...")),
        record(2, fetch("https://x.org/2", f"**Content of https://x.org/2:**\n\nSee [it](/{fact_check.split('/', 3)[3]})."),
               linked_urls=[fact_check]),
        # The model alone: it guesses the URL, which the reply only repeats, and cites it
        record(3, [{"turn": 0, "type": "reasoning", "text": "The slug is likely fact-check-the-claim-is-false.html"},
                   *fetch(fact_check, f"Error: Cannot fetch {fact_check}. fetch_url only fetches ...")],
               quoted_urls=[fact_check + "?utm_source=x"]),
        record(4, search("the claim", "[1] Other\nURL: https://x.org/4\nSnippet: Nothing to see")),
        record(5),
        record(6, encountered_urls=[f"https://web.archive.org/web/2026/{fact_check}"]),
    ]
    jsonl_path = tmp_path / "results_detailed.jsonl"
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    claims = [{"id": i, "review_url": fact_check} for i in (1, 2, 3, 4, 6)] + [{"id": 5}]

    rb.write_fact_check_exposure(jsonl_path, claims, tmp_path)

    stats = json.loads((tmp_path / "fact_check_exposure.json").read_text())["gemini"]
    assert stats["claims_by_id"] == {
        "1": {"search_preview": "Snippet: Fact Check: The claim ... fact-check-the-claim-is ..."},
        "2": {"linked": fact_check, "fetched_page": f"**Content of https://x.org/2:** See [it](/{fact_check.split('/', 3)[3]})."},
        "3": {"cited": fact_check + "?utm_source=x", "fetch_attempt": fact_check,
              "mentioned": "The slug is likely fact-check-the-claim-is-false.html"},
        "6": {"search_result": f"https://web.archive.org/web/2026/{fact_check}"},
    }
    assert (stats["claims"], stats["claims_with_review_url"], stats["shown_by_tools"], stats["only_by_model"]) == (6, 5, 3, 1)
    assert stats["ways"]["linked"] == 1 and stats["ways"]["retrieved"] == 0


def test_add_review_info():
    """The tools block the claims' reviews and their publishers' fact-checks; web_search searches
    before the first review."""
    claims = [
        {"id": 1, "review_url": "https://a.org/1", "date": "2026-07-01"},
        {"id": 2, "review_urls": ["https://b.org/2", "https://c.org/3"], "review_date": "2026-07-05T00:00:00"},
        {"id": 3, "review_url": "https://unknown.org/4"},
        {"id": 4},
    ]
    looked_up = []

    def lookup(urls):
        looked_up.extend(urls)
        return {"https://a.org/1": Review(datetime(2026, 7, 3, 9), ["a.org/fact-check"]),
                "https://b.org/2": Review(datetime(2026, 7, 9), ["b.org"]),
                "https://c.org/3": Review(datetime(2026, 7, 7), ["c.org/x", "b.org"])}

    enriched = rb.add_review_info(claims, lookup=lookup)

    assert looked_up == ["https://a.org/1", "https://b.org/2", "https://c.org/3", "https://unknown.org/4"]
    assert [(c["review_urls"], c["review_date"], c["fact_check_scopes"]) for c in enriched] == [
        (["https://a.org/1"], "2026-07-03T09:00:00", ["a.org/fact-check"]),
        # The dataset's review date comes first
        (["https://b.org/2", "https://c.org/3"], "2026-07-05T00:00:00", ["b.org", "c.org/x"]),
        (["https://unknown.org/4"], None, []),
        ([], None, []),
    ]
    assert enriched[0]["date"] == "2026-07-01"


def test_the_database_queries_fail_with_a_pointer_to_the_config(tmp_path):
    with pytest.raises(RuntimeError, match="database section"):
        lookup_reviews(["https://a.org/1"], config_path=tmp_path / "missing.yaml")
    with pytest.raises(RuntimeError, match="database section"):
        signatory_scopes(config_path=tmp_path / "missing.yaml")
