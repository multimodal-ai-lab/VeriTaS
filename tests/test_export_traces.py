import json

from eval.baselines.scripts.export_traces import CLAIM_FIELDS, README, export_runs, export_traces


def _record(claim_id, provider="openrouter", verdict="Intact (certain)", status="success"):
    return {"claim_id": claim_id, "claim_text": f"claim {claim_id}", "claim_text_clean": "", "claim_date": "2026-07-26",
            "review_date": "2026-07-27", "fact_check_scopes": ["example.org/fact-check"], "num_images": 0,
            "num_videos": 0, "ground_truth": {"integrity": 1.0}, "provider": provider, "status": status,
            "error_message": None, "duration_seconds": 1.0,
            "result": {"verdict": verdict, "reasoning": "answer", "citations": ["https://a.org"], "model": "m",
                       "provider": provider, "usage": {}, "justification": "j", "encountered_urls": ["https://a.org"],
                       "retrieved_urls": [], "linked_urls": [], "quoted_urls": [], "blocked": [],
                       "trace": [{"type": "prompt", "text": "..."}]}}


def test_one_file_per_claim_with_the_claim_and_the_result_but_its_citations(tmp_path):
    jsonl_path = tmp_path / "results_detailed.jsonl"
    failed = {**_record(2, status="error"), "result": None}
    lines = [_record(1, verdict=None, status="no_answer"), failed, _record(1)]  # Claim 1 retried
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"claim_id": 3, "provi')

    assert export_traces(jsonl_path, tmp_path / "traces") == 2

    exported = json.loads((tmp_path / "traces" / "1.json").read_text())
    assert list(exported) == [*CLAIM_FIELDS, "result"]
    assert "ground_truth" not in exported
    assert exported["result"]["verdict"] == "Intact (certain)"  # The last record of the claim
    assert list(exported["result"]) == ["verdict", "model", "justification", "encountered_urls", "retrieved_urls",
                                        "linked_urls", "quoted_urls", "blocked", "trace"]
    assert exported["result"]["trace"] == [{"type": "prompt", "text": "..."}]
    assert json.loads((tmp_path / "traces" / "2.json").read_text())["result"] is None


def test_each_run_gets_a_folder_and_the_readme_is_written_once(tmp_path):
    for run in ("claude-fable-5-1", "mimo-v2.6-pro"):
        (tmp_path / run / "results").mkdir(parents=True)
        (tmp_path / run / "results" / "results_detailed.jsonl").write_text(json.dumps(_record(1)) + "\n")
    jsonl_paths = [tmp_path / run / "results" / "results_detailed.jsonl" for run in ("claude-fable-5-1", "mimo-v2.6-pro")]

    assert export_runs(jsonl_paths, tmp_path / "traces") == {"claude-fable-5-1": 1, "mimo-v2.6-pro": 1}

    assert sorted(p.relative_to(tmp_path / "traces").as_posix() for p in (tmp_path / "traces").rglob("*")) == [
        "claude-fable-5-1", "claude-fable-5-1/1.json", "mimo-v2.6-pro", "mimo-v2.6-pro/1.json", "readme.txt"]
    assert (tmp_path / "traces" / "readme.txt").read_text() == README


def test_several_providers_get_one_file_each(tmp_path):
    jsonl_path = tmp_path / "results_detailed.jsonl"
    jsonl_path.write_text("".join(json.dumps(_record(1, provider)) + "\n" for provider in ("openai", "gemini")))

    export_traces(jsonl_path, tmp_path / "traces")

    assert sorted(p.name for p in (tmp_path / "traces").iterdir()) == ["1_gemini.json", "1_openai.json"]


def test_a_run_that_did_not_record_what_was_blocked_gets_blocked_null(tmp_path):
    jsonl_path = tmp_path / "results_detailed.jsonl"
    record = _record(1)
    del record["result"]["blocked"]
    jsonl_path.write_text(json.dumps(record) + "\n")

    export_traces(jsonl_path, tmp_path / "traces")

    result = json.loads((tmp_path / "traces" / "1.json").read_text())["result"]
    assert result["blocked"] is None
    assert list(result)[-2:] == ["blocked", "trace"]
