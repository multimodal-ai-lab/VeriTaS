"""Tests for the tool-calling loop every provider runs (BaseFactChecker.check_claim)."""

import io
from pathlib import Path

import anthropic
import ezmm
import httpx
import openai
import pytest
from baseline_fakes import ScriptedFactChecker, ezmm_registry, fake_search  # noqa: F401  # ezmm_registry is a fixture
from PIL import Image as PILImage
from google.genai import errors as genai_errors

from eval.baselines.common.media import Image, MediaLimits
from eval.baselines.providers import base
from eval.baselines.common.types import MAX_RETRIES, FactCheckError, NoAnswerError
from eval.baselines.providers.base import MAX_TURNS, Reasoning, Step, Text, ToolCall

FINAL = Step([Text('```json\n{"integrity": "intact", "certainty": "rather certain", "justification": "Confirmed."}\n```')],
             input_tokens=5, output_tokens=3)


def test_tool_results_go_back_to_the_model_and_usage_adds_up(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    checker = ScriptedFactChecker([
        Step([ToolCall("1", "web_search", {"query": "eiffel"})], input_tokens=10, output_tokens=2),
        FINAL,
    ])

    result = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05")

    assert checker.allow_tools == [True, True]
    tool_results = checker.conversation[2]
    assert tool_results[0][0] == "web_search"
    assert "https://example.com/a" in tool_results[0][1]
    assert result.verdict == "Intact (rather certain)"
    assert result.justification == "Confirmed."
    assert result.usage == {"input_tokens": 15, "output_tokens": 5, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert (result.model, result.provider) == ("scripted-model", "scripted")


def test_trace_has_everything_in_order(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    checker = ScriptedFactChecker([
        Step([Reasoning("I should search."), Text("Searching."), ToolCall("1", "web_search", {"query": "eiffel"})],
             stop_reason="tool_use"),
        Step([Reasoning("The result confirms it."), *FINAL.output], stop_reason="end_turn"),
    ])

    trace = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05").trace

    assert [(entry.get("turn"), entry["type"]) for entry in trace] == [
        (None, "prompt"), (0, "reasoning"), (0, "text"), (0, "tool_call"), (0, "stop"), (0, "tool_result"),
        (1, "reasoning"), (1, "text"), (1, "stop"),
    ]
    assert "The Eiffel Tower is in Paris" in trace[0]["text"]
    assert trace[1]["text"] == "I should search."
    assert trace[3] == {"turn": 0, "type": "tool_call", "id": "1", "name": "web_search", "args": {"query": "eiffel"}}
    assert trace[4] == {"turn": 0, "type": "stop", "reason": "tool_use"}
    assert (trace[5]["id"], trace[5]["name"]) == ("1", "web_search")
    assert "https://example.com/a" in trace[5]["text"]
    assert '"justification": "Confirmed."' in trace[7]["text"]


def _status_error(sdk: str, status: int) -> Exception:
    """The error each vendor's SDK raises for an HTTP status."""
    if sdk == "gemini":
        error_class = genai_errors.ServerError if status >= 500 else genai_errors.ClientError
        return error_class(status, {"error": {"code": status, "message": "Error", "status": "ERROR"}})
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.test"))
    module = {"openai": openai, "anthropic": anthropic}[sdk]
    return module.APIStatusError(f"Error code: {status}", response=response, body=None)


def _flaky_checker(error: Exception, failures: int = 1) -> tuple[ScriptedFactChecker, list]:
    """A checker whose first `failures` model calls raise `error`. Returns it and its calls."""
    checker = ScriptedFactChecker([FINAL])
    calls = []
    scripted_step = checker._step

    def flaky_step(conversation, allow_tools):
        calls.append(allow_tools)
        if len(calls) <= failures:
            raise error
        return scripted_step(conversation, allow_tools)

    checker._step = flaky_step
    return checker, calls


@pytest.mark.parametrize("sdk, status", [
    ("openai", 429), ("anthropic", 429), ("gemini", 429),
    ("anthropic", 529),  # Overloaded
    ("openai", 502), ("anthropic", 502), ("gemini", 504), ("openai", 500), ("gemini", 503),
])
def test_rate_limits_and_transient_server_errors_are_retried(monkeypatch, capsys, sdk, status):
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    checker, calls = _flaky_checker(_status_error(sdk, status))

    result = checker.check_claim("claim", claim_date="2024-02-05", claim_id=7)

    assert result.verdict == "Intact (rather certain)"
    assert len(calls) == 2
    assert "    [7] Transient error (" in capsys.readouterr().out


@pytest.mark.parametrize("error", [
    _status_error("openai", 400), _status_error("anthropic", 401), _status_error("gemini", 404),
    # A status code in the message alone doesn't count
    RuntimeError("Error code: 400 - prompt is too long: 215003 tokens > 200000 maximum"),
])
def test_other_errors_are_not_retried(monkeypatch, error):
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    checker, calls = _flaky_checker(error)

    with pytest.raises(FactCheckError) as failure:
        checker.check_claim("claim", claim_date="2024-02-05")
    assert type(failure.value.__cause__) is type(error)
    assert len(calls) == 1


def test_retries_give_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    checker, calls = _flaky_checker(_status_error("anthropic", 529), failures=MAX_RETRIES)

    with pytest.raises(FactCheckError) as failure:
        checker.check_claim("claim", claim_date="2024-02-05")
    assert isinstance(failure.value.__cause__, anthropic.APIStatusError)
    assert len(calls) == MAX_RETRIES


def test_last_turn_disables_tools_and_tells_the_model():
    checker = ScriptedFactChecker([Step([ToolCall("1", "unknown_tool", {})])])

    with pytest.raises(NoAnswerError) as no_answer:
        checker.check_claim("claim", claim_date="2024-02-05")
    assert checker.allow_tools == [True] * (MAX_TURNS - 1) + [False]
    # The message comes right before the last model call, after the last tool results (the
    # scripted model calls a tool even then, whose result follows)
    assert checker.conversation[-3:-1] == [base.LAST_TURN_MESSAGE, checker.steps[0]]
    assert isinstance(checker.conversation[-4], list)  # The tool results before it
    assert checker.conversation.count(base.LAST_TURN_MESSAGE) == 1
    trace = no_answer.value.result.trace
    assert {"turn": MAX_TURNS - 1, "type": "message", "text": base.LAST_TURN_MESSAGE} in trace


def test_a_model_done_before_the_last_turn_gets_no_message():
    checker = ScriptedFactChecker([FINAL])
    checker.check_claim("claim", claim_date="2024-02-05")
    assert base.LAST_TURN_MESSAGE not in checker.conversation


@pytest.mark.parametrize("final", [
    Step([Reasoning("Let me weigh the evidence")], stop_reason="max_tokens"),  # Cut off while thinking
    Step([], stop_reason="SAFETY"),  # Blocked
    Step([Text("Let me search for more.")], stop_reason="end_turn"),  # Ended without calling a tool
    Step([Text('{"integrity": "intact", "justification": "No certainty."}')], stop_reason="end_turn"),
])
def test_a_final_response_without_a_valid_answer_is_no_answer(monkeypatch, final):
    """Not a verdict of "Unknown": the claim is retried on --resume."""
    fake_search(monkeypatch, ["https://example.com/a"])
    checker = ScriptedFactChecker([Step([ToolCall("1", "web_search", {"query": "q"})]), final])

    with pytest.raises(NoAnswerError) as no_answer:
        checker.check_claim("claim", claim_date="2024-02-05")

    assert no_answer.value.status == "no_answer"
    assert f"(stop reason: {final.stop_reason})" in str(no_answer.value)
    # Everything up to it is kept
    result = no_answer.value.result
    assert result.verdict is None and result.encountered_urls == ["https://example.com/a"]
    assert result.trace[-1] == {"turn": 1, "type": "stop", "reason": final.stop_reason}


def test_unknown_is_a_valid_answer():
    checker = ScriptedFactChecker([Step([Text('{"integrity": "unknown", "certainty": null, "justification": "?"}')])])
    assert checker.check_claim("claim", claim_date="2024-02-05").verdict == "Unknown"


def test_a_failed_fact_check_keeps_its_trace_urls_and_usage(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    checker = ScriptedFactChecker([
        Step([Reasoning("I should search."), ToolCall("1", "web_search", {"query": "q"})], input_tokens=10,
             output_tokens=2),
        FINAL,
    ])
    scripted_step = checker._step

    def step_then_fail(conversation, allow_tools):
        if len(checker.allow_tools) == 1:
            raise ValueError("The API is down")
        return scripted_step(conversation, allow_tools)

    checker._step = step_then_fail

    with pytest.raises(FactCheckError) as failure:
        checker.check_claim("claim", claim_date="2024-02-05")

    assert failure.value.status == "error"
    assert str(failure.value) == "ValueError: The API is down"
    result = failure.value.result
    assert result.verdict is None
    assert result.encountered_urls == ["https://example.com/a"]
    assert result.usage == {"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert [entry["type"] for entry in result.trace] == ["prompt", "reasoning", "tool_call", "stop", "tool_result"]


def test_prompt_has_the_claim_day():
    checker = ScriptedFactChecker([FINAL])
    checker.check_claim("claim", claim_date="2024-02-05T12:22:02")

    assert '"claim"\nPublished on: February 05, 2024' in checker.conversation[0][0]


def _png(path, color="red", size=(40, 20)) -> str:
    PILImage.new("RGB", size, color).save(path, "PNG")
    return str(path)


def test_the_claim_media_follow_its_text_frames_without_native_video(monkeypatch, tmp_path, ezmm_registry):
    image_path = _png(tmp_path / "photo.png")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"mp4")
    frame = Image("image/jpeg", b"frame")
    monkeypatch.setattr(base, "extract_video_frames", lambda path, n, max_side: [frame])

    checker = ScriptedFactChecker([FINAL])
    checker.check_claim("claim", image_paths=[image_path], video_paths=[str(video_path)], claim_date="2024-02-05")
    before, image, between, frames, after = checker.conversation[0]
    assert before.endswith('# The Claim\n"claim <image:1>')
    # The claim's image as it is
    assert image == Image("image/png", Path(image_path).read_bytes())
    assert (between, frames) == ("<video:1> (1 frames of the video)", frame)
    assert after.startswith('"\nPublished on: February 05, 2024')

    native = ScriptedFactChecker([FINAL], native_video=True)
    native.check_claim("claim", video_paths=[str(video_path)], claim_date="2024-02-05")
    before, video, after = native.conversation[0]
    assert before.endswith('"claim <video:1>') and isinstance(video, ezmm.Video)


def test_the_tools_know_the_images_the_claim_takes_in_every_request(monkeypatch, tmp_path, ezmm_registry):
    """They count toward the cap on images per request: its images, and its videos' frames
    for models without native video."""
    image_path = _png(tmp_path / "photo.png")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"mp4")
    monkeypatch.setattr(base, "extract_video_frames", lambda path, n, max_side: [Image("image/jpeg", b"frame")] * n)
    sessions = []
    monkeypatch.setattr(base.ToolSession, "close", lambda session: sessions.append(session))

    for native_video in (False, True):
        checker = ScriptedFactChecker([FINAL], native_video=native_video, media_limits=MediaLimits(video_frames=3))
        checker.check_claim("claim", image_paths=[image_path], video_paths=[str(video_path)], claim_date="2024-02-05")
    assert [session.claim_images for session in sessions] == [4, 1]


def test_claim_images_are_scaled_down_only_when_too_large(tmp_path, ezmm_registry):
    small = ezmm.Image(file_path=_png(tmp_path / "small.png", size=(2000, 1000)))
    large = ezmm.Image(file_path=_png(tmp_path / "large.png", color="blue", size=(3000, 1500)))

    _, small_part, _, large_part = ScriptedFactChecker([FINAL])._parts([small, large], claim=True)

    assert small_part == Image("image/png", Path(small.file_path).read_bytes())
    assert large_part.mime_type == "image/jpeg"
    assert PILImage.open(io.BytesIO(large_part.data)).size == (2000, 1000)


def test_long_page_videos_reach_native_video_models_as_frames(monkeypatch, tmp_path, ezmm_registry):
    """Native video costs tokens by the second, every model call. The claim's videos stay native."""
    monkeypatch.setattr(base, "extract_video_frames", lambda path, n, max_side: [Image("image/jpeg", b"f")] * n)
    monkeypatch.setattr(ezmm.Video, "duration", property(lambda video: 121.0))
    (tmp_path / "v.mp4").write_bytes(b"mp4")
    video = ezmm.Video(file_path=str(tmp_path / "v.mp4"))
    checker = ScriptedFactChecker([FINAL], native_video=True, media_limits=MediaLimits(video_frames=3))

    assert checker._parts([video]) == [f"{video.reference} (3 frames of the video)", *[Image("image/jpeg", b"f")] * 3]
    assert checker._parts([video], claim=True) == [video.reference, video]
    monkeypatch.setattr(ezmm.Video, "duration", property(lambda video: 120.0))
    assert checker._parts([video]) == [video.reference, video]
    # A video whose length can't be read, as ezMM can't open it
    monkeypatch.setattr(ezmm.Video, "duration", property(_unopenable))
    monkeypatch.setattr(base, "extract_video_frames", lambda path, n, max_side: [])
    assert checker._parts([video]) == [f"{video.reference} (0 frames of the video)"]


def _unopenable(video):
    raise OSError(f"Failed to open video: {video.file_path}")


def test_sequences_become_text_with_each_medium_after_its_reference(monkeypatch, tmp_path, ezmm_registry):
    frames = [Image("image/jpeg", b"f1"), Image("image/jpeg", b"f2")]
    monkeypatch.setattr(base, "extract_video_frames", lambda path, n, max_side: frames)
    image = ezmm.Image(file_path=_png(tmp_path / "big.png", size=(3000, 1500)))
    (tmp_path / "v.mp4").write_bytes(b"mp4")
    video = ezmm.Video(file_path=str(tmp_path / "v.mp4"))
    sequence = ["Intro", image, "middle", video, "end"]

    parts = ScriptedFactChecker([FINAL], media_limits=MediaLimits(page_image_side=1000))._parts(sequence)
    assert [p if isinstance(p, str) else type(p).__name__ for p in parts] == [
        "Intro <image:1>", "Image", "middle <video:1> (2 frames of the video)", "Image", "Image", "end"]
    # Scaled down to max_side, as JPEG
    assert parts[1].mime_type == "image/jpeg"
    assert PILImage.open(io.BytesIO(parts[1].data)).size == (1000, 500)

    monkeypatch.setattr(ezmm.Video, "duration", property(lambda video: 10.0))
    native = ScriptedFactChecker([FINAL], native_video=True)._parts(sequence)
    assert native[3] is video and native[2] == "middle <video:1>"


def test_the_conversation_and_tool_session_are_ended_even_if_the_claim_fails(monkeypatch):
    checker = ScriptedFactChecker([FINAL])
    ended, closed = [], []
    monkeypatch.setattr(checker, "_end_conversation", ended.append)
    monkeypatch.setattr(base.ToolSession, "close", lambda session: closed.append(session))

    def failing_step(conversation, allow_tools):
        raise ValueError("The API is down")

    monkeypatch.setattr(checker, "_step", failing_step)

    with pytest.raises(FactCheckError):
        checker.check_claim("claim", claim_date="2024-02-05")
    assert ended == [checker.conversation]
    assert len(closed) == 1  # Its page media get deleted
