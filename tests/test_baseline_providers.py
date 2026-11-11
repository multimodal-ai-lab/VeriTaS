"""The real SDKs build the requests and parse the responses; a mock transport stands in for the
server (for Gemini, a fake client returns the SDK's own types)."""

import base64
import json
from datetime import date
from types import SimpleNamespace

import anthropic
import ezmm
import httpx
import pytest
from baseline_fakes import ezmm_registry, fake_search  # noqa: F401  # ezmm_registry is a fixture
from google.genai import types
from openai import OpenAI
from PIL import Image as PILImage

from eval.baselines import UnifiedFactChecker
from eval.baselines.common.media import Image
from eval.baselines.common.prompts import build_prompt
from eval.baselines.providers.base import ToolCall
from eval.baselines.providers import (
    AnthropicFactChecker,
    GeminiFactChecker,
    OpenAIFactChecker,
    OpenRouterFactChecker,
    SelfhostedFactChecker,
)

FINAL_TEXT = ('```json\n{"integrity": "intact", "certainty": "certain", '
              '"justification": "Per [A](https://example.com/a) it is."}\n```')
PROMPT, = build_prompt("The Eiffel Tower is in Paris", date(2024, 2, 5))
TRACE_TYPES = ["prompt", "reasoning", "tool_call", "stop", "tool_result", "reasoning", "text", "stop"]


def _mock_transport(responses: list[dict]) -> tuple[httpx.MockTransport, list]:
    """A server that replies with `responses` in turn. Returns the transport and the
    (path, JSON body) of each request."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=responses[len(requests) - 1])

    return httpx.MockTransport(handler), requests


def _mock_openai_client(base_url: str, responses: list[dict]) -> tuple[OpenAI, list]:
    transport, requests = _mock_transport(responses)
    return OpenAI(api_key="test", base_url=base_url, http_client=httpx.Client(transport=transport)), requests


def _trace_types(result) -> list[str]:
    return [entry["type"] for entry in result.trace]


def _stop_reasons(result) -> list[str]:
    return [entry["reason"] for entry in result.trace if entry["type"] == "stop"]


def test_openai_uses_the_responses_api_with_reasoning_summaries(monkeypatch, tmp_path, ezmm_registry):
    fake_search(monkeypatch, ["https://example.com/a"])
    image_path = tmp_path / "photo.png"
    PILImage.new("RGB", (40, 20), "red").save(image_path, "PNG")
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
             "input_tokens_details": {"cached_tokens": 4}, "output_tokens_details": {"reasoning_tokens": 1}}
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "I should search."}]}
    function_call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "web_search",
                     "arguments": '{"query": "eiffel"}', "status": "completed"}
    final_reasoning = {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "Confirmed."}]}
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": FINAL_TEXT, "annotations": []}]}
    checker = OpenAIFactChecker(api_key="test", model="gpt-test")
    checker.client, requests = _mock_openai_client("https://openai.test/v1", [
        {"id": "resp_1", "object": "response", "status": "completed", "output": [reasoning, function_call],
         "usage": usage},
        {"id": "resp_2", "object": "response", "status": "completed", "output": [final_reasoning, message],
         "usage": usage},
    ])

    result = checker.check_claim("The Eiffel Tower is in Paris", image_paths=[str(image_path)],
                                 claim_date="2024-02-05")

    (path, first), (_, second) = requests
    assert path == "/v1/responses"
    assert first["model"] == "gpt-test"
    assert "instructions" not in first  # No system prompt
    assert first["tool_choice"] == "auto"
    assert first["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert [(t["type"], t["name"], t["strict"]) for t in first["tools"]] == [
        ("function", "web_search", False), ("function", "fetch_url", False)]
    before, image, after = first["input"][0]["content"]
    assert before["text"].endswith('# The Claim\n"The Eiffel Tower is in Paris <image:1>')
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    assert image == {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "auto"}
    assert after["text"].startswith('"\nPublished on: February 05, 2024')

    assert second["input"][1:3] == [reasoning, function_call]
    output = second["input"][3]
    assert (output["type"], output["call_id"]) == ("function_call_output", "call_1")
    assert "https://example.com/a" in output["output"]

    assert result.verdict == "Intact (certain)"
    assert result.encountered_urls == ["https://example.com/a"]
    assert result.usage == {"input_tokens": 20, "output_tokens": 4, "cache_read_tokens": 8, "cache_write_tokens": 0}
    assert _trace_types(result) == TRACE_TYPES
    assert _stop_reasons(result) == ["completed", "completed"]
    assert [e["text"] for e in result.trace if e["type"] == "reasoning"] == ["I should search.", "Confirmed."]


def test_anthropic_requests_summarized_thinking_and_passes_it_back(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    usage = {"input_tokens": 10, "output_tokens": 2, "cache_creation_input_tokens": 5, "cache_read_input_tokens": 3}
    thinking = {"type": "thinking", "thinking": "I should search.", "signature": "sig_1"}
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "eiffel"}}

    def message(content: list[dict], stop_reason: str) -> dict:
        return {"id": "msg", "type": "message", "role": "assistant", "model": "claude-test", "content": content,
                "stop_reason": stop_reason, "stop_sequence": None, "usage": usage}

    transport, requests = _mock_transport([
        message([thinking, tool_use], "tool_use"),
        message([{"type": "thinking", "thinking": "Confirmed.", "signature": "sig_2"},
                 {"type": "text", "text": FINAL_TEXT}], "end_turn"),
    ])
    checker = AnthropicFactChecker(api_key="test", model="claude-test")
    checker.client = anthropic.Anthropic(api_key="test", base_url="https://anthropic.test",
                                         http_client=httpx.Client(transport=transport))

    result = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05")

    (path, first), (_, second) = requests
    assert path == "/v1/messages"
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert "system" not in first
    assert first["messages"][0]["content"] == [{"type": "text", "text": PROMPT}]
    assert second["messages"][1] == {"role": "assistant", "content": [thinking, tool_use]}
    assert second["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"

    assert result.verdict == "Intact (certain)"
    # Input tokens include those written to and read from the cache
    assert result.usage == {"input_tokens": 36, "output_tokens": 4, "cache_read_tokens": 6, "cache_write_tokens": 10}
    assert _trace_types(result) == TRACE_TYPES
    assert _stop_reasons(result) == ["tool_use", "end_turn"]
    assert [e["text"] for e in result.trace if e["type"] == "reasoning"] == ["I should search.", "Confirmed."]


def test_gemini_requests_thoughts_and_keeps_them_out_of_the_answer(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=2, thoughts_token_count=3, cached_content_token_count=4)

    def response(*parts: types.Part) -> types.GenerateContentResponse:
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)),
                                        finish_reason=types.FinishReason.STOP)],
            usage_metadata=usage,
        )

    replies = [
        response(types.Part(text="I should search.", thought=True),
                 types.Part(function_call=types.FunctionCall(name="web_search", args={"query": "eiffel"}))),
        response(types.Part(text="Confirmed.", thought=True), types.Part(text=FINAL_TEXT)),
    ]
    configs = []

    def generate_content(model, contents, config):
        configs.append(config)
        return replies[len(configs) - 1]

    checker = GeminiFactChecker(api_key="test", model="gemini-test")
    checker.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    result = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05")

    assert all(config.thinking_config.include_thoughts for config in configs)
    assert all(config.thinking_config.model_dump(mode="json")["thinking_level"] == "MEDIUM" for config in configs)
    assert all(config.system_instruction is None for config in configs)
    assert result.reasoning == FINAL_TEXT
    assert result.verdict == "Intact (certain)"
    assert result.usage == {"input_tokens": 20, "output_tokens": 10, "cache_read_tokens": 8, "cache_write_tokens": 0}
    assert _trace_types(result) == TRACE_TYPES
    assert _stop_reasons(result) == ["STOP", "STOP"]
    assert [e["text"] for e in result.trace if e["type"] == "reasoning"] == ["I should search.", "Confirmed."]


def test_selfhosted_uses_chat_completions_and_vllm_reasoning(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    tool_call = {"id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "eiffel"}'}}

    def completion(message: dict) -> dict:
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "usage": usage,
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]}

    checker = SelfhostedFactChecker(api_key="test", base_url="https://vllm.test/v1", model="llama-test")
    checker.client, requests = _mock_openai_client("https://vllm.test/v1", [
        completion({"role": "assistant", "content": None, "reasoning_content": "I should search.",
                    "tool_calls": [tool_call]}),
        # Newer vLLM versions name the field "reasoning"
        completion({"role": "assistant", "content": FINAL_TEXT, "reasoning": "Confirmed."}),
    ])

    result = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05")

    (path, first), (_, second) = requests
    assert path == "/v1/chat/completions"
    assert first["reasoning_effort"] == "medium"
    assert first["messages"] == [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    assert [t["function"]["name"] for t in first["tools"]] == ["web_search", "fetch_url"]
    assert second["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert second["messages"][2]["role"] == "tool" and second["messages"][2]["tool_call_id"] == "call_1"
    assert result.verdict == "Intact (certain)"
    assert result.usage == {"input_tokens": 20, "output_tokens": 4, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert _trace_types(result) == TRACE_TYPES
    assert _stop_reasons(result) == ["tool_calls", "stop"]
    assert [e["text"] for e in result.trace if e["type"] == "reasoning"] == ["I should search.", "Confirmed."]


def test_openrouter_pins_the_host_requires_tool_support_and_passes_reasoning_details_back(monkeypatch):
    fake_search(monkeypatch, ["https://example.com/a"])
    # OpenRouter's usage goes beyond the OpenAI API's: cache writes and what the call was billed
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0005,
             "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1}}
    tool_call = {"id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "eiffel"}'}}
    details = [{"type": "reasoning.text", "text": "I should search.", "signature": "sig_1",
                "format": "anthropic-claude-v1", "index": 0}]

    def completion(message: dict) -> dict:
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "usage": usage,
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]}

    checker = OpenRouterFactChecker(api_key="test", model="vendor/model-test", host="host/fp8")
    checker.client, requests = _mock_openai_client("https://openrouter.ai/api/v1", [
        completion({"role": "assistant", "content": None, "reasoning": "I should search.",
                    "reasoning_details": details, "tool_calls": [tool_call]}),
        completion({"role": "assistant", "content": FINAL_TEXT, "reasoning": "Confirmed."}),
    ])

    result = checker.check_claim("The Eiffel Tower is in Paris", claim_date="2024-02-05")

    (path, first), (_, second) = requests
    assert path == "/api/v1/chat/completions"
    assert first["provider"] == {"only": ["host/fp8"], "require_parameters": True}
    assert first["reasoning"] == {"effort": "medium"}
    assert "reasoning_effort" not in first
    assert [t["function"]["name"] for t in first["tools"]] == ["web_search", "fetch_url"]
    assert second["messages"][1]["reasoning_details"] == details
    assert second["messages"][2]["role"] == "tool" and second["messages"][2]["tool_call_id"] == "call_1"
    assert result.verdict == "Intact (certain)"
    assert result.usage == {"input_tokens": 20, "output_tokens": 4, "cache_read_tokens": 8, "cache_write_tokens": 2,
                            "cost": 0.001}
    assert _trace_types(result) == TRACE_TYPES
    assert _stop_reasons(result) == ["tool_calls", "stop"]
    assert [e["text"] for e in result.trace if e["type"] == "reasoning"] == ["I should search.", "Confirmed."]


def test_reasoning_effort_is_passed_to_every_provider():
    providers = ["openai", "gemini", "anthropic", "openrouter"]
    fc = UnifiedFactChecker(providers=providers, api_keys={p: "test" for p in providers}, reasoning_effort="high",
                            openrouter_host="host/fp8")
    assert [fc.get_provider(p).reasoning_effort for p in providers] == ["high"] * len(providers)
    assert fc.get_provider("openrouter").host == "host/fp8"
    assert UnifiedFactChecker(provider="openai", api_keys={"openai": "test"}).get_provider().reasoning_effort == "medium"


def test_openrouter_needs_a_host():
    """Without a pinned host, OpenRouter would spread the requests over hosts of different quantizations."""
    with pytest.raises(ValueError, match="host"):
        UnifiedFactChecker(provider="openrouter", api_keys={"openrouter": "test"})


def test_gemini_stop_reason_of_a_blocked_prompt():
    """A blocked prompt gets no candidate, only feedback saying why."""
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.PROHIBITED_CONTENT))
    assert GeminiFactChecker._stop_reason(blocked) == "PROHIBITED_CONTENT"
    assert GeminiFactChecker._stop_reason(types.GenerateContentResponse()) == "NO_CANDIDATES"


def test_gemini_2_models_get_no_thinking_level():
    """Gemini 2 models reject thinking levels: "Thinking level is not supported for this model"."""
    assert GeminiFactChecker(api_key="test", model="gemini-2.5-flash")._thinking_config().thinking_level is None
    assert GeminiFactChecker(api_key="test", model="gemini-3.8-flash")._thinking_config().thinking_level == "MEDIUM"


IMAGE = Image("image/jpeg", b"jpg")
PAGE = ["Content of the page <image:7>", IMAGE, "and more text"]
SEARCH = ["Search results"]
CALLS = [ToolCall("call_1", "fetch_url", {"url": "https://e.org"}), ToolCall("call_2", "web_search", {"query": "q"})]


def test_openai_puts_page_text_and_images_in_order_in_the_function_output():
    items = []
    OpenAIFactChecker(api_key="test")._add_tool_results(items, CALLS, [PAGE, SEARCH])

    assert items[0]["output"] == [
        {"type": "input_text", "text": "Content of the page <image:7>"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,anBn", "detail": "auto"},
        {"type": "input_text", "text": "and more text"},
    ]
    assert items[1]["output"] == "Search results"


def test_anthropic_puts_page_text_and_images_in_order_in_the_tool_result():
    messages = []
    AnthropicFactChecker(api_key="test")._add_tool_results(messages, CALLS, [PAGE, SEARCH])

    page, search = messages[0]["content"]
    assert page["content"] == [
        {"type": "text", "text": "Content of the page <image:7>"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "anBn"}},
        {"type": "text", "text": "and more text"},
    ]
    assert search["content"] == "Search results"


def test_gemini_gets_page_media_in_order_after_the_function_responses_and_videos_uploaded(tmp_path, ezmm_registry):
    """The model ignores uploaded videos in a function response's own parts."""
    (tmp_path / "v.mp4").write_bytes(b"mp4")
    video = ezmm.Video(file_path=str(tmp_path / "v.mp4"))
    uploads, deleted = [], []

    def upload(file):
        uploads.append(file)
        return SimpleNamespace(name="files/abc")

    checker = GeminiFactChecker(api_key="test")
    checker.client = SimpleNamespace(files=SimpleNamespace(
        upload=upload,
        get=lambda name: SimpleNamespace(state=SimpleNamespace(name="ACTIVE"), mime_type="video/mp4",
                                         uri="https://generativelanguage.googleapis.com/v1beta/files/abc"),
        delete=lambda name: deleted.append(name),
    ))
    history = []
    page = ["Content of the page <image:7>", IMAGE, "and more text <video:1>", video]
    checker._add_tool_results(history, CALLS, [page, SEARCH])

    fetch, search, text, image, more, video_part = history[0].parts
    assert fetch.function_response.response == {
        "result": "The content, with its images and videos, follows the function responses."}
    assert search.function_response.response == {"result": "Search results"}
    assert (text.text, image.inline_data.data, more.text) == ("Content of the page <image:7>", b"jpg", "and more text <video:1>")
    assert video_part.file_data.file_uri.endswith("/files/abc") and uploads == [str(video.file_path)]

    checker._end_conversation(history)
    assert deleted == ["files/abc"]


def test_chat_completions_send_page_text_and_images_in_order_in_a_user_message():
    """Tool messages take only text."""
    messages = []
    SelfhostedFactChecker(api_key="test", base_url="https://vllm.test/v1")._add_tool_results(messages, CALLS, [PAGE, SEARCH])

    assert [(m["role"], m["content"]) for m in messages[:2]] == [
        ("tool", "The content, with its images, follows in the next message."), ("tool", "Search results")]
    assert messages[2] == {"role": "user", "content": [
        {"type": "text", "text": "Content of the page <image:7>"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,anBn"}},
        {"type": "text", "text": "and more text"},
    ]}


def test_the_last_turn_message_follows_the_tool_results_in_each_api(tmp_path):
    text = "Answer now."
    items = []
    openai_checker = OpenAIFactChecker(api_key="test")
    openai_checker._add_tool_results(items, CALLS, [PAGE, SEARCH])
    openai_checker._add_text(items, text)
    assert items[-1] == {"role": "user", "content": [{"type": "input_text", "text": text}]}

    # Anthropic and Gemini: in the same user turn, as turns alternate
    messages = []
    anthropic_checker = AnthropicFactChecker(api_key="test")
    anthropic_checker._add_tool_results(messages, CALLS, [SEARCH, SEARCH])
    anthropic_checker._add_text(messages, text)
    assert len(messages) == 1 and messages[0]["content"][-1] == {"type": "text", "text": text}

    history = []
    gemini_checker = GeminiFactChecker(api_key="test")
    gemini_checker._add_tool_results(history, CALLS, [SEARCH, SEARCH])
    gemini_checker._add_text(history, text)
    assert len(history) == 1 and history[0].parts[-1].text == text

    chat_checker = SelfhostedFactChecker(api_key="test", base_url="https://vllm.test/v1")
    messages = []
    chat_checker._add_tool_results(messages, CALLS, [SEARCH, SEARCH])
    chat_checker._add_text(messages, text)
    assert messages[-1] == {"role": "user", "content": [{"type": "text", "text": text}]}
    messages = []
    chat_checker._add_tool_results(messages, CALLS, [PAGE, SEARCH])
    chat_checker._add_text(messages, text)
    assert [m["role"] for m in messages] == ["tool", "tool", "user"]
    assert messages[-1]["content"][-1] == {"type": "text", "text": text}
