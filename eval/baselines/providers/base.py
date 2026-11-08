"""Abstract base class for fact-checking providers: the tool-calling loop every provider runs."""

import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

import ezmm

from ..common.media import (Image, ItemSequence, MediaLimits, extract_video_frames, load_claim_image, load_image,
                            render)
from ..common.parsing import extract_justification, extract_urls, extract_verdict
from ..common.prompts import build_prompt
from ..common.tools import ToolSession, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from ..common.types import FactCheckError, FactCheckResult, NoAnswerError, MAX_RETRIES, BASE_DELAY, MAX_DELAY

try:
    from veritas import api_secrets
except ImportError:
    api_secrets = None

MAX_TURNS = 50
LAST_TURN_MESSAGE = ("You have used up your whole quota of tool calls. You must provide an answer given the "
                     "information you were able to gather so far.")

# What a provider turns into its API's content: text, images, and videos if it has native_video
Part = str | Image | ezmm.Video


def only_text(parts: list[Part]) -> str | None:
    return parts[0] if len(parts) == 1 and isinstance(parts[0], str) else None


DEFAULT_REASONING_EFFORT = "medium"

# 529: Anthropic's API is overloaded
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def is_retryable(error: Exception) -> bool:
    """The HTTP status is `status_code` in the OpenAI and Anthropic SDKs, `code` in Google's."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return status in RETRYABLE_STATUS_CODES


def resolve_api_key(api_key: str | None, secret_name: str, env_var: str) -> str:
    key = api_key or (api_secrets or {}).get(secret_name) or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"{secret_name} API key required. Either:\n"
            f"  1. Add it to config.yaml (api_secrets.{secret_name})\n"
            f"  2. Set {env_var} environment variable\n"
            f"  3. Pass api_key parameter"
        )
    return key


@dataclass
class ToolCall:
    id: str | None
    name: str
    args: dict | None  # None if the model's arguments were not a JSON object

    @classmethod
    def from_json(cls, id: str | None, name: str, arguments: str | None) -> "ToolCall":
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = None
        return cls(id, name, args)


@dataclass
class Reasoning:
    """The model's reasoning, as far as its API returns it: usually a summary."""

    text: str


@dataclass
class Text:
    text: str


@dataclass
class Step:
    """One model call. stop_reason is the API's own ("end_turn", "max_tokens", "SAFETY", ...).
    input_tokens counts the whole prompt, the tokens read from and written to the API's prompt
    cache included."""

    output: list[Reasoning | Text | ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    # Reported by Anthropic, and by OpenRouter for models whose cache writes are billed
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    cost: float | None = None  # What the call was billed, in USD, if the API says (OpenRouter)

    @property
    def text(self) -> str:
        return "\n".join(item.text for item in self.output if isinstance(item, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [item for item in self.output if isinstance(item, ToolCall)]


TRACE_TYPES = {Reasoning: "reasoning", Text: "text", ToolCall: "tool_call"}


class BaseFactChecker(ABC):
    """Abstract base class for fact-checking providers.

    Every provider runs the same baseline: the model verifies the claim with the
    web_search and fetch_url tools and answers with the claim's integrity, its certainty
    and a justification in JSON, which make a label of the 7-class scheme. The loop lives in
    `check_claim`; a provider only implements its API's message format in the abstract methods.
    """

    provider_name: str = "base"
    default_model: str = ""
    native_video: bool = False  # Whether videos reach the model as is, else as frames

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        media_limits: MediaLimits = MediaLimits(),
    ):
        """
        Initialize the fact-checker.

        Args:
            api_key: API key for the provider. If None, uses config/env var.
            model: Model identifier to use (must support tool use). Defaults to `default_model`.
            scrape_methods: Which scrapeMM backends fetch_url uses (subset of
                            integrations/browser/firecrawl/decodo, or "auto").
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.
            reasoning_effort: How much the model reasons, e.g. "low", "medium" or "high", passed
                              as is to the API's effort setting. Which values a model takes
                              depends on its vendor.
            media_limits: Which images and videos the model gets, and in what form.
        """
        self.model = model or self.default_model
        self.scrape_methods = scrape_methods
        self.max_searches = max_searches
        self.max_fetches = max_fetches
        self.reasoning_effort = reasoning_effort
        self.media_limits = media_limits
        self.client = self._create_client(api_key)

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
        claim_id: int | str | None = None,
        blocked_urls: list[str] | None = None,
        cutoff_date: str | datetime | None = None,
        blocked_scopes: list[str] | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include.
                         Note: Only Gemini supports native video. Other providers
                         will extract frames from videos and treat as images.
            claim_date: Date of the claim (ISO format string or datetime).
                        web_search only returns content published before this day, unless
                        cutoff_date is given.
            claim_id: ID of the claim, which tags the log lines of its tool calls and retries.
            blocked_urls: URLs the tools never show or fetch, e.g. the claim's own fact-check,
                          which would give the answer away.
            cutoff_date: web_search only returns content published before this day, e.g. the
                         day the claim's fact-check was published.
            blocked_scopes: Hosts, or hosts and paths, whose URLs the tools never show or fetch,
                            e.g. those under which the fact-check's publisher publishes
                            fact-checks ("leadstories.com/hoax-alert").

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.

        Raises:
            NoAnswerError: The model's final response holds no valid answer.
            FactCheckError: The fact-check failed, e.g. on an API error. Both hold the result
                            so far: its trace, URLs and usage.
        """
        session = ToolSession(
            claim_date=claim_date,
            scrape_methods=self.scrape_methods,
            max_searches=self.max_searches,
            max_fetches=self.max_fetches,
            media_limits=self.media_limits,
            blocked_urls=blocked_urls or (),
            claim_id=claim_id,
            cutoff_date=cutoff_date,
            blocked_scopes=blocked_scopes or (),
            claim_images=len(image_paths or [])
                         + (0 if self.native_video else self.media_limits.video_frames * len(video_paths or [])),
        )

        media = ([ezmm.Image(file_path=path) for path in image_paths or []]
                 + [ezmm.Video(file_path=path) for path in video_paths or []])
        prompt = build_prompt(claim, session.claim_day, media)
        conversation = self._start_conversation(self._parts(prompt, claim=True))

        tokens = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        usage = dict.fromkeys(tokens, 0)
        trace = [{"type": "prompt", "text": render(prompt)}]
        step = Step([])
        try:
            for turn in range(MAX_TURNS):
                last_turn = turn == MAX_TURNS - 1
                if last_turn:
                    self._add_text(conversation, LAST_TURN_MESSAGE)
                    trace.append({"turn": turn, "type": "message", "text": LAST_TURN_MESSAGE})
                step = self._call_with_retry(self._step, conversation, allow_tools=not last_turn, log=session.log)
                for key in tokens:
                    usage[key] += getattr(step, key)
                if step.cost is not None:
                    usage["cost"] = usage.get("cost", 0) + step.cost
                trace += [{"turn": turn, "type": TRACE_TYPES[type(item)], **asdict(item)} for item in step.output]
                trace.append({"turn": turn, "type": "stop", "reason": step.stop_reason})

                if not step.tool_calls:
                    break
                results = [session.call(call.name, call.args) for call in step.tool_calls]
                trace += [
                    {"turn": turn, "type": "tool_result", "id": call.id, "name": call.name, "text": result.text,
                     "media": result.media}
                    for call, result in zip(step.tool_calls, results)
                ]
                self._add_tool_results(conversation, step.tool_calls, [self._parts(result.content) for result in results])
        except Exception as e:
            raise FactCheckError(f"{type(e).__name__}: {e}", self._make_result("", session, usage, trace)) from e
        finally:
            self._end_conversation(conversation)
            session.close()

        result = self._make_result(step.text, session, usage, trace)
        if result.verdict is None:
            raise NoAnswerError(f"No valid answer in the final response (stop reason: {step.stop_reason})", result)
        return result

    def _parts(self, sequence: ItemSequence, claim: bool = False) -> list[Part]:
        """An ezMM sequence as the model gets it: its text, with each image and video right after
        its reference, so that the model can name it. Adjacent text is joined, as by ezMM."""
        limits = self.media_limits
        max_side = limits.claim_image_side if claim else limits.page_image_side
        parts = []
        for element in sequence:
            if isinstance(element, str):
                new = [element]
            elif isinstance(element, ezmm.Image):
                load = load_claim_image if claim else load_image
                new = [element.reference, load(element.file_path, max_side)]
            elif isinstance(element, ezmm.Video) and self.native_video and (claim or self._is_short(element)):
                new = [element.reference, element]
            elif isinstance(element, ezmm.Video):
                frames = extract_video_frames(str(element.file_path), limits.video_frames, max_side=max_side)
                new = [f"{element.reference} ({len(frames)} frames of the video)", *frames]
            else:  # Audio, files: no API here takes them
                new = [element.reference]
            for part in new:
                if isinstance(part, str) and parts and isinstance(parts[-1], str):
                    parts[-1] += " " + part
                else:
                    parts.append(part)
        return parts

    def _is_short(self, video: ezmm.Video) -> bool:
        """False if the length can't be read: ezMM raises for a file it can't open."""
        try:
            return video.duration <= self.media_limits.native_video_seconds
        except Exception:
            return False

    def _end_conversation(self, conversation: list):
        """Clean up after a fact-check, e.g. delete files uploaded for it."""

    @abstractmethod
    def _create_client(self, api_key: str | None):
        """Create the API client."""

    @abstractmethod
    def _start_conversation(self, prompt: list[Part]) -> list:
        """The conversation: the user turn with the prompt, the claim's media in their places."""

    @abstractmethod
    def _step(self, conversation: list, allow_tools: bool) -> Step:
        """Call the model on the conversation and append the model's turn to it."""

    @abstractmethod
    def _add_tool_results(self, conversation: list, tool_calls: list[ToolCall], results: list[list[Part]]):
        """Append the results of the model's tool calls to the conversation."""

    @abstractmethod
    def _add_text(self, conversation: list, text: str):
        """Add text to the conversation as the user's, after the last tool results."""

    def _make_result(self, final_text: str, session: ToolSession, usage: dict, trace: list[dict]) -> FactCheckResult:
        justification = extract_justification(final_text)
        citations = list(dict.fromkeys(
            session.encountered_urls + session.retrieved_urls + extract_urls(final_text)
        ))
        return FactCheckResult(
            verdict=extract_verdict(final_text),
            reasoning=final_text,
            citations=citations,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
            justification=justification,
            encountered_urls=list(session.encountered_urls),
            retrieved_urls=list(session.retrieved_urls),
            linked_urls=list(session.linked_urls),
            quoted_urls=extract_urls(justification),
            blocked=list(session.blocked),
            trace=trace,
        )

    def _call_with_retry(self, api_call_fn, *args, log: Callable[[str], None], **kwargs):
        """
        Call an API function with exponential backoff retry for rate limits.

        Args:
            api_call_fn: The API function to call.
            *args, **kwargs: Arguments to pass to the function.
            log: Prints a log line (of the claim's tool session).

        Returns:
            The result from the API call.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                return api_call_fn(*args, **kwargs)
            except Exception as e:
                if is_retryable(e):
                    last_exception = e
                    # Exponential backoff with jitter
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    log(f"Transient error ({e}), retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    # Not a retryable error, re-raise immediately
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception
