"""Anthropic fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via Anthropic's
tool use API.
"""

import anthropic as anthropic_sdk

from ..common.tools import anthropic_tools
from .base import BaseFactChecker, Part, Reasoning, Step, Text, ToolCall, only_text, resolve_api_key

# Output tokens per model call, thinking included. The SDK refuses non-streaming requests that
# could take more than 10 minutes, i.e. above about 21,000 tokens (Claude Opus 4 and 4.1: 8192).
MAX_TOKENS = 16384


class AnthropicFactChecker(BaseFactChecker):
    """Fact-checker using Anthropic Claude with the web_search and fetch_url tools."""

    provider_name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def _create_client(self, api_key: str | None) -> anthropic_sdk.Anthropic:
        return anthropic_sdk.Anthropic(api_key=resolve_api_key(api_key, "anthropic", "ANTHROPIC_API_KEY"))

    def _start_conversation(self, prompt: list[Part]) -> list:
        return [{"role": "user", "content": self._content(prompt)}]

    @staticmethod
    def _content(parts: list[Part]) -> list[dict]:
        """Text and images, in order, as content blocks."""
        return [
            {"type": "text", "text": part} if isinstance(part, str)
            else {"type": "image", "source": {"type": "base64", "media_type": part.mime_type, "data": part.base64}}
            for part in parts
        ]

    def _step(self, messages: list, allow_tools: bool) -> Step:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=anthropic_tools(),
            tool_choice={"type": "auto" if allow_tools else "none"},
            # Adaptive thinking, which Claude 5 models always do, with a summary of it in the
            # thinking blocks: they are empty by default. On Claude 4.6 models, this turns thinking on.
            thinking={"type": "adaptive", "display": "summarized"},
            extra_body={
                # In the body, as older SDKs lack the output_config parameter
                "output_config": {"effort": self.reasoning_effort},
                # Caches the conversation up to its last block, so each turn reads the turns
                # before it from the cache. In the body, as older SDKs lack the parameter.
                "cache_control": {"type": "ephemeral"},
            },
        )
        # Keep the whole turn: thinking blocks must be passed back unchanged
        messages.append({"role": "assistant", "content": response.content})

        output = []
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                output.append(Reasoning(block.thinking))
            elif block.type == "redacted_thinking":
                output.append(Reasoning("[redacted]"))
            elif block.type == "text" and block.text:
                output.append(Text(block.text))
            elif block.type == "tool_use":
                output.append(ToolCall(block.id, block.name, block.input))
        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        return Step(
            output,
            # input_tokens leaves out the tokens read from and written to the cache
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            stop_reason=response.stop_reason,
        )

    def _add_tool_results(self, messages: list, tool_calls: list[ToolCall], results: list[list[Part]]):
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": only_text(result) or self._content(result)}
            for call, result in zip(tool_calls, results)
        ]})

    def _add_text(self, messages: list, text: str):
        # In the user turn with the tool results, after them: user and assistant turns alternate
        messages[-1]["content"].append({"type": "text", "text": text})
