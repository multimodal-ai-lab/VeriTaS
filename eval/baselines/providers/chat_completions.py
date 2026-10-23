"""Base for fact-checkers on OpenAI-compatible chat completions APIs (vLLM, OpenRouter).

The model verifies the claim with the web_search and fetch_url tools via function calling.
Subclasses create the client.
"""

from ..common.tools import chat_completions_tools
from .base import BaseFactChecker, Part, Reasoning, Step, Text, ToolCall, only_text


class ChatCompletionsFactChecker(BaseFactChecker):
    """Fact-checker using chat completions with the web_search and fetch_url tools."""

    def _request_params(self) -> dict:
        """API-specific parameters of every request: the reasoning effort, as OpenAI's
        `reasoning_effort`, which vLLM takes."""
        return {"reasoning_effort": self.reasoning_effort}

    def _start_conversation(self, prompt: list[Part]) -> list:
        return [{"role": "user", "content": self._content(prompt)}]

    @staticmethod
    def _content(parts: list[Part]) -> list[dict]:
        """Text and images, in order, as message content."""
        return [
            {"type": "text", "text": part} if isinstance(part, str)
            else {"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{part.base64}"}}
            for part in parts
        ]

    def _step(self, messages: list, allow_tools: bool) -> Step:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=chat_completions_tools(),
            tool_choice="auto" if allow_tools else "none",
            **self._request_params(),
        )
        choice = response.choices[0]
        message = choice.message
        # The whole message goes back: OpenRouter requires reasoning_details to be passed back unchanged
        messages.append(message)

        # The reasoning of reasoning models comes in a field that is not part of the OpenAI API:
        # "reasoning_content" (vLLM), or "reasoning" (newer vLLM versions, OpenRouter)
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        output = [Reasoning(reasoning)] if reasoning else []
        if message.content:
            output.append(Text(message.content))
        output += [
            ToolCall.from_json(call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls or []
        ]
        usage = response.usage
        details = usage.prompt_tokens_details if usage else None
        return Step(
            output,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_read_tokens=(details.cached_tokens or 0) if details else 0,
            # Not in the OpenAI API: OpenRouter reports these, vLLM does not
            cache_write_tokens=(getattr(details, "cache_write_tokens", None) or 0) if details else 0,
            stop_reason=choice.finish_reason,  # "stop", "tool_calls", "length", "content_filter"
            cost=getattr(usage, "cost", None),
        )

    def _add_tool_results(self, messages: list, tool_calls: list[ToolCall], results: list[list[Part]]):
        # Tool messages take only text: a result with images follows in a user message, text and images in order
        with_images = []
        for call, result in zip(tool_calls, results):
            text = only_text(result)
            if text is None:
                with_images += result
                text = "The content, with its images, follows in the next message."
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if with_images:
            messages.append({"role": "user", "content": self._content(with_images)})

    def _add_text(self, messages: list, text: str):
        # After the tool messages; in the user message with their images if there is one, as some
        # chat templates reject two user messages in a row
        last = messages[-1]
        if isinstance(last, dict) and last["role"] == "user":
            last["content"].append({"type": "text", "text": text})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
