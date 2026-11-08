"""OpenAI fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via function calling
in OpenAI's Responses API. Current OpenAI models reject function tools in the chat
completions API unless reasoning is turned off.
"""

from openai import OpenAI

from ..common.tools import responses_tools
from .base import BaseFactChecker, Part, Reasoning, Step, Text, ToolCall, only_text, resolve_api_key


class OpenAIFactChecker(BaseFactChecker):
    """Fact-checker using OpenAI with the web_search and fetch_url tools."""

    provider_name = "openai"
    default_model = "gpt-5.2"

    def _create_client(self, api_key: str | None) -> OpenAI:
        return OpenAI(api_key=resolve_api_key(api_key, "openai", "OPENAI_API_KEY"))

    def _start_conversation(self, prompt: list[Part]) -> list:
        return [{"role": "user", "content": self._content(prompt)}]

    @staticmethod
    def _content(parts: list[Part]) -> list[dict]:
        return [
            {"type": "input_text", "text": part} if isinstance(part, str)
            else {"type": "input_image", "image_url": f"data:{part.mime_type};base64,{part.base64}", "detail": "auto"}
            for part in parts
        ]

    def _step(self, items: list, allow_tools: bool) -> Step:
        response = self.client.responses.create(
            model=self.model,
            input=items,
            tools=responses_tools(),
            tool_choice="auto" if allow_tools else "none",
            # Only a summary of the reasoning is available
            reasoning={"effort": self.reasoning_effort, "summary": "auto"},
        )
        # Keep the whole output: reasoning models need their reasoning items passed
        # back along with the function calls they led to
        items.extend(response.output)

        output = []
        for item in response.output:
            if item.type == "reasoning":
                summary = "\n\n".join(part.text for part in item.summary)
                if summary:
                    output.append(Reasoning(summary))
            elif item.type == "message":
                text = "".join(part.text if part.type == "output_text" else part.refusal for part in item.content)
                if text:
                    output.append(Text(text))
            elif item.type == "function_call":
                output.append(ToolCall.from_json(item.call_id, item.name, item.arguments))
        incomplete = response.incomplete_details
        return Step(
            output,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            cache_read_tokens=response.usage.input_tokens_details.cached_tokens if response.usage else 0,
            # "completed", or why the response is incomplete: "max_output_tokens", "content_filter"
            stop_reason=incomplete.reason if incomplete and incomplete.reason else response.status,
        )

    def _add_tool_results(self, items: list, tool_calls: list[ToolCall], results: list[list[Part]]):
        items.extend(
            {"type": "function_call_output", "call_id": call.id, "output": only_text(result) or self._content(result)}
            for call, result in zip(tool_calls, results)
        )

    def _add_text(self, items: list, text: str):
        items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
