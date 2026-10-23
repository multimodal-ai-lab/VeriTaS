"""Gemini fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via Gemini's
function calling. Videos, the claim's and the fetched pages', are passed natively.
"""

import time
from pathlib import Path

from google import genai
from google.genai import types

from ..common.media import Image
from ..common.tools import TOOL_DEFINITIONS
from .base import BaseFactChecker, Part, Reasoning, Step, Text, ToolCall, only_text, resolve_api_key


# Video MIME types supported by Gemini
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
    ".m4v": "video/mp4",
}

# Disables function calling, forcing the model to give its verdict
NO_FUNCTION_CALLING = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
)


def gemini_tool() -> types.Tool:
    """The tools in Gemini format."""
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["parameters"],
        )
        for tool in TOOL_DEFINITIONS
    ])


class GeminiFactChecker(BaseFactChecker):
    """Fact-checker using Gemini with the web_search and fetch_url tools."""

    provider_name = "gemini"
    default_model = "gemini-2.5-flash"
    native_video = True

    def _create_client(self, api_key: str | None) -> genai.Client:
        return genai.Client(api_key=resolve_api_key(api_key, "google", "GOOGLE_API_KEY"))

    def _start_conversation(self, prompt: list[Part]) -> list:
        return [types.Content(role="user", parts=self._gemini_parts(prompt))]

    def _gemini_parts(self, parts: list[Part]) -> list[types.Part]:
        """Text, images and videos, in order, as Gemini's parts. Videos are uploaded with the
        File API: inline, each later request, which repeats the conversation, would repeat them."""
        gemini_parts = []
        for part in parts:
            if isinstance(part, str):
                gemini_parts.append(types.Part.from_text(text=part))
            elif isinstance(part, Image):
                gemini_parts.append(types.Part.from_bytes(data=part.data, mime_type=part.mime_type))
            else:
                gemini_parts.append(self._upload_video(str(part.file_path))
                                    or types.Part.from_text(text="[The video could not be uploaded.]"))
        return gemini_parts

    def _step(self, history: list, allow_tools: bool) -> Step:
        response = self.client.models.generate_content(
            model=self.model,
            contents=history,
            config=types.GenerateContentConfig(
                tools=[gemini_tool()],
                tool_config=None if allow_tools else NO_FUNCTION_CALLING,
                thinking_config=self._thinking_config(),
            ),
        )
        content = response.candidates[0].content if response.candidates else None
        parts = (content.parts or []) if content else []
        if content:
            # Keep the model's whole turn, incl. thought signatures
            history.append(content)

        # Don't use response.text: it warns when there are non-text parts
        output = []
        for part in parts:
            if part.thought and part.text:
                output.append(Reasoning(part.text))
            elif part.text:
                output.append(Text(part.text))
            elif part.function_call:
                call = part.function_call
                output.append(ToolCall(call.id, call.name, dict(call.args or {})))

        usage = response.usage_metadata
        return Step(
            output,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            cache_read_tokens=(usage.cached_content_token_count or 0) if usage else 0,
            stop_reason=self._stop_reason(response),
        )

    @staticmethod
    def _stop_reason(response: types.GenerateContentResponse) -> str | None:
        """Why the model stopped: the candidate's finish reason ("STOP", "MAX_TOKENS", "SAFETY",
        "MALFORMED_FUNCTION_CALL", ...), or why the prompt was blocked if there is no candidate."""
        if response.candidates:
            reason = response.candidates[0].finish_reason
        else:
            feedback = response.prompt_feedback
            reason = feedback.block_reason if feedback and feedback.block_reason else "NO_CANDIDATES"
        return getattr(reason, "name", reason)

    def _thinking_config(self) -> types.ThinkingConfig:
        """Summaries of the model's thoughts, as parts marked `thought`, at the reasoning effort.
        Gemini 2 models have no thinking levels (the API rejects them), only a token budget:
        they think as much as they decide to."""
        if self.model.startswith(("gemini-2.", "models/gemini-2.")):
            return types.ThinkingConfig(include_thoughts=True)
        # The API's thinking levels are upper case; the SDK passes those it doesn't know (yet) on as they are
        return types.ThinkingConfig(include_thoughts=True, thinking_level=self.reasoning_effort.upper())

    def _add_tool_results(self, history: list, tool_calls: list[ToolCall], results: list[list[Part]]):
        # A function response holds JSON, and the model ignores uploaded videos in its own parts:
        # a result with images or videos follows the responses in the same turn, in order
        responses, with_media = [], []
        for call, result in zip(tool_calls, results):
            text = only_text(result)
            if text is None:
                with_media += result
                text = "The content, with its images and videos, follows the function responses."
            responses.append(types.Part.from_function_response(name=call.name, response={"result": text}))
        history.append(types.Content(role="user", parts=responses + self._gemini_parts(with_media)))

    def _add_text(self, history: list, text: str):
        # In the user turn with the function responses, after them
        history[-1].parts.append(types.Part.from_text(text=text))

    def _end_conversation(self, history: list):
        """Delete the files uploaded for the fact-check: until they expire after 48 hours,
        they count against the project's storage quota."""
        for content in history:
            for part in content.parts or []:
                if part.file_data and part.file_data.file_uri:
                    name = "files/" + part.file_data.file_uri.rsplit("/", 1)[-1]
                    try:
                        self.client.files.delete(name=name)
                    except Exception as e:
                        print(f"    Warning: Could not delete uploaded file {name}: {e}")

    def _upload_video(self, video_path: str) -> types.Part | None:
        """A video uploaded with the File API, by its URI. None if the upload fails."""
        try:
            uploaded_file = self.client.files.upload(file=video_path)
            # Wait for file to become ACTIVE (required before use)
            max_wait = 60  # seconds
            wait_interval = 2
            waited = 0
            while waited < max_wait:
                file_info = self.client.files.get(name=uploaded_file.name)
                if file_info.state.name == "ACTIVE":
                    mime_type = VIDEO_MIME_TYPES.get(Path(video_path).suffix.lower(), "video/mp4")
                    return types.Part.from_uri(file_uri=file_info.uri, mime_type=file_info.mime_type or mime_type)
                elif file_info.state.name == "FAILED":
                    print("    Warning: Video file processing failed")
                    return None
                time.sleep(wait_interval)
                waited += wait_interval
            print(f"    Warning: Video file not ready after {max_wait}s, skipping")
            return None
        except Exception as e:
            print(f"    Warning: Failed to upload video via File API: {e}")
            return None
