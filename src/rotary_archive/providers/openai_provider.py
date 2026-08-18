"""OpenAI provider.

Present so the club can switch backends with a config edit. Uses the chat
completions API with a JSON-schema response format; concurrency comes from the
base class's thread pool rather than OpenAI's batch endpoint, which needs a
file upload round trip that is not worth the complexity here.
"""

from __future__ import annotations

from typing import Any

from .base import AnalysisResult, Job, ProviderError, VisionProvider, extract_json


class OpenAIProvider(VisionProvider):
    name = "openai"
    supports_batch = False
    supports_schema = True

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model, options)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "The openai package is not installed. Run: pip install 'rotary-archive[openai]'"
            ) from exc

        try:
            self.client = OpenAI()
        except Exception as exc:
            raise ProviderError(
                "Could not construct the OpenAI client. Set OPENAI_API_KEY."
            ) from exc

        self.max_tokens = int(self.options.get("max_tokens", 8000))

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        data_url = f"data:{job.media_type};base64,{job.image_b64()}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "archive_item",
                        "strict": True,
                        "schema": schema,
                    },
                },
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {
                                "type": "text",
                                "text": job.context or "Catalogue this item.",
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error="response blocked by content filter",
            )

        text = choice.message.content or ""
        try:
            parsed = extract_json(text)
        except ValueError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=str(exc),
            )

        usage = getattr(response, "usage", None)
        return AnalysisResult(
            item_id=job.item_id,
            ok=True,
            data=parsed,
            raw=text,
            provider=self.name,
            model=self.model,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            },
        )
