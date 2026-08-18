"""Google Gemini provider."""

from __future__ import annotations

from typing import Any

from .base import AnalysisResult, Job, ProviderError, VisionProvider, extract_json


class GeminiProvider(VisionProvider):
    name = "gemini"
    supports_batch = False
    supports_schema = True

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model, options)
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError(
                "The google-genai package is not installed. Run: "
                "pip install 'rotary-archive[gemini]'"
            ) from exc

        self._genai = genai
        try:
            self.client = genai.Client()
        except Exception as exc:
            raise ProviderError(
                "Could not construct the Gemini client. Set GEMINI_API_KEY "
                "or GOOGLE_API_KEY."
            ) from exc

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=job.image_path.read_bytes(), mime_type=job.media_type
                    ),
                    job.context or "Catalogue this item.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

        text = getattr(response, "text", "") or ""
        try:
            parsed = extract_json(text)
        except ValueError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=str(exc),
            )

        usage = getattr(response, "usage_metadata", None)
        return AnalysisResult(
            item_id=job.item_id,
            ok=True,
            data=parsed,
            raw=text,
            provider=self.name,
            model=self.model,
            usage={
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            },
        )
