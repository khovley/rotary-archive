"""Ollama provider - a local, zero-cost option.

The reason this exists: a volunteer club may have no budget and no appetite for
sending its members' photographs to a third party. A vision model running on
someone's laptop is slower and less accurate than a frontier model, but it is
free, private, and good enough for a first pass that a human then corrects.

Talks to Ollama's HTTP API directly over the standard library, so selecting
this provider adds no dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import AnalysisResult, Job, VisionProvider, extract_json

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaProvider(VisionProvider):
    name = "ollama"
    supports_batch = False
    supports_schema = True   # Ollama accepts a JSON schema as `format`

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model, options)
        self.host = str(self.options.get("ollama_host", DEFAULT_HOST)).rstrip("/")
        # Local models are slow; a minute per item is not unusual on a laptop.
        self.timeout = float(self.options.get("ollama_timeout", 300))

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": job.context or "Catalogue this item.",
            "images": [job.image_b64()],
            "format": schema,
            "stream": False,
            "options": {"num_predict": int(self.options.get("max_tokens", 4096))},
        }

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=(
                    f"could not reach Ollama at {self.host} ({exc}). "
                    "Is `ollama serve` running?"
                ),
            )
        except Exception as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            parsed = extract_json(body.get("response", ""))
        except ValueError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=str(exc),
            )

        return AnalysisResult(
            item_id=job.item_id,
            ok=True,
            data=parsed,
            raw=body.get("response"),
            provider=self.name,
            model=self.model,
            usage={
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
            },
        )

    def estimate_cost(self, n_items: int) -> dict[str, Any] | None:
        return {"items": n_items, "model": self.model, "usd": 0.0, "local": True}
