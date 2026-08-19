"""Provider abstraction.

One interface, several backends. The club may not stay on Claude, and a local
option matters for a volunteer organisation with no budget, so nothing above
this layer knows which model is answering.

Two capabilities are negotiated rather than assumed:

  * `supports_batch` - some providers offer a genuine asynchronous batch
    endpoint at a discount. Those that do not get a threaded fallback here, so
    callers always use the same method.
  * `supports_schema` - some providers can be constrained to a JSON schema.
    Those that cannot get the schema described in the prompt instead, and their
    output is validated on the way back.
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

MEDIA_TYPES = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
}


class ProviderError(RuntimeError):
    """Configuration or transport failure. Not a per-item analysis failure."""


@dataclass
class Job:
    """One item to analyse."""

    item_id: str
    image_path: Path
    context: str = ""          # per-item hints (capture date, neighbours)

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES.get(self.image_path.suffix.lower(), "image/jpeg")

    def image_b64(self) -> str:
        return base64.standard_b64encode(self.image_path.read_bytes()).decode()


@dataclass
class AnalysisResult:
    """Outcome for one item. `ok` false means this item failed, not the run."""

    item_id: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    error: str | None = None


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Providers without schema enforcement wrap JSON in prose or fences however
    they like. Tries the whole string first, then a fenced block, then the
    outermost brace-balanced span - scanning for balance rather than regex
    matching, because a transcription field routinely contains braces of its
    own.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : index + 1])

    raise ValueError(f"no JSON object in response: {text[:200]}")


class VisionProvider(ABC):
    """Analyse an image against a JSON schema."""

    name: str = "base"
    supports_batch: bool = False
    supports_schema: bool = True

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    @abstractmethod
    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        """Analyse a single item. Must not raise for per-item failures."""

    def analyze_many(
        self,
        jobs: Iterable[Job],
        system: str,
        schema: dict[str, Any],
        *,
        max_concurrency: int = 4,
        progress: Any = None,
    ) -> Iterator[AnalysisResult]:
        """Analyse many items.

        The default is a bounded thread pool, which is the right shape for
        providers with only a synchronous endpoint. Providers with a real batch
        API override this to use it - the discount and the absence of
        rate-limit babysitting are worth the extra code there.
        """
        jobs = list(jobs)
        if not jobs:
            return

        workers = max(1, int(max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(
                lambda job: self._guarded(job, system, schema), jobs
            ):
                if progress is not None:
                    progress(result)
                yield result

    def _guarded(
        self, job: Job, system: str, schema: dict[str, Any]
    ) -> AnalysisResult:
        """One item's failure must never abort a thousand-item run.

        ProviderError is the deliberate exception. It signals a condition that
        applies to every item - bad credentials, an unreachable endpoint - so
        letting it propagate stops the run once, instead of reporting the same
        problem a thousand times and burying it.
        """
        try:
            return self.analyze(job, system, schema)
        except ProviderError:
            raise
        except Exception as exc:
            return AnalysisResult(
                item_id=job.item_id,
                ok=False,
                provider=self.name,
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

    def estimate_cost(self, n_items: int) -> dict[str, Any] | None:
        """Rough spend for `n_items`, or None if the provider is free or
        cannot be priced. Shown before a run so nobody is surprised."""
        return None


# ----------------------------------------------------------------- registry ---


def build_provider(
    llm_config: dict[str, Any], *, stage: str | None = None
) -> VisionProvider:
    """Construct the configured provider.

    Imports lazily so that a missing optional SDK only matters to the person
    who actually selected that provider.

    `stage` picks up an optional per-stage model override. The two stages have
    very different economics: segmentation runs once per *photograph* - perhaps
    a hundred calls for a whole collection - and carries the hard reasoning
    about which pieces belong together. Cataloguing runs once per *item*, ten
    times as often, and is mostly transcription. Being able to spend on the
    scarce hard calls and economise on the many easy ones is worth more than
    picking one model for both.
    """
    name = str(llm_config.get("provider", "anthropic")).lower()
    model = str(llm_config.get("model", "claude-opus-5"))
    if stage:
        model = str(llm_config.get(f"{stage}_model") or model)

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, llm_config)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, llm_config)
    if name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(model, llm_config)
    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(model, llm_config)
    if name == "claude_cli":
        from .claude_cli_provider import ClaudeCLIProvider

        return ClaudeCLIProvider(model, llm_config)

    raise ProviderError(
        f"Unknown provider {name!r}. Set [llm] provider in config.toml to one "
        "of: anthropic, openai, gemini, ollama, claude_cli."
    )
