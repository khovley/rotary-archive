"""Anthropic provider - the default and the reference implementation.

Three features carry the economics of a thousand-item run:

  * **Structured outputs** (`output_config.format`) constrain the response to
    the schema, so nothing downstream parses prose.
  * **Prompt caching** on the system block. The curator prompt and schema are
    byte-identical for every item and sit ahead of the image, so after the
    first call that prefix bills at roughly a tenth of the input rate.
  * **Message Batches** at half price. A first pass over a whole collection is
    not latency-sensitive - you start it and come back later - which is exactly
    what the batch endpoint is for.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator

from .base import AnalysisResult, Job, ProviderError, VisionProvider, extract_json

# Published list prices, USD per million tokens, for the cost estimate only.
# Batch runs bill at half these rates and cached prefix reads at about a tenth
# of the input rate; both are applied in estimate_cost.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# A 1600px rectified crop lands in this range; the system prompt is stable.
EST_IMAGE_TOKENS = 2200
EST_SYSTEM_TOKENS = 1600
EST_OUTPUT_TOKENS = 900

BATCH_POLL_SECONDS = 20
BATCH_MAX_WAIT_SECONDS = 24 * 60 * 60


class AnthropicProvider(VisionProvider):
    name = "anthropic"
    supports_batch = True
    supports_schema = True

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model, options)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The anthropic package is not installed. Run: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        try:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile - an unset API key does not mean there
            # are no credentials.
            self.client = anthropic.Anthropic()
        except Exception as exc:
            raise ProviderError(
                "Could not construct the Anthropic client. Set ANTHROPIC_API_KEY "
                "in your environment, or run `ant auth login`."
            ) from exc

        self.max_tokens = int(self.options.get("max_tokens", 8000))
        self.effort = self.options.get("effort")
        self.use_batch = bool(self.options.get("use_batch", True))

    # ------------------------------------------------------------ requests --

    def _system_blocks(self, system: str) -> list[dict[str, Any]]:
        """System prompt as a cacheable block.

        The breakpoint goes here, at the end of the only part of the request
        that is identical across every item. Everything volatile - the image,
        the per-item context - comes afterwards in the user turn, so the cached
        prefix is never invalidated.
        """
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _request_params(
        self, job: Job, system: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema}
        }
        if self.effort:
            output_config["effort"] = self.effort

        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system_blocks(system),
            "output_config": output_config,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": job.media_type,
                                "data": job.image_b64(),
                            },
                        },
                        {
                            "type": "text",
                            "text": job.context or "Catalogue this item.",
                        },
                    ],
                }
            ],
        }

    # ------------------------------------------------------------ responses --

    @staticmethod
    def _usage(message: Any) -> dict[str, Any]:
        usage = getattr(message, "usage", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(
                usage, "cache_read_input_tokens", None
            ),
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", None
            ),
        }

    def _from_message(self, item_id: str, message: Any) -> AnalysisResult:
        # A refusal returns HTTP 200 with empty or partial content, so this has
        # to be checked before touching content at all.
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return AnalysisResult(
                item_id=item_id,
                ok=False,
                provider=self.name,
                model=self.model,
                usage=self._usage(message),
                error=f"model declined to analyse this item (category: {category})",
            )

        text = "".join(
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        )

        if getattr(message, "stop_reason", None) == "max_tokens" and not text.strip():
            return AnalysisResult(
                item_id=item_id, ok=False, provider=self.name, model=self.model,
                usage=self._usage(message),
                error="hit max_tokens before producing output; raise [llm] max_tokens",
            )

        try:
            data = extract_json(text)
        except ValueError as exc:
            return AnalysisResult(
                item_id=item_id, ok=False, provider=self.name, model=self.model,
                usage=self._usage(message), error=str(exc),
            )

        return AnalysisResult(
            item_id=item_id,
            ok=True,
            data=data,
            raw=text,
            usage=self._usage(message),
            provider=self.name,
            model=self.model,
        )

    # -------------------------------------------------------------- single --

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        try:
            message = self.client.messages.create(**self._request_params(job, system, schema))
        except self._anthropic.APIStatusError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"{exc.status_code}: {getattr(exc, 'message', exc)}",
            )
        except self._anthropic.APIConnectionError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"connection error: {exc}",
            )
        except self._anthropic.AuthenticationError as exc:
            # Affects every item, so stop rather than failing them one by one.
            raise ProviderError(
                f"Authentication failed: {exc}. Set ANTHROPIC_API_KEY or run "
                "`ant auth login`."
            ) from exc
        except TypeError as exc:
            # What the SDK raises when no credential source resolves at all.
            if "authentication" in str(exc).lower():
                raise ProviderError(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY or "
                    "run `ant auth login`."
                ) from exc
            raise
        return self._from_message(job.item_id, message)

    # --------------------------------------------------------------- batch --

    def analyze_many(
        self,
        jobs: Iterable[Job],
        system: str,
        schema: dict[str, Any],
        *,
        max_concurrency: int = 4,
        progress: Any = None,
    ) -> Iterator[AnalysisResult]:
        jobs = list(jobs)
        if not jobs:
            return
        if not self.use_batch:
            yield from super().analyze_many(
                jobs, system, schema,
                max_concurrency=max_concurrency, progress=progress,
            )
            return
        yield from self._run_batch(jobs, system, schema, progress=progress)

    def _run_batch(
        self,
        jobs: list[Job],
        system: str,
        schema: dict[str, Any],
        *,
        progress: Any = None,
    ) -> Iterator[AnalysisResult]:
        requests = [
            {
                "custom_id": job.item_id,
                "params": self._request_params(job, system, schema),
            }
            for job in jobs
        ]

        try:
            batch = self.client.messages.batches.create(requests=requests)
        except Exception as exc:
            # The SDK resolves credentials lazily, so a missing key surfaces
            # here rather than at construction. Failing the whole submission
            # once is right: an auth problem affects every item, and reporting
            # it a thousand times would bury the one thing worth reading.
            raise ProviderError(
                f"Could not submit the batch: {type(exc).__name__}: {exc}\n"
                "If this is an authentication error, set ANTHROPIC_API_KEY or "
                "run `ant auth login`."
            ) from exc
        if progress is not None:
            progress(
                AnalysisResult(
                    item_id="", ok=True, provider=self.name, model=self.model,
                    data={"_status": "submitted", "_batch_id": batch.id,
                          "_count": len(jobs)},
                )
            )

        deadline = time.monotonic() + BATCH_MAX_WAIT_SECONDS
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise ProviderError(
                    f"batch {batch.id} did not finish within 24h; "
                    "retrieve it later with the Anthropic console or SDK"
                )
            if progress is not None:
                counts = getattr(batch, "request_counts", None)
                progress(
                    AnalysisResult(
                        item_id="", ok=True, provider=self.name, model=self.model,
                        data={
                            "_status": "waiting",
                            "_batch_id": batch.id,
                            "_succeeded": getattr(counts, "succeeded", None),
                            "_processing": getattr(counts, "processing", None),
                        },
                    )
                )
            time.sleep(BATCH_POLL_SECONDS)

        # Results arrive in arbitrary order - always key by custom_id, never
        # by position.
        seen: set[str] = set()
        for entry in self.client.messages.batches.results(batch.id):
            item_id = entry.custom_id
            seen.add(item_id)
            kind = entry.result.type

            if kind == "succeeded":
                result = self._from_message(item_id, entry.result.message)
            elif kind == "errored":
                error = getattr(entry.result, "error", None)
                result = AnalysisResult(
                    item_id=item_id, ok=False, provider=self.name, model=self.model,
                    error=f"batch error: {getattr(error, 'type', 'unknown')}",
                )
            else:  # canceled | expired
                result = AnalysisResult(
                    item_id=item_id, ok=False, provider=self.name, model=self.model,
                    error=f"batch request {kind}; resubmit this item",
                )

            if progress is not None:
                progress(result)
            yield result

        # Anything the batch silently dropped must still be accounted for, or
        # the caller would mark it analysed without an analysis.
        for job in jobs:
            if job.item_id not in seen:
                yield AnalysisResult(
                    item_id=job.item_id, ok=False, provider=self.name,
                    model=self.model, error="missing from batch results",
                )

    # ---------------------------------------------------------------- cost --

    def estimate_cost(self, n_items: int) -> dict[str, Any] | None:
        prices = PRICING.get(self.model)
        if not prices or n_items <= 0:
            return None
        input_rate, output_rate = prices

        # The system prefix is written to cache once and read thereafter; the
        # image is fresh input on every request.
        cached_reads = max(0, n_items - 1) * EST_SYSTEM_TOKENS * 0.1
        first_write = EST_SYSTEM_TOKENS * 1.25
        images = n_items * EST_IMAGE_TOKENS
        outputs = n_items * EST_OUTPUT_TOKENS

        cost = (
            (cached_reads + first_write + images) / 1_000_000 * input_rate
            + outputs / 1_000_000 * output_rate
        )
        if self.use_batch:
            cost *= 0.5

        return {
            "items": n_items,
            "model": self.model,
            "batch": self.use_batch,
            "usd": round(cost, 2),
            "usd_per_item": round(cost / n_items, 4),
        }
