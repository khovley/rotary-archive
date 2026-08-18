"""A scriptable provider for tests.

Lets the whole analyse stage be exercised deterministically and for free -
including the failure paths that are awkward to provoke against a real API:
refusals, malformed JSON, missing fields, and items dropped from a batch.
"""

from __future__ import annotations

from typing import Any, Callable

from rotary_archive.providers.base import AnalysisResult, Job, VisionProvider

GOOD_RESPONSE: dict[str, Any] = {
    "item_type": "newspaper_clipping",
    "title": "Rotary Club Funds New Library Wing",
    "summary": "The club presented a cheque for the new reading room.",
    "full_text": "ROTARY CLUB FUNDS NEW LIBRARY WING\n\nThe Rotary Club of "
                 "Brookfield presented a cheque for $4,200 to the town library.",
    "date_value": "1962-07-14",
    "date_precision": "day",
    "date_source": "printed",
    "date_note": "",
    "people": ["Harold Pratt", "Eleanor Voss"],
    "organizations": ["Rotary Club of Brookfield", "Brookfield Town Library"],
    "places": ["Brookfield"],
    "topics": ["fundraising", "libraries", "community service"],
    "rotary_context": "A club service project reported in the local press.",
    "presentation": "text",
    "legibility": 5,
    "condition_notes": "",
    "alt_text": "A newspaper clipping headlined 'Rotary Club Funds New "
                "Library Wing'.",
    "orientation_hint": "upright",
    "confidence": 0.93,
    "needs_human_review": False,
    "review_reason": "",
}


class FakeProvider(VisionProvider):
    """Returns scripted results. `responder` maps a Job to a dict or an
    AnalysisResult, so a test can vary behaviour per item."""

    name = "fake"
    supports_batch = False
    supports_schema = True

    def __init__(
        self,
        model: str = "fake-1",
        options: dict[str, Any] | None = None,
        responder: Callable[[Job], Any] | None = None,
    ) -> None:
        super().__init__(model, options)
        self.responder = responder or (lambda job: dict(GOOD_RESPONSE))
        self.calls: list[Job] = []
        self.systems: list[str] = []

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        self.calls.append(job)
        self.systems.append(system)

        outcome = self.responder(job)
        if isinstance(outcome, AnalysisResult):
            return outcome
        if isinstance(outcome, Exception):
            raise outcome

        return AnalysisResult(
            item_id=job.item_id,
            ok=True,
            data=outcome,
            raw=outcome,
            provider=self.name,
            model=self.model,
            usage={"input_tokens": 2000, "output_tokens": 700},
        )


class FakeBatchProvider(FakeProvider):
    """Exercises the batch path, including results arriving out of order and
    an item silently dropped."""

    name = "fake_batch"
    supports_batch = True

    def __init__(self, *args: Any, drop: set[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.drop = drop or set()

    def analyze_many(
        self, jobs, system, schema, *, max_concurrency=4, progress=None
    ):
        jobs = list(jobs)
        results = [
            self._guarded(job, system, schema)
            for job in jobs
            if job.item_id not in self.drop
        ]
        # Batch results come back in arbitrary order; reversing here makes any
        # positional assumption in the caller fail loudly.
        for result in reversed(results):
            if progress is not None:
                progress(result)
            yield result

        for job in jobs:
            if job.item_id in self.drop:
                yield AnalysisResult(
                    item_id=job.item_id, ok=False, provider=self.name,
                    model=self.model, error="missing from batch results",
                )
