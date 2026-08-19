"""Claude CLI provider - runs analysis through a Claude subscription.

Shells out to the `claude` CLI instead of the API, so a run costs nothing
beyond an existing subscription. The trade-offs are real and worth stating:
it is serial, noticeably slower per item, has no batch discount, and the CLI
returns prose that has to be parsed rather than a schema-constrained response.

Reasonable for a few dozen items or for trying the pipeline out before
committing to API spend. For a full collection, the Anthropic provider with
batching is both faster and more reliable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from .base import AnalysisResult, Job, ProviderError, VisionProvider, extract_json


class ClaudeCLIProvider(VisionProvider):
    name = "claude_cli"
    supports_batch = False
    supports_schema = False   # no structured-output enforcement over the CLI

    def __init__(self, model: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model, options)
        self.executable = str(self.options.get("claude_executable", "claude"))
        if shutil.which(self.executable) is None:
            raise ProviderError(
                f"`{self.executable}` is not on PATH. Install Claude Code, or "
                "set [llm] provider to 'anthropic'."
            )
        self.timeout = float(self.options.get("cli_timeout", 300))

    def analyze(self, job: Job, system: str, schema: dict[str, Any]) -> AnalysisResult:
        # The CLI has no system-prompt/image-attachment API, so the schema and
        # the image path both go in the prompt text and the model is asked to
        # read the file itself.
        prompt = (
            f"{system}\n\n"
            f"Read the image at this absolute path: {job.image_path}\n\n"
            f"{job.context or 'Catalogue this item.'}\n\n"
            "Respond with a single JSON object matching this schema, and "
            "nothing else - no commentary, no code fence:\n"
            f"{json.dumps(schema)}"
        )

        try:
            completed = subprocess.run(
                [
                    self.executable, "-p", prompt, "--model", self.model,
                    # The nested CLI needs exactly one capability: opening the
                    # image it is being asked about. Left unrestricted it
                    # treats the request as a task to work on, goes looking for
                    # the project, tries to run a script, hits a permission
                    # prompt it cannot answer, and reports that instead of
                    # returning the JSON - so the photo comes back with no
                    # items and no error anyone can act on.
                    "--allowed-tools", "Read",
                    "--permission-mode", "acceptEdits",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"claude CLI timed out after {self.timeout:.0f}s",
            )
        except Exception as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

        if completed.returncode != 0:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=f"claude CLI exited {completed.returncode}: "
                      f"{completed.stderr.strip()[:200]}",
            )

        try:
            parsed = extract_json(completed.stdout)
        except ValueError as exc:
            return AnalysisResult(
                item_id=job.item_id, ok=False, provider=self.name, model=self.model,
                error=str(exc),
            )

        return AnalysisResult(
            item_id=job.item_id,
            ok=True,
            data=parsed,
            raw=completed.stdout,
            provider=self.name,
            model=self.model,
        )

    def analyze_many(self, jobs, system, schema, *, max_concurrency=4, progress=None):
        # One CLI process at a time. Running several would contend for the same
        # subscription rate limit and produce failures rather than throughput.
        return super().analyze_many(
            jobs, system, schema, max_concurrency=1, progress=progress
        )

    def estimate_cost(self, n_items: int) -> dict[str, Any] | None:
        return {
            "items": n_items,
            "model": self.model,
            "usd": 0.0,
            "note": "billed to your Claude subscription, not per token",
        }
