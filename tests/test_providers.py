"""Provider layer: the contract every backend must honour."""

from __future__ import annotations

import pytest

from fake_provider import GOOD_RESPONSE, FakeProvider
from rotary_archive.providers.base import (
    AnalysisResult,
    Job,
    ProviderError,
    VisionProvider,
    build_provider,
)


def make_job(tmp_path, name="item-00"):
    path = tmp_path / f"{name}.webp"
    path.write_bytes(b"not-really-an-image")
    return Job(item_id=name, image_path=path)


# ------------------------------------------------------------------ jobs ----


@pytest.mark.parametrize(
    "suffix,expected",
    [
        (".webp", "image/webp"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".png", "image/png"),
        (".bmp", "image/jpeg"),   # unknown extensions fall back
    ],
)
def test_job_media_type(tmp_path, suffix, expected):
    path = tmp_path / f"x{suffix}"
    path.write_bytes(b"data")
    assert Job(item_id="x", image_path=path).media_type == expected


def test_job_base64_round_trips(tmp_path):
    import base64

    path = tmp_path / "x.webp"
    path.write_bytes(b"hello archive")
    encoded = Job(item_id="x", image_path=path).image_b64()
    assert base64.standard_b64decode(encoded) == b"hello archive"


# ---------------------------------------------------------- error handling --


def test_per_item_failure_does_not_escape(tmp_path):
    """A single bad item must come back as a failed result, not an exception -
    otherwise one unreadable file aborts a thousand-item run."""
    provider = FakeProvider(responder=lambda job: RuntimeError("boom"))
    results = list(
        provider.analyze_many([make_job(tmp_path)], "sys", {}, max_concurrency=1)
    )
    assert len(results) == 1
    assert results[0].ok is False
    assert "boom" in results[0].error


def test_provider_error_propagates(tmp_path):
    """ProviderError signals a whole-run condition such as bad credentials.
    Converting it to a per-item failure would repeat one message a thousand
    times and bury it."""
    provider = FakeProvider(responder=lambda job: ProviderError("no credentials"))
    with pytest.raises(ProviderError):
        list(provider.analyze_many([make_job(tmp_path)], "sys", {}, max_concurrency=1))


def test_analyze_many_handles_an_empty_job_list():
    assert list(FakeProvider().analyze_many([], "sys", {})) == []


def test_analyze_many_returns_one_result_per_job(tmp_path):
    jobs = [make_job(tmp_path, f"item-{i:02d}") for i in range(7)]
    results = list(FakeProvider().analyze_many(jobs, "sys", {}, max_concurrency=3))
    assert {r.item_id for r in results} == {j.item_id for j in jobs}


def test_system_prompt_is_identical_for_every_item(tmp_path):
    """The cached prefix only pays off if it is byte-identical across items.
    A per-item system prompt would silently disable prompt caching."""
    provider = FakeProvider()
    jobs = [make_job(tmp_path, f"item-{i:02d}") for i in range(4)]
    list(provider.analyze_many(jobs, "THE SYSTEM PROMPT", {}, max_concurrency=1))
    assert len(set(provider.systems)) == 1


# ---------------------------------------------------------------- registry --


def test_build_provider_rejects_an_unknown_name():
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_provider({"provider": "nonesuch", "model": "x"})


def test_build_provider_constructs_anthropic_by_default():
    from rotary_archive.providers.anthropic_provider import AnthropicProvider

    provider = build_provider({"model": "claude-opus-5"})
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5"


def test_build_provider_reports_a_missing_optional_sdk():
    """Selecting a provider whose SDK is absent must say so, not raise
    ImportError from somewhere deep in the call stack."""
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ProviderError, match="not installed"):
            build_provider({"provider": "openai", "model": "gpt-4o"})


# -------------------------------------------------------------------- cost --


def _pricer(model: str, batch: bool):
    from rotary_archive.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model, provider.use_batch, provider.options = model, batch, {}
    return provider


def test_batch_pricing_is_half_of_synchronous():
    assert _pricer("claude-opus-5", True).estimate_cost(500)["usd"] == pytest.approx(
        _pricer("claude-opus-5", False).estimate_cost(500)["usd"] / 2, rel=0.01
    )


def test_cost_scales_with_item_count():
    single = _pricer("claude-opus-5", True).estimate_cost(100)["usd"]
    tenfold = _pricer("claude-opus-5", True).estimate_cost(1000)["usd"]
    assert tenfold > single * 9  # not exactly 10x - the cache write is one-off


def test_cheaper_models_estimate_lower():
    opus = _pricer("claude-opus-5", True).estimate_cost(500)["usd"]
    haiku = _pricer("claude-haiku-4-5", True).estimate_cost(500)["usd"]
    assert haiku < opus


def test_unknown_model_has_no_estimate():
    assert _pricer("some-future-model", True).estimate_cost(500) is None


def test_zero_items_has_no_estimate():
    assert _pricer("claude-opus-5", True).estimate_cost(0) is None


def test_local_provider_reports_free(tmp_path):
    from rotary_archive.providers.ollama_provider import OllamaProvider

    estimate = OllamaProvider("llava").estimate_cost(500)
    assert estimate["usd"] == 0.0
    assert estimate["local"] is True


# --------------------------------------------------------------- interface --


def test_every_provider_declares_its_capabilities():
    """Callers branch on these, so a backend that forgot to set them would
    silently get the wrong treatment."""
    from rotary_archive.providers import (
        anthropic_provider, gemini_provider, ollama_provider, openai_provider,
    )

    classes = [
        anthropic_provider.AnthropicProvider,
        openai_provider.OpenAIProvider,
        gemini_provider.GeminiProvider,
        ollama_provider.OllamaProvider,
    ]
    for cls in classes:
        assert issubclass(cls, VisionProvider)
        assert isinstance(cls.name, str) and cls.name
        assert isinstance(cls.supports_batch, bool)
        assert isinstance(cls.supports_schema, bool)


def test_fake_provider_returns_a_well_formed_result(tmp_path):
    result = FakeProvider().analyze(make_job(tmp_path), "sys", {})
    assert isinstance(result, AnalysisResult)
    assert result.ok
    assert result.data["title"] == GOOD_RESPONSE["title"]


# ------------------------------------------------------ per-stage models ----


def test_stage_override_picks_a_different_model():
    """Segmentation runs once per photograph and carries the hard judgements;
    cataloguing runs once per item and is mostly transcription. Being able to
    spend on the few and economise on the many is the point."""
    config = {
        "provider": "claude_cli",
        "model": "claude-sonnet-5",
        "analyze_model": "claude-haiku-4-5",
    }
    assert build_provider(config, stage="segment").model == "claude-sonnet-5"
    assert build_provider(config, stage="analyze").model == "claude-haiku-4-5"


def test_no_override_falls_back_to_the_shared_model():
    config = {"provider": "claude_cli", "model": "claude-sonnet-5"}
    for stage in (None, "segment", "analyze"):
        assert build_provider(config, stage=stage).model == "claude-sonnet-5"


def test_an_empty_override_is_ignored_rather_than_used_as_a_model_name():
    """A commented-out or blank key in config.toml must not become the model."""
    config = {"provider": "claude_cli", "model": "claude-sonnet-5", "segment_model": ""}
    assert build_provider(config, stage="segment").model == "claude-sonnet-5"
