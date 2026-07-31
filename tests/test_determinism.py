"""
Determinism and cross-backend consistency tests.

These tests answer JOSS review issue #424: LLMs are stochastic samplers, so the
same input can yield different generations across runs, and different backends
can yield different generations from the same input. LAiSER addresses this by
(a) defaulting every backend to greedy decoding, and (b) projecting raw
generations onto a fixed taxonomy vocabulary in the alignment stage.

The tests below verify both halves without requiring a GPU, a model download,
or a paid API call:

- Decoding defaults are asserted directly against each backend's signature and,
  for the HTTP backends, against the request payload that would be sent.
- Alignment stability is asserted by running the real alignment path repeatedly
  over a fixed input and requiring byte-identical output.
"""

import importlib
import inspect

import pandas as pd
import pytest

from laiser.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, GENERATION_SEED


def _import_or_skip(module_path):
    """Import a backend module, skipping if its optional dependency is absent.

    Backends such as vLLM, torch, google-genai, and llama-cpp-python are
    optional extras, so a bare CPU install cannot import all of them. This
    helper skips rather than fails in that case. It is used in preference to
    pytest.importorskip because the latter re-raises ImportErrors that are not
    ModuleNotFoundError, which is exactly the shape raised by
    ``from google import genai`` when google-genai is not installed.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"{module_path} unavailable in this environment: {e}")


# ---------------------------------------------------------------------------
# 1. Decoding defaults
# ---------------------------------------------------------------------------


def test_config_defaults_are_deterministic():
    """The shipped configuration must select greedy decoding."""
    assert DEFAULT_TEMPERATURE == 0.0, "Default decoding must be greedy (temperature 0.0)"
    assert DEFAULT_TOP_P == 1.0, "Nucleus filtering must be inactive by default"
    assert GENERATION_SEED is not None, "A default generation seed must be defined"


@pytest.mark.parametrize(
    "module_path, func_name",
    [
        ("laiser.llm_models.openai", "openai_generate"),
        ("laiser.llm_models.anthropic", "anthropic_generate"),
        ("laiser.llm_models.gemini", "gemini_generate"),
        ("laiser.llm_models.hugging_face_llm", "llm_generate"),
        ("laiser.llm_models.hugging_face_llm", "llm_generate_vllm"),
        ("laiser.llm_models.llama_cpp_handler", "llama_cpp_chat"),
    ],
)
def test_backend_defaults_to_zero_temperature(module_path, func_name):
    """Every backend entry point must default to temperature 0.0.

    This is the guard against regression: adding a backend that inherits its
    provider's default temperature (typically 1.0) will fail here.
    """
    module = _import_or_skip(module_path)
    func = getattr(module, func_name)
    params = inspect.signature(func).parameters

    assert "temperature" in params, f"{func_name} does not expose a temperature parameter"
    assert params["temperature"].default == DEFAULT_TEMPERATURE, (
        f"{func_name} defaults to temperature {params['temperature'].default}, "
        f"expected {DEFAULT_TEMPERATURE}"
    )


@pytest.mark.parametrize(
    "module_path, func_name",
    [
        ("laiser.llm_models.gemini", "gemini_generate"),
        ("laiser.llm_models.hugging_face_llm", "llm_generate"),
        ("laiser.llm_models.hugging_face_llm", "llm_generate_vllm"),
        ("laiser.llm_models.llama_cpp_handler", "llama_cpp_chat"),
    ],
)
def test_seedable_backends_default_to_the_configured_seed(module_path, func_name):
    """Backends that accept a sampling seed must default to GENERATION_SEED.

    The OpenAI Responses API and the Anthropic Messages API are excluded because
    neither accepts a seed; on those backends reproducibility rests on greedy
    decoding alone.
    """
    module = _import_or_skip(module_path)
    params = inspect.signature(getattr(module, func_name)).parameters

    assert "seed" in params, f"{func_name} does not expose a seed parameter"
    assert params["seed"].default == GENERATION_SEED


def test_openai_payload_carries_deterministic_decoding(monkeypatch):
    """The OpenAI request body must actually contain temperature 0.0."""
    openai_backend = _import_or_skip("laiser.llm_models.openai")

    captured = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": "ok"}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Response()

    monkeypatch.setattr(openai_backend.requests, "post", _fake_post)
    openai_backend.openai_generate("prompt", api_key="test-key")

    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["top_p"] == DEFAULT_TOP_P


def test_anthropic_payload_carries_deterministic_decoding(monkeypatch):
    """The Anthropic request body must actually contain temperature 0.0."""
    anthropic_backend = _import_or_skip("laiser.llm_models.anthropic")

    captured = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Response()

    monkeypatch.setattr(anthropic_backend.requests, "post", _fake_post)
    anthropic_backend.anthropic_generate("prompt", api_key="test-key")

    assert captured["temperature"] == DEFAULT_TEMPERATURE


def test_router_forwards_deterministic_decoding(monkeypatch):
    """LLMRouter must pass its temperature and seed through to the backend."""
    router_module = _import_or_skip("laiser.llm_models.llm_router")

    captured = {}

    def _fake_vllm(prompt, llm, temperature=None, seed=None, max_tokens=200):
        captured["temperature"] = temperature
        captured["seed"] = seed
        return "ok"

    monkeypatch.setattr(router_module, "llm_generate_vllm", _fake_vllm)
    monkeypatch.setattr(router_module.LLMRouter, "_initialize_components", lambda self: None)

    router = router_module.LLMRouter(model_id="local-model", use_gpu=False)
    router.generate("prompt")

    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["seed"] == GENERATION_SEED


# ---------------------------------------------------------------------------
# 2. Alignment stability
# ---------------------------------------------------------------------------


RAW_SKILLS = [
    "Python programming",
    "python programming skills",
    "Experience programming in Python",
    "statistical analysis",
    "Data visualization techniques",
    "machine learning model development",
]


@pytest.fixture(scope="module")
def alignment_service():
    data_access = _import_or_skip("laiser.data_access")
    services = _import_or_skip("laiser.services")
    DataAccessLayer = data_access.DataAccessLayer
    FAISSIndexManager = data_access.FAISSIndexManager
    SkillAlignmentService = services.SkillAlignmentService

    da = DataAccessLayer()
    fm = FAISSIndexManager(da)
    try:
        fm.initialize_index(force_rebuild=False)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Skipping alignment determinism test: index init failed: {repr(e)}")
    return SkillAlignmentService(data_access=da, faiss_manager=fm)


@pytest.mark.alignment
def test_alignment_is_stable_across_repeated_runs(alignment_service):
    """The alignment stage must be reproducible.

    Retrieval uses an exact inner-product index over deterministic
    sentence-transformer embeddings, so repeated alignment of an identical raw
    skill list must produce an identical frame, including similarity scores.
    """
    runs = [
        alignment_service.align_skills_to_taxonomy(list(RAW_SKILLS), document_id="doc-1")
        for _ in range(3)
    ]

    first = runs[0].reset_index(drop=True)
    for i, other in enumerate(runs[1:], start=2):
        pd.testing.assert_frame_equal(
            first, other.reset_index(drop=True), check_exact=True,
            obj=f"alignment run 1 vs run {i}",
        )


@pytest.mark.alignment
def test_alignment_output_space_is_closed(alignment_service):
    """Aligned output must be drawn from the taxonomy, never from free text.

    This is the mechanism that makes aligned results comparable across
    backends: whatever a model generates, the reported canonical skill is
    always an entry that already exists in the taxonomy.
    """
    aligned = alignment_service.align_skills_to_taxonomy(list(RAW_SKILLS), document_id="doc-2")

    if aligned.empty:  # pragma: no cover - environment dependent
        pytest.skip("Alignment returned no rows; taxonomy index may be unavailable")

    metadata = alignment_service.faiss_manager.get_metadata()
    vocabulary = set(metadata["skill"].astype(str))

    unknown = set(aligned["Taxonomy Skill"]) - vocabulary
    assert not unknown, f"Aligned output contains entries absent from the taxonomy: {unknown}"


@pytest.mark.alignment
def test_identical_raw_skills_map_to_identical_taxonomy_entries(alignment_service):
    """The same raw string must always resolve to the same taxonomy entry.

    Alignment is a pure function of the raw string, so a phrase repeated within
    one document, or emitted by two different backends, cannot produce two
    different canonical skills or two different similarity scores.
    """
    duplicated = ["statistical analysis", "Python programming", "statistical analysis"]
    aligned = alignment_service.align_skills_to_taxonomy(duplicated, document_id="doc-3")

    if aligned.empty:  # pragma: no cover - environment dependent
        pytest.skip("Alignment returned no rows; taxonomy index may be unavailable")

    for raw_skill, group in aligned.groupby("Raw Skill"):
        assert group["Taxonomy Skill"].nunique() == 1, (
            f"Raw skill {raw_skill!r} resolved to multiple taxonomy entries: "
            f"{set(group['Taxonomy Skill'])}"
        )
        assert group["Correlation Coefficient"].nunique() == 1, (
            f"Raw skill {raw_skill!r} produced inconsistent similarity scores: "
            f"{set(group['Correlation Coefficient'])}"
        )
