"""Decoding determinism and alignment stability.

Language models sample their output, so the same input can produce different
generations across runs and across backends (JOSS review #424). LAiSER limits
this in two ways, and these tests check both without a GPU, a model download or
a paid API call:

- Decoding: every backend defaults to greedy decoding and a fixed seed. Checked
  against each entry point's defaults and against the parameters each backend
  actually receives.
- Alignment: generated phrases are matched to a fixed taxonomy by exact search,
  so repeated alignment of the same phrases returns an identical result.
"""

import importlib
import inspect
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from laiser import config

pytestmark = pytest.mark.determinism


def _import_or_skip(module_path):
    """Import an optional backend module, skipping when its dependency is absent.

    pytest.importorskip only skips on ModuleNotFoundError, but ``from google import
    genai`` without google-genai installed raises a plain ImportError.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"{module_path} unavailable in this environment: {e}")


@pytest.fixture
def reload_config(monkeypatch):
    """Reload laiser.config under a controlled environment, then restore it.

    laiser.config reads LAISER_TEMPERATURE and LAISER_TOP_P at import. Reloading
    lets a test set or clear them without depending on the runner's environment;
    teardown restores the variables and reloads again.
    """
    yield lambda: importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


def test_shipped_defaults_are_greedy(monkeypatch, reload_config):
    monkeypatch.delenv("LAISER_TEMPERATURE", raising=False)
    monkeypatch.delenv("LAISER_TOP_P", raising=False)

    fresh = reload_config()

    assert fresh.DEFAULT_TEMPERATURE == 0.0
    assert fresh.DEFAULT_TOP_P == 1.0
    assert fresh.GENERATION_SEED == 42


def test_environment_can_override_decoding_defaults(monkeypatch, reload_config):
    monkeypatch.setenv("LAISER_TEMPERATURE", "0.7")
    monkeypatch.setenv("LAISER_TOP_P", "0.9")

    fresh = reload_config()

    assert fresh.DEFAULT_TEMPERATURE == 0.7
    assert fresh.DEFAULT_TOP_P == 0.9


# ---------------------------------------------------------------------------
# 2. Entry-point defaults
# ---------------------------------------------------------------------------

SEEDABLE_ENTRY_POINTS = [
    ("laiser.llm_models.gemini", "gemini_generate"),
    ("laiser.llm_models.gemini", "GeminiAPI"),
    ("laiser.llm_models.hugging_face_llm", "llm_generate"),
    ("laiser.llm_models.hugging_face_llm", "llm_generate_vllm"),
    ("laiser.llm_models.llama_cpp_handler", "llama_cpp_chat"),
    ("laiser.llm_models.llama_cpp_handler", "LlamaCppBackend"),
    ("laiser.llm_models.llm_router", "LLMRouter"),
    ("laiser.services", "SkillExtractionService"),
    ("laiser.skill_extractor_refactored", "SkillExtractorRefactored"),
]

# The OpenAI Responses API and the Anthropic Messages API accept no seed.
ENTRY_POINTS = SEEDABLE_ENTRY_POINTS + [
    ("laiser.llm_models.openai", "openai_generate"),
    ("laiser.llm_models.anthropic", "anthropic_generate"),
]


@pytest.mark.parametrize("module_path, name", ENTRY_POINTS)
def test_entry_point_defaults_to_greedy_decoding(module_path, name):
    """Guards against a backend inheriting its provider's default temperature, usually 1.0."""
    params = inspect.signature(getattr(_import_or_skip(module_path), name)).parameters

    assert "temperature" in params, f"{name} has no temperature parameter"
    assert params["temperature"].default == config.DEFAULT_TEMPERATURE


@pytest.mark.parametrize("module_path, name", SEEDABLE_ENTRY_POINTS)
def test_seedable_entry_point_defaults_to_generation_seed(module_path, name):
    params = inspect.signature(getattr(_import_or_skip(module_path), name)).parameters

    assert "seed" in params, f"{name} has no seed parameter"
    assert params["seed"].default == config.GENERATION_SEED


# ---------------------------------------------------------------------------
# 3. Decoding parameters each backend actually receives
# ---------------------------------------------------------------------------


class _JSONResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _capture_post(monkeypatch, module, body):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json or {})
        return _JSONResponse(body)

    monkeypatch.setattr(module.requests, "post", fake_post)
    return sent


def test_openai_request_uses_greedy_decoding(monkeypatch):
    openai_backend = _import_or_skip("laiser.llm_models.openai")
    sent = _capture_post(monkeypatch, openai_backend, {"output_text": "ok"})

    openai_backend.openai_generate("prompt", api_key="test-key")

    assert sent["temperature"] == config.DEFAULT_TEMPERATURE
    assert sent["top_p"] == config.DEFAULT_TOP_P
    assert "seed" not in sent


def test_anthropic_request_uses_greedy_decoding(monkeypatch):
    anthropic_backend = _import_or_skip("laiser.llm_models.anthropic")
    sent = _capture_post(monkeypatch, anthropic_backend, {"content": [{"type": "text", "text": "ok"}]})

    anthropic_backend.anthropic_generate("prompt", api_key="test-key")

    assert sent["temperature"] == config.DEFAULT_TEMPERATURE
    # top_p is only sent when it has been moved off 1.0.
    assert ("top_p" in sent) == (config.DEFAULT_TOP_P != 1.0)


def test_gemini_request_uses_greedy_decoding_and_seed(monkeypatch):
    gemini = _import_or_skip("laiser.llm_models.gemini")
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["config"] = config
            return SimpleNamespace(text="ok", parsed=None)

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(gemini.genai, "Client", FakeClient)

    gemini.gemini_generate("prompt", api_key="test-key")

    sent = captured["config"]
    assert sent.temperature == config.DEFAULT_TEMPERATURE
    assert sent.max_output_tokens == gemini.DEFAULT_MAX_OUTPUT_TOKENS
    if gemini._SUPPORTS_SEED:
        assert sent.seed == config.GENERATION_SEED


class _Encoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text, return_tensors):
        return _Encoded(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        return "ok"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


def test_transformers_at_temperature_zero_turns_sampling_off(monkeypatch):
    hf = _import_or_skip("laiser.llm_models.hugging_face_llm")
    seeded = []
    monkeypatch.setattr(hf.torch, "manual_seed", seeded.append)
    model = _FakeModel()

    hf.llm_generate("prompt", _FakeTokenizer(), model, "some/model", use_gpu=False, temperature=0.0)

    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs
    assert seeded == []  # nothing is sampled, so there is nothing to seed


def test_transformers_sampling_is_seeded(monkeypatch):
    hf = _import_or_skip("laiser.llm_models.hugging_face_llm")
    seeded = []
    monkeypatch.setattr(hf.torch, "manual_seed", seeded.append)
    model = _FakeModel()

    hf.llm_generate("prompt", _FakeTokenizer(), model, "some/model", use_gpu=False, temperature=0.7, seed=123)

    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.7
    assert seeded == [123]


def test_vllm_sampling_params_are_greedy_and_seeded(monkeypatch):
    hf = _import_or_skip("laiser.llm_models.hugging_face_llm")
    recorded = {}
    monkeypatch.setattr(hf, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(hf, "SamplingParams", lambda **kwargs: recorded.update(kwargs) or kwargs)
    engine = SimpleNamespace(
        generate=lambda prompts, sampling_params: [SimpleNamespace(outputs=[SimpleNamespace(text=" ok ")])]
    )

    assert hf.llm_generate_vllm("prompt", engine) == "ok"
    assert recorded["temperature"] == config.DEFAULT_TEMPERATURE
    assert recorded["top_p"] == config.DEFAULT_TOP_P
    assert recorded["seed"] == config.GENERATION_SEED


class _SeedAwareLlama:
    def __init__(self):
        self.kwargs = None

    def create_chat_completion(self, messages, max_tokens=None, stop=None, temperature=None, top_p=None, seed=None):
        self.kwargs = {"temperature": temperature, "top_p": top_p, "seed": seed}
        return {"choices": [{"message": {"content": "ok"}}]}


class _LlamaWithoutSeed:
    def __init__(self):
        self.kwargs = None

    def create_chat_completion(self, messages, max_tokens=None, stop=None, temperature=None, top_p=None):
        self.kwargs = {"temperature": temperature, "top_p": top_p}
        return {"choices": [{"message": {"content": "ok"}}]}


def test_llama_cpp_passes_greedy_decoding_and_seed():
    llama_cpp = _import_or_skip("laiser.llm_models.llama_cpp_handler")
    llama = _SeedAwareLlama()

    llama_cpp.llama_cpp_chat("prompt", llama)

    assert llama.kwargs == {
        "temperature": config.DEFAULT_TEMPERATURE,
        "top_p": config.DEFAULT_TOP_P,
        "seed": config.GENERATION_SEED,
    }


def test_llama_cpp_omits_seed_on_builds_that_do_not_accept_one():
    llama_cpp = _import_or_skip("laiser.llm_models.llama_cpp_handler")
    llama = _LlamaWithoutSeed()

    llama_cpp.llama_cpp_chat("prompt", llama)  # would raise TypeError if a seed were sent

    assert llama.kwargs["temperature"] == config.DEFAULT_TEMPERATURE


# ---------------------------------------------------------------------------
# 4. Forwarding through the router and the public extractor
# ---------------------------------------------------------------------------

ROUTES = [
    # (route, router attributes, function the router calls, whether it takes a seed)
    ("vllm", {"llm": "engine"}, "llm_generate_vllm", True),
    ("transformers", {"model": "model", "tokenizer": "tokenizer"}, "llm_generate", True),
    ("llama_cpp", {"backend": "llama_cpp", "llm": "engine"}, "llama_cpp_chat", True),
    ("gemini", {"model_id": "gemini"}, "gemini_generate", True),
    ("openai", {"model_id": "openai"}, "openai_generate", False),
]


def _router(monkeypatch, attrs):
    router_module = _import_or_skip("laiser.llm_models.llm_router")
    monkeypatch.setattr(router_module.LLMRouter, "_initialize_components", lambda self: None)
    attrs = dict(attrs)
    router = router_module.LLMRouter(
        attrs.pop("model_id", "some/model"), use_gpu=False, backend=attrs.pop("backend", None)
    )
    for key, value in attrs.items():
        setattr(router, key, value)
    return router_module, router


def _record_calls(monkeypatch, module, name):
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(module, name, record)
    return calls


@pytest.mark.parametrize("route, attrs, target, takes_seed", ROUTES, ids=[r[0] for r in ROUTES])
def test_router_forwards_decoding_defaults(monkeypatch, route, attrs, target, takes_seed):
    router_module, router = _router(monkeypatch, attrs)
    calls = _record_calls(monkeypatch, router_module, target)

    router.generate("prompt")

    assert calls[0]["temperature"] == config.DEFAULT_TEMPERATURE
    if takes_seed:
        assert calls[0]["seed"] == config.GENERATION_SEED


@pytest.mark.parametrize("route, attrs, target, takes_seed", ROUTES, ids=[r[0] for r in ROUTES])
def test_router_honours_per_call_overrides(monkeypatch, route, attrs, target, takes_seed):
    """Review of #426 found llama.cpp ignored a caller's seed; overrides must reach every backend."""
    router_module, router = _router(monkeypatch, attrs)
    calls = _record_calls(monkeypatch, router_module, target)

    router.generate("prompt", temperature=0.5, seed=7)

    assert calls[0]["temperature"] == 0.5
    if takes_seed:
        assert calls[0]["seed"] == 7


def test_extractor_passes_decoding_settings_to_the_router(monkeypatch):
    services = _import_or_skip("laiser.services")
    extractor_module = _import_or_skip("laiser.skill_extractor_refactored")

    class NoOp:
        def __init__(self, *args, **kwargs):
            pass

        def initialize_index(self, *args, **kwargs):
            return None

    for name in ("DataAccessLayer", "FAISSIndexManager", "KnowledgeFAISSIndexManager", "TaskFAISSIndexManager"):
        monkeypatch.setattr(services, name, NoOp)
    monkeypatch.setattr(services, "AlignmentService", NoOp)

    received = {}

    class RecordingRouter:
        def __init__(self, *args, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(services, "LLMRouter", RecordingRouter)

    extractor_module.SkillExtractorRefactored(model_id="some/model", use_gpu=False, temperature=0.3, seed=9)

    assert received["temperature"] == 0.3
    assert received["seed"] == 9


# ---------------------------------------------------------------------------
# 5. Alignment stability
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

    da = data_access.DataAccessLayer()
    fm = data_access.FAISSIndexManager(da)
    try:
        fm.initialize_index(force_rebuild=False)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Taxonomy index unavailable: {e!r}")
    return services.SkillAlignmentService(data_access=da, faiss_manager=fm)


@pytest.mark.alignment
def test_alignment_is_identical_across_repeated_runs(alignment_service):
    """Exact search over fixed embeddings: repeated runs must match, scores included."""
    runs = [alignment_service.align_skills_to_taxonomy(list(RAW_SKILLS), document_id="doc-1") for _ in range(3)]

    first = runs[0].reset_index(drop=True)
    for i, other in enumerate(runs[1:], start=2):
        pd.testing.assert_frame_equal(
            first, other.reset_index(drop=True), check_exact=True, obj=f"alignment run 1 vs run {i}"
        )


@pytest.mark.alignment
def test_aligned_output_is_drawn_from_the_taxonomy(alignment_service):
    """Whatever a model generates, the reported entry already exists in the taxonomy."""
    aligned = alignment_service.align_skills_to_taxonomy(list(RAW_SKILLS), document_id="doc-2")
    if aligned.empty:  # pragma: no cover - environment dependent
        pytest.skip("Alignment returned no rows")

    vocabulary = set(alignment_service.faiss_manager.get_metadata()["skill"].astype(str).str.strip())

    unknown = set(aligned["Taxonomy Skill"]) - vocabulary
    assert not unknown, f"Aligned output contains entries absent from the taxonomy: {unknown}"


@pytest.mark.alignment
def test_the_same_phrase_always_resolves_to_the_same_entry(alignment_service):
    aligned = alignment_service.align_skills_to_taxonomy(
        ["statistical analysis", "Python programming", "statistical analysis"], document_id="doc-3"
    )
    if aligned.empty:  # pragma: no cover - environment dependent
        pytest.skip("Alignment returned no rows")

    for raw, group in aligned.groupby("Raw Skill"):
        assert group["Taxonomy Skill"].nunique() == 1, f"{raw!r} resolved to several entries"
        assert group["Correlation Coefficient"].nunique() == 1, f"{raw!r} produced different scores"
