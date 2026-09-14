"""Offline tests for local-model routing. No model weights are downloaded."""

import pytest
import torch

from laiser.exceptions import LAiSERError
from laiser.llm_models import hugging_face_llm, llm_router, model_loader
from laiser.llm_models.llm_router import LLMRouter


class _Encoded(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.seen_text = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "CHAT:" + messages[0]["content"]

    def __call__(self, text, return_tensors):
        self.seen_text = text
        return _Encoded(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        return " ".join(str(i) for i in ids.tolist())


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        # The prompt tokens (1, 2, 3) come back first, then the completion.
        return torch.tensor([[1, 2, 3, 7, 8]])


def _bare_router(**attrs):
    """An LLMRouter with no model loading, so routing can be tested in isolation."""
    router = LLMRouter.__new__(LLMRouter)
    router.model_id = "some/model"
    router.use_gpu = False
    router.hf_token = router.api_key = router.backend = None
    router.llm = router.model = router.tokenizer = router.nlp = None
    for key, value in attrs.items():
        setattr(router, key, value)
    return router


def test_transformers_model_is_routed_to_transformers_generate(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_router, "llm_generate", lambda *args, **kwargs: calls.append(args) or "ok")
    monkeypatch.setattr(llm_router, "llm_generate_vllm", lambda *args, **kwargs: pytest.fail("routed to vLLM"))

    router = _bare_router(tokenizer=FakeTokenizer(), model=FakeModel())

    assert router.generate("prompt") == "ok"
    assert calls and calls[0][0] == "prompt"


def test_vllm_engine_still_routes_to_vllm(monkeypatch):
    monkeypatch.setattr(llm_router, "llm_generate", lambda *args, **kwargs: pytest.fail("routed to transformers"))
    monkeypatch.setattr(llm_router, "llm_generate_vllm", lambda prompt, llm: f"vllm:{llm}")

    assert _bare_router(llm="engine").generate("prompt") == "vllm:engine"


def test_llm_generate_returns_only_the_completion():
    model = FakeModel()

    result = hugging_face_llm.llm_generate("p", FakeTokenizer(), model, "some/model", use_gpu=False)

    assert result == "7 8"
    assert model.generate_kwargs["max_new_tokens"] == hugging_face_llm.MAX_NEW_TOKENS


def test_llm_generate_applies_chat_template_when_present():
    tokenizer = FakeTokenizer(chat_template="{{ template }}")

    hugging_face_llm.llm_generate("describe", tokenizer, FakeModel(), "some/model", use_gpu=False)

    assert tokenizer.seen_text == "CHAT:describe"


@pytest.fixture
def recorded_loads(monkeypatch):
    loads = []

    def fake_model_from_pretrained(name, **kwargs):
        loads.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", lambda name, **kwargs: FakeTokenizer())
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(model_loader, "BitsAndBytesConfig", lambda **kwargs: "8bit")
    return loads


def test_cpu_load_skips_quantization(monkeypatch, recorded_loads):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model_loader.load_model_from_transformer("some/model", use_gpu=True)

    assert "quantization_config" not in recorded_loads[0][1]


def test_gpu_load_keeps_quantization(monkeypatch, recorded_loads):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model_loader.load_model_from_transformer("some/model", use_gpu=True)

    assert recorded_loads[0][1]["quantization_config"] == "8bit"


def test_cpu_load_failure_raises_instead_of_falling_back(monkeypatch):
    attempted = []

    def failing_from_pretrained(name, **kwargs):
        attempted.append(name)
        raise OSError("not found")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", failing_from_pretrained)

    with pytest.raises(OSError):
        model_loader.load_model_from_transformer("missing/model", use_gpu=False)
    assert attempted == ["missing/model"]


# The tests below go through LLMRouter's constructor, which is what
# SkillExtractorRefactored builds, rather than the loader alone.


def _failing_load(*args, **kwargs):
    raise OSError("model could not be loaded")


def test_cpu_router_raises_when_the_model_cannot_load(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(llm_router, "load_model_from_transformer", _failing_load)

    with pytest.raises(LAiSERError, match="model could not be loaded"):
        LLMRouter("missing/model", use_gpu=False)


def test_gpu_router_raises_when_vllm_and_transformers_both_fail(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(llm_router, "load_model_from_vllm", _failing_load)
    monkeypatch.setattr(llm_router, "load_model_from_transformer", _failing_load)

    with pytest.raises(LAiSERError, match="model could not be loaded"):
        LLMRouter("missing/model", use_gpu=True)


def test_gpu_router_falls_back_to_transformers_when_vllm_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(llm_router, "load_model_from_vllm", _failing_load)
    monkeypatch.setattr(llm_router, "load_model_from_transformer", lambda *a, **k: (FakeTokenizer(), FakeModel()))

    router = LLMRouter("some/model", use_gpu=True)

    assert router.llm is None and router.model is not None


def test_hosted_provider_loads_no_local_weights(monkeypatch):
    monkeypatch.setattr(llm_router, "load_model_from_transformer", lambda *a, **k: pytest.fail("loaded local weights"))

    router = LLMRouter("openai", use_gpu=False)

    assert router.model is None and router.llm is None
