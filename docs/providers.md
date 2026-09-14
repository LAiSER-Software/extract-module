# Model providers

The language model does the extraction step; alignment always runs locally against the bundled indexes.
Choose the model when you construct the extractor.

| You pass | Runs on | Needs |
|---|---|---|
| `model_id="gemini"` | Google Gemini API | a Gemini API key, passed as `api_key` |
| `model_id="openai"` | OpenAI API | `api_key`, or `OPENAI_API_KEY` in the environment |
| `model_id="<Hugging Face model id>"`, `use_gpu=False` | your CPU, through transformers | nothing — no key, no GPU |
| `model_id="<Hugging Face model id>"`, `use_gpu=True` | your GPU, through vLLM | `laiser[gpu]` and a CUDA device |
| `backend="llama_cpp"` | your machine, through llama.cpp | `llama-cpp-python` and a GGUF file |

`hf_token` is only needed for gated or private Hugging Face models.

## Gemini

```python
import os
from laiser.skill_extractor_refactored import SkillExtractorRefactored

extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
)
```

The key must be passed as `api_key`; it is not read from the environment automatically. Optional settings,
read from environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LAISER_GEMINI_MODEL` | `gemini-2.5-flash` | which Gemini model to call |
| `LAISER_GEMINI_TIMEOUT` | `60` | request timeout in seconds |
| `LAISER_GEMINI_MAX_OUTPUT_TOKENS` | `4096` | maximum length of each response |

## OpenAI

```python
extractor = SkillExtractorRefactored(model_id="openai")  # uses OPENAI_API_KEY
```

Requests go to the OpenAI Responses API with the `gpt-4.1-mini` model. The model is not currently
configurable through LAiSER.

## Local model on CPU

Any Hugging Face causal language model can run on CPU, with no API key:

```python
extractor = SkillExtractorRefactored(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    use_gpu=False,
)
```

The weights download on first use and are cached by Hugging Face. Instruction-tuned models work best,
because LAiSER formats the prompt with the model's chat template when it has one. Each response is capped at
1,000 new tokens.

!!! warning "Model size is a trade-off"
    Small models such as the 0.5B example above run in seconds to minutes per document on a laptop CPU,
    but extract fewer and noisier concepts than a hosted model. Larger models extract better but may be
    too slow without a GPU. Alignment quality does not depend on the model — only which phrases reach it.

## Local model on GPU

With `use_gpu=True` on a machine with CUDA, LAiSER loads the model with vLLM, falling back to transformers
with 8-bit quantization if vLLM cannot load it. Install the GPU extra first:

```bash
pip install "laiser[gpu]"
```

If a GPU model fails to load, LAiSER falls back to `TheBloke/Mixtral-7B-Instruct-v0.1-AWQ`. On CPU there is
no fallback: a model that fails to load raises the original error.

## llama.cpp (local GGUF)

Quantized GGUF models run through llama.cpp on CPU or GPU:

```bash
pip install llama-cpp-python
export LAISER_LLAMA_CPP_MODEL_PATH=/path/to/model.gguf
```

```python
extractor = SkillExtractorRefactored(backend="llama_cpp")
```

| Variable | Default | Effect |
|---|---|---|
| `LAISER_LLAMA_CPP_MODEL_PATH` | *(required)* | path to the `.gguf` model file |
| `LLAMA_CPP_CTX` | `4096` | context window in tokens |
| `LLAMA_CPP_THREADS` | llama.cpp default | CPU threads to use |

!!! note "Anthropic"
    The package contains an Anthropic client, but it is not yet selectable through `model_id`.
