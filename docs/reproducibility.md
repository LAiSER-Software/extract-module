# Reproducibility

Language models generate text by sampling, so left to their providers' defaults two runs over the same input
can return different skills. LAiSER is configured so that, by default, they do not.

## Defaults

Every backend decodes greedily: at temperature `0.0` the model always takes its most probable next token, so
the same prompt produces the same text. A fixed seed (`42`) is also supplied wherever a backend accepts one,
which keeps runs reproducible if you deliberately raise the temperature.

| Backend | Temperature | Seed |
|---|---|---|
| Gemini | `0.0` | `42`, on google-genai releases that accept a seed |
| OpenAI | `0.0` | not accepted by the API |
| Local model through Transformers | greedy — sampling switched off | `42`, used when temperature is above `0.0` |
| Local model through vLLM | `0.0` | `42` |
| llama.cpp | `0.0` | `42`, on llama-cpp-python releases that accept a seed |

For Transformers models, sampling is turned off explicitly rather than left to the checkpoint. Some checkpoints
ship a `generation_config.json` that enables sampling, which would otherwise make repeated runs differ.

Alignment is deterministic as well. Each extracted phrase is embedded with a fixed sentence-transformer model
and matched by exact search against the bundled taxonomy index, so the same phrase always resolves to the same
taxonomy entry with the same similarity score.

## Changing the defaults

Pass `temperature` and `seed` when creating an extractor:

```python
extractor = SkillExtractorRefactored(model_id="gemini", api_key=api_key, temperature=0.7, seed=123)
```

Pass `seed=None` to send no seed. The defaults can also be changed for a whole process through environment
variables, which LAiSER reads when it is imported:

| Variable | Default | Effect |
|---|---|---|
| `LAISER_TEMPERATURE` | `0.0` | default temperature for every backend |
| `LAISER_TOP_P` | `1.0` | nucleus sampling cut-off; `1.0` leaves it off |

## Reference configuration

To let others reproduce your results, pin every setting that affects them:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored

extractor = SkillExtractorRefactored(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # pin the model
    use_gpu=False,
    temperature=0.0,  # the default: greedy decoding
    seed=42,  # the default
)

results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
    allowed_sources=["esco", "onet", "osn", "ukos"],
    similarity_thresholds={"skill": 0.60, "knowledge": 0.50, "task": 0.50},  # the defaults
)
```

Also record:

- **the LAiSER version** — it fixes the bundled taxonomy data and the embedding model used for alignment,
  `sentence-transformers/all-MiniLM-L6-v2`;
- **the model revision**, for Hugging Face models that may be updated under the same name.

## What this does not guarantee

- **Hosted models can change.** A provider can update the model behind a name, and providers do not
  guarantee identical output even at temperature `0.0`.
- **Different models extract different phrases.** Alignment can map different wordings of a skill onto the
  same taxonomy entry, but each phrase is matched on its own, so that is likely rather than guaranteed — and
  alignment cannot recover a skill a model never extracted. Compare models on the aligned columns, and measure
  their agreement rather than assuming it.
- **Hardware and library versions matter for local models.** Floating-point results can differ between
  devices and library releases.

## How it is tested

`tests/test_determinism.py` runs in CI without a GPU, an API key or a language model download. The alignment
checks use the real embedding model, which Hugging Face downloads on first use. It checks that:

- every public entry point defaults to temperature `0.0`, and every seedable one to seed `42`;
- the decoding parameters each backend actually receives match those defaults, and per-call overrides reach
  every backend;
- repeated alignment of the same phrases returns an identical table, similarity scores included.
