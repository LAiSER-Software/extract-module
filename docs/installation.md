# Installation

## Requirements

- Python **3.10 or later**. CI tests 3.10, 3.11, 3.12 and 3.13.
- A GPU is optional. Hosted providers and small local models run on CPU; larger local models are
  practical only on a GPU.

## Install from PyPI

```bash
pip install laiser
laiser setup
```

Run `laiser setup` once before extracting. It downloads the package's taxonomy bundles from GitHub Releases,
checks their integrity, and prepares the alignment embedding model. No provider API key is needed for setup.

The taxonomy bundles total about 125 MiB compressed and are cached under `~/.cache/laiser` by default.
Set `LAISER_CACHE_DIR` to use a different taxonomy cache directory. The embedding model uses the
Hugging Face cache, which can be set with `HF_HOME`. Use the same cache settings for setup and extraction.
Re-running setup reuses the prepared files.

### GPU inference with vLLM

```bash
pip install "laiser[gpu]"
laiser setup
```

The `gpu` extra adds [vLLM](https://docs.vllm.ai/) for faster local inference on CUDA devices, plus
`bitsandbytes` and `accelerate`, which LAiSER uses to load a model in 8-bit when vLLM cannot. Check that
PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Local GGUF models with llama.cpp

To run a quantized GGUF model through llama.cpp, install its Python bindings separately:

```bash
pip install llama-cpp-python
```

See [Model providers](providers.md#llamacpp-local-gguf) for how to point LAiSER at the model file.

## Install from source

```bash
git clone https://github.com/LAiSER-Software/extract-module.git
cd extract-module
pip install -e ".[dev]"
```

The `dev` extra adds the test and lint tooling used in CI.

The checkout includes the taxonomy files. Run `laiser setup` to prepare its alignment embedding model
ahead of time as well.

## Language models

Setup prepares the taxonomy data and the embedding model used for alignment. If you choose a local
language model for extraction, that model has its own download. Hosted providers such as Gemini still
need network access when extracting from text.
