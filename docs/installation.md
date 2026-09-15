# Installation

## Requirements

- Python **3.10 or later**. CI tests 3.10, 3.11, 3.12 and 3.13.
- A GPU is optional. Hosted providers and small local models run on CPU; larger local models are
  practical only on a GPU.

## Install from PyPI

```bash
pip install laiser
```

This installs everything needed to extract with a hosted provider or a local transformers model, and to
align results against the bundled taxonomies.

### GPU inference with vLLM

```bash
pip install "laiser[gpu]"
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

## What downloads on first use

!!! note "First run is slower"
    The taxonomy indexes are part of the package, but two things are fetched the first time they are
    needed and cached afterwards by Hugging Face:

    - the sentence-embedding model used for alignment and deduplication
      (`sentence-transformers/all-MiniLM-L6-v2`), and
    - the language model itself, if you use a local model rather than a hosted API.
