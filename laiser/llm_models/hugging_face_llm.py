"""
Module Description:
-------------------
Class to extract skills from text and align them to existing taxonomy

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Institute of Public Policy
        Program on Skills, Credentials and Workforce Policy
        Media and Public Affairs Building
        805 21st Street NW
        Washington, DC 20052
        PSCWP@gwu.edu
        https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2024 George Washington University Institute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Input Requirements:
-------------------
- All the libraries in the requirements.txt should be installed

Output/Return Format:
----------------------------
- List of extracted skills from text

"""

"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     6/30/2025      Anket Patil          Modularize LLM generation logic for transformers and vLLM
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from laiser.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, GENERATION_SEED, MAX_NEW_TOKENS

try:
    from vllm import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    SamplingParams = None  # Optional fallback


def llm_generate(
    prompt: str,
    tokenizer,
    model,
    model_id: str,
    use_gpu: bool,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = GENERATION_SEED,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """Generate text with a local HuggingFace Transformers model.

    Decoding is deterministic by default. A temperature of 0.0 disables
    sampling entirely (``do_sample=False``, i.e. greedy decoding), which also
    prevents a checkpoint's bundled ``generation_config.json`` from silently
    re-enabling sampling. When a caller raises the temperature above zero,
    ``seed`` is applied through ``torch.manual_seed`` so the run stays
    reproducible.

    Only the newly generated tokens are decoded. Returning the prompt as well
    would hand the downstream response parser the prompt's own JSON examples
    alongside the model's answer.
    """
    if tokenizer is None or model is None:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision="main")  # nosec B615
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision="main", quantization_config=quantization_config, device_map="auto"  # nosec B615
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature is not None and temperature > 0.0
    if seed is not None:
        torch.manual_seed(seed)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = DEFAULT_TOP_P

    outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)


def llm_generate_vllm(
    prompt,
    llm,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = GENERATION_SEED,
    max_tokens: int = 200,
):
    """Generate text with a local vLLM engine.

    Decoding is deterministic by default: ``temperature`` defaults to
    ``config.DEFAULT_TEMPERATURE`` (0.0, i.e. greedy decoding) rather than to
    vLLM's own default of 1.0, and ``config.GENERATION_SEED`` is passed so that
    runs remain reproducible if a caller raises the temperature.
    """
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed. Please install it to use this function."
        )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=DEFAULT_TOP_P,
        seed=seed,
    )
    result = llm.generate([prompt], sampling_params=sampling_params)
    raw_text = result[0].outputs[0].text.strip()
    return raw_text
