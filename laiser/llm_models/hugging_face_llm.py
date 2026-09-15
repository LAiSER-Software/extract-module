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


from laiser.config import MAX_NEW_TOKENS
from laiser.llm_models.model_loader import load_model_from_transformer

try:
    from vllm import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    SamplingParams = None  # Optional fallback


def llm_generate(
    prompt: str, tokenizer, model, model_id: str, use_gpu: bool, max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate a completion with a transformers model.

    Returns only the newly generated text. ``model.generate`` returns the prompt
    tokens followed by the completion; decoding all of it hands the prompt's own
    example JSON to the response parser, which can match it instead of the answer.
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_model_from_transformer(model_id, use_gpu=use_gpu)

    text = prompt
    if getattr(tokenizer, "chat_template", None):
        # Instruct models expect their chat format; raw text is often echoed or ignored.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completion = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def llm_generate_vllm(prompt, llm):
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Please install it to use this function.")

    sampling_params = SamplingParams(max_tokens=200, seed=42)
    result = llm.generate([prompt], sampling_params=sampling_params)
    raw_text = result[0].outputs[0].text.strip()
    return raw_text
