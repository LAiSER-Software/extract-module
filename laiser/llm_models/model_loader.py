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
[1.0.0]     6/30/2025      Anket Patil          Support transformer and vLLM model initialization
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import EntryNotFoundError, RepositoryNotFoundError
from typing import Tuple, Optional

from laiser.config import DEFAULT_TRANSFORMER_MODEL_ID, DEFAULT_VLLM_MODEL_ID
from laiser.exceptions import ModelLoadError, VLLMNotAvailableError

# Handle optional import for vLLM
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    VLLM_AVAILABLE = False

def load_model_from_transformer(model_id: str = None, token: str = ""):
    """Load transformer model with proper error handling"""
    model_id = model_id or DEFAULT_TRANSFORMER_MODEL_ID
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=token,
            quantization_config=quantization_config,
            device_map="auto"
        )
    except (RepositoryNotFoundError, EntryNotFoundError, OSError) as e:
        print(f"[WARN] Failed to load model '{model_id}': {e}")
        print(f"[INFO] Falling back to default model: {DEFAULT_TRANSFORMER_MODEL_ID}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TRANSFORMER_MODEL_ID, use_auth_token=token)
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_TRANSFORMER_MODEL_ID,
                use_auth_token=token,
                quantization_config=quantization_config,
                device_map="auto"
            )
        except Exception as fallback_error:
            raise ModelLoadError(f"Failed to load both requested and default models: {fallback_error}")

    return tokenizer, model


def load_model_from_vllm(model_id: str = None, token: str = None):
    """Load vLLM model with proper error handling"""
    if not VLLM_AVAILABLE:
        raise VLLMNotAvailableError("vLLM is not installed. Cannot load model using vLLM backend.")

    model_id = model_id or DEFAULT_VLLM_MODEL_ID

    try:
        llm = LLM(
            model=model_id,
            dtype="float16",
            quantization="gptq",
        )
        print(f"[INFO] Successfully loaded vLLM model: {model_id}")
    except Exception as e:
        print(f"[WARN] Failed to load model '{model_id}': {e}")
        print(f"[INFO] Falling back to default model: {DEFAULT_VLLM_MODEL_ID}")
        try:
            llm = LLM(
                model=DEFAULT_VLLM_MODEL_ID,
                dtype="float16", 
                quantization="gptq",
            )
        except Exception as fallback_error:
            raise ModelLoadError(f"Failed to load both requested and default vLLM models: {fallback_error}")
    
    return llm