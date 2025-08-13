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

# Handle optional import for vLLM
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    VLLM_AVAILABLE = False

DEFAULT_TRANSFORMER_MODEL_ID = "google/gemma-2-9b-it"

def load_model_from_transformer( model_id: str = None,token: str = ""):
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
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TRANSFORMER_MODEL_ID, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_TRANSFORMER_MODEL_ID,
            use_auth_token=token,
            quantization_config=quantization_config,
            device_map="auto"
        )

    return tokenizer, model


DEFAULT_VLLM_MODEL_ID = "marcsun13/gemma-2-9b-it-GPTQ"

def load_model_from_vllm(model_id: str = None, token: str = None, dtype: str = None):

    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Cannot load model using vLLM backend.")

    model_id = model_id or DEFAULT_VLLM_MODEL_ID

    # Determine the appropriate dtype based on model type if not explicitly provided
    def get_dtype_for_model(model_name):
        """Get appropriate dtype for specific model types"""
        model_name_lower = model_name.lower()
        if 'gemma2' in model_name_lower or 'gemma-2' in model_name_lower:
            return "bfloat16"  # Gemma2 requires bfloat16 or float32
        else:
            return "float16"   # Default for other models

    # Use provided dtype or auto-detect
    if dtype is None:
        dtype = get_dtype_for_model(model_id)

    try:
        llm = LLM(
            model=model_id,
            dtype=dtype,
            quantization="gptq",
        )
        print(f"[INFO] Successfully loaded vLLM model: {model_id} with dtype: {dtype}")
    except Exception as e:
        print(f"[WARN] Failed to load model '{model_id}' with dtype '{dtype}': {e}")
        print(f"[INFO] Falling back to default model: {DEFAULT_VLLM_MODEL_ID}")
        fallback_dtype = get_dtype_for_model(DEFAULT_VLLM_MODEL_ID)
        llm = LLM(
            model=DEFAULT_VLLM_MODEL_ID,
            dtype=fallback_dtype,
            quantization="gptq",
        )
        print(f"[INFO] Loaded fallback model with dtype: {fallback_dtype}")
    
    return llm