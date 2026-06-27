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
[1.0.0]     6/30/2025      Anket Patil          Centralize LLM dispatch logic using router function
"""

import os

import torch

from laiser._internal.model_loader import load_model_from_transformer, load_model_from_vllm
from laiser.exceptions import LAiSERError
from laiser.providers.llama_cpp_handler import llama_cpp_chat

MODEL_PATH = os.getenv("LAISER_LLAMA_CPP_MODEL_PATH", "")
LLAMA_CPP_CTX = int(os.getenv("LLAMA_CPP_CTX", "4096"))
LLAMA_CPP_THREADS = os.getenv("LLAMA_CPP_THREADS")

# Import with error handling for optional dependencies
try:
    from laiser.providers.gemini import gemini_generate
except ImportError as e:
    _IMPORT_WARNINGS = [f"Gemini support not available: {e}"]

    def gemini_generate(*args, **kwargs):
        raise ImportError("Gemini support is not available. Please install google-genai.")


# Import with error handling for optional dependencies
try:
    from laiser.providers.openai import openai_generate
except ImportError as e:
    _IMPORT_WARNINGS = globals().get("_IMPORT_WARNINGS", [])
    _IMPORT_WARNINGS.append(f"openai support not available: {e}")

try:
    from laiser.providers.hugging_face_llm import llm_generate_vllm
except ImportError as e:
    _IMPORT_WARNINGS = globals().get("_IMPORT_WARNINGS", [])
    _IMPORT_WARNINGS.append(f"HuggingFace LLM support not available: {e}")

    def llm_generate_vllm(*args, **kwargs):
        raise ImportError("HuggingFace LLM support is not available. Please install required packages.")


class LLMRouter:

    def __init__(self, model_id: str, use_gpu: bool, hf_token=None, api_key=None, backend=None):
        self.model_id = model_id
        self.use_gpu = use_gpu
        self.hf_token = hf_token
        self.api_key = api_key
        self.backend = backend

        self.llm = None
        self.model = None
        self.tokenizer = None
        self.nlp = None

        self._initialize_components()

    # ---------------- ROUTER ----------------
    def generate(self, prompt: str, **kwargs):
        if self.model_id == "gemini":
            return gemini_generate(prompt, self.api_key, **kwargs)

        if self.model_id == "openai":
            return openai_generate(prompt, self.api_key)

        # If a local GGUF model was loaded with llama-cpp-python, use it
        if self.backend == "llama_cpp":
            return llama_cpp_chat(prompt, self.llm)

        return llm_generate_vllm(prompt, self.llm)

    # ---------------- INIT ----------------
    def _initialize_components(self):
        try:
            if self.backend == "llama_cpp":
                try:
                    from llama_cpp import Llama
                except ImportError as e:
                    raise LAiSERError(
                        "llama-cpp-python is not installed. Install it to use backend='llama_cpp'."
                    ) from e

                self.llm = Llama(
                    model_path=str(MODEL_PATH),
                    n_ctx=LLAMA_CPP_CTX,
                    n_threads=LLAMA_CPP_THREADS or None,
                    n_gpu_layers=-1,  # Use GPU if available, else CPU
                    # logits_all=False,
                    # chat_format="chatml",
                )
                return

            if self.model_id == "gemini":
                return

            elif self.use_gpu and torch.cuda.is_available():
                try:
                    self._initialize_vllm()
                    if self.llm is not None:
                        return
                except Exception:
                    pass

                try:
                    self._initialize_transformer()
                    if self.model is not None:
                        return
                except Exception:
                    pass

            else:
                try:
                    self._initialize_transformer()
                    if self.model is not None:
                        return
                except Exception:
                    pass

            if globals().get("_IMPORT_WARNINGS"):
                raise LAiSERError("; ".join(globals()["_IMPORT_WARNINGS"]))

        except Exception as e:
            raise LAiSERError(f"Critical failure during component initialization: {e}")

    def _initialize_vllm(self):
        self.llm = load_model_from_vllm(self.model_id, self.hf_token)

    def _initialize_transformer(self):
        self.tokenizer, self.model = load_model_from_transformer(self.model_id, self.hf_token)
