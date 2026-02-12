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

# Import with error handling for optional dependencies
try:
    from laiser.llm_models.gemini import gemini_generate
except ImportError as e:
    print(f"Warning: Gemini support not available: {e}")
    def gemini_generate(*args, **kwargs):
        raise ImportError("Gemini support is not available. Please install google-generativeai package.")
# Import with error handling for optional dependencies
try:
    from laiser.llm_models.openai import openai_generate
except ImportError as e:
    print(f"Warning: openai support not available: {e}")

try:
    from laiser.llm_models.hugging_face_llm import llm_generate_vllm
except ImportError as e:
    print(f"Warning: HuggingFace LLM support not available: {e}")
    def llm_generate_vllm(*args, **kwargs):
        raise ImportError("HuggingFace LLM support is not available. Please install required packages.")

def llm_router(prompt: str, model_id: str, use_gpu: bool, llm, tokenizer=None, model=None, api_key=None):
    """
    Route LLM requests to appropriate model implementation.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    model_id : str
        Model identifier ('gemini' or other)
    use_gpu : bool
        Whether to use GPU
    llm : object
        vLLM model instance
    tokenizer : object, optional
        Tokenizer for transformer models
    model : object, optional
        Model instance for transformer models
    api_key : str, optional
        API key for external services
    
    Returns
    -------
    str
        Generated response from the model
    """
    if model_id == 'gemini':
        return gemini_generate(prompt, api_key)
    if model_id == 'openai':
        return openai_generate(prompt, api_key)

    # Fallback: Hugging Face LLM
    return llm_generate_vllm(prompt, llm)
