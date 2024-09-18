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
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
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
[1.0.0]     07/10/2024      Satya Phanindra K.  Define all the LLM methods being used in the project
[1.0.1]     07/19/2024      Satya Phanindra K.  Add descriptions to each method

TODO:
-----

"""

import re
import torch
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.cuda.empty_cache()

def fetch_model_output(response):
    """
    Format the model's output to extract the skill keywords from the get_completion() response
    
    Parameters
    ----------
    input_text : text
        The model's response after processing the prompt. 
        Contains special tags to identify the start and end of the model's response.
        
    Returns
    -------
    list: List of extracted skills from text
    
    """
    # Find the content between the model start tag and the last <eos> tag
    pattern = r'<start_of_turn>model\s*<eos>(.*?)<eos>\s*$'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        content = match.group(1).strip()

        # Split the content by lines and filter out empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Extract skills (lines starting with '-')
        skills = [line[1:].strip() for line in lines if line.startswith('-')]

        return skills

def get_completion_batch(queries: list, model, tokenizer, batch_size=2) -> list:
    """
    Get completions for a list of queries using the model
    
    Parameters
    ----------
    queries : list
        List of queries to get completions for using the model
    model : model
        The model to use for generating completions
    tokenizer : tokenizer
        The tokenizer to use for encoding the queries
    batch_size : int, optional
        Preferred batch size to use for generating completions
        
    Returns
    -------
    list: List of extracted skills from the text(s)
    
    """
    
    device = "cuda:0"
    results = []

    prompt_template = """
    <start_of_turn>user
    Name all the skills present in the following description in a single list. Response should be in English and have only the skills, no other information or words. Skills should be keywords, each being no more than 3 words.
    Below text is the Description:

    {query}
    <end_of_turn>\n<start_of_turn>model
    """

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        prompts = [prompt_template.format(query=query) for query in batch]

        encodeds = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True)
        model_inputs = encodeds.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for full_output in decoded:
            # Extract only the model's response
            response = full_output.split("<start_of_turn>model<eos>")[-1].strip()
            processed_response = fetch_model_output(response)
            results.append(processed_response)

        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()

        print(f"Processed batch {i//batch_size + 1}/{(len(queries)-1)//batch_size + 1}")

    return results

def get_completion(query: str, model, tokenizer) -> str:
    """
    Get completion for a single query using the model
    
    Parameters
    ----------
    query : str
        The query to get completions for using the model
    model : model
        The model to use for generating completions
    tokenizer : tokenizer
        The tokenizer to use for encoding the queries
        
    Returns
    -------
    list: List of extracted skills from the text
        
    """
    
    device = "cuda:0"

    prompt_template = """
    <start_of_turn>user
    Name all the skills present in the following description in a single list. Response should be in English and have only the skills, no other information or words. Skills should be keywords, each being no more than 3 words.
    Below text is the Description:

    {query}
    <end_of_turn>\n<start_of_turn>model
    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)


    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    response = decoded.split("<start_of_turn>model<eos>")[-1].strip()
    processed_response = fetch_model_output(response)
    return (processed_response)
