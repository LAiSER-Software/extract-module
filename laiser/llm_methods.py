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
[1.0.0]     07/10/2024      Satya Phanindra K.  Define all the LLM methods being used in the project
[1.0.1]     07/19/2024      Satya Phanindra K.  Add descriptions to each method
[1.0.2]     11/24/2024      Prudhvi Chekuri     Add support for skills extraction from syllabi data
[1.0.3]     03/12/2025      Prudhvi Chekuri     Implement functions to extract levels, KSAs from job descriptions and syllabi data using vLLM
[1.0.4]     03/15/2025      Prudhvi Chekuri     Add exception handling
[1.0.5]     06/29/2025      Anket Patil         Integrate LLM Router for multi-LLM support


TODO:
-----

"""

import re
import torch
import numpy as np
import json

from laiser.utils import get_top_esco_skills
from laiser.llm_models.llm_router import llm_router

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

def get_completion(input_text, text_columns, input_type, model, tokenizer) -> str:
    """
    Get completion for a single query using the model
    
    Parameters
    ----------
    input_text : pandas Series with text data related to Job Description / Syllabus Description / Course Outcomes etc.
        The query to get completions for using the model
    text_columns : list
        List of columns in the input_text dataframe that contain the text data. (Default: ['description'])
    input_type : str
        Type of input data - 'job_desc' / 'syllabus' etc. (Default: 'job_desc')
    model : model
        The model to use for generating completions
    tokenizer : tokenizer
        The tokenizer to use for encoding the queries
        
    Returns
    -------
    list: List of extracted skills from the text
        
    """
    
    device = "cuda:0"

    if input_type == "job_desc":
        prompt_template = """
        <start_of_turn>user
        Name all the skills present in the following description in a single list. Response should be in English and have only the skills, no other information or words. Skills should be keywords, each being no more than 3 words.
        Below text is the Description:

        {query}
        <end_of_turn>\n<start_of_turn>model
        """
        prompt = prompt_template.format(query=input_text[text_columns[0]])
    elif input_type == "syllabus":
        prompt_template = """
        <start_of_turn>user
        Name all the skills present in the following course details in a single list. Response should be in English and have only the skills, no other information or words. Skills should be keywords, each being no more than 3 words.

        Course Description:
        {description}


        Learning Outcomes:
        {learning_outcomes}

        <end_of_turn>
        <start_of_turn>model
        """
        prompt = prompt_template.format(description=input_text[text_columns[0]], learning_outcomes=input_text[text_columns[1]])

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)


    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    response = decoded.split("<start_of_turn>model<eos>")[-1].strip()
    processed_response = fetch_model_output(response)
    return (processed_response)


def parse_output_vllm(response):
    
    """
    Parse the model's response to extract key skills, knowledge required, and task abilities.
    
    Parameters
    ----------
    response : str
        The model's response after processing the prompt.
        
    Returns
    -------
    list: List of dictionaries that has levels, KSAs for all the data points in the input text.
    
    """
    
    out = []
    # Split into items, handling optional '->' prefix and multi-line input
    items = [item.strip() for item in response.split('->') if item.strip()]

    for item in items:
        skill_data = {}
        try:
            # Extract skill
            skill_match = re.search(r"Skill:\s*([^,\n]+)", item)
            if skill_match:
                skill_data['Skill'] = skill_match.group(1).strip()

            # Extract level
            level_match = re.search(r"Level:\s*(\d+)", item)
            if level_match:
                skill_data['Level'] = int(level_match.group(1).strip())

            # Extract knowledge required (multi-line support with re.DOTALL)
            knowledge_match = re.search(r"Knowledge Required:\s*(.*?)(?=\s*Task Abilities:|\s*$)", item, re.DOTALL)
            if knowledge_match:
                knowledge_raw = knowledge_match.group(1).strip()
                skill_data['Knowledge Required'] = [k.strip() for k in knowledge_raw.split(',') if k.strip()]

            # Extract task abilities (multi-line support with re.DOTALL)
            task_match = re.search(r"Task Abilities:\s*(.*?)(?=\s*$)", item, re.DOTALL)
            if task_match:
                task_raw = task_match.group(1).strip()
                skill_data['Task Abilities'] = [t.strip() for t in task_raw.split(',') if t.strip()]

            out.append(skill_data)
        except:
            continue

    return out


def create_ksa_prompt(query, input_type, num_key_skills, num_key_kr, num_key_tas):
    # TODO: Verify the docstring and update missing/incorrect information
    """
    Create a structured prompt for the KSA (Knowledge, Skills, Abilities) extraction task.
    
    Parameters
    ----------
    query : dict
        A dictionary containing the input data, including 'description' and optionally 'learning_outcomes'.
    input_type : str
        The type of input data - 'job_desc' or 'syllabi'.
    num_key_skills : int
        The number of key skills to extract.
    num_key_kr : str
        The number of key knowledge areas to extract (e.g., '3-5').
    num_key_tas : str
        The number of key task abilities to extract (e.g., '3-5').
    Returns
    ------- 
    str
        The formatted prompt for the KSA extraction task.       
    
    """

    prompt_template = """user
**Objective:** Given a {input_desc}, complete the following tasks with structured outputs.

### Semantic matches from Taxonomy Skills:
{esco_context_block}

### Tasks:
1. **Skills Extraction:** Identify {num_key_skills} key skills mentioned in the {input_desc}.
  - Extract/Filter contextually relevant skill keywords or phrases from taxonomy semantic matches.

2. **Skill Level Assignment:** Assign a proficiency level to each extracted skill based on the SCQF Level Descriptors (see below).

3. **Knowledge Required:** For each skill, list {num_key_kr} broad areas of understanding or expertise necessary to develop the skill.

4. **Task Abilities:** For each skill, list {num_key_tas} general tasks or capabilities enabled by the skill.

### Guidelines:
- **Skill Extraction:** 
    - If the Semantic matches from the taxonomy skills are provided, use them to identify relevant skills.
    - If none of the semantic matches are contextually relevant to the {input_desc}, infer skills from the {input_desc} directly.

- **Skill Level Assignment:** Use the SCQF Level Descriptors to classify proficiency:
  - 1: Basic awareness of simple concepts.
  - 2: Limited operational understanding, guided application.
  - 3: Moderate knowledge, supervised application of techniques.
  - 4: Clear understanding, independent work in familiar contexts.
  - 5: Advanced knowledge, autonomous problem-solving.
  - 6: Specialized knowledge, critical analysis within defined areas.
  - 7: Advanced specialization, leadership in problem-solving.
  - 8: Expert knowledge, innovation in complex contexts.
  - 9: Highly specialized expertise, contributing original thought.
  - 10: Sustained mastery, influential in areas of specialization.
  - 11: Groundbreaking innovation, professional or academic mastery.
  - 12: Global expertise, leading advancements at the highest level.

- **Knowledge and Task Abilities:**
  - **Knowledge Required:** Broad areas, e.g., "data visualization techniques."
  - **Task Abilities:** General tasks or capabilities, e.g., "data analysis."
  - Each item in these two lists should be no more than three words.
  - Avoid overly specific or vague terms.

### Answer Format:
- Use this format strictly in the response:
  -> Skill: [Skill Name], Level: [1–12], Knowledge Required: [list], Task Abilities: [list].

{input_text}

**Response:** Provide only the requested structured information without additional explanations.


model
"""

    input_desc = "job description" if input_type == "syllabi" else "course syllabus description and its learning outcomes"
    if input_type == "syllabi":
        input_text = f"""### Input:\n**Course Description:** {query["description"]}\n**Learning Outcomes:** {query["learning_outcomes"]}"""
    else:
        input_text = f"""### Input:\n{query["description"]}"""

    # Prepare ESCO context for each input
    esco_context = []
    top_esco = get_top_esco_skills(input_text, top_k=10)
    esco_context.append(", ".join([s['Skill'] for s in top_esco]))
    prompt = prompt_template.format(input_desc=input_desc, num_key_skills=num_key_skills, num_key_kr=num_key_kr, num_key_tas=num_key_tas, input_text=input_text, esco_context_block=esco_context)
    return prompt


def vllm_generate(llm, queries, input_type, batch_size, num_key_skills=5, num_key_kr='3-5', num_key_tas='3-5'):
    """
    Generate completions for the whole data using the LLM model with vLLM.
    
    Parameters
    ----------
    llm : model
        The model to use for generating completions
    queries : pandas DataFrame
        The queries to get completions for using the model
    input_type : str
        Type of input data - 'job_desc' / 'syllabus' etc. (Default: 'job_desc')
    batch_size : int, optional
        Preferred batch size to use for generating completions
    num_key_skills : int, optional
        Number of key skills to extract from the input text
    num_key_kr : str, optional
        Number of key knowledge required items to extract from the input text
    num_key_tas : str, optional
        Number of key task abilities items to extract from the input text
        
    Returns
    -------
    list: List of completions generated by the model for the input queries
    
    """

    result = []

    sampling_params = SamplingParams(max_tokens=1000, seed=42)

    for i in range(0, len(queries), batch_size):
        prompts = [create_ksa_prompt(queries.iloc[i], input_type, num_key_skills, num_key_kr, num_key_tas) for i in range(i, min(i+batch_size, len(queries)))]

        try:
            output = llm.generate(prompts, sampling_params=sampling_params)
            result.extend(output)
        except Exception as e:
            result.extend([None]*batch_size)
            print(f"Error generating batch at index {i}: {e}")
            continue

    return result


def get_completion_vllm(input_text, text_columns, id_column, input_type, llm, batch_size) -> list:

    """
    Get completions for whole input data and parse the required KSAs from the model responses. The input data can be a job description or syllabi data.
    
    Parameters
    ----------
    input_text : pandas DataFrame
        The input data to get completions for using the model
    text_columns : list (optional)
        List of columns in the input_text dataframe that contain the text data. (Default: ['description'])
    id_column : str
        Column name in the input_text dataframe that contains the unique identifier for each row
    input_type : str
        Type of input data - 'job_desc' / 'syllabus' etc. (Default: 'job_desc')
    llm : model
        The model to use for generating completions
    batch_size : int, optional
        Preferred batch size to use for generating completions
        
    Returns
    -------
    list: List of dictionaries that has levels, KSAs for all the data points in the input text. 
    """

    try:
        result = vllm_generate(llm, input_text, input_type=input_type, batch_size=batch_size)
    except Exception as e:
        print(f"Error in vLLM generation: {e}")
        return []
    
    parsed_output = []
    for i in range(len(result)):
        if result[i] is not None:
            try:
                parsed = parse_output_vllm(result[i].outputs[0].text)
                for item in parsed:
                    item[id_column] = input_text.iloc[i][id_column]
                    item['description'] = input_text.iloc[i]['description']
                    if 'learning_outcomes' in input_text.columns:
                        item['learning_outcomes'] = input_text.iloc[i]['learning_outcomes']
                parsed_output.extend(parsed)
            except Exception as e:
                print(f"Error parsing output for index {i}: {e}")
                continue

    return parsed_output

# ----------------------------------------------------------------------------------
# NEW HELPER: get_ksa_details
# ----------------------------------------------------------------------------------

def get_ksa_details(skill: str, description: str, model_id, use_gpu, llm, tokenizer=None, model=None, api_key=None,num_key_kr: int = 3, num_key_tas: int = 3):    
    """
    Generate Knowledge Required and Task Abilities for a given skill in the context of the supplied description.

    Parameters
    ----------
    skill : str
        The skill name for which to generate KSAs.
    description : str
        The textual description (job description, syllabus, etc.) providing context.
    llm : vllm.LLM
        The LLM instance already initialised by the caller.
    num_key_kr : int, optional
        Maximum length of the Knowledge Required list (default 3).
    num_key_tas : int, optional
        Maximum length of the Task Abilities list (default 3).

    Returns
    -------
    Tuple[list, list]
        knowledge_required, task_abilities – both are lists of strings.  Empty lists are
        returned if generation/parsing fails.
    """

    prompt = (
        "user\n"
        f"Given the following context, provide concise lists for the specified skill.\n\n"
        f"Skill: {skill}\n\n"
        "Context:\n"
        f"{description}\n\n"
        f"For the skill above produce:\n"
        f"- Knowledge Required: {num_key_kr} bullet items, each ≤ 3 words.\n"
        f"- Task Abilities: {num_key_tas} bullet items, each ≤ 3 words.\n\n"
        "Respond strictly in valid JSON with the exact keys 'Knowledge Required' and 'Task Abilities'.\n"
        "model"
    )

    try:
        raw_text = llm_router(prompt, model_id, use_gpu,llm, tokenizer, model,api_key)       
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            return [], []

        parsed = json.loads(json_match.group())
        knowledge = parsed.get("Knowledge Required", [])
        task_abilities = parsed.get("Task Abilities", [])

        # Ensure they are lists
        if not isinstance(knowledge, list):
            knowledge = [str(knowledge)]
        if not isinstance(task_abilities, list):
            task_abilities = [str(task_abilities)]

        return knowledge, task_abilities
    except Exception as e:
        print(f"[get_ksa_details] Generation/parsing error for skill '{skill}': {e}")
        return [], []