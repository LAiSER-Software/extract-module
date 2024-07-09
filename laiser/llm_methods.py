import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from params import AI_MODEL_ID

torch.cuda.empty_cache()

model_id = "google/gemma-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, padding_side='left')


def finetune(self, input_text):
    """
    The function extracts skills from text using HuggingFace's API

    Parameters
    ----------
    input_text : text
        Job advertisement / Job Description / Syllabus Description / Course Outcomes etc.

    Returns
    -------
    list: List of extracted skills from text

    Notes
    -----
    The Function is fine-tuned only to return list of skills based on prompt passed to it.

    """
    tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(AI_MODEL_ID)
    
    # use the model variable to generate the list of skills form the input_text
    model_output = model.generate(
        tokenizer(input_text, return_tensors="pt").input_ids,
        max_length=100,
        num_return_sequences=1,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    extracted_skills = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    extracted_skills_set = set(extracted_skills)
        
    return list(extracted_skills_set)


def fetch_model_output(response):
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

