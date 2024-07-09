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
- Pandas Dataframe with ID and Text Column

Output/Return Format:
----------------------------
- Pandas dataframe with below columns:
    - "Research ID": text_id
    - "Skill Name": Raw skill extracted,
    - "Skill Tag": skill tag from taxonomy,
    - "Correlation Coefficient": similarity_score


"""
"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     05/30/2024      Vedant M.           Initial Version
[1.0.1]     06/01/2024      Vedant M.           Referencing utils.py and params.py
[1.0.2]     06/08/2024      Satya Phanindra K.  Modify get_aligned_skills function to JSON output
[1.0.3]     06/10/2024      Vedant M.           Updated functions extract_raw and align_skills for input and output
[1.0.4]     06/13/2024      Vedant M.           Added function extractor to encapsulate both functions
[1.0.5]     06/15/2024      Satya Phanindra K.  Replaced OpenAI API with HuggingFace API for skill extraction
[1.0.6]     06/20/2024      Satya Phanindra K.  Added function to extract skills from text using Fine-Tuned Language Model's API

TODO:
-----
- 1: Add references to utils and global parameter file
- 2: sort taxonomy inputs
- 3: include rsd_name instead of keywords from osn
- 4: Optimize the `align_skills` function.
"""

# native packages
import sys
import os

# installed packages
import pandas as pd
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM


# internal packages
from laiser.utils import get_embedding, cosine_similarity
from laiser.params import AI_MODEL_ID, API_KEY, SIMILARITY_THRESHOLD, SKILL_DB_PATH


class Skill_Extractor:
    """
    Class to extract skills from text and align them to existing taxonomy
    ...

    Attributes
    ----------
    client : HuggingFace API client
    nlp : spacy nlp model
        Short description

    Parameters
    ----------


    Methods
    -------
    extract_raw(input_text: text)
        The function extracts skills from text using NER model

    align_skills(raw_skills: list, document_id='0': string):
        This function aligns the skills provided to the desired taxonomy
        
    extractor(data: pandas dataframe, id_column='Research ID', text_column='Text'):
        Function takes text dataset to extract and aligns skills based on available taxonomies
    ....

    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        return

    # Declaring a private method for extracting raw skills from input text
    def extract_raw(self, input_text):
        """
        The function extracts skills from text using Fine-Tuned Language Model's API

        Parameters
        ----------
        input_text : text
            Job advertisement / Job Description / Syllabus Description / Course Outcomes etc.

        Returns
        -------
        list: List of extracted skills from text

        Notes
        -----
        More details on which (pre-trained) language model is fine-tuned can be found in llm_methods.py
        The Function is designed only to return list of skills based on prompt passed to OpenAI's Fine-tuned model.

        """
        tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(AI_MODEL_ID)
        
        # TODO: optimize the model usage by loading it once and using it multiple times from the  llm_methods.py file
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

    def align_skills(self, raw_skills, document_id='0'):
        """
        This function aligns the skills provided to the available taxonomy

        Parameters
        ----------
        raw_skills : list
            Provide list of skill extracted from Job Descriptions / Syllabus.
        document_id: string
            ID of the document or text from where skills were extracted
            Defaults to '0'

        Returns
        -------
        list: List of taxonomy skills from text in JSON format
            [
                {
                    "Research ID": text_id,
                    "Skill Name": Raw skill extracted,
                    "Skill Tag": taxonomy skill tag,
                    "Correlation Coefficient": similarity_score
                },
                ...
            ]

        """
        # dataframe for skill taxonomy database
        skill_db_df = pd.read_csv(SKILL_DB_PATH)

        skill_matches = pd.DataFrame(columns=['Research ID', 'Raw Skill', 'Skill Tag', 'Correlation Coefficient'])

        # iterate over extracted skills
        for raw_skill in raw_skills:

            # get vectorized embedding for raw skill
            raw_skill_embedding = get_embedding(self.nlp, raw_skill)

            matched_skill_set = set()

            # iterate over each row in skill taxonomy db
            for index, row in skill_db_df.iterrows():
                tag = row['SkillTag']
                label = row['SkillLabel']

                # get vectorized embedding for skill in taxonomy db
                db_skill_embedding = get_embedding(self.nlp, label)

                # get cosine similarity between raw skill and skill from taxonomy db
                similarity = cosine_similarity(raw_skill_embedding, db_skill_embedding)

                # if cosine similarity > threshold and not already added then add to the matched skills
                if similarity > SIMILARITY_THRESHOLD and tag not in matched_skill_set:
                    temp = pd.DataFrame([{
                        "Research ID": document_id,
                        "Raw Skill": raw_skill,
                        "Skill Tag": tag,
                        "Correlation Coefficient": similarity
                    }])
                    skill_matches = pd.concat([skill_matches, temp], ignore_index=True)
                    matched_skill_set.add(tag)

        return skill_matches.to_dict(orient='records')


    def extractor(self, data, id_column='Research ID', text_column='Text'):
        """
        Function takes text dataset to extract and aligns skills based on available taxonomies

        Parameters
        ----------
        data : pandas dataframe
            Dataset containing text id and actual text to extract skills.
        id_column: string
            Name of id column in the dataset. Defaults to 'Research ID'
        text_column: string
            Name of the text column in the dataset. Defaults to 'Text'

        Returns
        -------
        list: List of skill tags and similarity_score for all texts in  from text in JSON format
            [
                {
                    "Research ID": text_id
                    "Skill Name": Raw skill extracted,
                    "Skill Tag": taxonomy skill tag,
                    "Correlation Coefficient": similarity_score
                },
                ...
            ]

        """
        extracted = pd.DataFrame(columns=['Research ID', 'Raw Skill', 'Skill Tag', 'Correlation Coefficient'])
        for index, row in data.iterrows():
            research_id = row[id_column]
            input_text = row[text_column]
            raw_skills = self.extract_raw(input_text)
            aligned_skills = self.align_skills(raw_skills, research_id)
            extracted = extracted._append(aligned_skills, ignore_index=True)
        return extracted