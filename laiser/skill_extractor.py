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
[1.0.7]     07/03/2024      Satya Phanindra K.  Added CONDITIONAL GPU support for Fine-Tuned Language Model and error handling
[1.0.9]     07/08/2024      Satya Phanindra K.  Added support for SkillNer model for skill extraction, if GPU not available
[1.0.8]     07/11/2024      Satya Phanindra K.  Calculate cosine similarities in bulk for optimal performance.
[1.0.9]     07/15/2024      Satya Phanindra K.  Error handling for empty list outputs from extract_raw function


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
import spacy
import torch
import numpy as np
import pandas as pd
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from sklearn.metrics.pairwise import cosine_similarity
from skillNer.skill_extractor_class import SkillExtractor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.spatial.distance import cdist


# internal packages
from laiser.utils import get_embedding, cosine_similarity
from laiser.params import AI_MODEL_ID, SIMILARITY_THRESHOLD, SKILL_DB_PATH
from laiser.llm_methods import get_completion


class Skill_Extractor:
    """
    Class to extract skills from text and align them to existing taxonomy
    ...

    Attributes
    ----------
    client : HuggingFace API client
    nlp : spacy nlp model
    
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
        self.model_id = AI_MODEL_ID
        self.nlp = spacy.load("en_core_web_lg")
        self.skill_db_df = pd.read_csv(SKILL_DB_PATH)
        self.skill_db_embeddings = np.array([get_embedding(self.nlp, label) for label in self.skill_db_df['SkillLabel']])
        if torch.cuda.is_available():
            print("GPU is available. Using GPU for Fine-tuned Language model initialization.")
            torch.cuda.empty_cache()
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=self.bnb_config,
                device_map={"": 0},
                token="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"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, add_eos_token=True, padding_side='left', token="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")
        else:
            print("GPU is not available. Using CPU for SkillNer model initialization.")
            self.ner_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
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
        
        if torch.cuda.is_available():
            # GPU is available. Using Language model for extraction.
            extracted_skills = get_completion(input_text, self.model, self.tokenizer)
            extracted_skills_set = set(extracted_skills)
            torch.cuda.empty_cache()   
        else: 
            # GPU is not available. Using SkillNer model for extraction.
            ner_extractor = self.ner_extractor
            extracted_skills_set = set()
            annotations = None
            try:
                annotations = ner_extractor.annotate(input_text)
            except ValueError as e:
                print(f"Skipping example, ValueError encountered: {e}")
            except Exception as e:
                print(f"Skipping example, An unexpected error occurred: {e}")

            for item in annotations['results']['full_matches']:
                extracted_skills_set.add(item['doc_node_value'])

            # get ngram_scored
            for item in annotations['results']['ngram_scored']:
                extracted_skills_set.add(item['doc_node_value'])

            return list(extracted_skills_set)
            
        return list(extracted_skills_set)

    def align_skills(self, raw_skills, document_id='0'):
        """
        This function aligns the skills provided to the available taxonomy

        Parameters
        ----------
        raw_skills : list
            Provide list of skill extracted from Job Descriptions / Syllabus.

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
        raw_skill_embeddings = np.array([get_embedding(self.nlp, skill) for skill in raw_skills])

        # Calculate cosine similarities in bulk
        similarities = 1 - cdist(raw_skill_embeddings, self.skill_db_embeddings, metric='cosine')

        matches = []
        for i, raw_skill in enumerate(raw_skills):
            skill_matches = np.where(similarities[i] > SIMILARITY_THRESHOLD)[0]
            for match in skill_matches:
                matches.append({
                    "Research ID": document_id,
                    "Raw Skill": raw_skill,
                    "Skill Tag": self.skill_db_df.iloc[match]['SkillTag'],
                    "Correlation Coefficient": similarities[i, match]
                })

        return matches

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