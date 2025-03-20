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
[1.0.10]    11/24/2024      Prudhvi Chekuri     Added support for skills extraction from syllabi data
[1.1.0]     03/12/2025      Prudhvi Chekuri     Added support for extracting KSAs from text and aligning them to the taxonomy
[1.1.1]     03/15/2025      Prudhvi Chekuri     Add exception handling


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
from vllm import LLM
from tqdm.auto import tqdm
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from sklearn.metrics.pairwise import cosine_similarity
from skillNer.skill_extractor_class import SkillExtractor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.spatial.distance import cdist


# internal packages
from laiser.utils import get_embedding, cosine_similarity
from laiser.params import SIMILARITY_THRESHOLD, SKILL_DB_PATH
from laiser.llm_methods import get_completion, get_completion_vllm


class Skill_Extractor:
    """
    Class to extract skills from text and align them to existing taxonomy
    ...

    Attributes
    ----------
    model_id: string
        Model ID for Large Language Model
    HF_TOKEN: string
        HuggingFace Token for restricted models under gated HF repos.
    use_gpu: boolean
        Flag to use GPU for Large Language Model
    nlp: spacy model
        Spacy model for NER
    skill_db_df: pandas dataframe
        Dataframe containing taxonomy skills
    skill_db_embeddings: numpy array
        Array containing embeddings of taxonomy skills
    llm: LLM model
        Large Language Model for skill extraction
    ner_extractor: SkillExtractor
        SkillNer model for CPU skill extraction
    
    Methods
    -------
    extract_raw(input_text: text)
        The function extracts skills from text using NER model
        
    extractor(data: pandas dataframe, id_column='Research ID', text_column='Text'):
        Function takes text dataset to extract and aligns skills based on available taxonomies
        
    align_KSAs(extracted_df: pandas dataframe, id_column='Research ID'):
        This function aligns the KSAs provided to the available taxonomy    
    ....

    """

    def __init__(self, AI_MODEL_ID=None, HF_TOKEN=None, use_gpu=None):
        self.model_id = AI_MODEL_ID
        self.HF_TOKEN=HF_TOKEN
        self.use_gpu=use_gpu if use_gpu else torch.cuda.is_available()
        
        try:
            print("Found 'en_core_web_lg' model. Loading...")
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading 'en_core_web_lg' model...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")

        self.skill_db_df = pd.read_csv(SKILL_DB_PATH)
        self.skill_db_embeddings = np.array([get_embedding(self.nlp, label) for label in self.skill_db_df['SkillLabel']])
        if torch.cuda.is_available() and self.use_gpu:
            print("GPU is available. Using GPU for Large Language model initialization...")
            
            try:
                torch.cuda.empty_cache()
            
                # Use quantization to reduce the model size and memory usage
                if self.model_id:
                    self.llm = LLM(model=self.model_id, dtype="float16", quantization='gptq')
                else:
                    self.llm = LLM(model="marcsun13/gemma-2-9b-it-GPTQ", dtype="float16", quantization='gptq')   
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
                raise
        else:
            print("GPU is not available. Using CPU for SkillNer model initialization...")
            self.ner_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
        return


    # Declaring a private method for extracting raw skills from input text
    def extract_raw(self, input_text, text_columns, id_column, input_type, batch_size):
        """
        The function extracts skills from text using Fine-Tuned Language Model's API

        Parameters
        ----------
        input_text : pandas Series with text data
            Job advertisement / Job Description / Syllabus Description / Course Outcomes etc.
        id_column: string
            Name of id column in the dataset. Defaults to 'Research ID'
        text_columns: list
            Name of the text columns in the dataset. Defaults to 'description'
        input_type: string
            Type of input data. Defaults to 'job_desc'
        batch_size: int
            Number of examples to process in each batch. Defaults to 32

        Returns
        -------
        list: List of extracted skills from text

        """    
        
        if torch.cuda.is_available() and self.use_gpu:
            # GPU is available. Using Language model for extraction.
            extracted_skills_set = get_completion_vllm(input_text, text_columns, id_column, input_type, self.llm, batch_size)
            torch.cuda.empty_cache()
        else: 
            # GPU is not available. Using SkillNer model for extraction.
            ner_extractor = self.ner_extractor
            extracted_skills_set = set()
            annotations = None
            try:
                
                if input_type == "job_desc":
                  input_text = input_text[text_columns][0]
                elif input_type == "syllabus":
                  input_text = f"Course Description: {input_text[text_columns[0]]}\n\nLearning Outcomes: {input_text[text_columns[1]]}"
                
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


    def align_KSAs(self, extracted_df, id_column):
        
        """
        This function aligns the KSAs provided to the available taxonomy
        
        Parameters
        ----------
        extracted_df : pandas dataframe
            Dataset containing extracted KSAs from text and their details.
        id_column: string
            Name of id column in the dataset. Defaults to 'Research ID'
            
        Returns
        -------
        list: List of taxonomy skills from text in JSON format

            [
                {
                    "Research ID": text_id,
                    "Description": String, (optional)
                    "Learning Outcomes": List of Strings, (optional)
                    "Raw Skill": String,
                    "Level": int, (optional)
                    "Knowledge Required": String, (optional)
                    "Task Abilities": String, (optional)
                    "Skill Tag": String,
                    "Correlation Coefficient": float
                },
                ...
            ]
        """
        
        matches = []

        extracted_df = extracted_df[
                extracted_df['Skill'].apply(lambda x: isinstance(x, str) and x.strip() != '')
            ]


        try:
            raw_skill_embeddings = np.array([get_embedding(self.nlp, skill) for skill in extracted_df['Skill']])
            # Calculate cosine similarities in bulk
            similarities = 1 - cdist(raw_skill_embeddings, self.skill_db_embeddings, metric='cosine')
        except Exception as e:
            print(f"Error computing embeddings or similarities in align_KSAs: {e}")
            return []

        for i, raw_skill in tqdm(enumerate(extracted_df['Skill'])):
            try:
                skill_matches = np.where(similarities[i] > SIMILARITY_THRESHOLD)[0]
                for match in skill_matches:
                    matches.append({
                        "Research ID": extracted_df.iloc[i][id_column],
                        "Description": extracted_df.iloc[i]['description'],
                        "Learning Outcomes": extracted_df.iloc[i]['learning_outcomes'],
                        "Raw Skill": raw_skill,
                        "Level": extracted_df.iloc[i]['Level'],
                        "Knowledge Required": extracted_df.iloc[i]['Knowledge Required'],
                        "Task Abilities": extracted_df.iloc[i]['Task Abilities'],
                        "Skill Tag": f"ESCO.{match}",
                        "Correlation Coefficient": similarities[i, match]
                    })
            except:
                continue

        return matches


    def extractor(self, data, id_column='Research ID', text_columns=["description"], input_type="job_desc", levels=False, batch_size=32):
        """
        Function takes text dataset to extract and aligns skills based on available taxonomies

        Parameters
        ----------
        data : pandas dataframe
            Dataset containing text id and actual text to extract skills.
        id_column: string
            Name of id column in the dataset. Defaults to 'Research ID'
        text_columns: list
            Name of the text columns in the dataset. Defaults to 'description'
        input_type: string
            Type of input data. Defaults to 'job_desc'
        levels: boolean
            Whether to include skill levels in the output. Defaults to False.
        batch_size: int
            Number of examples to process in each batch. Defaults to 32

        Returns
        -------
        For CPU:
        list: List of skill tags and similarity_score for all texts in  from text in JSON format
            [
                {
                    "Research ID": text_id,
                    "Skill Tag": String, 
                    "Description": String, (optional)
                    "Learning Outcomes": List of Strings, (optional)
                    "Raw Skill": String,
                    "Level": int, (optional)
                    "Knowledge Required": String, (optional)
                    "Task Abilities": String, (optional)
                    "Correlation Coefficient": float
                },
                ...
            ]
            
        For GPU:
        pandas dataframe with below columns:
            - "Research ID": text_id
            - "Description": text description
            - "Learning Outcomes": learning outcomes
            - "Raw Skill": Raw skill extracted
            - "Level": Level of the skill
            - "Knowledge Required": Knowledge required for the skill
            - "Task Abilities": Task abilities
            - "Skill Tag": taxonomy skill tag
            - "Correlation Coefficient": similarity_score

        """

        if torch.cuda.is_available() and self.use_gpu:
            
            assert not data.empty, "Input data is empty, pass a valid input..."
            
            try:
                KSAs = self.extract_raw(data, text_columns, id_column, input_type)
            
                extracted_df = pd.DataFrame(KSAs)
                
                if input_type != 'syllabus':
                    extracted_df["learning_outcomes"] = ''

                selected_columns = [id_column, "learning_outcomes", "Skill", "Knowledge Required", "Task Abilities"]
                if levels:
                    selected_columns.insert(3, "Level")  # Insert "Level" at the correct position
                
                extracted_df = extracted_df[selected_columns]
                matches = self.align_KSAs(extracted_df, id_column)
                
                extracted_columns = ['Research ID', 'Learning Outcomes', 'Raw Skill', 'Knowledge Required', 'Task Abilities', 'Skill Tag', 'Correlation Coefficient']
                if levels:
                    extracted_columns.insert(3, 'Level')  # Include "Level" column in the final dataframe

                extracted = pd.DataFrame(columns=extracted_columns)
                extracted = extracted._append(matches, ignore_index=True)

                if input_type != "syllabus":
                    extracted.drop("Learning Outcomes", axis=1, inplace=True)
                
            except Exception as e:
                print(f"Error in extraction pipeline: {e}")
                extracted = pd.DataFrame()
        else:
            extracted = pd.DataFrame(columns=['Research ID', 'Raw Skill', 'Skill Tag', 'Correlation Coefficient'])
            for _, row in data.iterrows():
                research_id = row[id_column]
                raw_skills = self.extract_raw(row, text_columns, id_column, input_type)
                if len(raw_skills) == 0:
                    continue
                else:
                    aligned_skills = self.align_skills(raw_skills, research_id)
                    extracted = extracted._append(aligned_skills, ignore_index=True)

        return extracted

