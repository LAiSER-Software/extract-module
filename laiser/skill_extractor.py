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
[1.1.2]     03/20/2025      Prudhvi Chekuri     Fix Levels Toggle
[1.2.0]     06/12/2025      Satya Phanindra K.  Added support for ESCO skills using FAISS index and SentenceTransformers
[1.2.1]     06/29/2025      Anket Patil         Added support for API-based LLM models 



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
from pathlib import Path

# installed packages
import spacy
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from sklearn.metrics.pairwise import cosine_similarity
from skillNer.skill_extractor_class import SkillExtractor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import faiss
from scipy.spatial.distance import cdist
import requests

# internal packages
from laiser.utils import get_embedding
from laiser.params import SIMILARITY_THRESHOLD, SKILL_DB_PATH
from laiser.llm_methods import get_completion, get_completion_vllm,get_ksa_details
from laiser.llm_models.model_loader import load_model_from_vllm

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
    def build_faiss_index_esco(self):
        """
        Builds a FAISS index for ESCO skills

        Returns
        -------
        faiss index object
        """

        # Embed ESCO skills using SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("Embedding ESCO skills...")
        esco_embeddings = model.encode(self.skill_names, convert_to_numpy=True, show_progress_bar=True)

        # ⚡ Normalize & Index using FAISS (cosine sim = L2 norm + dot product)
        dimension = esco_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(esco_embeddings)
        index.add(esco_embeddings)
        
        # save the index to disk
        # TODO: switch from FAISS to a more persistent and cloud supported vector database.
        print("Saving FAISS index to disk...")
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create input directory if it doesn't exist
        input_dir = os.path.join(script_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        # Save index to input directory
        index_path = os.path.join(input_dir, "esco_faiss_index.index")
        faiss.write_index(index, index_path)
        print("FAISS index for ESCO skills built and saved for reusability.")
        self.index = index
        return index
    
    def load_faiss_index_esco(self):
        """
        Loads a FAISS index for ESCO skills

        Returns
        -------
        faiss index object
        """
        try:
            print("Loading FAISS index for ESCO skills...")

            # 1. Path to package-included file
            local_index_path = Path(__file__).parent / "public" / "esco_faiss_index.index"

            # 2. GitHub fallback URL
            raw_url = "https://raw.githubusercontent.com/LAiSER-Software/extract-module/main/laiser/input/esco_faiss_index.index"

            # 3. Check local package path
            if not local_index_path.exists():
                print(f"FAISS index not found locally at {local_index_path}. Downloading from {raw_url} ...")
                response = requests.get(raw_url, timeout=10)  # Add timeout to prevent hanging requests
                response.raise_for_status()
                
                # Validate response content
                if response.headers.get("Content-Type") != "application/octet-stream":
                    raise ValueError(f"Unexpected content type: {response.headers.get('Content-Type')}")
                
                local_index_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                with open(local_index_path, "wb") as f:
                    f.write(response.content)
                print("Download complete.")

            # 4. Load FAISS index
            index = faiss.read_index(str(local_index_path))
            print("FAISS index for ESCO skills loaded successfully.")
            self.index = index
            return index

        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("ESCO FAISS index not found. Building a new index on system memory...")
            self.index = None
            return None

    def __init__(self, AI_MODEL_ID=None, HF_TOKEN=None,api_key=None, use_gpu=None):
        self.model_id = AI_MODEL_ID
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.HF_TOKEN=HF_TOKEN
        self.api_key = api_key
        self.use_gpu = use_gpu if use_gpu is not None else False
        # Ensure the FAISS index attribute is always defined
        self.index = None  # Will be populated after loading or building the index
        # Load ESCO data and FAISS index
        print("Loading ESCO skill taxonomy data...")
        self.esco_df = pd.read_csv("https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/ESCO_skills_Taxonomy.csv")
        self.skill_names = self.esco_df["preferredLabel"].tolist()
        
        # Load FAISS index        
        self.load_faiss_index_esco()
        if self.index is None:
            print("ESCO FAISS index not found. Building a new index on system memory...")
            self.build_faiss_index_esco()
        
        try:
            print("Found 'en_core_web_lg' model. Loading...")
            self.nlp = spacy.load("en_core_web_lg")
            
        except OSError:
            print("Downloading 'en_core_web_lg' model...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")

        self.skill_db_df = pd.read_csv(SKILL_DB_PATH)
        self.skill_db_embeddings = np.array([get_embedding(self.nlp, label) for label in self.skill_db_df['SkillLabel']])
        
        # Initialize llm attribute
        self.llm = None
        
        if torch.cuda.is_available() and self.use_gpu:
            print("GPU is available. Using GPU for Large Language model initialization...")
            
            try:
                torch.cuda.empty_cache()
                self.llm = load_model_from_vllm( self.model_id,self.HF_TOKEN)
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
                raise
        elif self.model_id == 'gemini':
            print("Using Gemini API for skill extraction...")
            # For Gemini, llm remains None as it's handled by the API
        else:
            if self.model_id != 'gemini':
                print("GPU is not available. Using CPU for SkillNer model initialization...")
                self.ner_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
        return


    # TODO: Deprecate flow, handle this abandoned flow in future releases
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
            
    # TODO: Deprecate flow, handle this abandoned flow in future releases
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

    # TODO: Deprecate flow, handle this abandoned flow in future releases
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

   
    def get_top_esco_skills(self, input_text, top_k=25):
        """
        Retrieve the top-k ESCO skills most semantically similar to the input text using the
        Sentence-Transformer embeddings and the pre-built FAISS index that were both loaded
        during __init__.

        Parameters
        ----------
        input_text : str
            The text (e.g., job description) for which to retrieve similar ESCO skills.
        top_k : int, optional
            Number of top skills to return (default 25).

        Returns
        -------
        list[dict]
            Each dictionary contains:
                - "Skill": str  – ESCO preferredLabel.
                - "index": int  – Position of the skill in the taxonomy list (used for tagging).
                - "score": float – Cosine similarity score (0-1).
        """
        # Encode the input text once using the cached embedding model
        emb = self.embedding_model.encode(input_text, convert_to_numpy=True)
        faiss.normalize_L2(emb.reshape(1, -1))

        # Search within the FAISS index
        scores, indices = self.index.search(emb.reshape(1, -1), top_k)

        return [
            {
                "Skill": self.skill_names[idx],
                "index": int(idx),
                "score": float(scores[0][rank])
            }
            for rank, idx in enumerate(indices[0])
        ]

        
    def extractor(self, data, id_column='Research ID', text_columns=["description"], input_type="job_desc", top_k= None, levels=False, batch_size=32, warnings=True):
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
        warnings: boolean
            Whether to display development mode warnings. Defaults to True.

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
        # ------------------------------------------------------------------
        # NEW OUTPUT PIPELINE (v1.2.0 – 2025-06-12)
        # Produces the required columns:
        # Research ID, Description, Raw Skill, Knowledge Required, Task Abilities,
        # Skill Tag, Correlation Coefficient
        # ------------------------------------------------------------------
        output_records = []
        for _, row in data.iterrows():
            research_id = row[id_column]

            # Build the description text that we will use for semantic search & LLM context
            if input_type == "job_desc":
                description_text = row[text_columns[0]]
            elif input_type == "syllabus" and len(text_columns) >= 2:
                description_text = (
                    f"Course Description: {row[text_columns[0]]}\n\n"
                    f"Learning Outcomes: {row[text_columns[1]]}"
                )
            else:
                # Fallback – concatenate all specified text columns separated by space
                description_text = " ".join([str(row[col]) for col in text_columns])

            # Retrieve the top-k ESCO skills via FAISS semantic search
            try:
                top_k_val = top_k if (top_k is not None and top_k > 0) else 25
                top_esco = self.get_top_esco_skills(description_text, top_k=top_k_val)

            except Exception as e:
                print(f"[Skill_Extractor.extractor] ESCO search failed for ID {research_id}: {e}")
                continue

            for skill_obj in top_esco:
                raw_skill = skill_obj["Skill"]
                coeff = skill_obj["score"]
                tag = f"ESCO.{skill_obj['index']}"

                # Fetch Knowledge Required & Task Abilities using the LLM if GPU/LLM available
                knowledge_required, task_abilities = [], []
                if (torch.cuda.is_available() and self.use_gpu) or self.model_id == 'gemini':
                    knowledge_required, task_abilities = get_ksa_details(raw_skill, description_text, self.model_id,self.use_gpu,llm = self.llm, tokenizer=None,model=None,api_key=self.api_key)

                output_records.append({
                    "Research ID": research_id,
                    "Description": description_text,
                    "Raw Skill": raw_skill,
                    "Knowledge Required": knowledge_required,
                    "Task Abilities": task_abilities,
                    "Skill Tag": tag,
                    "Correlation Coefficient": coeff
                })

        # If we successfully generated any records with the new specification, we can
        # return them immediately and skip the legacy pipeline that follows.
        if len(output_records) > 0:
            return pd.DataFrame(output_records)

        # ------------------------------------------------------------------
        # Legacy pipeline retained below for backwards compatibility
        # ------------------------------------------------------------------
        # Display development mode warning if warnings are enabled
        try:    
            if warnings:
                print("\033[93m" + "=" * 80 + "\033[0m")
                print("\033[93m\033[1m⚠️  WARNING: LAiSER is currently in development mode. Features may be experimental. Use with caution! ⚠️\033[0m")
                print("\033[93m" + "=" * 80 + "\033[0m")

            if torch.cuda.is_available() and self.use_gpu:
                KSAs = self.extract_raw(data, text_columns, id_column, input_type, batch_size)
            
                extracted_df = pd.DataFrame(KSAs)
                if input_type != 'syllabus':
                    extracted_df["learning_outcomes"] = ''

            if torch.cuda.is_available() and self.use_gpu:
                
                assert not data.empty, "Input data is empty, pass a valid input..."
                
                try:
                    KSAs = self.extract_raw(data, text_columns, id_column, input_type, batch_size)
                
                    extracted_df = pd.DataFrame(KSAs)
                    
                    if input_type != 'syllabus':
                        extracted_df["learning_outcomes"] = ''

                    selected_columns = [id_column, "description", "learning_outcomes", "Skill", "Level", "Knowledge Required", "Task Abilities"]
                    
                    extracted_df = extracted_df[selected_columns]
                    matches = self.align_KSAs(extracted_df, id_column)
                    
                    extracted_columns = ['Research ID', "Description", 'Learning Outcomes', 'Raw Skill', 'Level', 'Knowledge Required', 'Task Abilities', 'Skill Tag', 'Correlation Coefficient']

                    extracted = pd.DataFrame(columns=extracted_columns)
                    extracted = extracted._append(matches, ignore_index=True)

                    if input_type != "syllabus":
                        extracted.drop("Learning Outcomes", axis=1, inplace=True)

                    if not levels:
                        extracted.drop("Level", axis=1, inplace=True)

                    
                except Exception as e:
                    print(f"Error in extraction pipeline: {e}")
                    extracted = pd.DataFrame()
            else:
                extracted = pd.DataFrame(columns=['Research ID', 'Raw Skill', 'Skill Tag', 'Correlation Coefficient'])
                for _, row in data.iterrows():
                    research_id = row[id_column]
                    raw_skills = self.extract_raw(row, text_columns, id_column, input_type, batch_size)
                    if len(raw_skills) == 0:
                        continue
                    else:
                        aligned_skills = self.align_skills(raw_skills, research_id)
                        extracted = extracted._append(aligned_skills, ignore_index=True)
        except Exception as e:
            print(f"\033[91m❌ Legacy pipeline could not be executed: {e}\033[0m")
            extracted = pd.DataFrame()

        return extracted
