"""
Module Description:
-------------------
A Class with utility functions

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


"""
"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     06/01/2024      Vedant M.           Initial Version
[1.0.1]     06/10/2024      Vedant M.           added logging function


TODO:
-----
- 1:
"""

import numpy as np
import psutil
import logging

# Handle optional dependencies with proper fallbacks
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Some features may be limited.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Some features may be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some features may be limited.")

def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between 2 vectors

    Parameters
    ----------
    vec1, vec2 : numpy array of vectorized text

    Returns
    -------
    numeric value
    """
    product_of_magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if product_of_magnitude == 0.0:
        return 0.0
    return np.dot(vec1, vec2) / product_of_magnitude

# TODO: Clean/Refactor the following functions to use FAISS for ESCO skills
def build_faiss_index_esco():
    """
    Builds a FAISS index for ESCO skills

    Returns
    -------
    faiss index object
    """
    
    # Load ESCO skill data
    esco_df = pd.read_csv("https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/ESCO_skills_Taxonomy.csv")  # replace with your file if needed
    skill_names = esco_df["preferredLabel"].tolist()

    # Embed ESCO skills using SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Embedding ESCO skills...")
    esco_embeddings = model.encode(skill_names, convert_to_numpy=True, show_progress_bar=True)

    # ⚡ Normalize & Index using FAISS (cosine sim = L2 norm + dot product)
    dimension = esco_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(esco_embeddings)
    index.add(esco_embeddings)
    
    # save the index to disk
    # TODO: store this in a persistent storage like S3 or a database and remove this step from initialization
    print("Saving FAISS index to disk...")
    faiss.write_index(index, "esco_faiss_index.index")
    print("FAISS index for ESCO skills built and saved for reusability.")
    return index
    
def load_faiss_index_esco():
    """
    Loads a FAISS index for ESCO skills

    Returns
    -------
    faiss index object
    """
    if not FAISS_AVAILABLE:
        print("Warning: faiss not available. Cannot load FAISS index.")
        return None
        
    try:
        # set the path to your FAISS index file in the input directory
        print("Loading FAISS index for ESCO skills...")
        
        import os
        index_path = os.path.join(os.path.dirname(__file__), "input/esco_faiss_index.index")
        if not os.path.exists(index_path):
            print(f"Warning: FAISS index file not found at {index_path}. Some features may be limited.")
            return None
        index = faiss.read_index(index_path)
        print("FAISS index for ESCO skills loaded successfully.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def get_top_esco_skills(input_text, top_k=10):
    """
    Retrieves top ESCO skills based on cosine similarity with input text
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: sentence-transformers not available. Returning empty list.")
        return []
    
    if not FAISS_AVAILABLE:
        print("Warning: faiss not available. Returning empty list.")
        return []
    
    if not PANDAS_AVAILABLE:
        print("Warning: pandas not available. Returning empty list.")
        return []
    
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        index = load_faiss_index_esco()
        if index is None:
            print("Warning: FAISS index for ESCO skills didn't load properly. Returning empty list.")
            return []
        
        skill_names = pd.read_csv("https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/ESCO_skills_Taxonomy.csv")["preferredLabel"].tolist()
        emb = model.encode(input_text, convert_to_numpy=True)
        faiss.normalize_L2(emb.reshape(1, -1))
        scores, indices = index.search(emb.reshape(1, -1), top_k)
        return [{"Skill": skill_names[i], "index": int(i), "score": float(scores[0][j])} for j, i in enumerate(indices[0])]
    except Exception as e:
        print(f"Error in get_top_esco_skills: {e}")
        return []

def get_embedding(nlp, input_text):
    """
    Creates vector embeddings for input text based on nlp object

    Parameters
    ----------
    nlp : object of spacy nlp model
    input_text : text
        Provide text to be vectorized, usually skill, extracted of referenced

    Returns
    -------
    numpy array of vectorized text


    """
    doc = nlp(input_text)
    if len(doc) == 0:
        return np.zeros(300)  # Return zeros for empty texts
    return np.mean([word.vector for word in doc], axis=0)


def log_performance(function_name, start_time, end_time):
    """
    Utility function to log performance in unit of time for a function

    Parameters
    ----------
    function_name : text
        Name of the function
    start_time : time
        execution start time of the function
    end_time : time
        execution end time of the function

    """
    execution_time = end_time - start_time
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB

    log_message = (
        f"Function: {function_name}\n"
        f"Execution time: {execution_time:.2f} seconds\n"
        f"CPU usage: {cpu_percent:.2f}%\n"
        f"Memory usage: {memory_usage:.2f} MB\n"
        "-------------------------------"
    )
    logging.info(log_message)
    print(log_message)
