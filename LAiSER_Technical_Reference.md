# LAiSER Technical Reference Guide

## Leveraging Artificial Intelligence for Skill Extraction & Research

This technical reference guide provides detailed information about the LAiSER package's implementation, APIs, and internal architecture.

## Table of Contents

1. [Package Structure](#package-structure)
2. [Core Classes and Methods](#core-classes-and-methods)
3. [Data Flow](#data-flow)
4. [LLM Integration](#llm-integration)
5. [FAISS Implementation](#faiss-implementation)
6. [Prompt Engineering](#prompt-engineering)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Testing](#testing)
10. [Advanced Usage](#advanced-usage)

## Package Structure

The LAiSER package is organized into several modules, each with a specific responsibility:

```
laiser/
├── __init__.py                  # Package initialization and version
├── config.py                    # Configuration constants and templates
├── exceptions.py                # Custom exception classes
├── data_access.py               # Data loading and FAISS management
├── services.py                  # Business logic services
├── skill_extractor.py           # Legacy skill extractor
├── skill_extractor_refactored.py # New refactored interface
├── llm_methods.py               # LLM utility methods
├── params.py                    # Legacy parameters (deprecated)
├── utils.py                     # Utility functions
├── llm_models/                  # LLM integration modules
│   ├── __init__.py
│   ├── model_loader.py          # Model loading utilities
│   ├── llm_router.py            # LLM routing logic
│   ├── gemini.py                # Gemini API integration
│   └── hugging_face_llm.py      # HuggingFace integration
└── public/                      # Public assets
    ├── combined.csv             # Combined skills data
    ├── skills_df.json           # Skills data in JSON format
    ├── skills_v03.index         # FAISS index for skills
    └── esco_faiss_index.index   # FAISS index for ESCO skills
```

## Core Classes and Methods

### SkillExtractorRefactored

The main entry point for the LAiSER package, providing a clean interface for skill extraction and alignment.

#### Key Methods

```python
def __init__(self, model_id=None, hf_token=None, api_key=None, use_gpu=None):
    """Initialize the skill extractor with model configuration."""
    
def _initialize_components(self):
    """Initialize required components based on configuration."""
    
def _initialize_spacy(self):
    """Initialize SpaCy model."""
    
def _initialize_vllm(self):
    """Initialize vLLM model."""
    
def _initialize_transformer(self):
    """Initialize transformer model."""
    
def align_skills(self, raw_skills, document_id='0', description=''):
    """Align raw skills to taxonomy."""
    
def extract_and_map_skills(self, input_data, text_columns):
    """Extract skills from input data using LLM."""
    
def extract_and_align(self, data, id_column='Research ID', text_columns=None, 
                     input_type="job_desc", top_k=None, levels=False, 
                     batch_size=32, warnings=True):
    """Extract and align skills from a dataset."""
```

### SkillExtractionService

Handles the core business logic for skill extraction and alignment.

#### Key Methods

```python
def extract_skills_basic(self, input_data, input_type="job_desc"):
    """Extract basic skills from text using simple extraction."""
    
def extract_skills_with_ksa(self, input_data, input_type="job_desc", 
                           num_skills=5, num_knowledge="3-5", num_abilities="3-5"):
    """Extract skills with Knowledge, Skills, Abilities details."""
    
def get_skill_details(self, skill, context, num_knowledge=3, num_abilities=3):
    """Get detailed KSA information for a specific skill."""
    
def align_extracted_skills(self, raw_skills, document_id='0', description=''):
    """Align extracted skills to taxonomy."""
```

### FAISSIndexManager

Manages FAISS indices for efficient skill similarity search.

#### Key Methods

```python
def initialize_index(self, force_rebuild=False):
    """Initialize FAISS index (load or build)."""
    
def search_similar_skills(self, query_embedding, top_k=25):
    """Search for similar skills using FAISS index."""
```

### DataAccessLayer

Handles data loading and external API calls.

#### Key Methods

```python
def get_embedding_model(self):
    """Get or initialize the embedding model."""
    
def load_esco_skills(self):
    """Load ESCO skills taxonomy data."""
    
def load_combined_skills(self):
    """Load combined skills taxonomy data."""
    
def build_faiss_index(self, skill_names):
    """Build FAISS index for given skill names."""
    
def save_faiss_index(self, index, file_path):
    """Save FAISS index to file."""
    
def load_faiss_index(self, file_path):
    """Load FAISS index from file."""
    
def download_faiss_index(self, url, local_path):
    """Download FAISS index from URL."""
```

## Data Flow

The data flow in LAiSER follows these steps:

1. **Input Processing**:
   - User provides a DataFrame with text data (job descriptions, syllabi)
   - Text columns are identified and extracted

2. **Skill Extraction**:
   - Text is preprocessed to remove irrelevant information
   - LLM is prompted to extract skills
   - Response is parsed to extract skill names

3. **Taxonomy Alignment**:
   - Extracted skills are embedded using SentenceTransformers
   - FAISS index is used to find similar skills in the taxonomy
   - Matches above a similarity threshold are retained

4. **KSA Analysis**:
   - For each aligned skill, knowledge and task abilities are extracted
   - LLM is prompted with the skill and context to generate KSA details

5. **Result Compilation**:
   - Results are compiled into a structured DataFrame
   - Output includes original IDs, raw skills, aligned skills, and KSA details

## LLM Integration

LAiSER supports multiple LLM backends through a flexible routing system.

### LLM Router

The `llm_router` function in `llm_models/llm_router.py` routes requests to the appropriate LLM implementation:

```python
def llm_router(prompt, model_id, use_gpu, llm, tokenizer=None, model=None, api_key=None):
    """Route LLM requests to appropriate model implementation."""
    if model_id == 'gemini':
        return gemini_generate(prompt, api_key)
    return llm_generate_vllm(prompt, llm)
```

### Supported Models

1. **Gemini API**:
   - Cloud-based inference using Google's Gemini models
   - Requires an API key
   - Implementation in `llm_models/gemini.py`

2. **HuggingFace Transformers**:
   - Local inference using various transformer models
   - Supports quantization for efficient inference
   - Implementation in `llm_models/hugging_face_llm.py`

3. **vLLM**:
   - Optimized inference for faster processing
   - Supports various quantization methods (AWQ, GPTQ)
   - Implementation in `llm_models/model_loader.py`

### Model Loading

Models are loaded using the functions in `llm_models/model_loader.py`:

```python
def load_model_from_transformer(model_id=None, token=""):
    """Load model using transformers library."""
    
def load_model_from_vllm(model_id=None, token=None, dtype=None, quantization=None):
    """Load model using vLLM library."""
```

## FAISS Implementation

LAiSER uses FAISS (Facebook AI Similarity Search) for efficient similarity search between extracted skills and taxonomy skills.

### Index Creation

FAISS indices are created using the `build_faiss_index` method in `DataAccessLayer`:

```python
def build_faiss_index(self, skill_names):
    """Build FAISS index for given skill names."""
    model = self.get_embedding_model()
    embeddings = model.encode(skill_names, convert_to_numpy=True, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index
```

### Similarity Search

Similarity search is performed using the `search_similar_skills` method in `FAISSIndexManager`:

```python
def search_similar_skills(self, query_embedding, top_k=25):
    """Search for similar skills using FAISS index."""
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    faiss.normalize_L2(q)
    
    scores, indices = self.index.search(q, top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue
        if 0 <= idx < len(self.skill_names):
            results.append({
                "Skill": self.skill_names[idx],
                "Similarity": float(score),
                "Rank": rank
            })
    
    return results
```

## Prompt Engineering

LAiSER uses carefully crafted prompts to extract skills and KSA details from text. These prompts are defined in `config.py`.

### Skill Extraction Prompt

```
[INST]user
Name all the skills present in the following description in a single list. Response should be in English and have only the skills, no other information or words. Skills should be keywords, each being no more than 3 words.
Below text is the Description:

{query}
[/INST]
[INST]model
```

### KSA Extraction Prompt

```
[INST]user
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
{scqf_levels}

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

[/INST]
[INST]model
```

### KSA Details Prompt

```
[INST]user
Given the following context, provide concise lists for the specified skill.

Skill: {skill}

Context:
{description}

For the skill above produce:
- Knowledge Required: {num_key_kr} bullet items, each ≤ 3 words.
- Task Abilities: {num_key_tas} bullet items, each ≤ 3 words.

Respond strictly in valid JSON with the exact keys 'Knowledge Required' and 'Task Abilities'.
[/INST]
[INST]model
```

## Error Handling

LAiSER uses a comprehensive error handling system with custom exceptions defined in `exceptions.py`.

### Custom Exceptions

```python
class LAiSERError(Exception):
    """Base exception class for LAiSER project"""
    pass

class ModelLoadError(LAiSERError):
    """Raised when model loading fails"""
    pass

class VLLMNotAvailableError(LAiSERError):
    """Raised when vLLM is required but not available"""
    pass

class SkillExtractionError(LAiSERError):
    """Raised when skill extraction fails"""
    pass

class FAISSIndexError(LAiSERError):
    """Raised when FAISS index operations fail"""
    pass

class InvalidInputError(LAiSERError):
    """Raised when input validation fails"""
    pass
```

### Error Handling Strategy

LAiSER follows these error handling principles:

1. **Graceful Degradation**: If a preferred method fails, fall back to alternatives
2. **Informative Messages**: Provide clear error messages with context
3. **Logging**: Log errors and warnings for debugging
4. **Recovery**: Attempt to recover from errors when possible

Example from `SkillExtractorRefactored._initialize_components`:

```python
try:
    # Initialize SpaCy model
    self._initialize_spacy()
    
    # Initialize LLM components
    if self.model_id == 'gemini':
        print("Using Gemini API for skill extraction...")
        # No local model needed for Gemini
        return
    elif self.use_gpu and torch.cuda.is_available():
        print("GPU available. Attempting to initialize vLLM model...")
        try:
            self._initialize_vllm()
            if self.llm is not None:
                print("vLLM initialization successful!")
                return
        except Exception as e:
            print(f"WARNING: vLLM initialization failed: {e}")
            print("Falling back to transformer model...")
            
        # Fallback to transformer
        try:
            self._initialize_transformer()
            if self.model is not None:
                print("Transformer model fallback successful!")
                return
        except Exception as e:
            print(f"WARNING: Transformer model fallback also failed: {e}")
    else:
        print("Using CPU/transformer model...")
        try:
            self._initialize_transformer()
            if self.model is not None:
                print("Transformer model initialization successful!")
                return
        except Exception as e:
            print(f"WARNING: Transformer model initialization failed: {e}")
    
    # If all else fails, warn but continue
    print("WARNING: No model successfully initialized. Extraction methods may have limited functionality.")
    print("TIP: Consider using Gemini API by setting model_id='gemini' and providing an api_key.")
        
except Exception as e:
    raise LAiSERError(f"Critical failure during component initialization: {e}")
```

## Performance Considerations

### Memory Usage

- FAISS indices can consume significant memory, especially for large taxonomies
- LLMs require substantial GPU memory (15+ GB recommended)
- Batch processing helps manage memory usage for large datasets

### Optimization Techniques

1. **vLLM Acceleration**: Uses vLLM for optimized inference when available
2. **Quantization**: Supports 8-bit quantization for transformer models
3. **Batch Processing**: Processes data in configurable batches
4. **FAISS Indexing**: Uses efficient vector similarity search
5. **Lazy Loading**: Loads resources only when needed

### GPU Considerations

- LAiSER automatically detects GPU availability using `torch.cuda.is_available()`
- When a GPU is available, LAiSER attempts to use vLLM for optimized inference
- If vLLM initialization fails, it falls back to transformer models with quantization
- For CPU-only environments, LAiSER uses transformer models with 8-bit quantization or SkillNer

### Scaling Considerations

- For processing large datasets (1000+ items), consider:
  - Increasing batch size for more efficient processing
  - Using a more powerful GPU for faster inference
  - Using the Gemini API to offload computation to the cloud
  - Splitting the dataset into smaller chunks

## Testing

LAiSER includes test scripts to verify functionality:

### run_extract_and_align.py

Tests the full extraction and alignment pipeline:

```python
import pandas as pd
from laiser.skill_extractor_refactored import SkillExtractorRefactored

# Sample data
data = pd.DataFrame([
    {
        "Research ID": "aetna_trainer_001",
        "description": "POSITION SUMMARY: This position requires curriculum development..."
    },
    {
        "Research ID": "aetna_trainer_002",
        "description": "Looking for someone with a strong technical training background..."
    },
])

# Initialize extractor
extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key="your_api_key",
    use_gpu=False
)

# Extract and align skills
results = extractor.extract_and_align(
    data=data,
    id_column='Research ID',
    text_columns=['description'],
    warnings=True
)

print(results)
```

### run_align_test.py

Tests the skill alignment functionality:

```python
from laiser.services import SkillAlignmentService, DataAccessLayer, FAISSIndexManager

# Sample skills
raw_skills = [
    "Curriculum development",
    "Technical training",
    "Medicaid state websites",
    # ...
]

# Initialize services
data_access = DataAccessLayer()
faiss_manager = FAISSIndexManager(data_access)
aligner = SkillAlignmentService(data_access, faiss_manager)

# Align skills
aligned = aligner.align_skills_to_taxonomy(
    raw_skills=raw_skills,
    document_id="aetna_trainer_001",
    description="POSITION SUMMARY...",
    similarity_threshold=0.30,
    top_k=10
)

print("\nAligned Skills:")
for skill in aligned:
    print(f" - {skill}")
```

## Advanced Usage

### Custom Taxonomies

LAiSER can be extended to use custom skill taxonomies:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize extractor
extractor = SkillExtractorRefactored()

# Load custom taxonomy
custom_taxonomy = pd.read_csv("path/to/custom_taxonomy.csv")

# Replace the default taxonomy
extractor.skill_service.data_access._combined_df = custom_taxonomy

# Build a new FAISS index
skill_names = custom_taxonomy["skill_name"].tolist()
index = extractor.skill_service.data_access.build_faiss_index(skill_names)
extractor.skill_service.alignment_service.faiss_manager.index = index
extractor.skill_service.alignment_service.faiss_manager.skill_names = skill_names

# Use the extractor with custom taxonomy
results = extractor.extract_and_align(data, id_column="id", text_columns=["description"])
```

### Customizing Prompts

You can customize the prompts used for skill extraction:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize extractor
extractor = SkillExtractorRefactored()

# Customize the skill extraction prompt
extractor.skill_extraction_prompt = lambda cleaned_description: f"""
task: "Domain-Specific Skill Extraction"

description: |
You are an expert AI system specialized in extracting technical skills from {your_domain} job descriptions.
Your goal is to analyze the following job description and output only the specific skill names.

job_description: |
{cleaned_description}

### OUTPUT FORMAT
{{
"skills": [
    "skill1",
    "skill2",
    "skill3"
]
}}
"""

# Use the extractor with custom prompt
results = extractor.extract_and_align(data, id_column="id", text_columns=["description"])
```

### Batch Processing with Progress Tracking

For large datasets, you can implement progress tracking:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd
from tqdm.auto import tqdm

# Load large dataset
data = pd.read_csv("large_dataset.csv")

# Initialize extractor
extractor = SkillExtractorRefactored()

# Process in smaller batches with progress tracking
batch_size = 50
results = []

for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
    batch = data.iloc[i:i+batch_size]
    batch_results = extractor.extract_and_align(
        data=batch,
        id_column='id',
        text_columns=['description'],
        batch_size=10  # Sub-batch size for LLM processing
    )
    results.append(batch_results)

# Combine results
final_results = pd.concat(results, ignore_index=True)
```

### Advanced Error Handling

Implement more robust error handling for production use:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
from laiser.exceptions import LAiSERError, ModelLoadError, FAISSIndexError
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='laiser.log'
)
logger = logging.getLogger('laiser')

# Initialize extractor with error handling
try:
    extractor = SkillExtractorRefactored(
        model_id="gemini",
        api_key="your_api_key"
    )
except ModelLoadError as e:
    logger.error(f"Model loading failed: {e}")
    # Fall back to CPU mode
    extractor = SkillExtractorRefactored(use_gpu=False)
except LAiSERError as e:
    logger.error(f"LAiSER initialization failed: {e}")
    raise

# Process data with error handling
try:
    results = extractor.extract_and_align(
        data=data,
        id_column='id',
        text_columns=['description']
    )
except FAISSIndexError as e:
    logger.error(f"FAISS index error: {e}")
    # Try rebuilding the index
    extractor.skill_service.faiss_manager.initialize_index(force_rebuild=True)
    results = extractor.extract_and_align(
        data=data,
        id_column='id',
        text_columns=['description']
    )
except LAiSERError as e:
    logger.error(f"Extraction failed: {e}")
    raise
```
