# Leveraging Artificial Intelligence for Skill Extraction & Research (LAiSER): A Taxonomy-Aware Framework for Labor Market Intelligence

**Satya Phanindra Kumar Kalaga**  
*Program on Skills, Credentials and Workforce Policy*  
*Institute of Public Policy*  
*The George Washington University*  
*Washington, DC 20052*  
*Email: pscwp@gwu.edu*

**Date:** November 2025

---

## Abstract

This paper presents LAiSER, a comprehensive artificial intelligence framework designed to extract, standardize, and align skill information from unstructured textual data with established skill taxonomies. The system addresses a critical challenge in workforce development and educational policy: the lack of standardized, machine-readable skill information that can facilitate communication between learners, educators, and employers. LAiSER employs a novel two-stage pipeline that combines advanced language models with semantic vector search to achieve high-precision skill extraction and taxonomy alignment. The framework supports multiple computational backends, including GPU-accelerated inference, cloud-based APIs, and CPU-only environments, making it accessible to diverse research and operational contexts. Initial deployments demonstrate the system's capability to process large-scale datasets while maintaining accuracy in skill identification and classification.

**Keywords:** skill extraction, natural language processing, labor market intelligence, taxonomy alignment, large language models, workforce analytics, semantic similarity, FAISS, ESCO

---

## 1. Introduction

### 1.1 Motivation

The contemporary labor market is characterized by rapid technological change, evolving skill requirements, and a growing disconnect between educational curricula and industry needs (Autor & Dorn, 2013). This misalignment creates significant challenges for multiple stakeholders: learners struggle to identify relevant competencies, educators lack precise understanding of labor market demands, and employers face difficulties in communicating skill requirements effectively. A fundamental barrier to addressing these challenges is the absence of a standardized, computationally tractable representation of skills that can bridge the semantic gap between different domains and stakeholders.

Traditional approaches to skill extraction rely heavily on manual annotation, expert judgment, or simple keyword matching—methods that are labor-intensive, error-prone, and fail to capture the semantic richness of skill descriptions (Boselli et al., 2018). Furthermore, the proliferation of domain-specific vocabularies for describing competencies has resulted in a fragmented landscape where the same skill may be described using vastly different terminology across contexts.

### 1.2 Contribution

This paper introduces LAiSER, a comprehensive framework that addresses these limitations through three key innovations:

1. **Intelligent Text Preprocessing**: A domain-aware preprocessing stage that removes extraneous information (company branding, legal boilerplate, benefits descriptions) while preserving task-relevant skill information.

2. **Multi-Model Skill Extraction**: A flexible architecture supporting multiple large language models (LLMs) with automatic fallback mechanisms, enabling deployment across different computational environments and budget constraints.

3. **Taxonomy-Aware Alignment**: A semantic similarity-based alignment system leveraging FAISS vector search to map extracted skills to standardized taxonomies (e.g., ESCO), providing interoperability and enabling cross-domain skill analysis.

LAiSER is designed with research reproducibility, operational scalability, and practical utility as core principles. The system has been deployed in multiple research contexts and demonstrates robust performance across diverse text sources, including job advertisements, course syllabi, and training program descriptions.

## 2. System Architecture

### 2.1 Design Principles

LAiSER's architecture reflects four fundamental design principles:

**Modularity**: The system is organized into loosely coupled service layers (data access, skill extraction, taxonomy alignment) that can be independently modified, tested, and deployed.

**Flexibility**: Multiple computational backends (vLLM, Transformer models, cloud APIs) ensure the framework can adapt to varied resource constraints and deployment scenarios.

**Robustness**: Comprehensive error handling, graceful degradation, and fallback mechanisms ensure system reliability even when individual components fail.

**Extensibility**: Well-defined interfaces and service abstractions enable researchers to incorporate new models, taxonomies, or extraction methodologies without extensive system modifications.

### 2.2 Core Components

#### 2.2.1 Configuration and Data Access Layer

The `config.py` module centralizes all system parameters, including model identifiers, similarity thresholds, prompt templates, and SCQF (Scottish Credit and Qualifications Framework) level descriptors. This centralization facilitates reproducible research by documenting all experimental parameters in a single location.

The `data_access.py` module implements a data access layer with support for both remote and local data sources. The `DataAccessLayer` class manages taxonomy loading, embedding model initialization, and skill-to-tag mapping, while the `FAISSIndexManager` handles vector index operations with intelligent caching to minimize redundant computations.

#### 2.2.2 Service Layer Architecture

The service layer (`services.py`) encapsulates the core business logic through four primary service classes:

**PromptBuilder**: Generates contextually appropriate prompts for different extraction tasks. The service supports multiple input types (job descriptions, course syllabi) and extraction modes (basic skill extraction, KSA framework extraction), ensuring that language model queries are optimally structured for each use case.

**ResponseParser**: Implements robust parsing logic to extract structured skill information from LLM responses. The parser employs multiple fallback strategies to handle malformed JSON, unexpected formatting, and model-specific output variations, significantly improving system reliability.

**SkillAlignmentService**: Manages the alignment of extracted skills with established taxonomies using semantic similarity search. The service leverages FAISS (Facebook AI Similarity Search) indices to efficiently retrieve the most relevant canonical skills from large taxonomies, with configurable similarity thresholds and top-k retrieval parameters.

**SkillExtractionService**: Serves as the primary orchestrator, coordinating interactions between the PromptBuilder, ResponseParser, and SkillAlignmentService to execute complete skill extraction pipelines.

#### 2.2.3 Multi-Model LLM Integration

The `llm_models` package provides a unified interface to multiple language model backends:

- **Gemini API Integration** (`gemini.py`): Enables cloud-based inference using Google's Gemini models, suitable for scenarios where local computational resources are limited.

- **HuggingFace Model Integration** (`hugging_face_llm.py`): Supports local inference using HuggingFace Transformers, with optimizations for both CPU and GPU environments.

- **vLLM Integration** (`model_loader.py`): Provides GPU-accelerated inference for high-throughput scenarios, with automatic memory management and batching.

- **LLM Router** (`llm_router.py`): Implements intelligent routing logic with automatic fallback, allowing the system to gracefully degrade from vLLM to Transformer models to API-based inference based on resource availability.

#### 2.2.4 Skill Extractor Interface

The framework provides two interface levels:

**Legacy Interface** (`Skill_Extractor`): Maintains backward compatibility with existing research pipelines while incorporating enhanced taxonomy-aware functionality.

**Refactored Interface** (`SkillExtractorRefactored`): Offers a modern, clean API with explicit method signatures, comprehensive documentation, and improved error reporting. The refactored interface separates skill extraction from alignment, enabling researchers to use these operations independently or in combination based on their specific requirements.

### 2.3 Two-Stage Skill Extraction Pipeline

LAiSER employs a sophisticated two-stage pipeline to maximize extraction accuracy:

#### Stage 1: Intelligent Preprocessing

The preprocessing stage applies a domain-aware cleaning operation that removes:
- Company branding and marketing language
- Geographical location and contact information
- Compensation details and scheduling information
- HR/legal boilerplate (EEO statements, diversity policies)
- Benefits descriptions
- Generic motivational language ("fast-paced environment", "self-starter")

This preprocessing is crucial for focusing the language model's attention on task-relevant information, significantly reducing false positive skill extractions. The preprocessing itself is performed by an LLM with explicit instructions to preserve job duties, technical responsibilities, required skills, qualifications, and tools.

#### Stage 2: Structured Skill Extraction

The extraction stage uses carefully engineered prompts to elicit structured skill information from the preprocessed text. The prompts explicitly constrain the model to:
- Extract concrete, job-relevant skills (excluding soft traits and generic workplace behaviors)
- Return skills as concise noun phrases (1-5 words)
- Produce valid JSON output with a standardized schema
- Avoid introducing skills not present or strongly implied in the source text

This constrained generation approach dramatically improves the consistency and parsability of model outputs.

### 2.4 Taxonomy Alignment via Semantic Similarity

Following extraction, raw skills are aligned to a standardized taxonomy (ESCO by default) using a semantic similarity approach:

1. **Embedding Generation**: Each extracted skill and taxonomy skill is encoded using a sentence transformer model (default: `all-MiniLM-L6-v2`), producing dense vector representations that capture semantic meaning.

2. **Vector Search**: FAISS indices enable efficient approximate nearest neighbor search over large taxonomy vocabularies (10,000+ skills), retrieving the top-k most similar canonical skills.

3. **Similarity Filtering**: Alignment candidates are filtered using a configurable similarity threshold (default: 0.20), with higher thresholds increasing precision at the cost of recall.

4. **Tag Assignment**: Each aligned skill receives its corresponding taxonomy tag (e.g., ESCO skill code), enabling integration with labor market information systems and standardized reporting.

This alignment process ensures that skills extracted from diverse sources can be compared, aggregated, and analyzed using a common vocabulary.

## 3. Knowledge, Skills, and Abilities (KSA) Framework

Beyond basic skill extraction, LAiSER supports extraction within the KSA (Knowledge, Skills, Abilities) framework, a widely adopted model in workforce development and human resource management. The KSA extraction pipeline provides:

**Skill Identification**: Extraction of key skills from input text, contextualized using semantic matches from the taxonomy.

**Proficiency Level Assignment**: Automatic classification of skill proficiency using the Scottish Credit and Qualifications Framework (SCQF), a 12-level scale ranging from basic awareness to global expertise. This classification enables fine-grained analysis of skill requirements across occupations and educational programs.

**Knowledge Requirements**: For each skill, the system identifies broad areas of understanding necessary to develop competency (e.g., "data visualization techniques" for a data analysis skill).

**Task Abilities**: For each skill, the system identifies general tasks or capabilities enabled by the skill (e.g., "statistical modeling", "report generation").

This structured output facilitates downstream analyses such as skills gap identification, curriculum mapping, and job-candidate matching.

## 4. Technical Implementation

### 4.1 Dependencies and Computational Requirements

LAiSER is implemented in Python 3.9+ with the following key dependencies:

- **PyTorch**: Deep learning framework for model inference
- **Transformers** (HuggingFace): Pre-trained language model access
- **vLLM**: GPU-accelerated inference engine (optional)
- **spaCy**: Natural language processing utilities
- **FAISS**: Efficient similarity search
- **Sentence-Transformers**: Semantic embedding generation
- **pandas**: Data manipulation and result formatting

**Computational Requirements**: The framework supports three deployment modes:
- **GPU Mode** (Recommended): 15GB+ VRAM, enables vLLM acceleration for high-throughput processing
- **CPU Mode**: No special hardware requirements, suitable for small-scale experiments
- **API Mode**: No local compute requirements, leverages cloud-based inference

### 4.2 Input and Output Specifications

**Input Requirements**:
- Pandas DataFrame containing textual descriptions (job advertisements, course descriptions, etc.)
- ID column for document identification
- One or more text columns containing skill-bearing content

**Output Format**:
The system produces a structured DataFrame with the following columns:
- `Research ID`: Document identifier
- `Description`: Source text (full or excerpt)
- `Raw Skill`: Skill as extracted by the LLM
- `Taxonomy Skill`: Aligned canonical skill from taxonomy
- `Skill Tag`: Taxonomy identifier (e.g., ESCO code)
- `Correlation Coefficient`: Semantic similarity score between raw and canonical skill (0-1)

For KSA extraction, additional columns include:
- `Level`: SCQF proficiency level (1-12)
- `Knowledge Required`: List of knowledge areas
- `Task Abilities`: List of enabled capabilities

## 5. Use Cases and Applications

### 5.1 Workforce Analytics and Labor Market Intelligence

LAiSER enables large-scale analysis of skill distributions in labor markets. Researchers can process thousands of job advertisements to identify:
- Emerging skills and declining competencies
- Skill co-occurrence patterns
- Regional variations in skill demand
- Industry-specific skill requirements
- Temporal trends in skill evolution

By standardizing skill descriptions through taxonomy alignment, the framework enables cross-temporal and cross-regional comparisons that would be impossible with raw text analysis.

### 5.2 Educational Program Assessment and Curriculum Development

Educators can extract skills from course syllabi and learning outcomes, then compare these against labor market demand signals derived from job advertisements. This application supports:
- Curriculum-industry alignment assessment
- Identification of skill gaps in educational programs
- Evidence-based curriculum redesign
- Accreditation self-studies
- Program-level learning outcome validation

### 5.3 Skills-Based Hiring and Talent Acquisition

Organizations can use LAiSER to standardize job descriptions and candidate resumes, enabling:
- Automated job-candidate matching based on skill overlap
- Identification of transferable skills across occupations
- Skills-based job architecture development
- Bias reduction in hiring through focus on demonstrable competencies

### 5.4 Career Pathway Analysis and Guidance

By extracting and aligning skills across multiple occupations, LAiSER can identify career transition pathways, revealing:
- Skill similarities between occupations
- Minimal skill gaps between roles
- Efficient upskilling routes
- Career lattice structures within industries

### 5.5 Policy Research and Workforce Development

Policymakers and workforce development agencies can use the framework to:
- Conduct regional skills assessments
- Evaluate workforce training program effectiveness
- Inform funding allocation for education and training
- Develop evidence-based workforce development strategies
- Support labor market information system development

## 6. Strengths and Innovations

### 6.1 Methodological Strengths

**Semantic Alignment Over Keyword Matching**: Unlike traditional keyword-based approaches, LAiSER's embedding-based alignment captures semantic similarity, correctly matching "machine learning" with "predictive modeling" even when exact terms differ.

**Context-Aware Preprocessing**: The intelligent preprocessing stage reduces noise and false positives, a persistent challenge in skill extraction from marketing-heavy job advertisements.

**Fallback Mechanisms**: The multi-model architecture with automatic fallback ensures system availability even when preferred computational resources are unavailable, crucial for operational deployments.

**Standardization Without Information Loss**: The dual reporting of raw extracted skills and taxonomy-aligned skills preserves both the original language used in source documents and the standardized representation, enabling both qualitative and quantitative analysis.

### 6.2 Technical Innovations

**Modular Service Architecture**: The separation of concerns into service layers (data access, skill extraction, taxonomy alignment) enables independent evolution of system components and facilitates testing and validation.

**Configurable Extraction Strategies**: Support for multiple extraction modes (basic, KSA framework, custom prompts) allows researchers to adapt the system to specific research questions and data characteristics.

**FAISS-Accelerated Search**: The use of FAISS indices for taxonomy alignment enables sub-second retrieval even with taxonomies containing tens of thousands of skills, making the system viable for real-time applications.

**Proficiency Level Classification**: Integration of the SCQF framework for automated skill level assignment enables analyses that account for skill depth, not just presence/absence.

### 6.3 Practical Advantages

**Accessibility**: Support for CPU-only deployment and cloud APIs ensures the framework is accessible to researchers and practitioners without access to expensive GPU infrastructure.

**Reproducibility**: Centralized configuration management, explicit version tracking, and comprehensive documentation facilitate reproducible research—a critical concern in computational social science.

**Extensibility**: Well-defined service interfaces and plugin architecture enable researchers to incorporate new taxonomies (beyond ESCO), alternative embedding models, or custom extraction logic without modifying core system code.

**Scalability**: Batch processing capabilities, GPU acceleration, and efficient vector search enable processing of millions of documents, supporting population-level labor market analyses.

## 7. Limitations and Future Directions

### 7.1 Current Limitations

**Language Model Dependence**: Extraction quality depends on the capabilities and biases of the underlying language models. Models trained primarily on English-language data may exhibit reduced performance on non-English texts or domain-specific jargon.

**Taxonomy Coverage**: Alignment quality is constrained by the comprehensiveness and currency of the reference taxonomy. Emerging skills or highly specialized competencies may lack suitable canonical representations.

**Computational Cost**: While the framework supports CPU-only operation, optimal performance requires GPU resources, potentially limiting accessibility for some researchers.

**Context Sensitivity**: The current implementation processes documents independently, without leveraging document metadata (industry, occupation, education level) that could improve extraction accuracy.

### 7.2 Future Research Directions

**Multi-Taxonomy Alignment**: Extending the framework to simultaneously align skills to multiple taxonomies (ESCO, O*NET, proprietary industry frameworks) would increase utility for diverse stakeholders.

**Multilingual Support**: Incorporating multilingual language models and cross-lingual embedding spaces would enable skill extraction from non-English sources, critical for international labor market research.

**Contextual Extraction**: Augmenting the extraction pipeline with document-level metadata and context (e.g., job level, industry sector) could improve extraction precision through more targeted prompts.

**Confidence Estimation**: Developing probabilistic models of extraction and alignment confidence would enable researchers to quantify uncertainty in derived skill measurements.

**Longitudinal Analysis Tools**: Building specialized modules for temporal analysis of skill evolution would support research on skill obsolescence, skill emergence, and labor market dynamics.

**Interactive Validation Interfaces**: Creating human-in-the-loop validation tools would facilitate active learning approaches, continuously improving extraction quality through expert feedback.

## 8. Conclusion

LAiSER represents a significant advance in automated skill extraction and standardization, addressing critical needs in workforce development, educational policy, and labor market research. By combining modern natural language processing techniques with careful system engineering, the framework achieves both high accuracy and practical deployability. The system's modular architecture, multi-model support, and taxonomy-aware alignment capabilities position it as a versatile tool for researchers and practitioners working at the intersection of education, employment, and skills.

The open-source nature of LAiSER, combined with comprehensive documentation and extensibility features, aims to foster a research community engaged in continuous improvement of skill extraction methodologies. As labor markets continue to evolve and skill requirements become increasingly complex, tools like LAiSER will be essential for maintaining the "mutually intelligible" skill information ecosystems that benefit learners, educators, and employers alike.

## 9. Data and Code Availability

**Source Code**: The LAiSER codebase is publicly available at [https://github.com/LAiSER-Software/extract-module](https://github.com/LAiSER-Software/extract-module) under the MIT License.

**Sample Datasets**: Example datasets for testing and validation are available at [https://github.com/LAiSER-Software/datasets](https://github.com/LAiSER-Software/datasets).

**Installation**: The framework can be installed via pip:
```bash
# For GPU-accelerated environments
pip install laiser[gpu]

# For CPU-only environments  
pip install laiser[cpu]
```

**Documentation**: Comprehensive architecture documentation, API references, and usage examples are maintained in the repository's `docs/` directory and `ARCHITECTURE.md`.

## 10. Example Usage

### Basic Skill Extraction and Alignment

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize extractor with Gemini API (no GPU required)
extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key="your-api-key",
    use_gpu=False
)

# Load job descriptions
data = pd.read_csv("job_descriptions.csv")

# Extract and align skills
results = extractor.extract_and_align(
    data,
    id_column="job_id",
    text_columns=["title", "description"],
    input_type="job_desc",
    batch_size=32
)

# Save standardized skill data
results.to_csv("aligned_skills.csv", index=False)
```

### KSA Extraction with Proficiency Levels

```python
from laiser.services import SkillExtractionService

# Initialize service layer directly
service = SkillExtractionService()

# Extract skills with KSA framework
job_description = {
    "description": "Software engineer role requiring Python, machine learning, and cloud deployment experience..."
}

ksa_results = service.extract_skills_with_ksa(
    input_data=job_description,
    input_type="job_desc",
    num_skills=5,
    num_knowledge="3-5",
    num_abilities="3-5"
)

# Results include: Skill, Level (1-12), Knowledge Required, Task Abilities
for result in ksa_results:
    print(f"Skill: {result['Skill']}")
    print(f"SCQF Level: {result['Level']}")
    print(f"Knowledge Required: {result['Knowledge Required']}")
    print(f"Task Abilities: {result['Task Abilities']}\n")
```


## 11. Acknowledgments

The author gratefully acknowledges the George Washington University Institute of Public Policy and the Program on Skills, Credentials, and Workforce Policy (PSCWP) for institutional support and research guidance. This project was supported by grants from the Walmart Foundation and the Gates Foundation. 

The author thanks the open-source community, particularly GW Open Source Program Office and the developers of HuggingFace Transformers, FAISS, and spaCy, whose tools form the technical foundation of this framework. Special appreciation is extended to the contributors who volunteered their time to test and refine the LAiSER project on GitHub.

## 12. References

Autor, D. H., & Dorn, D. (2013). The growth of low-skill service jobs and the polarization of the US labor market. *American Economic Review*, 103(5), 1553-1597.

Boselli, R., Cesarini, M., Mercorio, F., & Mezzanzanica, M. (2018). Classifying online job advertisements through machine learning. *Future Generation Computer Systems*, 86, 319-328.

European Commission. (2020). *ESCO: European Skills, Competences, Qualifications and Occupations*. Retrieved from [https://ec.europa.eu/esco](https://ec.europa.eu/esco)

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982-3992.

Scottish Credit and Qualifications Framework Partnership. (2019). *SCQF Handbook: User Guide*. Retrieved from [https://scqf.org.uk](https://scqf.org.uk)
