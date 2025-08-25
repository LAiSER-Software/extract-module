# LAiSER Changelog

This document outlines the changes between versions of the LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) package.

## Version 0.2.4 (Current)

### Major Changes
- Implemented a new taxonomy-aware approach to extract skills aligned with the ESCO taxonomy before LLM inference for KSAs
- Restructured the output format to provide a clean table with standardized columns:
  - Research ID
  - Description
  - Raw Skill
  - Knowledge Required
  - Task Abilities
  - Skill Tag
  - Correlation Coefficient

### Technical Improvements
- Added `get_ksa_details()` helper function that calls vLLM and parses JSON lists for Knowledge Required & Task Abilities
- Converted `build_faiss_index_esco` & `load_faiss_index_esco` to instance methods
- Re-wrote `get_top_esco_skills` as an instance method using cached SentenceTransformer & FAISS
- Injected new pipeline at the top of the extractor method that executes the plan and returns the new-format DataFrame
- Maintained legacy flow as a fallback for backward compatibility

### API Changes
- `get_top_esco_skills()` now returns `{Skill, index, score}` (includes taxonomy index)
- The `Skill_Extractor` class now ensures `self.index` is always defined
- FAISS index is now loaded or built lazily

### Dependencies
- Added support for Google's Gemini API
- Added FAISS for efficient similarity search
- Added SentenceTransformer for embedding generation

### Pending for Future Releases
1. Remove or deprecate legacy `align_skills`, `align_KSAs`, and the old code path
2. Add unit tests for:
   - FAISS search output structure
   - `get_ksa_details` JSON parsing robustness
   - End-to-end `Skill_Extractor.extractor` on CPU-only and GPU paths
3. Update README.md and examples to showcase the new output format
4. Investigate CPU-only fallback for Knowledge/Task inferences
5. Persist ESCO vector index in a cloud-ready vector store
6. Profile & batch calls to `get_ksa_details` for speed and cost
7. Clean up duplicate `import json` lines in `llm_methods.py`
8. Verify that `requirements.txt` includes `faiss-cpu` for environments without GPU FAISS
9. Implement batched LLM calls to reduce cost (planned for v0.3)

## Version 0.2.2

- Updated documentation
- Integrated FAISS index search + Sentence Transformers for better retrieval
- Added Warnings for developmental features
- Default toggle `Levels` output to False for stability improvements. 

## Version <0.2.0 (legacy)

### Core Functionality
- Initial implementation of skill extraction using LLM or SkillNer
- Basic alignment of extracted skills with taxonomy using cosine similarity
- Support for both job descriptions and syllabi data
- GPU acceleration for LLM-based extraction

### Technical Features
- Integration with Hugging Face models
- Support for vLLM for faster inference
- Basic error handling and logging
- Configurable similarity threshold
