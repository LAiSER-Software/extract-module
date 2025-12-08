# Adding Batched vLLM Inference

This guide walks through the steps required to process job descriptions in batches when using the vLLM backend. The goal is to turn the two per-row LLM calls in `laiser/skill_extractor_refactored.py` into batched calls while keeping the public API unchanged.

---

## 1. Understand the Current Flow

- `SkillExtractorRefactored.extract_and_align` loops over each row and calls `extract_and_map_skills` per row.
- `extract_and_map_skills` invokes `llm_router` twice (cleaning prompt, extraction prompt).
- `llm_router` routes to `llm_generate_vllm` which wraps `llm.generate([prompt], sampling_params=...)`.
- Skills are parsed with `_parse_skills_from_response`, then aligned locally via `SkillExtractionService.align_extracted_skills`.

Key takeaway: there are two sequential LLM calls per row, each with a single prompt in a list.

---

## 2. Expose Batch-Friendly Helpers

Create helper methods inside `SkillExtractorRefactored` to prepare and post-process batches.

1. **Prompt preparation**
   - Add a method `_build_cleaning_prompt_batch(rows, text_columns)` that returns a list of strings (one prompt per row) plus metadata (IDs, original text).
   - Add a method `_build_extraction_prompt_batch(cleaned_texts)` producing the extraction prompts.

2. **Response parsing**
   - Adapt `_parse_skills_from_response` to operate on individual strings (already true) and add `_parse_skills_batch(responses)` that returns a list of skill lists aligned with the prompts.

3. **Alignment**
   - Create `_align_batch(skills_batch, metadata)` that loops through each skills list and calls `align_skills`. No vLLM changes needed here.

---

## 3. Batch Calls Through `llm_router`

1. Update `llm_router` to accept either a `str` or a `List[str]`.
   - Detect the input type; if the prompt is a list, pass it directly to `llm_generate_vllm` without wrapping it in `[prompt]`.
2. Modify `llm_generate_vllm` accordingly:
   - Expect `prompts: List[str]`.
   - Call `llm.generate(prompts, sampling_params=...)`.
   - Return a list of response strings.
3. Ensure Gemini or transformer fallbacks still accept single strings. (You can keep existing single-string path and wrap list prompts in a loop, or add explicit handling to keep legacy behavior intact.)

---

## 4. Rewrite `extract_and_align` Loop

Replace the current per-row loop with batching logic:

1. **Accumulate rows**
   - Introduce a `batch_size` parameter (default to `DEFAULT_BATCH_SIZE`) and accumulate input rows until the batch is full or you reach the end of the DataFrame.

2. **Batch cleaning call**
   - Build cleaning prompts for the batch using `_build_cleaning_prompt_batch`.
   - Call `llm_router` once with the list of prompts.
   - Split the concatenated responses to extract the cleaned text for each row.

3. **Batch extraction call**
   - Build extraction prompts using the cleaned descriptions.
   - Call `llm_router` once with the prompt list.
   - Parse skills for each entry with `_parse_skills_batch`.

4. **Alignment**
   - For each batch element, call `align_skills` with the parsed skills, ID, and original description.
   - Collect the resulting records into `results`.

5. **Edge cases**
   - If the LLM returns fewer responses than prompts, print a warning and skip the affected rows.
   - If parsing yields no skills, handle it as today (log warning, continue).

---

## 5. Maintain Backward Compatibility

- Preserve the existing `extract_and_align` signature and behavior for callers.
- Keep the default `batch_size` equal to 1 if you want to enable batching gradually.
- Ensure logging and warnings remain useful (e.g., include row IDs when a batch member fails).

---

## 6. Testing Checklist

1. **Unit tests**
   - Add tests covering:
     - Single-row behavior (batch size 1).
     - Multi-row batch (batch size >1) verifying prompt counts and parsing alignment.
     - Responses where some entries fail to parse.

2. **Integration tests**
   - Run `extract_and_align` on a small DataFrame with `batch_size=1` and `batch_size>1` to confirm identical outputs.

3. **Performance smoke test**
   - Measure runtime on a sample dataset to confirm reduced total LLM calls (should be number of batches × 2).

---

## 7. Optional Enhancements

- **Dynamic batch sizing**: auto-adjust based on token counts to avoid GPU OOM.
- **Streaming or async**: use vLLM streaming APIs if you need real-time updates.
- **Fallback handling**: if the router detects Gemini/transformer, automatically fall back to single-row mode to avoid wiring changes elsewhere.

---

## 8. Deployment Notes

- Review any timeouts or rate limits for hosted LLM endpoints.
- Update documentation (`ARCHITECTURE.md` or `README.md`) to mention batch inference.
- Communicate the change to downstream users so they set `batch_size` appropriately when instantiating `SkillExtractorRefactored`.

By following these steps, you’ll enable batched vLLM inference while keeping the existing API stable and ensuring the rest of the pipeline continues to function as expected.
