# Usage

All extraction goes through `SkillExtractorRefactored.extract_concepts`. This page covers its options; the
[API reference](reference.md) lists the full signature.

## Input

| Parameter | Description |
|---|---|
| `data` | a pandas `DataFrame`, one row per document |
| `id_column` | column holding each document's identifier; copied to the output. Default `"Research ID"` |
| `text_columns` | columns holding the text to analyse; joined with spaces for job descriptions |
| `input_type` | what kind of text the rows contain — selects the prompt |

### Input types

| `input_type` | Use for | Text used |
|---|---|---|
| `"job_desc"` (or `"job_description"`) | job postings and role descriptions | every column in `text_columns` |
| `"syllabus"`, `"course_syllabus"`, `"course_syllabi"` | courses and programs | the `description` and `learning_outcomes` columns — list both in `text_columns` |

!!! note "List both syllabus columns"
    The syllabus prompt treats the course description and its learning outcomes separately, reading them
    from the `description` and `learning_outcomes` columns. Only columns named in `text_columns` are read,
    so pass `text_columns=["description", "learning_outcomes"]`. With only `description`, the learning
    outcomes reach the model empty.

## What to extract

`concepts` selects the concept types:

| Value | Returns |
|---|---|
| `["skills"]` *(default)* | skills only — one model call per document |
| `["skills", "knowledge", "tasks"]` or `["all"]` | skills, plus the knowledge and tasks behind each skill — a second model call per document |

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills", "knowledge", "tasks"],
)
```

## Controlling alignment

| Parameter | Default | Description |
|---|---|---|
| `allowed_sources` | all | taxonomies to search — see [Taxonomies](taxonomies.md) |
| `top_k` | `25` | maximum aligned matches per concept type, per document — up to 75 rows with all three types |
| `similarity_thresholds` | skill `0.60`, knowledge `0.50`, task `0.50` | minimum similarity per type, e.g. `{"task": 0.6}` |
| `similarity_threshold` | *(unset)* | one minimum for every type; `similarity_thresholds` still overrides the types it names |

Raise a threshold for fewer, closer matches; lower it to see weaker ones.

## Output

By default `extract_concepts` returns a `DataFrame` with one row per aligned phrase. Each extracted phrase is
matched to its single closest taxonomy entry and kept only if that match clears the threshold for its type.

| Column | Meaning |
|---|---|
| `Research ID` | the document the match came from |
| `Type` | skill, knowledge or task |
| `Raw Concept` | the phrase the model extracted |
| `Taxonomy Concept` | the taxonomy entry it matched |
| `Taxonomy Description` | that entry's description |
| `Taxonomy Source` | `esco`, `onet`, `osn` or `ukos` |
| `Source Url` | link to the entry, where available |
| `Correlation Coefficient` | similarity between phrase and entry; higher is closer |

!!! note "One schema for every concept type"
    Each concept type is aligned separately, into its own columns (`Raw Skill` / `Taxonomy Skill`,
    `Raw Knowledge` / `Taxonomy Knowledge`, `Raw Task` / `Taxonomy Task`). Before results are returned,
    those are merged into `Raw Concept` and `Taxonomy Concept`, with `Type` recording which kind each row
    is. Filter on `Type` to get one kind back, for example `results[results["Type"] == "skill"]`.

### Writing to CSV

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    output_csv_path="results.csv",
)
```

The file is written only when `output_csv_path` is given; the `DataFrame` is returned either way.

### Graph output

With `return_edges=True` the result is a dict of two `DataFrame`s:

```python
graph = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    concepts=["skills", "knowledge", "tasks"],
    return_edges=True,
)

nodes = graph["nodes"]  # the normal results table
edges = graph["edges"]  # Research ID, Skill, Knowledge, Task, Edge Type, confidence
```

Each edge is an `ENABLES` relationship from a knowledge area to a task, derived from knowledge and tasks the
model attributed to the same skill. Knowledge and tasks come from a second model call, which only runs when
`concepts` includes `"knowledge"` or `"tasks"` — with `concepts=["skills"]` the edges table is empty.

## Other options

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `32` | accepted for compatibility; currently has no effect — rows are processed one at a time |
| `levels` | `False` | accepted for compatibility; currently has no effect |
| `extract` | `None` | older name for `concepts`; passing both with different values raises `ValueError` |
| `timing` | `False` | accepted for compatibility with benchmark scripts; currently has no effect |
| `warnings` | `False` | print a warning for each row that raises an error; the row is skipped either way |

## Skills-only wrapper

`extract_and_align` predates `extract_concepts` and is kept for existing code. It behaves like
`extract_concepts` with `concepts=["skills"]`:

```python
results = extractor.extract_and_align(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
)
```

More copy-paste snippets are on the [Examples](examples.md) page.
