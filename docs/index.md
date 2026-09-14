# LAiSER

**LAiSER** (Leveraging Artificial Intelligence for Skill Extraction & Research) is a Python package
that reads free text — job postings, course descriptions, syllabi — and returns the skills, knowledge
areas and tasks it describes, each matched to an entry in a published taxonomy.

The point of the matching step is comparability. Two job postings that say "Python scripting" and
"writing Python code" should end up pointing at the same taxonomy skill, so the results can be counted,
joined and compared across employers, programs and datasets.

## How it works

For every row in an input table, LAiSER runs four stages:

1. **Extraction.** The document text is placed into a prompt for its input type (job description or
   syllabus) and sent to a language model, which returns candidate skills — and, if requested, the
   knowledge and tasks behind them.
2. **Parsing and deduplication.** The model's response is parsed into a list of phrases. Exact
   duplicates are removed, then near-duplicates are collapsed using embedding similarity.
3. **Taxonomy alignment.** Each phrase is embedded and matched to its closest entry in bundled FAISS
   indexes of taxonomy entries. Matches above a similarity threshold are kept, up to `top_k` per concept
   type.
4. **Output.** Matches are returned as one table with a row per match: the phrase as extracted, the
   taxonomy entry it matched, which taxonomy that entry comes from, and the similarity score.

The language model can be a hosted API or a model that runs on your own machine — see
[Model providers](providers.md). The taxonomy indexes ship with the package, so alignment runs locally
either way.

## Bundled taxonomies

| Source | `allowed_sources` value | Covers |
|---|---|---|
| ESCO — European Skills, Competences, Qualifications and Occupations | `esco` | skills, knowledge, tasks |
| O\*NET — US Occupational Information Network | `onet` | skills, knowledge, tasks |
| UK Skills Classification | `ukos` | skills, knowledge, tasks |
| Open Skills Network | `osn` | skills |

Details and entry counts are on the [Taxonomies](taxonomies.md) page.

## Where to go next

- **[Installation](installation.md)** — install from PyPI, with or without GPU support.
- **[Quickstart](quickstart.md)** — a complete run, including one that needs no API key.
- **[Usage](usage.md)** — every option of `extract_concepts` and what the output columns mean.
- **[Cookbook notebooks](cookbook.md)** — end-to-end analyses you can open in Google Colab.
- **[API reference](reference.md)** — generated from the source.
