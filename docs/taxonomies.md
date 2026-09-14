# Taxonomies

Extracted concepts are aligned against taxonomy indexes that ship inside the package. Each index is a
FAISS inner-product index over sentence embeddings of the entries' names and descriptions, built from the
CSV files in `laiser/public/`.

## Sources

| `allowed_sources` | Taxonomy | Entries link to |
|---|---|---|
| `esco` | [ESCO](https://esco.ec.europa.eu/) — European Skills, Competences, Qualifications and Occupations | `data.europa.eu/esco/...` |
| `onet` | [O\*NET](https://www.onetcenter.org/) — US Occupational Information Network (database 30.2) | `onetcenter.org` |
| `ukos` | [UK Skills Classification](https://skillsclassification.org/) | `skillsclassification.org` |
| `osn` | [Open Skills Network](https://www.openskillsnetwork.org/) | `osmt.wgu.edu` |

## Entries per concept type

| Source | Skills | Knowledge | Tasks |
|---|---:|---:|---:|
| `esco` | 10,715 | 3,219 | 10,715 |
| `onet` | 8,810 | 33 | 18,796 |
| `ukos` | 6,693 | 5,056 | 22,583 |
| `osn` | 932 | — | — |

The Open Skills Network contributes skills only, so `allowed_sources=["osn"]` returns no knowledge or task
matches.

## Choosing sources

Pass `allowed_sources` to `extract_concepts` to restrict alignment. Omit it to search every source.

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    concepts=["skills", "knowledge", "tasks"],
    allowed_sources=["esco", "ukos"],
)
```

Values are case- and whitespace-insensitive. `uk` is accepted as an alias for `ukos`.

### Older source names

Before version 1.0.1, taxonomy entries were labelled by provider *and* type, such as `esco_task` or
`onet_knowledge`. Those names are still accepted and map onto their provider:

| Accepted | Treated as |
|---|---|
| `esco_knowledge`, `esco_task` | `esco` |
| `onet_knowledge`, `onet_task`, `onet_skill`, `onet_tech` | `onet` |

Because concept type is chosen with `concepts`, `allowed_sources=["esco_task"]` now matches every ESCO
entry of the requested types, not only tasks.
