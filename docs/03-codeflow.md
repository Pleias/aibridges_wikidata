# Code flow

## Data preparation

```mermaid
flowchart LR
  Formatted["Formatted statement batches<br/>WIKIDATA_FORMATTED_SOURCE"] --> BG["build_graph.py"]
  Extracted["Languages, instance_of,<br/>sitelinks, property translations<br/>WIKIDATA_EXTRACTED_SOURCE"] --> BG
  BG --> Edges["edges_fwd.arrow<br/>edges_bwd.arrow"]
  BG --> Nodes["nodes.parquet"]
  BG --> Props["properties.json"]
  Ranks["Ranks and references<br/>WIKIDATA_HF_RANKS_SOURCE"] --> BA["build_annotations.py"]
  Nodes --> BA
  BA --> Ann["references/ · ranks.parquet"]
  Formatted --> BE["build_entities.py"]
  Nodes --> BE
  Ann --> BE
  BE --> Records["entities/ · entities_index.json"]
  Nodes --> BL["build_label_index.py"]
  BL --> Labels["label_index/"]
```

All outputs go to `WIKIDATA_GRAPH_DIR`. The build order is:

1. `build_graph.py` (edges, then nodes and property labels)
2. `build_annotations.py`
3. `build_entities.py`
4. `build_label_index.py`

Every artifact keys on the integer QID. The build can run again from the raw
inputs at any time.

## Question construction

```mermaid
flowchart LR
  Cards["question_design_cards.yaml"] --> Select["Select an uncovered card"]
  Catalogue["relation_patterns.parquet<br/>hub_relations.parquet"] --> Select
  Select --> Inspect["Inspect candidates with WDGraphEnv"]
  Inspect --> Gold["Compute the complete reference answer"]
  Gold --> Review["Manual wording and ambiguity review"]
  Review --> Task["tasks/cards/*.json"]
```

Cards set the experimental coverage. The hub catalogue narrows the graph
search. The person who designs the task is responsible for semantic necessity,
naturalness, completeness and exact evidence. See
[06 Question design](06-question-design.md).

## One run

```mermaid
sequenceDiagram
  participant R as rlm_loop.py
  participant M as Model endpoint
  participant E as WDGraphEnv
  participant S as Sub-query
  participant A as Run archive

  R->>M: system prompt + question
  M-->>R: reasoning + one Python cell
  R->>E: execute the cell in the persistent namespace
  E-->>R: logged result
  opt the cell calls llm_query or llm_map
    R->>S: supplied text + instruction
    S-->>R: verdicts
  end
  R->>A: checkpoint (messages, reads, sub-queries, reasoning)
  R->>M: cell output + turn counter
  M-->>R: FINAL(JSON)
  R->>R: shape and provenance checks
  R->>A: answer.md + metadata.json
```

## Run archive

Each run writes one directory, `<task_id>_<timestamp>/`:

| File | Contents |
|---|---|
| `question.txt` | The question shown to the model |
| `system_prompt.txt` | The complete system prompt |
| `messages.json` | All model replies and harness feedback, in order |
| `reasoning.json` | The thinking text of each turn (when thinking is on) |
| `read_log.json` | Every environment call, its arguments and the identifiers it returned |
| `sub_queries.json` | Every sub-query: turn, instruction, supplied text and response (when the run delegates) |
| `answer.md` | The accepted final answer |
| `metadata.json` | Status, turns, reads, tokens, timing and the generation settings |

## One batch

`run_batch.py` starts one subprocess per task. A failed API call costs one task
and not the batch. At the end it writes:

| File | Contents |
|---|---|
| `batch.json` | Task manifest, git revision, model, generation settings, progress and wall time |
| `scores.json` | One scored record per run (see [07 Evaluation](07-evaluation.md)) |
| `failures.json` | Subprocess diagnostics for tasks that produced no archive |

`scripts/run_reference.sh` adds the text reports `report_summary.txt`,
`report_scores.txt`, `report_coverage.txt`, `report_grounding.txt` and
`report_timing.txt`, and the endpoint probe report `probe.json`.
