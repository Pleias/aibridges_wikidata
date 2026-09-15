# Scope

## Purpose

The system tests whether a Recursive Language Model (RLM) can answer difficult
Wikidata questions. The model explores a frozen graph. It does not receive the
dataset in its context.

The model writes Python in a persistent REPL. It reads only the evidence it
needs and keeps intermediate results in variables across turns. It can
delegate semantic interpretation to sub-queries through `llm_query` and
`llm_map`. It finishes with a JSON answer that a program can score.

The main research artifact is the trajectory. A trajectory records the model
messages, the reasoning, the executed code, the graph reads, the sub-queries,
the token usage and the final answer.

## Why an environment and not a long context

Wikidata is too large for a context window. When a model reads many
descriptions in its context and decides from memory several turns later, its
accuracy decreases without a visible error ("context rot"). The harness keeps
the data outside the context:

- The graph stays in the environment. The model reads slices of it.
- Candidate sets stay in Python variables. Code does the set operations.
- Semantic judgement over many items goes to sub-queries in batches.
- Every identifier in a final answer must come from a read in the same run.

## The benchmark

The benchmark contains 56 tasks:

- 50 reviewed tasks under `tasks/cards/`, built from 18 design cards;
- 3 certified hard tasks under `tasks/hard/`;
- 3 tasks under `tasks/nameonly/` that identify entities by name only.

The tasks vary:

- logical against semantic dependencies (the LT, ST, SST and PST execution
  patterns);
- candidate volume and exhaustive selection;
- obscure entities and relations;
- multilingual evidence in six languages, especially Arabic and Russian;
- the objective: retrieval, aggregation or data-quality audit.

[06 Question design](06-question-design.md) describes each card and gives a
concrete task for each one.

## Success criteria

A task is valid when these conditions are true:

- The frozen graph contains a complete, deterministic reference answer.
- The question states a credible information need. It does not give QIDs,
  PIDs or an execution recipe.

A run is good when these conditions are true:

- Every identifier in the final answer comes from data that the run read.
- The final answer matches the reference answer under a normalized, LLM-free
  comparison.
- The archive is sufficient to explain the result and the model strategy.

## Boundaries

- The environment is a frozen local snapshot. It is not a live Wikidata
  service, and it does not execute SPARQL.
- The number of graph hops alone does not define reasoning difficulty.
- Question wording is reviewed by a person. It is not generated without
  review.
- Model memory is not evidence.
