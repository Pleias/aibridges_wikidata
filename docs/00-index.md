# Documentation

This is the entry point to the documentation.

## Read order

| Document | Question it answers |
|---|---|
| [01 Scope](01-scope.md) | What does the system do, and what is a good result? |
| [02 Architecture](02-architecture.md) | What are the components, and what does each one own? |
| [03 Code flow](03-codeflow.md) | How do data, tasks, runs and scores connect? |
| [04 Setup](04-setup.md) | How do I install the software, the graph and the model? |
| [05 Running](05-running.md) | How do I serve the model, run tasks and inspect results? |
| [06 Question design](06-question-design.md) | What do the 56 tasks test, and how were they built? |
| [07 Evaluation](07-evaluation.md) | How are runs scored and audited? |
| [08 Results](08-results.md) | What does the reference configuration achieve? |

## Terms

| Term | Meaning |
|---|---|
| Task | One JSON file: a question, the allowed functions, a turn limit and a reference answer. |
| Run | One execution of one task. It writes one run directory. |
| Batch | A set of runs that `run_batch.py` starts together. It writes one batch directory. |
| Turn | One model reply and the execution of its Python cell. |
| Trajectory | The complete record of a run. |
| Root model | The model that writes the Python cells. |
| Sub-query | A model call made from inside a cell with `llm_query` or `llm_map`. |
| Graph read | One charged call to the graph environment. A repeated read of cached data is free. |
| Reference configuration | The settings in `config/runs/reference.env`. The published results use them. |
| Execution pattern | LT, ST, SST or PST. See [06 Question design](06-question-design.md). |

## Main entry points

| Area | Source of truth |
|---|---|
| Benchmark tasks | `tasks/{cards,hard,nameonly}/*.json` |
| Question coverage plan | `config/taxonomy/question_design_cards.yaml` |
| Hub discovery | `scripts/taxonomy/discover_hubs.py` |
| Graph environment | `scripts/wd_graph_env.py` |
| RLM harness | `scripts/rlm_loop.py` |
| Batch runner | `scripts/bench/run_batch.py` |
| Scorers and audits | `scripts/bench/{score,grading,coverage,grounding,audit_gold}.py` |
| Run browser | `scripts/bench/view.py` |
