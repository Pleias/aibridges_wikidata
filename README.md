# Wikidata RLM harness

This repository contains a Recursive Language Model (RLM) harness, a frozen
Wikidata graph environment and a benchmark of 56 reviewed tasks.

The model does not receive the graph in its context. It writes Python in a
persistent REPL (Read-Eval-Print Loop), reads only the graph data it needs, and
can delegate semantic judgement to sub-queries. Every run keeps a complete
trajectory: model messages, reasoning, executed code, graph reads, sub-queries
and the final answer. A deterministic scorer compares the final answer with a
reference answer. No language model judges correctness.

## Current results

The reference configuration serves Qwen3.8-27B-FP8 with vLLM on one GPU, on
the current harness. The same 56 tasks also ran with two other models on the
first harness version:

| Metric | Gemini 3 Flash<br/>(first harness version) | Qwen3.6-35B-A3B<br/>(first harness version) | **Qwen3.8-27B-FP8<br/>(current harness)** |
|---|---:|---:|---:|
| Exact answers | 28/56 (50.0%) | 10/56 (17.9%) | **38/56 (67.9%)** |
| Runs that reached `FINAL` | 50/56 | 22/56 | **53/56** |
| Runs that used all turns | 6/56 | 34/56 | **3/56** |
| Tokens, all runs | 2,475,818 | 2,110,283 | **1,644,407** |
| Content-exact answers | — | — | 43/56 (76.8%) |
| Graph reads, all runs | — | — | 10,301 |

A dash (—) shows a metric that is not available for that batch. The
complete results, with the metric definitions, the harness differences and the
grounding audit, are in [docs/08-results.md](docs/08-results.md).

## Quick start

1. Install the harness environment:

   ```bash
   uv sync
   ```

2. Copy the site configuration and set your paths:

   ```bash
   cp config/local.env.example config/local.env
   ```

3. Put the graph in `WIKIDATA_GRAPH_DIR` and the model weights in `MODELS_DIR`
   (see [docs/04-setup.md](docs/04-setup.md)).

4. Start the model server on the GPU machine:

   ```bash
   bash scripts/serve/serve_vllm.sh
   ```

5. In a second shell, run the benchmark:

   ```bash
   bash scripts/run_reference.sh
   ```

On a Slurm cluster, edit the account lines in
[jobs/slurm/reference56.slurm](jobs/slurm/reference56.slurm) and submit it. The
job starts the server and runs the benchmark on one node.

## Repository layout

| Path | Contents |
|---|---|
| `env.sh` | Environment for all scripts. Resolves paths from the repository root. |
| `config/local.env.example` | Site paths and endpoint. Copy to `config/local.env`. |
| `config/runs/reference.env` | Generation, harness and server settings of the published results. |
| `config/taxonomy/question_design_cards.yaml` | The 18 design cards. |
| `tasks/cards/` | 50 reviewed tasks. |
| `tasks/hard/` | 3 certified hard tasks. |
| `tasks/nameonly/` | 3 tasks that identify entities by name only. |
| `scripts/rlm_loop.py` | The RLM harness: one task, one run. |
| `scripts/wd_graph_env.py` | The graph environment: the functions the model calls. |
| `scripts/build_*.py` | Build the frozen graph from Wikidata Parquet inputs. |
| `scripts/bench/` | Batch runner, scorers, audits, reports and run browser. |
| `scripts/serve/` | vLLM server start and readiness check. |
| `scripts/run_reference.sh` | Runs the 56 tasks and writes the reports. |
| `jobs/slurm/` | Slurm job template. |
| `tests/` | Unit tests. They need no graph, no GPU and no server. |
| `docs/` | Documentation. |

Graph data, model weights, run results and `config/local.env` are not part of
the repository.

## Documentation

Start with [docs/00-index.md](docs/00-index.md).
