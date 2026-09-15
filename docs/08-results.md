# Results

This document gives the results of the reference configuration on the 56-task
benchmark. [07 Evaluation](07-evaluation.md) defines each metric.

## Configuration

| Item | Value |
|---|---|
| Model | `unsloth/Qwen3.8-27B-FP8`, served with vLLM on one NVIDIA H100 80 GB |
| Root generation | Thinking on, reasoning effort `low`, temperature 0, top-p 0.95, top-k 20, 32,768 output tokens |
| Harness | `llm_map` on, search hints on, sub-query cap 80, one task at a time |
| Server | Context 131,072 tokens, FP8 KV cache, MTP with 3 speculative tokens |
| Tasks | 50 reviewed cards, 3 certified hard tasks, 3 name-only tasks |

`config/runs/reference.env` contains these settings. `scripts/run_reference.sh`
uses them.

## Answer quality

| Metric | Result |
|---|---:|
| Exact answers | **38/56 (67.9%)** |
| Content-exact answers | **43/56 (76.8%)** |
| Runs that reached `FINAL` | 53/56 |

<!-- REVIEW: per-pattern table (LT / ST / SST / PST / HARD / HIDE) to add from
     scripts/bench/summary.py on the reference batch. -->

## Comparison with the first harness version

The same 56 tasks ran with two other models on the first harness version. The
reference configuration runs Qwen3.8-27B-FP8 on the current harness.

| Metric | Gemini 3 Flash<br/>(first harness version) | Qwen3.6-35B-A3B<br/>(first harness version) | **Qwen3.8-27B-FP8<br/>(current harness)** |
|---|---:|---:|---:|
| Exact answers | 28/56 (50.0%) | 10/56 (17.9%) | **38/56 (67.9%)** |
| Runs that reached `FINAL` | 50/56 | 22/56 | **53/56** |
| Runs that used all turns (`max_iters`) | 6/56 | 34/56 | **3/56** |
| Turns, all runs | 325 | 542 | 389 |
| Mean turns per run | 5.80 | 9.68 | 6.95 |
| Tokens, all runs | 2,475,818 | 2,110,283 | **1,644,407** |
| Content-exact answers | — | — | 43/56 (76.8%) |
| Graph reads, all runs | — | — | 10,301 |

A dash (—) shows a metric that is not available for that batch.

The Gemini 3 Flash column uses the latest archived run of each task, through an
OpenAI-compatible API endpoint. The Qwen3.6-35B-A3B column is one batch served
with vLLM on one NVIDIA H100 80 GB.

The current harness adds these items to the first version:

- the echo of a bare expression on the last line of a cell;
- `llm_map`, the batched delegation primitive;
- search hints for an empty name search;
- thinking support: no stop sequence while thinking is on, and the reasoning
  archived in `reasoning.json`;
- the content-exact verdict, the graded field detail, the coverage report and
  the grounding report.

The three columns do not isolate one variable. The model, the harness version
and the generation settings all differ between them.

## Cost and efficiency

| Metric | Result |
|---|---:|
| Turns, all runs | 389 |
| Graph reads, all runs | 10,301 |
| Tokens, all runs (root and sub-queries) | 1,644,407 |
| Name searches | 251 |
| Name searches with an empty result | 56 |
| Batch wall time | 38 min |

The model does not receive the graph. It read 10,301 slices of a graph with
48 million nodes and 262 million edges to answer 56 questions.

## Grounding of quotations

Many answer formats ask for the exact description text that supports an
entity. `grounding.py` checked the 174 quotations in the final answers:

| Verdict | Quotations |
|---|---:|
| `GROUNDED`: stored, and read by the run | 134 |
| `BLIND`: stored, but not read by the run | 2 |
| `NOT_VERBATIM`: not identical to a stored description | 19 |
| `UNCHECKED`: no entity to attribute it to | 19 |
| **Total** | **174** |

A manual reading of the 19 `NOT_VERBATIM` quotations gave these results:

- 16 contain the stored description inside added words.
- 3 differ from the stored description only in typographic punctuation.
- 0 contain text that is not in the graph.

No checked quotation contains text that is not in the graph.

## Delegation example

<!-- REVIEW: confirm steps 1 (name search), 2 and 4 against the archived trajectory. -->

`PST_RETRIEVE_ALL_SEMANTIC_FILTER_001` asks for every Amnesty International
employee whose description combines literary or artistic work with activism,
with the exact supporting descriptions. The question gives no QID. The
trajectory shows the intended decomposition:

1. The model found the organization by name. It read the incoming "employer"
   links with one call: `edges('Q42970', direction='in', pid='P108')`. The
   result was a pool of 114 employees.
2. It fetched the descriptions of the pool in code and kept them in a
   variable.
3. It sent the descriptions to `llm_map` with one inclusion rule. The harness
   sent them in 12 batches of up to 10 items and returned one verdict per
   employee.
4. It selected the matches in Python and built the final answer from the
   identifiers and descriptions it had read.

The semantic judgement over 114 descriptions stayed out of the root context.
Every sub-query exchange is in `sub_queries.json`.

## Reproduce the results

1. Complete [04 Setup](04-setup.md).
2. Start the server:

   ```bash
   bash scripts/serve/serve_vllm.sh
   ```

3. In a second shell, run the benchmark:

   ```bash
   bash scripts/run_reference.sh
   ```

4. Read `report_summary.txt` and `report_grounding.txt` in the batch directory.

Temperature 0 makes the root model close to deterministic, but not fully
deterministic: GPU kernels and batching can change a token. A repeated batch
can therefore differ on a small number of tasks. Use `compare.py` to list them.
