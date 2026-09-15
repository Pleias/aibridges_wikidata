# Note for Mohamed

This repository is the first version of the folder that we will share with the
client. It shows the current state of the work: the 56-task benchmark and the
pipeline configuration that gives the best results. It does not show the
experiment history, negative results or future plans.

Read this note before you change the repository. Remove this file before the
repository goes to the client.

## Source

- All code, tasks and tests come from `Pleias/wikimedia_rlm`, branch
  `feat/qwen38-probe`, revision `f4dabf1`.
- Your later work on `feat/qwen-harness-evaluation` is already in that branch.
  The branch point is `240a7f2`.
- `docs/06-question-design.md` is your enriched document from commit `b23bb39`,
  with the small changes listed below.

## Contents

| Area | Items |
|---|---|
| Code | The harness (`rlm_loop.py`), the graph environment (`wd_graph_env.py`), the build scripts, the scorers and the audits (`scripts/bench/`), hub discovery |
| New code | `scripts/bench/summary.py`: batch totals and one row per execution pattern. `tests/test_summary.py` tests it. |
| Tasks | The 56 tasks: `tasks/cards` (50), `tasks/hard` (3), `tasks/nameonly` (3). No reference answer changed. |
| Configuration | `env.sh`: the environment for all scripts. `config/local.env.example`: site paths and endpoint. `config/runs/reference.env`: the exact settings of the best run (`20260827-1719_T0-map-hints56`). |
| Scripts | `scripts/serve/serve_vllm.sh`: starts vLLM with the reference serving settings. `scripts/serve/wait_for_server.sh`: waits until the server can generate. `scripts/run_reference.sh`: runs the 56 tasks and writes the reports. |
| Slurm | `jobs/slurm/reference56.slurm`: a generic template. The account and partition are placeholders. |
| Documents | `README.md` and `docs/00` to `docs/08`: index, scope, architecture, code flow, setup, running, question design, evaluation, results |

Verification done:

- 104 tests pass (`python -m unittest discover -s tests`).
- All Python files compile. All shell scripts pass `bash -n`.
- `env.sh` applies the correct order of precedence: shell, then
  `config/local.env`, then `config/runs/reference.env`.
- `run_reference.sh` stops with a clear message when the graph or the server is
  missing.

Not verified: a complete run of this repository on a GPU. Jean Zay did not
answer during the preparation.

## Excluded from the source

- `experiments/`, `docs/history/`, `qwen38_changes.md`, `notebooks/`
- `ollama_shim.py`, `data/tasks/`, the old `requirements.txt`
- `jobs/jean_zay/` and `_env_vars.sh` (replaced by the generic scripts above)

## Changes from the source

### Paths

No path is fixed. Paths come from environment variables or are relative to the
repository root.

| Variable | Used by |
|---|---|
| `WIKIDATA_GRAPH_DIR` | `wd_graph_env.py`, all build scripts, `discover_hubs.py` |
| `WIKIDATA_FORMATTED_SOURCE` | `build_graph.py`, `build_entities.py`. Default `data/raw/wikidata_formatted` (was `wikidata_formatted_jeanzay`). |
| `WIKIDATA_EXTRACTED_SOURCE` | `build_graph.py` |
| `WIKIDATA_HF_RANKS_SOURCE` | `build_annotations.py` |
| `MODELS_DIR`, `RLM_STORAGE_ROOT` | `download_model.py`, `serve_vllm.sh`, caches |

`WIKIDATA_GRAPH_OUT` is replaced by `WIKIDATA_GRAPH_DIR`. Log lines print
absolute paths, not `relative_to(ROOT)`, so an output outside the repository
does not raise an error.

### Behaviour

The harness behaves the same as in the best run. These changes do not change a
result:

- `API_KEY` is the documented key name. `LITELLM_API_KEY` is still accepted.
- When `SUBQ_MODEL_NAME` is not set, sub-queries use `MODEL_NAME` (resolved at
  call time). The old default named a Gemini model.
- The module-level copy of `llm_map` in `rlm_loop.py` is removed. It called an
  undefined `query` and the harness never used it. The copy inside `run_loop`,
  which calls `map_over`, is unchanged.
- `build_graph.py`: the `--migrate` option and `migrate()` are removed. French
  log messages are now in English.
- `download_model.py`: the shared-account owner check now compares against
  `RLM_STORAGE_ROOT`.
- `run_batch.py`: `generation_settings()` no longer records `RLM_STORAGE_ROOT`,
  because it is a path and not a setting.
- `grounding.py`: the verdict `FABRICATED` is now `NOT_VERBATIM`. None of the
  19 flagged quotations of the best run contains text that is absent from the
  graph (16 are the stored text inside added words, 3 differ only in
  punctuation), so "fabricated" was not accurate.

### Comments and docstrings

Removed from code comments, docstrings and test docstrings: Jean Zay, `compil`,
user names, job IDs, dated anecdotes, the history of failed experiments and
references to internal specification files. The technical reason for each
design decision stays.

### Documents

- `docs/06-question-design.md`: a new section on the 3 hard tasks and the 3
  name-only tasks, the `answer.bookkeeping` and `qids` fields in the task
  contract, and the hub catalogue named by its producing script.
- `docs/08-results.md` and `README.md`: the comparison table. Gemini 3 Flash
  and Qwen3.6-35B-A3B are marked "first harness version", Qwen3.8-27B-FP8 is
  marked "current harness". A note says that the columns do not isolate one
  variable.
- All new documents use ASD-STE100 style.

## Review items

1. **Per-pattern results are missing.** `docs/08-results.md` has a `REVIEW`
   comment where the LT / ST / SST / PST / HARD / HIDE table goes. Run
   `scripts/bench/summary.py` on
   `results/runs/20260827-1719_T0-map-hints56` and paste the table.
2. **Check the delegation example.** `docs/08-results.md` describes
   `PST_RETRIEVE_ALL_SEMANTIC_FILTER_001`. The `edges('Q42970', 'in', 'P108')`
   call, the pool of 114 and the 12 `llm_map` batches are confirmed. Steps 1
   (name search), 2 and 4 are not confirmed against the archived trajectory. A
   `REVIEW` comment marks them.
3. **Sub-query count row.** The first-version runs made 74 (Gemini) and 4
   (Qwen3.6) `llm_query` calls. The count for the best run is not available
   locally, so the row is not in the table. Add it if you want it.
4. **Model name.** The table uses `Qwen3.6-35B-A3B`, the name in the source
   documents. Confirm it.
5. **Grounding typography bug.** `grounding.py` does not fold typographic
   punctuation, but `canonical.py` does. A fix moves 3 quotations out of
   `NOT_VERBATIM`, and the grounding numbers in `08-results.md` then need a
   recomputation. Not fixed.
6. **Raw graph inputs.** `docs/04-setup.md` describes the format of the raw
   inputs, not their origin. Decide how the client gets the graph: a copy of
   the built graph directory, or a build from the raw inputs. Then document the
   origin of the formatted statement batches.
7. **Licence.** There is no `LICENSE` file. The contract says CC0 for the
   outputs. The code licence is not decided.
8. **Reference run archive.** The best-run batch directory is not included.
   Decide whether the client receives it, for example as a release asset.
9. **Search hints default.** `RLM_SEARCH_HINTS` is off by default in the code
   and on in `reference.env`. The provided scripts load `reference.env`. A
   direct call to `rlm_loop.py` without `source env.sh` does not.
10. **Settings are read at import.** `rlm_loop.py` reads `RLM_*` and `SUBQ_*`
    at import, before `load_dotenv()`. Settings in a `.env` file therefore do
    not reach those variables. The documents tell the user to `source env.sh`.
11. **vLLM version.** The documents require vLLM 0.17.0 or later. Add the exact
    version of the best run (probably 0.27.1) after you check the server log.
12. **Remaining technical anecdotes.** Some docstrings still describe a failure
    case as the reason for a design, for example the `HARD_01` fixture in
    `tests/test_rlm_loop.py`. Decide how deep the scrub goes.
13. **Remove this note** before the client receives the repository.
