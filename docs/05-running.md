# Running

Complete [04 Setup](04-setup.md) before these procedures. Run all commands from
bash.

## 1. Start the model server

1. On the GPU machine, start the server:

   ```bash
   bash scripts/serve/serve_vllm.sh
   ```

   The script loads `env.sh`, checks the versions and the checkpoint, and
   starts vLLM in the foreground. It serves the model as `$MODEL_NAME` on
   `$VLLM_HOST:$VLLM_PORT` (default `127.0.0.1:8000`).

2. Wait for the model load and compilation. This takes several minutes.
3. In a second shell, check that the server can generate:

   ```bash
   source env.sh
   bash scripts/serve/wait_for_server.sh
   ```

   The check sends a one-token completion. A `/health` answer alone does not
   show that the server can generate.

4. To stop the server, press Ctrl-C in its shell.

The server settings come from `config/runs/reference.env`. Change one setting
for one start with an environment variable:

```bash
MAX_NUM_SEQS=16 bash scripts/serve/serve_vllm.sh
```

## 2. Check the endpoint protocol (optional)

`probe_endpoint.py` checks that the served model can run the harness protocol.
It needs no graph.

```bash
source env.sh
.venv/bin/python scripts/bench/probe_endpoint.py --skip-effort
```

Four gates are hard. A failed hard gate gives exit code 1.

| Gate | Question |
|---|---|
| `served_model` | Does the endpoint serve the name that the batch requests? |
| `sub_query_endpoint` | Does `llm_query` get an answer from the endpoint? |
| `stop_honoured` | Does the closing-fence stop sequence work when thinking is off? |
| `turn_yields_a_cell` | Do the harness settings give one usable Python cell per turn? |

The probe also reports the multi-token prediction (MTP) acceptance rate and the
message fields that the endpoint returns. Without `--skip-effort`, it also
measures the cost of each reasoning effort.

## 3. Run the 56-task benchmark

1. Load the environment:

   ```bash
   source env.sh
   ```

2. Start the benchmark:

   ```bash
   bash scripts/run_reference.sh
   ```

The script does these steps:

1. It checks that the graph directory contains the runtime files.
2. It waits until the server can generate.
3. It runs the endpoint probe. Set `SKIP_PROBE=1` to skip it.
4. It runs the 56 tasks in `tasks/cards`, `tasks/hard` and `tasks/nameonly`
   with the reference configuration.
5. It writes the reports into the batch directory.

The batch directory is `results/runs/<YYYYMMDD-HHMM>_<TAG>/`. The default tag
is `reference56`.

For a short test, run the first task of each task folder:

```bash
TAG=smoke bash scripts/run_reference.sh --limit 1
```

## 4. Run on a Slurm cluster

`jobs/slurm/reference56.slurm` starts the server and runs the benchmark on one
GPU node.

1. Open `jobs/slurm/reference56.slurm`.
2. Replace `<your-account>` and `<gpu-partition>` with the values for your
   cluster.
3. Add the `module load` lines that your cluster needs.
4. Adjust `--time`. The reference batch took 38 minutes after the server was
   ready.
5. Submit the job from the repository root:

   ```bash
   sbatch jobs/slurm/reference56.slurm
   ```

The job log is `logs/rlm-reference56_<job-id>.out`. The server log is
`logs/vllm_<job-id>.log`. The job stops the server when the benchmark ends.

## 5. Run one task

Use one task to test a change or to read a trajectory.

```bash
source env.sh
.venv/bin/python scripts/rlm_loop.py \
  --task-file tasks/cards/ST_RETRIEVE_ONE_CROSS_LINGUAL_001.json \
  --batch one-task
```

The run directory is under `results/runs/<YYYYMMDD-HHMM>_one-task/`.

To run a selection of tasks as one batch, give the files to the batch runner:

```bash
.venv/bin/python scripts/bench/run_batch.py \
  --task-files tasks/cards/PST_RETRIEVE_ALL_SEMANTIC_FILTER_001.json \
               tasks/hard/HARD_02_homonym_needs_native_description.json \
  --workers 1 --tag selection
```

## 6. Settings

`config/runs/reference.env` holds the settings of the published results. A
variable set in the shell has priority, so one command can change one setting:

```bash
RLM_REASONING_EFFORT=medium bash scripts/run_reference.sh
```

| Variable | Reference | Effect |
|---|---|---|
| `RLM_ENABLE_THINKING` | `1` | Thinking on or off |
| `RLM_REASONING_EFFORT` | `low` | Reasoning effort sent to the endpoint |
| `RLM_TEMPERATURE` | `0` | Sampling temperature of the root model |
| `RLM_TOP_P` | `0.95` | Nucleus sampling |
| `RLM_TOP_K` | `20` | Top-k sampling |
| `RLM_MAX_TOKENS` | `32768` | Output token budget per request |
| `RLM_LLM_MAP` | `1` | Makes `llm_map` available to the model |
| `RLM_SEARCH_HINTS` | `1` | Explains an empty name search |
| `RLM_FINAL_GUARD` | `0` | Optional check of quotations at `FINAL` |
| `SUBQ_LIMIT` | `80` | Whole-run cap on sub-query calls |
| `WORKERS` | `1` | Concurrent tasks |
| `MAX_NUM_SEQS` | `8` | vLLM concurrency ceiling |
| `MAX_MODEL_LEN` | `131072` | vLLM context length |
| `MTP_TOKENS` | `3` | Speculative tokens. `0` disables MTP. |
| `REASONING_PARSER` | `qwen3` | `none` requires `RLM_ENABLE_THINKING=0` |

Every run records its generation settings under `generation` in
`metadata.json`. Every batch records them in `batch.json`.

To use a different OpenAI-compatible endpoint, set `BASE_URL`, `API_KEY` and
`MODEL_NAME`. `SUBQ_MODEL_NAME` sets a different model for sub-queries. By
default, sub-queries use `MODEL_NAME`.

## 7. Inspect results

| Command | Output |
|---|---|
| `.venv/bin/python scripts/bench/summary.py results/runs/<batch>` | Totals and a table per execution pattern |
| `.venv/bin/python scripts/bench/score.py results/runs/<batch>/*/` | One line per run: verdict, reads, turns, status, wrong fields |
| `.venv/bin/python scripts/bench/coverage.py results/runs/<batch>/*/` | Share of each candidate set that the run read |
| `.venv/bin/python scripts/bench/grounding.py results/runs/<batch>/*/` | Quotation grounding verdicts |
| `.venv/bin/python scripts/bench/timing.py results/runs/<batch>` | Wall time and generation rate |
| `.venv/bin/python scripts/bench/thinking.py results/runs/<batch>` | Share of output tokens spent in reasoning |
| `.venv/bin/python scripts/bench/compare.py results/runs/<a> results/runs/<b>` | Tasks fixed and tasks changed between two batches |
| `.venv/bin/python scripts/bench/audit_gold.py` | Reference answers checked against the graph |
| `.venv/bin/python scripts/bench/audit_tasks.py` | Task files that are valid for the harness |

The run browser shows every trajectory turn by turn: prompt, reasoning, code,
output, sub-queries and answer.

```bash
.venv/bin/python scripts/bench/view.py --open
```

The browser reads `results/runs/` on each refresh. It listens on
`127.0.0.1:8765` by default.

## 8. Hub discovery

`discover_hubs.py` builds the hub catalogue that supports question design (see
[06 Question design](06-question-design.md)).

```bash
source env.sh
.venv/bin/python scripts/taxonomy/discover_hubs.py \
  --max-hubs 500 \
  --output-dir data/generation/taxonomy/hub_discovery_500
```

The command does not overwrite an existing output directory unless you add
`--force`.
