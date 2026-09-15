# Setup

This procedure prepares one machine or cluster to run the benchmark. Paths are
not fixed: `env.sh` reads them from `config/local.env` and resolves relative
paths against the repository root.

## Requirements

| Item | Requirement |
|---|---|
| Operating system | Linux for the model server. The harness and scorers also run on macOS. |
| Python | 3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) (recommended) or pip |
| GPU (server only) | One GPU with 80 GB of memory. The reference run used one NVIDIA H100 80 GB. |
| vLLM (server only) | 0.17.0 or later, with transformers 5.8.0 or later |
| Disk | The graph directory, plus at least 40 GB for model weights and caches |

The harness, the scorers and the tests do not need a GPU. Only the model server
needs a GPU. The harness can connect to a server on another machine.

## 1. Get the repository

Put the repository in any directory. All scripts find the repository root from
their own location.

## 2. Install the harness environment

1. Go to the repository root.
2. Create the environment from the lock file:

   ```bash
   uv sync
   ```

   Without uv, use pip in a Python 3.13 virtual environment:

   ```bash
   python3.13 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. Run the unit tests. They need no graph, no GPU and no server:

   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```

## 3. Install vLLM on the server machine

vLLM is not in `pyproject.toml`, because its build depends on the CUDA version
of the machine.

1. Activate the environment that will run the server. It can be the harness
   environment.
2. Install vLLM and transformers for your CUDA version:

   ```bash
   uv pip install "vllm>=0.17.0" "transformers>=5.8.0"
   ```

   `env.sh` sets `UV_NO_SYNC=1`. This stops `uv run` from removing the
   hand-installed packages. Run `uv sync` only when you want to reset the
   environment to the lock file.

## 4. Configure the site

1. Copy the example configuration:

   ```bash
   cp config/local.env.example config/local.env
   ```

2. Edit `config/local.env`. Set these values:

   | Variable | Meaning |
   |---|---|
   | `RLM_STORAGE_ROOT` | Root for caches and temporary files |
   | `MODELS_DIR` | Directory that contains the model weights |
   | `WIKIDATA_GRAPH_DIR` | The built graph directory |
   | `BASE_URL` | The endpoint URL, when the server is on another machine |
   | `API_KEY` | The endpoint key. A local vLLM server accepts any non-empty value. |
   | `MODEL_NAME` | The served model name. Default: `Qwen3.8-27B-FP8`. |

3. Load the environment in each new bash shell:

   ```bash
   source env.sh
   ```

   The script prints the repository, graph, model and endpoint that it
   resolved. Read these lines before a large download or a long run.

A variable that is already set in the shell has priority over
`config/local.env`. `config/local.env` has priority over
`config/runs/reference.env`.

Do not write tokens or passwords into `config/local.env`. Set `HF_TOKEN` in the
shell when a download needs one.

## 5. Get the model weights

The server needs internet access for this step only. On many clusters, run it
on a login node.

1. Load the environment:

   ```bash
   source env.sh
   ```

2. Download the checkpoint:

   ```bash
   .venv/bin/python scripts/download_model.py
   ```

   The script writes `$MODELS_DIR/unsloth_Qwen3.8-27B-FP8`. It skips GGUF and
   `original/` files. An interrupted download continues when you run the
   command again.

3. Verify that the download is complete:

   ```bash
   .venv/bin/python scripts/download_model.py --verify
   ```

## 6. Get the graph

The runtime needs these items in `WIKIDATA_GRAPH_DIR`:

| Item | Built by | Used for |
|---|---|---|
| `meta.json` | `build_graph.py` | Graph size and definition |
| `nodes.parquet` | `build_graph.py` | Labels, descriptions, aliases, degrees, sitelinks |
| `edges_fwd.arrow`, `edges_bwd.arrow` | `build_graph.py` | Navigation in both directions |
| `properties.json` | `build_graph.py` | Property labels in six languages |
| `references/`, `ranks.parquet`, `annotations_index.json` | `build_annotations.py` | Inputs of the entity records |
| `entities/`, `entities_index.json` | `build_entities.py` | Entity records with claims |
| `label_index/` | `build_label_index.py` | Name search |

The graph is read-only at runtime. Several users and jobs can share one copy.
Point `WIKIDATA_GRAPH_DIR` at the shared copy.

### Use a prebuilt graph

Copy the prebuilt graph directory to a disk that the run machine can read. Set
`WIKIDATA_GRAPH_DIR` to that directory.

### Build the graph

The build needs a CPU machine with enough memory for the node table and disk
for the outputs. It needs no GPU.

1. Put the raw inputs on disk:

   | Variable | Contents |
   |---|---|
   | `WIKIDATA_FORMATTED_SOURCE` | `batch_*.parquet` statement batches with the columns `subject_id`, `subject_label`, `statement_id`, `property_id`, `property_label`, `value_id`, `value_label`, `value_type` and the qualifier columns |
   | `WIKIDATA_EXTRACTED_SOURCE` | `languages/batch_*.parquet`, `instance_of/batch_*.parquet`, `sitelinks.parquet`, `wikidata_property_translations.parquet` |
   | `WIKIDATA_HF_RANKS_SOURCE` | `ranks/chunk_*.parquet` and `references/chunk_*.parquet` |

2. Set the three variables in `config/local.env`.
3. Load the environment:

   ```bash
   source env.sh
   ```

4. Build the navigation graph:

   ```bash
   .venv/bin/python scripts/build_graph.py
   ```

5. Build the annotations:

   ```bash
   .venv/bin/python scripts/build_annotations.py
   ```

6. Build the entity records:

   ```bash
   .venv/bin/python scripts/build_entities.py
   ```

7. Build the name index:

   ```bash
   .venv/bin/python scripts/build_label_index.py
   ```

`build_graph.py --only nodes` rebuilds the node table without a new edge scan.
`build_entities.py --batches 40` builds a small sample for a test.

## 7. Check the installation

1. Start the server check without serving:

   ```bash
   bash scripts/serve/serve_vllm.sh --check
   ```

   The check verifies the vLLM and transformers versions and the checkpoint.

2. Continue with [05 Running](05-running.md).
