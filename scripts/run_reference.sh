#!/usr/bin/env bash
# Run the 56-task benchmark with the reference configuration, then write the
# score, coverage, grounding and timing reports into the batch directory.
#
# Prerequisites: a running endpoint (scripts/serve/serve_vllm.sh, or any
#   OpenAI-compatible server that serves $MODEL_NAME), the graph under
#   $WIKIDATA_GRAPH_DIR, and the harness environment (docs/04-setup.md).
#
# Usage:
#   bash scripts/run_reference.sh                     # all 56 tasks
#   TAG=smoke bash scripts/run_reference.sh --limit 1 # first task of each task folder
#   SKIP_PROBE=1 bash scripts/run_reference.sh        # no protocol probe
#
# Extra arguments go to scripts/bench/run_batch.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$REPO_ROOT/env.sh"
cd "$REPO_ROOT"

TAG="${TAG:-reference56}"
BATCH="$REPO_ROOT/results/runs/$(date +%Y%m%d-%H%M)_${TAG}"

for required in meta.json nodes.parquet edges_fwd.arrow edges_bwd.arrow \
                properties.json entities_index.json label_index; do
    if [[ ! -e "$WIKIDATA_GRAPH_DIR/$required" ]]; then
        echo "missing $WIKIDATA_GRAPH_DIR/$required (see docs/04-setup.md)" >&2
        exit 1
    fi
done

bash scripts/serve/wait_for_server.sh "$BASE_URL" "$MODEL_NAME"

mkdir -p "$BATCH"
if [[ "${SKIP_PROBE:-0}" != "1" ]]; then
    # Hard gates: served model name, sub-query endpoint, stop handling, and
    # one usable Python cell per turn. A failure stops the run here.
    "$PYTHON" scripts/bench/probe_endpoint.py --skip-effort \
        --effort "$RLM_REASONING_EFFORT" --report "$BATCH/probe.json"
fi

"$PYTHON" scripts/bench/run_batch.py \
    --families cards hard nameonly \
    --workers "${WORKERS:-1}" \
    --output-dir "$BATCH" \
    "$@"

"$PYTHON" scripts/bench/summary.py "$BATCH"            | tee "$BATCH/report_summary.txt"
"$PYTHON" scripts/bench/score.py "$BATCH"/*/          > "$BATCH/report_scores.txt"
"$PYTHON" scripts/bench/coverage.py "$BATCH"/*/       > "$BATCH/report_coverage.txt"
"$PYTHON" scripts/bench/grounding.py "$BATCH"/*/      > "$BATCH/report_grounding.txt"
"$PYTHON" scripts/bench/timing.py "$BATCH"            > "$BATCH/report_timing.txt"

echo
echo "batch:   $BATCH"
echo "reports: $BATCH/report_*.txt"
echo "browse:  $PYTHON scripts/bench/view.py --open"
