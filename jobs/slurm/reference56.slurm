#!/bin/bash
# Slurm template: serve the model and run the 56-task benchmark on one GPU node.
#
# Edit the #SBATCH lines for your cluster (account, partition, constraint).
# Submit from the repository root:
#   sbatch jobs/slurm/reference56.slurm
#
# The node needs one 80 GB GPU, read access to $WIKIDATA_GRAPH_DIR and the
# weights under $MODELS_DIR. Compute nodes need no internet access.
#
#SBATCH --job-name=rlm-reference56
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
mkdir -p logs

# Load site modules here when the cluster needs them, for example:
# module load cuda/13.0

source ./env.sh
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export TAG="${TAG:-reference56_${SLURM_JOB_ID}}"
SERVER_LOG="logs/vllm_${SLURM_JOB_ID}.log"

bash scripts/serve/serve_vllm.sh >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Model load and compilation take several minutes. Wait up to 20 minutes.
if ! bash scripts/serve/wait_for_server.sh "$BASE_URL" "$MODEL_NAME" 240; then
    echo "the server did not start; last lines of $SERVER_LOG:" >&2
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
fi

bash scripts/run_reference.sh
