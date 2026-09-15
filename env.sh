#!/usr/bin/env bash
# Environment for the harness, the build scripts and the vLLM server.
#
# Usage (from bash, in any directory):
#   source /path/to/repository/env.sh
#
# Order of precedence, highest first:
#   1. variables already set in the shell
#   2. config/local.env        (site paths and endpoint; copy the .example)
#   3. config/runs/reference.env (generation and serving settings)
#   4. the defaults below
#
# Relative paths are resolved against the repository root, so the repository
# can move to another machine without edits.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "env.sh: source this file from bash." >&2
    return 1 2>/dev/null || exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT

# Export KEY=VALUE lines from a file. A key that is already set is kept.
_rlm_load_env_file() {
    local file="$1" line key value
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [[ -z "${!key+x}" ]]; then
            export "$key=$value"
        fi
    done < "$file"
}

# Make a path absolute against the repository root.
_rlm_abs() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s' "$REPO_ROOT/$1" ;;
    esac
}

_rlm_load_env_file "$REPO_ROOT/config/local.env"
_rlm_load_env_file "$REPO_ROOT/config/runs/reference.env"

# --- Storage ----------------------------------------------------------------
export RLM_STORAGE_ROOT="$(_rlm_abs "${RLM_STORAGE_ROOT:-storage}")"
export MODELS_DIR="$(_rlm_abs "${MODELS_DIR:-$RLM_STORAGE_ROOT/models}")"
export WIKIDATA_GRAPH_DIR="$(_rlm_abs "${WIKIDATA_GRAPH_DIR:-data/graph}")"
for _rlm_var in WIKIDATA_FORMATTED_SOURCE WIKIDATA_EXTRACTED_SOURCE \
                WIKIDATA_HF_RANKS_SOURCE; do
    if [[ -n "${!_rlm_var:-}" ]]; then
        export "$_rlm_var=$(_rlm_abs "${!_rlm_var}")"
    fi
done
unset _rlm_var

# Library caches stay under the storage root, not in the home directory.
export HF_HOME="${HF_HOME:-$MODELS_DIR/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RLM_STORAGE_ROOT/cache/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$RLM_STORAGE_ROOT/cache/flashinfer}"
export TORCH_HOME="${TORCH_HOME:-$RLM_STORAGE_ROOT/cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RLM_STORAGE_ROOT/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RLM_STORAGE_ROOT/cache/triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RLM_STORAGE_ROOT/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RLM_STORAGE_ROOT/cache/pip}"

# vLLM is installed by hand into the environment (docs/04-setup.md). This
# stops `uv run` from removing it when it reconciles against uv.lock.
export UV_NO_SYNC=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1

# --- Endpoint ---------------------------------------------------------------
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export BASE_URL="${BASE_URL:-http://$VLLM_HOST:$VLLM_PORT/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.8-27B-FP8}"
export SUBQ_MODEL_NAME="${SUBQ_MODEL_NAME:-$MODEL_NAME}"

# --- Python -----------------------------------------------------------------
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
if [[ "$PYTHON" == */* ]]; then
    PYTHON="$(_rlm_abs "$PYTHON")"
fi
export PYTHON

unset -f _rlm_load_env_file _rlm_abs

echo "env.sh: repository $REPO_ROOT" >&2
echo "env.sh: graph      $WIKIDATA_GRAPH_DIR" >&2
echo "env.sh: models     $MODELS_DIR" >&2
echo "env.sh: endpoint   $BASE_URL ($MODEL_NAME)" >&2
