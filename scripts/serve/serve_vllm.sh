#!/usr/bin/env bash
# Serve Qwen3.8-27B-FP8 on an OpenAI-compatible endpoint with the reference
# serving configuration.
#
# Prerequisites: one GPU with 80 GB of memory, vLLM installed in the active
#   environment (docs/04-setup.md), and the weights under
#   $MODELS_DIR/unsloth_Qwen3.8-27B-FP8 (scripts/download_model.py).
#
# Usage:
#   bash scripts/serve/serve_vllm.sh            # check, then serve (foreground)
#   bash scripts/serve/serve_vllm.sh --check    # check only
#
# The server listens on $VLLM_HOST:$VLLM_PORT (default 127.0.0.1:8000) and
# serves the model under the name $MODEL_NAME. Stop it with Ctrl-C.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../env.sh
source "$REPO_ROOT/env.sh"

MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/unsloth_Qwen3.8-27B-FP8}"
VLLM_BIN="${VLLM_BIN:-vllm}"

# The weights are local. The server does not need the network.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# The reference run used 0: the FlashInfer FP8 blockscale kernel is JIT-built
# and its build must match the CUDA runtime of torch. Set 1 when your
# FlashInfer build supports it.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"

# --language-model-only: Qwen3.8 is a vision-language model and the harness
#   is text-only. The vision encoder memory goes to the KV cache.
# --max-model-len 131072: only 16 of the 64 layers keep a growing KV cache;
#   the other 48 hold a constant recurrent state, so a long context fits.
# --reasoning-parser qwen3: keeps <think> out of `content`. The harness
#   depends on it while thinking is on (see turn_stop in scripts/rlm_loop.py).
# --speculative-config mtp: multi-token prediction with 3 draft tokens.
SERVE_ARGS=(
    --served-model-name "$MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --max-model-len "${MAX_MODEL_LEN:-131072}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
    --language-model-only
)

REASONING_PARSER="${REASONING_PARSER:-qwen3}"
if [[ "$REASONING_PARSER" != "none" ]]; then
    SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
elif [[ "${RLM_ENABLE_THINKING:-0}" == "1" ]]; then
    echo "REASONING_PARSER=none requires RLM_ENABLE_THINKING=0." >&2
    exit 1
fi

MTP_TOKENS="${MTP_TOKENS:-3}"
if [[ "$MTP_TOKENS" != "0" ]]; then
    SERVE_ARGS+=(--speculative-config
                 "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

# Qwen3.8 is a hybrid Gated-DeltaNet model. An older vLLM cannot build it,
# and the failure comes only after the weights are on the GPU.
check_versions() {
    "$PYTHON" - <<'VERSIONS'
import re

import transformers
import vllm


def parts(version):
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


print(f"vllm {vllm.__version__} | transformers {transformers.__version__}")
assert parts(vllm.__version__) >= (0, 17, 0), "Qwen3.8 needs vLLM 0.17.0 or later"
assert parts(transformers.__version__) >= (5, 8, 0), \
    "Qwen3.8 needs transformers 5.8.0 or later"
VERSIONS
}

# config.json alone does not prove a complete checkpoint. An interrupted
# download keeps it and a few shards, and can lose the tokenizer.
check_checkpoint() {
    "$PYTHON" - "$MODEL_PATH" <<'CHECKPOINT'
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

path = Path(sys.argv[1])
shards = sorted(path.glob("*.safetensors"))
assert shards, f"no *.safetensors in {path}. Run scripts/download_model.py"
AutoConfig.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
for marker in ("<think>", "</think>"):
    assert tokenizer.encode(marker, add_special_tokens=False), (
        f"the tokenizer cannot encode {marker}; the checkpoint is incomplete. "
        f"Run: python scripts/download_model.py --verify")
print(f"checkpoint ok: {len(shards)} shards, vocab {len(tokenizer)}")
CHECKPOINT
}

check_versions
check_checkpoint

if [[ "${1:-}" == "--check" ]]; then
    echo "checks passed; not serving (--check)"
    exit 0
fi

echo "serving $MODEL_PATH as $MODEL_NAME on $VLLM_HOST:$VLLM_PORT"
printf '  %s\n' "${SERVE_ARGS[@]}"
exec "$VLLM_BIN" serve "$MODEL_PATH" "${SERVE_ARGS[@]}"
