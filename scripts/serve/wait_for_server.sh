#!/usr/bin/env bash
# Wait until the endpoint can COMPLETE a request, not only answer /health.
#
# A vLLM server answers /health some time before it can generate. A batch that
# starts in that interval loses its first tasks to connection errors. This
# check sends a one-token completion and returns only when it succeeds.
#
# Usage:
#   bash scripts/serve/wait_for_server.sh [BASE_URL] [MODEL_NAME] [TRIES]
# Defaults come from the environment (source env.sh). One try every 5 seconds.
set -euo pipefail
BASE="${1:-${BASE_URL:?set BASE_URL or pass it as the first argument}}"
MODEL="${2:-${MODEL_NAME:?set MODEL_NAME or pass it as the second argument}}"
TRIES="${3:-120}"

for i in $(seq 1 "$TRIES"); do
    if curl -fsS --max-time 20 "$BASE/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${API_KEY:-EMPTY}" \
            -d "{\"model\":\"$MODEL\",\"max_tokens\":1,
                 \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
            >/dev/null 2>&1; then
        echo "server can generate (after ${i} attempt(s))"
        exit 0
    fi
    sleep 5
done
echo "server did not complete a request after $((TRIES * 5))s; batch not started" >&2
exit 1
