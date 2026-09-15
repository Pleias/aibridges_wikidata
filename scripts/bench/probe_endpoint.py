"""probe_endpoint.py — prove a served model can run the RLM protocol.

Prerequisites: a reachable OpenAI-compatible endpoint (BASE_URL, API_KEY,
    MODEL_NAME), normally the vLLM server started by scripts/serve/
    serve_vllm.sh. The graph is NOT needed; this probe never reads it.
Outputs:       a table on stdout and, with --report, one JSON file. Exit code
    1 when a hard gate fails, so a job can stop before the batch.

Contract: the probe imports CELL_END, SUBQ_SYSTEM and extract_code from
    rlm_loop rather than restating them. A local copy of the stop sequence
    would let the probe pass while the harness fails on the same server.

Why: a batch measurement means nothing if the served model ignores `stop`
    or cannot return a Python cell. Thinking enlarges that risk rather than
    removing it: `stop` is
    matched against the generated stream, the reasoning block is part of that
    stream, and a code fence written inside <think> can therefore end the turn
    before any Python cell exists. That failure is invisible in a score file
    and expensive in GPU hours, so it is measured here first.

Usage:
    uv run python scripts/bench/probe_endpoint.py
    uv run python scripts/bench/probe_endpoint.py --repeats 5
    uv run python scripts/bench/probe_endpoint.py --report results/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rlm_loop import (CELL_END, ENABLE_THINKING, REASONING_EFFORT,
                      SUBQ_SYSTEM, extra_field, extract_code,
                      load_dotenv, make_client, reasoning_text,
                      turn_stop)

# A word the model is asked to write AFTER the closing fence. If the server
# honours `stop`, generation ends at the fence and the word never appears.
SENTINEL = "BANANA_AFTER_FENCE"

STOP_PROBE = (
    f"Write exactly one ```python block that prints the number 1. "
    f"Immediately after the closing fence, write the single word "
    f"{SENTINEL} on its own line."
)

# The decisive prompt. It asks for a fenced block INSIDE the reasoning, which
# is what a real turn produces naturally when the model drafts code before
# committing to it.
FENCE_IN_THINKING_PROBE = (
    "Before you answer, draft the code inside your reasoning: write it there "
    "as a fenced ```python block first and check it. Then give your final "
    "reply as exactly one ```python block that prints the number 1."
)

# A realistic root turn: the same shape the harness sends, without the graph.
EFFORT_PROBE_SYSTEM = (
    "You are a Recursive Language Model (RLM) agent exploring a frozen "
    "Wikidata snapshot through Python functions in a persistent REPL.\n\n"
    "## Rules\n"
    "1. ONE ```python block per turn.\n"
    "2. Keep data in variables and print only useful summaries.\n"
    "3. Every final fact must come from a function call made during this run.\n"
    "4. Use FINAL(value) to finish.\n\n"
    "## Available in the REPL\n"
    "- search_entity(text, limit=10, lang='en') -> (total, [{'qid', 'label', "
    "'description'}])\n"
    "- edges(qid, direction='both', pid=None) -> {'out': [...], 'in_': [...]}\n"
    "- descriptions(qid) -> descriptions in six languages\n"
    "- FINAL(value) -> terminate the run\n"
)

EFFORT_PROBE_USER = (
    "Question: Among people recorded as employees of PBS, identify the person "
    "described simply as a documentary producer. Follow that person's notable "
    "work. In which available descriptions is the work attributed to the Coen "
    "brothers?"
)

EFFORTS = ("low", "medium", "xhigh")

# vLLM exposes speculative-decoding counters only when a draft method is
# configured. Their absence means MTP is off, not that the server is broken.
MTP_METRICS = (
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
)


def metrics_url() -> str:
    """The /metrics path of the same server BASE_URL points at."""
    base = os.environ.get("BASE_URL", "").rstrip("/")
    return base[: -len("/v1")] + "/metrics" if base.endswith("/v1") else \
        base + "/metrics"


def read_mtp_counters() -> dict[str, float]:
    """Current speculative-decoding totals, or {} when MTP is not configured."""
    try:
        response = requests.get(metrics_url(), timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return {}
    found: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        for name in MTP_METRICS:
            if line.startswith(name):
                with_value = line.rsplit(" ", 1)
                if len(with_value) == 2:
                    try:
                        found[name] = found.get(name, 0.0) + float(with_value[1])
                    except ValueError:
                        pass
    return found


def call(client, model: str, messages: list[dict], *, stop=None,
         thinking: bool | None = None, effort: str | None = None,
         effort_in_template: bool = False, max_tokens: int = 4096,
         temperature: float = 0.7) -> dict:
    """One completion, reported as plain data rather than an SDK object."""
    template: dict = {}
    if thinking is not None:
        template["enable_thinking"] = thinking
    if effort and effort_in_template:
        template["reasoning_effort"] = effort
    kwargs: dict = {}
    if effort and not effort_in_template:
        kwargs["reasoning_effort"] = effort
    started = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **({"stop": stop} if stop else {}),
        **({"extra_body": {"chat_template_kwargs": template}} if template
           else {}),
        **kwargs,
    )
    choice = response.choices[0]
    message = choice.message
    extra = getattr(choice, "model_extra", None) or {}
    # vLLM reports the matched stop string here. The SDK may expose it as a
    # real attribute or leave it in the extra fields, and either can be None.
    stop_reason = extra_field(choice, "stop_reason")
    reasoning = reasoning_text(message)
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "content": message.content or "",
        "reasoning": reasoning,
        # What the endpoint actually returns on the message, so an absent
        # field can be told apart from one this client failed to read.
        "message_fields": sorted(
            set(message.model_dump(exclude_none=True))
            | set(getattr(message, "model_extra", None) or {})),
        "finish_reason": choice.finish_reason,
        "stop_reason": stop_reason,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        # Measured length, not the reported count. Some servers report
        # reasoning_tokens=0 while the model is thinking, so the usage field
        # cannot be used to decide whether an effort level did anything.
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "reasoning_chars": len(reasoning),
        "wall_s": round(time.time() - started, 1),
    }


def one_cell(content: str) -> bool:
    """True when the reply carries exactly one executable Python block."""
    return isinstance(extract_code(content), str)


def check_served_model(client, model: str) -> dict:
    """Gate: the endpoint must serve the exact name the batch will request."""
    served = sorted(entry.id for entry in client.models.list().data)
    return {"check": "served_model", "gate": True, "pass": model in served,
            "detail": {"requested": model, "served": served}}


def check_stop_without_thinking(client, model: str) -> dict:
    """Gate: `stop` must be honoured, or one-cell-per-turn is unenforceable."""
    result = call(client, model, [{"role": "user", "content": STOP_PROBE}],
                  stop=CELL_END, thinking=False, temperature=0.7)
    matched = result["stop_reason"] in CELL_END
    fenced = "```" in result["content"]
    absent = SENTINEL not in result["content"]
    # Only a hard gate when the harness actually sends a stop. With thinking
    # on it sends none (see turn_stop), and failing a run over an unused
    # feature would block a configuration that works.
    # A reply with no fence at all cannot demonstrate anything: the sentinel is
    # then missing because the model never got that far, not because `stop`
    # fired. Treat it as a failed gate so a human reads the transcript.
    return {
        "check": "stop_honoured", "gate": turn_stop() is not None,
        "pass": bool(matched or (fenced and absent)),
        "detail": {
            "used_by_this_configuration": turn_stop() is not None,
            "stop_reason_matched_cell_end": matched,
            "produced_a_fence": fenced,
            "sentinel_absent": absent,
            "inconclusive": not fenced and not matched,
            "finish_reason": result["finish_reason"],
            "content_tail": result["content"][-200:],
        },
    }


def check_turn_yields_a_cell(client, model: str, repeats: int,
                             effort: str) -> dict:
    """Gate: the harness's OWN turn settings must produce one usable cell.

    The configuration under test is whatever `rlm_loop` will send — thinking
    per ENABLE_THINKING, stop per turn_stop() — so this cannot pass while the
    real batch fails. A contrast run flips only the stop sequence, which is
    what separates a protocol failure from a model declining the instruction.

    The failure signature to watch is specific: empty content, non-empty
    reasoning, finish_reason 'stop'. That is `stop` firing inside <think>.
    """
    stop = turn_stop()
    messages = [{"role": "user", "content": FENCE_IN_THINKING_PROBE}]

    def sample(with_stop: bool) -> list[dict]:
        rows = []
        for _ in range(repeats):
            result = call(client, model, messages,
                          stop=CELL_END if with_stop else None,
                          thinking=ENABLE_THINKING, effort=effort,
                          max_tokens=8192, temperature=1.0)
            result["cell_ok"] = one_cell(result["content"])
            result["stopped_in_thinking"] = bool(
                not result["content"].strip() and result["reasoning"].strip()
                and result["finish_reason"] == "stop")
            rows.append(result)
        return rows

    configured = sample(bool(stop))
    contrast = sample(not stop)

    def rate(rows):
        return round(sum(r["cell_ok"] for r in rows) / max(1, len(rows)), 2)

    killed = sum(r["stopped_in_thinking"] for r in configured)
    cell_rate = rate(configured)
    return {
        "check": "turn_yields_a_cell", "gate": True,
        "pass": cell_rate >= 0.5 and killed == 0,
        "warn": cell_rate < 1.0,
        "detail": {
            "thinking": ENABLE_THINKING, "stop": stop, "effort": effort,
            "repeats": repeats,
            "cell_rate_as_configured": cell_rate,
            "cell_rate_with_stop_flipped": rate(contrast),
            "turns_killed_inside_thinking": killed,
            "message_fields": configured[0]["message_fields"] if configured
            else [],
            "runs": [{k: r[k] for k in (
                "cell_ok", "stopped_in_thinking", "finish_reason",
                "stop_reason", "reasoning_tokens", "reasoning_chars",
                "completion_tokens", "wall_s")} for r in configured],
        },
    }


def check_reasoning_effort(client, model: str) -> dict:
    """Report only: what each effort level costs, and which channel sets it.

    The model card passes `reasoning_effort` as a top-level API parameter; the
    vLLM recipe passes it through chat_template_kwargs. Both are tried, and
    the token counts say which one the server actually acts on.
    """
    messages = [
        {"role": "system", "content": EFFORT_PROBE_SYSTEM},
        {"role": "user", "content": EFFORT_PROBE_USER},
    ]
    rows = []
    for effort in EFFORTS:
        for in_template in (False, True):
            result = call(client, model, messages, stop=CELL_END,
                          thinking=True, effort=effort,
                          effort_in_template=in_template, max_tokens=16384,
                          temperature=1.0)
            rows.append({
                "effort": effort,
                "channel": "chat_template_kwargs" if in_template else "api",
                "reasoning_tokens": result["reasoning_tokens"],
                "reasoning_chars": result["reasoning_chars"],
                "completion_tokens": result["completion_tokens"],
                "finish_reason": result["finish_reason"],
                "truncated": result["finish_reason"] == "length",
                "cell_ok": one_cell(result["content"]),
                "wall_s": result["wall_s"],
            })
    by_api = [r["completion_tokens"] for r in rows if r["channel"] == "api"]
    responsive = len(set(by_api)) > 1
    return {"check": "reasoning_effort", "gate": False, "pass": True,
            "detail": {
                "effort_changes_token_count": responsive,
                "reasoning_content_returned": any(r["reasoning_chars"]
                                                  for r in rows),
                "reasoning_tokens_reported": any(r["reasoning_tokens"]
                                                 for r in rows),
                "rows": rows}}


def check_sub_query(client, model: str, subq_model: str) -> dict:
    """Gate: llm_query must reach an endpoint that answers.

    When SUBQ_MODEL_NAME names a model the endpoint does not serve, every
    delegated call fails, and the delegation column of the result table is
    silently zero.
    """
    result = call(client, subq_model, [
        {"role": "system", "content": SUBQ_SYSTEM},
        {"role": "user", "content": "Reply with the single word OK."},
    ], thinking=False, max_tokens=64, temperature=0.7)
    return {"check": "sub_query_endpoint", "gate": True,
            "pass": bool(result["content"].strip()),
            "detail": {"subq_model": subq_model,
                       "same_as_root": subq_model == model,
                       "reply": result["content"][:200]}}


def summarize_mtp(before: dict, after: dict) -> dict:
    """Report only: draft acceptance measured across this probe's generation."""
    if not before or not after:
        return {"check": "mtp", "gate": False, "pass": True,
                "detail": {"configured": False,
                           "note": "no spec_decode counters; MTP is not on"}}
    accepted = after.get(MTP_METRICS[0], 0) - before.get(MTP_METRICS[0], 0)
    draft = after.get(MTP_METRICS[1], 0) - before.get(MTP_METRICS[1], 0)
    return {"check": "mtp", "gate": False, "pass": True,
            "detail": {"configured": True,
                       "accepted_tokens": accepted, "draft_tokens": draft,
                       "acceptance_rate": round(accepted / draft, 3)
                       if draft else None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    ap.add_argument("--subq-model",
                    default=os.environ.get("SUBQ_MODEL_NAME"))
    ap.add_argument("--repeats", type=int, default=3,
                    help="samples for the fence-inside-thinking gate")
    ap.add_argument("--effort", default="medium",
                    help="effort used for the thinking gate")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--skip-effort", action="store_true",
                    help="drop the six-call effort sweep; it is the slow part")
    args = ap.parse_args()

    load_dotenv()
    model = args.model or os.environ.get("MODEL_NAME")
    if not model:
        raise SystemExit("--model or MODEL_NAME is required")
    subq_model = args.subq_model or model

    client = make_client()
    mtp_before = read_mtp_counters()

    results = [check_served_model(client, model)]
    if results[0]["pass"]:
        results.append(check_sub_query(client, model, subq_model))
        results.append(check_stop_without_thinking(client, model))
        results.append(
            check_turn_yields_a_cell(client, model, args.repeats,
                                     args.effort))
        if not args.skip_effort:
            results.append(check_reasoning_effort(client, model))
    results.append(summarize_mtp(mtp_before, read_mtp_counters()))

    print(f"\n{'check':<28} {'gate':<6} {'verdict':<8} detail")
    for row in results:
        verdict = "PASS" if row["pass"] else "FAIL"
        if row["pass"] and row.get("warn"):
            verdict = "WARN"
        print(f"{row['check']:<28} {'hard' if row['gate'] else 'info':<6} "
              f"{verdict:<8} "
              f"{json.dumps(row['detail'], ensure_ascii=False)[:120]}")

    failed = [row["check"] for row in results if row["gate"] and not row["pass"]]
    report = {"model": model, "subq_model": subq_model,
              "base_url": os.environ.get("BASE_URL"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "cell_end": CELL_END, "turn_stop": turn_stop(),
              "enable_thinking": ENABLE_THINKING,
              "reasoning_effort": REASONING_EFFORT,
              "checks": results, "failed_gates": failed}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.report}")

    if failed:
        print(f"\nFAILED GATES: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
    print("\nall hard gates passed")


if __name__ == "__main__":
    main()
