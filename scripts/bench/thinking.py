"""thinking.py — how much of a batch's output was reasoning.

Prerequisites: a batch directory whose runs carry reasoning.json (written when
    RLM_ENABLE_THINKING=1). For token counts rather than characters, a local
    tokenizer directory — by default $MODELS_DIR/<model recorded in metadata>.
Outputs:       a table on stdout; with --json, one JSON object per batch.

Contract: the endpoint's `output_tokens` counts EVERY generated token, thinking
    included, so the reasoning share is reasoning tokens over output tokens.
    vLLM 0.27.1 reports completion_tokens_details.reasoning_tokens as 0 for
    Qwen3.8, which is why the split is recomputed here from the archived text
    instead of read from usage. If the recount exceeds output_tokens the
    assumption is wrong for that server, and the tool says so rather than
    printing a share above 100%.

Why: reasoning effort is a knob with no cost signal attached — `medium` and
    `xhigh` differed 4x on a single synthetic prompt, which is not enough to
    choose on. This turns a finished batch into that measurement.

Usage:
    uv run python scripts/bench/thinking.py results/runs/<batch>
    uv run python scripts/bench/thinking.py --chars-only results/runs/<batch>
    uv run python scripts/bench/thinking.py --tokenizer "$MODELS_DIR/x" <batch>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Task ids start with their execution pattern, so the family breakdown needs no
# lookup into tasks/ — which may hold a different revision than the batch ran.
FAMILY = re.compile(r"^(HIDE|HARD|LT|SST|PST|ST)_")


def family_of(task: str) -> str:
    match = FAMILY.match(task)
    return match.group(1) if match else "other"


def load_tokenizer(path: str | None):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception:                             # noqa: BLE001 — optional
        return None


def default_tokenizer_path(runs: list[dict]) -> str | None:
    models_dir = os.environ.get("MODELS_DIR")
    if not models_dir or not runs:
        return None
    model = runs[0].get("model") or ""
    for candidate in Path(models_dir).glob("*"):
        # metadata records the SERVED name ("Qwen3.8-27B-FP8"); the directory
        # is the repo id ("unsloth_Qwen3.8-27B-FP8"). Match on the suffix.
        if candidate.is_dir() and candidate.name.endswith(model):
            return str(candidate)
    return None


def collect(batch: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(batch.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        thoughts = []
        reasoning_path = meta_path.parent / "reasoning.json"
        if reasoning_path.is_file():
            try:
                thoughts = json.loads(reasoning_path.read_text())
            except json.JSONDecodeError:
                thoughts = []
        rows.append({
            "task": meta.get("task", meta_path.parent.name),
            "family": family_of(meta.get("task", "")),
            "output_tokens": meta.get("output_tokens") or 0,
            "total_tokens": meta.get("total_tokens") or 0,
            "iterations": meta.get("iterations") or 0,
            "reasoning_turns": len(thoughts),
            "reasoning_chars": sum(len(t.get("reasoning", "")) for t in thoughts),
            "texts": [t.get("reasoning", "") for t in thoughts],
        })
    return rows


def summarize(batch: Path, tokenizer, chars_only: bool) -> dict:
    rows = collect(batch)
    if not rows:
        return {}
    for row in rows:
        if tokenizer is not None and not chars_only and row["texts"]:
            row["reasoning_tokens"] = sum(
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in row["texts"])
        else:
            row["reasoning_tokens"] = None
        row.pop("texts")

    out = sum(r["output_tokens"] for r in rows)
    counted = [r for r in rows if r["reasoning_tokens"] is not None]
    think = sum(r["reasoning_tokens"] for r in counted) if counted else None
    return {
        "batch": batch.name,
        "runs": len(rows),
        "runs_with_reasoning": sum(1 for r in rows if r["reasoning_turns"]),
        "output_tokens": out,
        "reasoning_tokens": think,
        "reasoning_chars": sum(r["reasoning_chars"] for r in rows),
        "reasoning_share": (think / out) if think and out else None,
        # A share above 1 means output_tokens does not include the thinking on
        # this server, and every derived number below would be wrong.
        "share_is_impossible": bool(think and out and think > out),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--tokenizer", default=None,
                    help="local model directory; default resolves from MODELS_DIR")
    ap.add_argument("--chars-only", action="store_true",
                    help="skip tokenization and report characters")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for batch in args.batches:
        if not batch.is_dir():
            continue
        rows = collect(batch)
        path = args.tokenizer or default_tokenizer_path(
            [json.loads(p.read_text())
             for p in sorted(batch.glob("*/metadata.json"))[:1]])
        tokenizer = None if args.chars_only else load_tokenizer(path)
        report = summarize(batch, tokenizer, args.chars_only)
        if not report:
            print(f"{batch.name}: no run metadata")
            continue
        if args.json:
            print(json.dumps(report))
            continue

        print(f"\n{report['batch']}")
        print(f"  runs with reasoning   {report['runs_with_reasoning']}"
              f"/{report['runs']}")
        print(f"  reasoning chars       {report['reasoning_chars']:,}")
        if report["reasoning_tokens"] is None:
            print("  reasoning tokens      not counted "
                  "(no tokenizer; pass --tokenizer or use --chars-only)")
        else:
            print(f"  reasoning tokens      {report['reasoning_tokens']:,}")
            print(f"  output tokens         {report['output_tokens']:,}")
            if report["share_is_impossible"]:
                print("  share                 IMPOSSIBLE — reasoning exceeds "
                      "output_tokens, so output does not include thinking here")
            else:
                print(f"  reasoning share       "
                      f"{report['reasoning_share']:.1%} of generated tokens")

        print(f"\n  {'family':<8}{'runs':>6}{'think turns':>13}"
              f"{'chars':>12}{'tokens':>10}{'chars/turn':>12}")
        families: dict[str, list[dict]] = {}
        for row in report["rows"]:
            families.setdefault(row["family"], []).append(row)
        for name in sorted(families, key=lambda k: -sum(
                r["reasoning_chars"] for r in families[k])):
            group = families[name]
            turns = sum(r["reasoning_turns"] for r in group)
            chars = sum(r["reasoning_chars"] for r in group)
            toks = sum(r["reasoning_tokens"] or 0 for r in group)
            print(f"  {name:<8}{len(group):>6}{turns:>13}{chars:>12,}"
                  f"{toks if toks else '-':>10}"
                  f"{chars // turns if turns else 0:>12,}")


if __name__ == "__main__":
    main()
