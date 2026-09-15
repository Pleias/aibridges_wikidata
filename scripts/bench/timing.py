"""timing.py — wall time and throughput for an archived batch.

Prerequisites: a batch directory written by run_batch.py, holding per-task run
    directories and, when the batch completed, batch.json.
Outputs:       a table on stdout; with --json, one JSON object.

Contract: generation rate is reported as output tokens over the summed
    per-RUN wall time, never over the batch wall time. Those differ whenever
    --workers > 1, and dividing by the batch clock would silently inflate the
    rate by the worker count.

Why: the score reports give tokens, reads and turns, and no time at all.
    "How long does a 56-task batch take" and "what does this configuration
    cost per token" are answered from values every archive already records.

Usage:
    uv run python scripts/bench/timing.py results/runs/<batch>
    uv run python scripts/bench/timing.py --json results/runs/<batch>
    uv run python scripts/bench/timing.py results/runs/<a> results/runs/<b>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_runs(batch: Path) -> list[dict]:
    runs = []
    for path in sorted(batch.glob("*/metadata.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return runs


def summarize(batch: Path) -> dict | None:
    runs = load_runs(batch)
    if not runs:
        return None
    walls = [r.get("wall_time_s") or 0.0 for r in runs]
    run_seconds = sum(walls)

    def total(field: str) -> int:
        return sum(r.get(field) or 0 for r in runs)

    manifest = {}
    manifest_path = batch / "batch.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    workers = manifest.get("workers") or 1
    batch_seconds = manifest.get("wall_time_s")
    calls = total("model_calls")
    output_tokens = total("output_tokens")

    return {
        "batch": batch.name,
        "model": manifest.get("model") or runs[0].get("model"),
        "tasks": len(runs),
        "workers": workers,
        # The batch clock includes server startup and queueing between tasks;
        # the summed run clock is time actually spent inside runs.
        "batch_wall_s": batch_seconds,
        "run_wall_s": round(run_seconds, 1),
        "overhead_s": (round(batch_seconds - run_seconds, 1)
                       if batch_seconds and workers == 1 else None),
        "median_task_s": round(statistics.median(walls), 1),
        "max_task_s": round(max(walls), 1),
        "model_calls": calls,
        "s_per_call": round(run_seconds / calls, 2) if calls else None,
        "input_tokens": total("input_tokens"),
        "output_tokens": output_tokens,
        "total_tokens": total("total_tokens"),
        # Output tokens per second of run wall time. This is the number to
        # compare across configurations; it is not the server's peak rate.
        "output_tok_per_s": round(output_tokens / run_seconds, 1)
        if run_seconds else None,
        "total_tok_per_s": round(total("total_tokens") / run_seconds, 1)
        if run_seconds else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [row for row in (summarize(b) for b in args.batches if b.is_dir())
            if row]
    if not rows:
        raise SystemExit("no batch directory contained run metadata")

    if args.json:
        for row in rows:
            print(json.dumps(row))
        return

    for row in rows:
        print(f"\n{row['batch']}  ({row['model']}, {row['tasks']} tasks, "
              f"{row['workers']} worker(s))")
        pairs = [
            ("batch wall", f"{row['batch_wall_s']} s"
             if row["batch_wall_s"] else "unknown (batch.json absent)"),
            ("summed run wall", f"{row['run_wall_s']} s "
             f"({row['run_wall_s'] / 3600:.2f} h)"),
            ("startup + gaps", f"{row['overhead_s']} s"
             if row["overhead_s"] is not None else "n/a"),
            ("median / max task", f"{row['median_task_s']} s / "
             f"{row['max_task_s']} s"),
            ("model calls", f"{row['model_calls']} "
             f"({row['s_per_call']} s per call)"),
            ("output tokens", f"{row['output_tokens']:,}"),
            ("total tokens", f"{row['total_tokens']:,}"),
            ("output tok/s", str(row["output_tok_per_s"])),
            ("total tok/s", str(row["total_tok_per_s"])),
        ]
        for label, value in pairs:
            print(f"  {label:<20} {value}")


if __name__ == "__main__":
    main()
