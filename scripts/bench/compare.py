"""compare.py — what changed between two runs of the same tasks.

Prerequisites: two batch directories written by run_batch.py, ideally over the
    same task list. Tasks present in only one side are reported, not silently
    dropped.
Outputs:       a table on stdout; with --json, one JSON object.

Contract: tasks are matched by `task`, never by run directory name, which
    carries a timestamp. A task that appears twice in a batch (a retry) is
    represented by its LAST run, matching how run_batch's own scores.json
    selects one.

Why: a configuration change is a two-arm comparison, and a headline count
    hides the shape of it. Four tasks fixed and four broken is not the same
    result as zero and zero, and both read as "no change" in a total.

Usage:
    uv run python scripts/bench/compare.py results/runs/<a> results/runs/<b>
    uv run python scripts/bench/compare.py --json <a> <b>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(batch: Path) -> dict[str, dict]:
    """Scored rows by task id, preferring scores.json when the batch finished."""
    rows: list[dict] = []
    scores = batch / "scores.json"
    if scores.is_file():
        try:
            rows = json.loads(scores.read_text())
        except json.JSONDecodeError:
            rows = []
    if not rows:
        # A batch still running, or one that died before scoring, still has
        # per-run metadata worth comparing.
        for meta_path in sorted(batch.glob("*/metadata.json")):
            meta = json.loads(meta_path.read_text())
            rows.append({"task": meta.get("task"), "exact": None,
                         "status": meta.get("status"),
                         "iterations": meta.get("iterations"),
                         "reads": meta.get("charged", meta.get("reads")),
                         "total_tokens": meta.get("total_tokens"),
                         "wrong_fields": []})
    return {r["task"]: r for r in rows if r.get("task")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a, b = load(args.before), load(args.after)
    shared = sorted(set(a) & set(b))
    fixed = [t for t in shared if not a[t].get("exact") and b[t].get("exact")]
    broken = [t for t in shared if a[t].get("exact") and not b[t].get("exact")]
    same = [t for t in shared if a[t].get("exact") == b[t].get("exact")]

    def total(rows, key):
        return sum(rows[t].get(key) or 0 for t in shared)

    report = {
        "before": args.before.name, "after": args.after.name,
        "compared": len(shared),
        "only_in_before": sorted(set(a) - set(b)),
        "only_in_after": sorted(set(b) - set(a)),
        "exact_before": sum(bool(a[t].get("exact")) for t in shared),
        "exact_after": sum(bool(b[t].get("exact")) for t in shared),
        "fixed": fixed, "broken": broken, "unchanged": len(same),
        "tokens_before": total(a, "total_tokens"),
        "tokens_after": total(b, "total_tokens"),
        "reads_before": total(a, "reads"), "reads_after": total(b, "reads"),
        "turns_before": total(a, "iterations"), "turns_after": total(b, "iterations"),
    }
    if args.json:
        print(json.dumps(report))
        return

    print(f"\n  {args.before.name}\n  {args.after.name}\n")
    print(f"  compared        {report['compared']} task(s)")
    print(f"  exact           {report['exact_before']} -> "
          f"{report['exact_after']}  "
          f"({report['exact_after'] - report['exact_before']:+d})")
    print(f"  fixed           {len(fixed)}")
    print(f"  broken          {len(broken)}")
    print(f"  unchanged       {report['unchanged']}")
    for label, key in (("turns", "turns"), ("reads", "reads"),
                       ("tokens", "tokens")):
        before, after = report[f"{key}_before"], report[f"{key}_after"]
        delta = f"{after - before:+,}" if before or after else "0"
        print(f"  {label:<15} {before:,} -> {after:,}  ({delta})")
    for name, tasks in (("FIXED", fixed), ("BROKEN", broken)):
        if not tasks:
            continue
        print(f"\n  {name}:")
        for t in tasks:
            was = a[t].get("wrong_fields") or []
            now = b[t].get("wrong_fields") or []
            detail = f"was {was}" if name == "FIXED" else f"now {now}"
            print(f"    {t[:52]:<54} {detail}")
    for side, key in (("before", "only_in_before"), ("after", "only_in_after")):
        if report[key]:
            print(f"\n  only in {side}: {', '.join(report[key][:6])}")


if __name__ == "__main__":
    main()
