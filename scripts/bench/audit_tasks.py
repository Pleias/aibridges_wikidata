"""audit_tasks.py — report which task files are valid and have archived runs.

Prerequisites: tasks/ and any persisted results under results/.
Outputs:       a table on stdout, and results/task_audit.json.

Contract: a task is reported, never repaired. It must satisfy
    rlm_loop.load_task and name only functions exposed by the harness.

Why: task files are written by generators and by hand, and the harness
    interface can change. Which files are still usable is not something to
    remember; it is something to check.

Usage:
    uv run python scripts/bench/audit_tasks.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import rlm_loop  # noqa: E402


def scores_by_task() -> dict[str, list[dict]]:
    """Every persisted run, newest batch last, keyed by task id.

    Two layouts: bench writes results/<batch>_scores.json, se writes
    scores.json inside its batch directory. The generic RLM runner instead
    writes one metadata.json per run. Reading only score tables made manually
    reviewed card tasks look unrun even though their traces were archived.
    """
    out: dict[str, list[dict]] = collections.defaultdict(list)
    files = [(p, p.name[:-len("_scores.json")])
             for p in sorted((ROOT / "results").glob("*_scores.json"))]
    files += [(p, p.parent.name)
              for p in sorted((ROOT / "results" / "runs").rglob("scores.json"))]
    for path, batch in sorted(files, key=lambda pair: pair[1]):
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("task"):
                out[row["task"]].append({**row, "batch": batch})

    known_run_ids = {
        row.get("run_id")
        for rows in out.values()
        for row in rows
        if row.get("run_id")
    }
    for path in sorted((ROOT / "results" / "runs").rglob("metadata.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task = row.get("task")
        if not task or row.get("run_id") in known_run_ids:
            continue
        out[task].append({
            **row,
            "batch": path.parents[1].name,
            # Raw archives do not contain a scorer verdict. Do not infer one
            # from a syntactically valid FINAL answer.
            "exact": None,
        })

    for rows in out.values():
        rows.sort(key=lambda row: (row.get("batch", ""), row.get("run_id", "")))
    return out


def check(path: Path) -> dict:
    """loads / tools, without touching the graph."""
    record = {"file": str(path.relative_to(ROOT)), "problems": []}
    try:
        task = rlm_loop.load_task(path)
    except Exception as error:                     # noqa: BLE001 - reporting
        record["problems"].append(f"does not load: {error}")
        record["id"] = path.stem
        return record
    record["id"] = task["id"]
    record["family"] = path.parent.name
    record["type"] = task["taxonomy"]["type"]
    record["turns"] = task["limits"]["turns"]
    record["grounded_code"] = bool(task.get("grounded_code"))
    record["min_reads"] = task.get("metrics", {}).get("minimum_reads")
    record["certificate"] = "certificate" in task
    unknown = sorted(set(task["allowed_functions"]) - set(rlm_loop.FUNCTION_DOCS))
    if unknown:
        record["problems"].append(f"unknown functions: {', '.join(unknown)}")
    return record


def main() -> None:
    scores = scores_by_task()
    # manifest.json and similar bookkeeping files are not tasks
    rows = [check(p) for p in sorted((ROOT / "tasks").rglob("*.json"))
            if p.name not in {"manifest.json", "index.json"}]
    for row in rows:
        runs = scores.get(row["id"], [])
        row["runs"] = len(runs)
        row["last_batch"] = runs[-1]["batch"] if runs else None
        scored = [run for run in runs if isinstance(run.get("exact"), bool)]
        row["last_exact"] = scored[-1]["exact"] if scored else None
        if not runs:
            row["problems"].append("never run")

    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_family[row.get("family", "?")].append(row)

    print(f"{'family':14} {'tasks':>5} {'valid':>6} {'run':>5} {'exact':>6} "
          f"{'cert':>5} {'gated':>6}")
    for family in sorted(by_family):
        group = by_family[family]
        valid = sum(not r["problems"] or r["problems"] == ["never run"]
                    for r in group)
        run = sum(r["runs"] > 0 for r in group)
        exact = sum(r["last_exact"] is True for r in group)
        print(f"{family:14} {len(group):5} {valid:6} {run:5} {exact:6} "
              f"{sum(r.get('certificate', False) for r in group):5} "
              f"{sum(r.get('grounded_code', False) for r in group):6}")

    broken = [r for r in rows if any("never run" not in p for p in r["problems"])]
    if broken:
        print("\nproblems")
        for row in broken:
            print(f"  {row['id'][:52]:54} {'; '.join(row['problems'])[:70]}")
    never = [r for r in rows if r["runs"] == 0]
    print(f"\nnever run: {len(never)} of {len(rows)}")
    for row in never[:12]:
        print(f"  {row['file']}")
    if len(never) > 12:
        print(f"  … and {len(never) - 12} more")

    out = ROOT / "results" / "task_audit.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
