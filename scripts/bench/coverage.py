"""coverage.py — did the trajectory sweep the set, measured from its own reads.

Prerequisites: run directories holding read_log.json, and the graph
    directory. No model and no GPU are needed.
Outputs:       a table on stdout; with --json, one object per run.

Contract: nothing here reads the task file. The anchor and predicate come
    from the trajectory's own calls, and the true size comes from the graph.
    A question the author has never seen scores exactly the same way.

Why: `pool_size` and `reviewed` ask the model to self-report how much it
    examined, and their meaning depends on parsing the question's prose
    scope. When a question folds a filter into its opening clause, the field
    becomes ambiguous, and that does not scale to large generated question
    sets.

    Enumeration is measurable without asking. The read log already records
    every call and every id it returned, so completeness is a comparison
    rather than a claim: the model decides HOW to explore, the scorer decides
    WHETHER it explored.

Usage:
    python scripts/bench/coverage.py results/runs/<batch>/<run> ...
    python scripts/bench/coverage.py --json results/runs/<batch>/*/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

QID = re.compile(r"^Q\d+$")


def sweeps(read_log: list[dict]) -> dict[tuple, set]:
    """Every (anchor, direction, predicate) the trajectory expanded, and the
    entities it actually obtained from that slice.

    Pages are unioned: edges(..., offset=K) is the same slice as offset=0, so
    a model that pages correctly must not be scored as three partial sweeps.
    """
    out: dict[tuple, set] = {}
    for entry in read_log:
        if entry.get("fn") != "edges":
            continue
        args = entry.get("args") or []
        if not args:
            continue
        anchor = str(args[0])
        direction = str(args[1]) if len(args) > 1 else "both"
        pid = str(args[2]) if len(args) > 2 else "None"
        key = (anchor, direction, pid)
        got = out.setdefault(key, set())
        for ident in entry.get("ids") or []:
            if QID.match(str(ident)) and str(ident) != anchor:
                got.add(str(ident))
    return out


def measure(run_dir: Path, env) -> dict | None:
    log_path = run_dir / "read_log.json"
    if not log_path.exists():
        return None
    meta = json.loads((run_dir / "metadata.json").read_text())
    found = []
    for (anchor, direction, pid), got in sweeps(json.loads(log_path.read_text())).items():
        try:
            true = env.count_edges(
                anchor, direction=("in" if direction == "in_" else direction),
                pid=None if pid in ("None", "") else pid)
        except (KeyError, ValueError):
            continue
        total = true["total"] if direction == "both" else \
            true["in_"] if direction == "in" else true["out"]
        if total <= 0:
            continue
        found.append({"anchor": anchor, "direction": direction, "pid": pid,
                      "swept": len(got), "true": total,
                      "coverage": round(len(got) / total, 4)})
    if not found:
        return {"run": run_dir.name, "task": meta.get("task"), "sweeps": [],
                "widest": None, "coverage": None}
    # The widest slice a trajectory touched is its candidate pool in every
    # case observed. Reported explicitly rather than averaged, because a
    # hundred one-edge lookups would otherwise drown one incomplete sweep.
    widest = max(found, key=lambda f: f["true"])
    return {"run": run_dir.name, "task": meta.get("task"),
            "sweeps": sorted(found, key=lambda f: -f["true"])[:6],
            "widest": widest, "coverage": widest["coverage"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from wd_graph_env import WDGraphEnv
    env = WDGraphEnv(read_budget=None)

    rows = [r for r in (measure(d, env) for d in args.run_dirs if d.is_dir()) if r]
    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return

    print(f"\n{'task':<44}{'widest slice':<26}{'swept':>7}{'true':>8}{'cov':>8}")
    for row in rows:
        w = row["widest"]
        if not w:
            print(f"{str(row['task'])[:42]:<44}{'(no expansion)':<26}")
            continue
        slice_name = f"{w['anchor']} {w['direction']} {w['pid']}"
        print(f"{str(row['task'])[:42]:<44}{slice_name[:24]:<26}"
              f"{w['swept']:>7}{w['true']:>8}{w['coverage']:>8.2f}")


if __name__ == "__main__":
    main()
