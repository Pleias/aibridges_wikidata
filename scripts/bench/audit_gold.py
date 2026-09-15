"""audit_gold.py — check every gold value that the graph can settle.

Prerequisites: tasks/ and data/graph/. No model, no GPU: the graph
    environment is memory-mapped Arrow plus numpy, so this runs on any CPU
    machine that can reach the graph directory.
Outputs:       a table on stdout, and results/gold_audit.json.

Contract: a gold is REPORTED, never repaired. Some disagreements are gold
    defects and some are deliberate conventions; the script cannot tell which,
    and guessing would quietly rewrite reference answers.

Why: an answer can recover every entity and still differ from the gold on a
    surface form -- `"United States"`, which is Q30's stored English label,
    against a gold that says "American". Neither form is wrong unless the task
    says which it wants. This audit shows which conventions the golds follow,
    and which gold values the graph confirms.

Usage:
    python scripts/bench/audit_gold.py
    python scripts/bench/audit_gold.py --json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

NAME_KEYS = ("name",)
TEXT_KEYS = ("evidence", "text", "description")
LANG_KEYS = ("language", "lang")
COUNT_KEYS = ("count",)


def rows(node, inherited: dict | None = None):
    """Every dict in an answer tree, with the nearest enclosing qid in scope.

    A qid can sit beside the fields it describes, or one level up — the
    cross-lingual families nest evidence under the entity it belongs to.
    """
    inherited = inherited or {}
    if isinstance(node, dict):
        scope = dict(inherited)
        for key in ("qid", "id", "entity", "item"):
            value = node.get(key)
            if isinstance(value, str) and value[:1].upper() == "Q":
                scope["qid"] = value
                break
        yield node, scope
        for value in node.values():
            yield from rows(value, scope)
    elif isinstance(node, list):
        for value in node:
            yield from rows(value, inherited)


def audit_task(task: dict, env) -> list[dict]:
    found = []

    def note(kind, **kw):
        found.append({"task": task["id"], "check": kind, **kw})

    expected = task["answer"].get("expected")
    if not isinstance(expected, (dict, list)):
        return found

    seen_qids = set()
    for row, scope in rows(expected):
        qid = scope.get("qid")
        if not qid:
            continue

        if qid not in seen_qids:
            seen_qids.add(qid)
            if env.name(qid) is None:
                note("qid_absent", qid=qid)
                continue

        # A. does `name` match the stored English label?
        for key in NAME_KEYS:
            if key in row and isinstance(row[key], str):
                stored = env.name(qid)
                if stored is not None and row[key] != stored:
                    note("name_vs_label_en", qid=qid, gold=row[key],
                         label_en=stored)

        # B. does `evidence` occur verbatim among the stored descriptions?
        lang = next((row[k] for k in LANG_KEYS
                     if isinstance(row.get(k), str)), None)
        for key in TEXT_KEYS:
            text = row.get(key)
            if not isinstance(text, str) or not text:
                continue
            stored = env.descriptions(qid) or {}
            if lang and lang in stored:
                if stored[lang] != text:
                    note("evidence_not_stored", qid=qid, lang=lang,
                         gold=text[:70], stored=str(stored[lang])[:70])
            elif lang:
                note("evidence_lang_absent", qid=qid, lang=lang,
                     available=sorted(stored))
            elif text not in stored.values():
                note("evidence_unattributed", qid=qid, gold=text[:70])

        # C. is the language code one the environment actually carries?
        for key in LANG_KEYS:
            code = row.get(key)
            if isinstance(code, str) and code not in (env.descriptions(qid) or {}):
                note("lang_not_available", qid=qid, lang=code,
                     available=sorted(env.descriptions(qid) or {}))

        # E. can the graph produce this string at all? A field whose gold
        # exists nowhere in the graph cannot be recomputed or checked — it is
        # a CONVENTION, and the answer format has to state it or the model
        # must guess. This is the list the spec pass needs.
        for key, value in row.items():
            if key in ("qid", "id") or not isinstance(value, str) or not value:
                continue
            if len(value) > 120:
                continue
            stored = set()
            label = env.name(qid)
            if label:
                stored.add(label)
            stored.update(str(v) for v in (env.labels(qid) or {}).values())
            stored.update(str(v) for v in (env.descriptions(qid) or {}).values())
            if value in stored:
                continue
            if key in LANG_KEYS and len(value) <= 5:
                continue                       # a language code, checked above
            note("not_in_graph", qid=qid, field=key, gold=value[:60])

    # D. does a count field agree with the list it counts?
    if isinstance(expected, dict):
        lists = {k: v for k, v in expected.items() if isinstance(v, list)}
        for key in COUNT_KEYS:
            value = expected.get(key)
            if isinstance(value, int) and len(lists) == 1:
                name, items = next(iter(lists.items()))
                if value != len(items):
                    note("count_mismatch", field=key, gold=value,
                         list_field=name, list_len=len(items))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from wd_graph_env import WDGraphEnv
    env = WDGraphEnv(read_budget=None)

    findings, tasks = [], 0
    for path in sorted((ROOT / "tasks").glob("*/*.json")):
        task = json.loads(path.read_text())
        tasks += 1
        findings.extend(audit_task(task, env))

    out = ROOT / "results" / "gold_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False))

    if args.json:
        print(json.dumps(findings, ensure_ascii=False))
        return

    kinds = collections.Counter(f["check"] for f in findings)
    print(f"\n{tasks} tasks audited, {len(findings)} findings "
          f"across {len({f['task'] for f in findings})} tasks\n")
    for kind, n in kinds.most_common():
        print(f"  {kind:<24} {n}")
    for kind, _ in kinds.most_common():
        print(f"\n--- {kind} ---")
        for f in [x for x in findings if x["check"] == kind][:8]:
            rest = {k: v for k, v in f.items() if k not in ("task", "check")}
            print(f"  {f['task'][:44]:<46} {rest}")
        extra = kinds[kind] - 8
        if extra > 0:
            print(f"  ... and {extra} more")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
