"""grounding.py — did the trajectory read the text it quotes?

Prerequisites: run directories holding read_log.json and answer.md, and
    the graph directory. No model and no GPU are needed.
Outputs:       a table on stdout; with --json, one object per run.

Contract: reads no task file and no gold. A quotation is checked against the
    graph and against the trajectory's own calls, so a question nobody has
    seen is judged by the same rule.

Why: coverage catches "swept a fifth of the set and answered anyway". It does
    not catch "quoted a description it never fetched". The reasoning-quality
    filter asks for genuine graph navigation rather than shortcuts, and a
    correct-looking quotation produced from parametric memory is the shortcut
    that filter exists to remove.

Four verdicts:
    GROUNDED      the text is stored, and the trajectory read it
    BLIND         the text is stored, but the trajectory never read it
    NOT_VERBATIM  the text equals no stored description of the entity: the
                  stored text inside added prose, changed punctuation, or
                  text that is not in the graph. Read the flagged examples.
    UNCHECKED     no entity or no language to attribute it to

BLIND is the interesting one. It is not a wrong answer; it is a right answer
    the trajectory cannot account for, which is worse in a corpus because it
    teaches the model to answer from priors and looks correct while doing it.

Usage:
    python scripts/bench/grounding.py results/runs/<batch>/*/
    python scripts/bench/grounding.py --json results/runs/<batch>/*/
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from canonical import InvalidAnswer, parse_answer  # noqa: E402

QID = re.compile(r"^Q\d+$")

# Calls that put DESCRIPTION text in front of the model, and the languages
# each one exposes. `edges` and `name` show labels only, so they cannot
# ground a description quotation.
ALL_LANGS = ("en", "fr", "de", "zh", "ar", "ru")
TEXT_CALLS = {
    "descriptions": ALL_LANGS,
    "describe": ALL_LANGS,
    "description": None,        # one language, taken from the call arguments
    "entity": ("en",),          # label + English description
    "search_entity": ("en",),   # each hit carries a description
}
TEXT_KEYS = ("evidence", "text", "description", "subject_evidence")
LANG_KEYS = ("language", "lang", "evidence_language", "subject_language")


def seen_text(read_log: list[dict]) -> dict[str, set]:
    """Entity -> the languages whose description the trajectory actually saw."""
    seen: dict[str, set] = collections.defaultdict(set)
    for entry in read_log:
        langs = TEXT_CALLS.get(entry.get("fn"))
        if langs is None and entry.get("fn") != "description":
            continue
        if entry.get("fn") == "description":
            args = entry.get("args") or []
            langs = (str(args[1]),) if len(args) > 1 else ALL_LANGS
        for ident in entry.get("ids") or []:
            if QID.match(str(ident)):
                seen[str(ident)].update(langs)
    return seen


def claims(node, scope=None):
    """Every (qid, lang, text) the answer asserts, from a generic walk."""
    scope = scope or {}
    if isinstance(node, dict):
        here = dict(scope)
        for key in ("qid", "id", "entity", "item"):
            value = node.get(key)
            if isinstance(value, str) and QID.match(value):
                here["qid"] = value
                break
        else:
            # The single-entity families nest the subject one level down --
            # {"entity": {"qid":..}, "evidence": ".."} -- so the quotation sits
            # beside the wrapper, not beside the id. Inherit only when exactly
            # ONE nested dict carries a qid: two would be ambiguous, and
            # guessing between them would attribute a quotation to the wrong
            # entity, which is worse than declining to check it.
            nested = [v.get("qid") for v in node.values()
                      if isinstance(v, dict) and isinstance(v.get("qid"), str)
                      and QID.match(v["qid"])]
            if len(nested) == 1:
                here["qid"] = nested[0]
        for key in LANG_KEYS:
            value = node.get(key)
            if isinstance(value, str) and 2 <= len(value) <= 5:
                here["lang"] = value
                break
        for key in TEXT_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                yield here.get("qid"), here.get("lang"), value.strip()
        for value in node.values():
            yield from claims(value, here)
    elif isinstance(node, list):
        for value in node:
            yield from claims(value, scope)


def verdict(qid, lang, text, seen, env) -> str:
    if not qid:
        return "UNCHECKED"
    try:
        stored = env.descriptions(qid) or {}
    except KeyError:
        return "UNCHECKED"
    matches = [code for code, value in stored.items() if str(value) == text]
    if not matches:
        return "NOT_VERBATIM"
    read = seen.get(qid, set())
    return "GROUNDED" if (read & set(matches)) else "BLIND"


def measure(run_dir: Path, env) -> dict | None:
    log_path, answer_path = run_dir / "read_log.json", run_dir / "answer.md"
    if not (log_path.exists() and answer_path.exists()):
        return None
    meta = json.loads((run_dir / "metadata.json").read_text())
    try:
        answer = parse_answer(answer_path.read_text())
    except InvalidAnswer:
        return {"run": run_dir.name, "task": meta.get("task"),
                "counts": {}, "blind": [], "not_verbatim": []}
    seen = seen_text(json.loads(log_path.read_text()))
    counts, blind, not_verbatim = collections.Counter(), [], []
    for qid, lang, text in claims(answer):
        v = verdict(qid, lang, text, seen, env)
        counts[v] += 1
        if v == "BLIND":
            blind.append(f"{qid}:{text[:40]}")
        elif v == "NOT_VERBATIM":
            not_verbatim.append(f"{qid}:{text[:40]}")
    return {"run": run_dir.name, "task": meta.get("task"),
            "counts": dict(counts), "blind": blind[:6],
            "not_verbatim": not_verbatim[:6]}


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

    total = collections.Counter()
    print(f"\n{'task':<44}{'GROUND':>8}{'BLIND':>7}{'NOTVERB':>8}{'UNCHK':>7}")
    for row in rows:
        c = row["counts"]
        total.update(c)
        if not c:
            continue
        print(f"{str(row['task'])[:42]:<44}{c.get('GROUNDED',0):>8}"
              f"{c.get('BLIND',0):>7}{c.get('NOT_VERBATIM',0):>8}"
              f"{c.get('UNCHECKED',0):>7}")
    print(f"\n{'TOTAL':<44}{total.get('GROUNDED',0):>8}{total.get('BLIND',0):>7}"
          f"{total.get('NOT_VERBATIM',0):>8}{total.get('UNCHECKED',0):>7}")
    for name in ("BLIND", "NOT_VERBATIM"):
        hits = [(r["task"], x) for r in rows for x in r[name.lower()]]
        if hits:
            print(f"\n{name}:")
            for task, x in hits[:10]:
                print(f"   {task[:40]:<42} {x}")


if __name__ == "__main__":
    main()
