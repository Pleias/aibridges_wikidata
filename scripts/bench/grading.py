"""grading.py — graded, per-field scoring beside the exact verdict.

Prerequisites: none.
Outputs:       none; pure functions used by score.py.

Contract: `grade()` NEVER decides pass or fail. `score.py` keeps computing
    `exact` exactly as before, and this module only adds detail beside it. No
    past verdict changes, so archived batches stay comparable.

Why: the exact verdict cannot distinguish a run that found 3 of 12 entities
    from one that found 11, and so cannot say whether a failure is a broken
    sweep or a long tail. This reports which identifiers were missed, not
    merely that some were.

Design follows the measured shape of the 56 answer schemas: 107 scalar
    fields, 43 nested dicts and 32 lists, of which 31 carry a natural row
    identity (24 `qid`, 3 `language`, 4 plain scalars). Rows are therefore
    matched BY IDENTITY, not by optimal assignment: with a QID in hand,
    precision/recall names the missing entity, which an alignment score
    cannot. The one identity-less field falls back to multiset overlap, so
    no assignment solver — and no new dependency — is needed.
"""

from __future__ import annotations

import json
from collections import Counter

from canonical import canonical

# Row keys that can identify a row, in preference order. Every one of them
# appears as an identity in the corpus; the list is not speculative.
IDENTITY_KEYS = ("qid", "id", "entity", "item", "language", "lang")

MISSING = object()


def _key(value) -> str:
    return json.dumps(canonical(value), sort_keys=True, ensure_ascii=False)


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return round(2 * precision * recall / (precision + recall), 4)


def row_identity(rows: list) -> str | None:
    """The key that identifies these rows, or None to fall back to overlap.

    A key qualifies only when every row carries it AND its values are unique;
    a repeated identity would silently merge two distinct rows.
    """
    if not rows or not all(isinstance(r, dict) for r in rows):
        return None
    shared = set.intersection(*(set(r) for r in rows))
    for name in IDENTITY_KEYS:
        if name not in shared:
            continue
        values = [_key(r[name]) for r in rows]
        if len(set(values)) == len(values):
            return name
    return None


def _overlap(pred: list, target: list) -> dict:
    """Multiset overlap — for rows with no identity to match on."""
    p, t = Counter(_key(x) for x in pred), Counter(_key(x) for x in target)
    tp, fp, fn = (p & t).total(), (p - t).total(), (t - p).total()
    return {"kind": "set", "identity": "positional", "f1": _f1(tp, fp, fn),
            "precision": round(tp / len(pred), 4) if pred else 0.0,
            "recall": round(tp / len(target), 4) if target else 0.0,
            "missing": fn, "extra": fp}


def _keyed(pred: list, target: list, name: str) -> dict:
    """Identity-keyed precision and recall, plus a detail check on matches."""
    def index(rows):
        out = {}
        for row in rows:
            if isinstance(row, dict) and name in row:
                out.setdefault(_key(row[name]), row)
        return out

    p, t = index(pred), index(target)
    matched = sorted(set(p) & set(t))
    missing = sorted(set(t) - set(p))
    extra = sorted(set(p) - set(t))
    # Rows the model emitted that carry no identity at all are false positives.
    tp, fp, fn = len(matched), len(extra) + (len(pred) - len(p)), len(missing)

    detail = sum(
        1 for k in matched
        if canonical({a: b for a, b in p[k].items() if a != name})
        == canonical({a: b for a, b in t[k].items() if a != name}))

    def plain(keys):
        out = []
        for k in keys[:12]:
            try:
                out.append(json.loads(k))
            except json.JSONDecodeError:
                out.append(k)
        return out

    return {"kind": "set", "identity": name, "f1": _f1(tp, fp, fn),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "missing": plain(missing), "extra": plain(extra),
            "detail_ok": f"{detail}/{len(matched)}"}


def _grade_list(expected: list, got) -> dict:
    if got is MISSING or not isinstance(got, list):
        return {"kind": "set", "identity": "n/a", "f1": 0.0,
                "precision": 0.0, "recall": 0.0,
                "missing": "<field absent or not a list>", "extra": []}
    name = row_identity(expected)
    return _keyed(got, expected, name) if name else _overlap(got, expected)


def _grade_dict(expected: dict, got) -> dict:
    if got is MISSING or not isinstance(got, dict):
        return {"kind": "group", "ok": False, "wrong": ["<field absent>"]}
    wrong = sorted(k for k in set(expected) | set(got)
                   if canonical(expected.get(k)) != canonical(got.get(k)))
    return {"kind": "group", "ok": not wrong, "wrong": wrong}


def grade(expected, got, bookkeeping=()) -> dict:
    """Per-field detail beside the exact verdict. Never a pass/fail decision.

    `bookkeeping` names the fields that count the CANDIDATE SET the search
    examined rather than the answer. It comes from the task file, so the
    harness holds no field names of its own and a generated task declares
    its own. Those fields still get scored and reported; they are excluded
    only from `content_exact`.

    Why the caller must declare it: it cannot be derived. A gold integer that
    matches no list length looks like bookkeeping, but so do `maximum`,
    `species_count` and `start_year`, which are facts about the world.
    """
    if not isinstance(expected, dict) or not isinstance(got, dict):
        ok = canonical(expected) == canonical(got)
        return {"set_f1": 1.0 if ok else 0.0,
                "scalars_ok": "1/1" if ok else "0/1",
                "fields": {"<whole answer>": {"kind": "scalar", "ok": ok}}}
    fields, f1s, scalars = {}, [], []
    for name, want in expected.items():
        have = got.get(name, MISSING)
        if isinstance(want, list):
            fields[name] = _grade_list(want, have)
            f1s.append(fields[name]["f1"])
        elif isinstance(want, dict):
            fields[name] = _grade_dict(want, have)
            scalars.append(fields[name]["ok"])
        else:
            ok = have is not MISSING and canonical(want) == canonical(have)
            fields[name] = {"kind": "scalar", "ok": ok}
            if not ok:
                fields[name]["expected"] = want
                fields[name]["got"] = None if have is MISSING else have
            scalars.append(ok)
    for name in set(got) - set(expected):
        fields[name] = {"kind": "unexpected", "ok": False}
    # Everything the ANSWER claims, ignoring what the search claims about
    # itself. A trajectory can group every entity correctly and still write a
    # wrong number into a bookkeeping field such as a candidate count; exact
    # scoring cannot tell that answer from one that swept nothing.
    content = [
        (v.get("ok") if v["kind"] in ("scalar", "group", "unexpected")
         else v.get("f1") == 1.0)
        for name, v in fields.items() if name not in set(bookkeeping)]
    return {
        "set_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "scalars_ok": f"{sum(scalars)}/{len(scalars)}" if scalars else "0/0",
        "content_exact": all(content) if content else None,
        "bookkeeping": sorted(bookkeeping),
        "fields": fields,
    }
