#!/usr/bin/env python3
"""
wd_graph_env.py — full-graph navigation environment over data/graph/
(nodes.parquet + edges_{fwd,bwd}.arrow, built by build_graph.py).

The graph location is data/graph/ in this repository, or the directory named by
the WIKIDATA_GRAPH_DIR environment variable.

World: nodes are the 48M en-labeled
Q-entities of the Feb-2026 formatted dump; edges are deduplicated mainsnak
Q->Q triples between nodes. Ranks are invisible in this dump, so historical/
deprecated relations are traversable — this environment teaches navigation,
not current-facts. Banned nodes/pids simply do not appear in output: the env
IS the world-definition.

Harness interface: repl_namespace(allowed=), read_log, seen_ids(), save_log(),
reads_left(), BudgetExhausted.

Scale conventions (deliberate, documented for the prompt):
  - edges() returns a dict with degree counts and per-direction lists, capped
    at `limit` per direction (hubs here have 100k+ incoming edges; returning
    them all would blow the stdout budget instantly). Truncation is explicit.
  - in_slice flags are gone: every returned id is expandable by construction.

Memory: edge arrays are opened memory-mapped; lookups touch O(log n) pages.
Node labels stay in one Arrow table (compact strings), not Python dicts.
"""

import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LANGS = ("en", "fr", "de", "zh", "ar", "ru")

# Explain an empty search instead of returning silence. Off by default in
# the code and on in the reference configuration (config/runs/reference.env).
SEARCH_HINTS = os.environ.get("RLM_SEARCH_HINTS", "0") == "1"


class BudgetExhausted(RuntimeError):
    """Raised when a run exceeds its read budget."""


REPO = pathlib.Path(__file__).resolve().parent.parent
# The graph is large and often lives outside the checkout. WIKIDATA_GRAPH_DIR
# points at it without a symlink; the repository default is data/graph/.
DEFAULT_DIR = pathlib.Path(os.environ.get("WIKIDATA_GRAPH_DIR")
                           or REPO / "data" / "graph")


def _qint(id_):
    if isinstance(id_, (int, np.integer)):
        return int(id_)
    if isinstance(id_, str) and id_[:1] == "Q" and id_[1:].isdigit():
        return int(id_[1:])
    raise KeyError(
        f"{id_!r} is not a QID. Known ids only — do not guess QIDs; use the "
        f"ids returned by edges().")


class Struct(dict):
    """A fixed-schema result that refuses invented keys, even through .get().

    The most expensive failure mode measured on this environment is not a wrong
    answer, it is a silent one: a model guesses a key name, writes
    ``result.get("out_count", 0)`` as a defensive reflex, receives the default,
    and every downstream count collapses to zero without anything being raised
    or logged. Three separate guesses did this — ``out_count`` for ``deg_out``,
    a top-level ``sitelinks`` for the per-language ``sitelink`` — and each
    produced a well-formed, internally consistent, entirely wrong answer.

    A KeyError the model can read and repair is strictly better than a plausible
    default it cannot detect. Only dictionaries with a CLOSED schema are wrapped:
    a mapping keyed by language stays an ordinary dict, because a missing
    language is real absence rather than a typo.
    """

    def __missing__(self, key):
        raise KeyError(
            f"{key!r} is not a key of this result. The keys are "
            f"{sorted(self)}. Use one of those — do not guess key names.")

    def get(self, key, default=None):
        if key not in self:
            raise KeyError(
                f"{key!r} is not a key of this result, so .get({key!r}, "
                f"{default!r}) would silently return {default!r} and corrupt "
                f"everything computed from it. The keys are {sorted(self)}.")
        return self[key]


class WDGraphEnv:
    def __init__(self, dir_path=DEFAULT_DIR, read_budget=None,
                 nav_banned_nodes=None, nav_banned_pids=None,
                 nav_policy=None, edge_limit=None):
        d = pathlib.Path(dir_path)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        # projection is load-bearing: nodes.parquet now carries the multilingual
        # entity layer (28 columns, ~12 GB in memory), so only ask for the four
        # navigation columns. Other languages are read on demand via labels().
        nodes = pq.read_table(
            d / "nodes.parquet",
            columns=["qid", "label_en", "deg_out", "deg_in"])
        self._qid = nodes.column("qid").to_numpy()          # sorted
        self._label = nodes.column("label_en")              # arrow, aligned
        self._deg_out = nodes.column("deg_out").to_numpy()
        self._deg_in = nodes.column("deg_in").to_numpy()
        edges = lambda n: pa.ipc.open_file(  # noqa: E731
            pa.memory_map(str(d / n), "rb")).read_all()
        fwd, bwd = edges("edges_fwd.arrow"), edges("edges_bwd.arrow")
        self._f_src, self._f_prop, self._f_dst = (
            fwd.column("src").to_numpy(), fwd.column("prop").to_numpy(),
            fwd.column("dst").to_numpy())
        self._b_src, self._b_prop, self._b_dst = (
            bwd.column("src").to_numpy(), bwd.column("prop").to_numpy(),
            bwd.column("dst").to_numpy())

        # properties.json maps a PID to a per-language label map; the chunk
        # fallback maps it straight to an English string
        self._plabels = {}
        self._plabels_all = {}          # pid -> {lang: label}, for search_property
        for cand in (d / "properties.json", REPO / "data" / "chunk_0.labels.json"):
            if cand.exists():
                raw = {k: v for k, v in json.loads(cand.read_text()).items()
                       if k[:1] == "P"}
                self._plabels_all = {
                    k: (v if isinstance(v, dict) else {"en": v})
                    for k, v in raw.items()}
                self._plabels = {k: v.get("en")
                                 for k, v in self._plabels_all.items()}
                break

        self._nav_banned = {_qint(q) for q in (nav_banned_nodes or ())}
        self._nav_banned_pids = {int(str(p).lstrip("P"))
                                 for p in (nav_banned_pids or ())}
        if nav_policy:
            self._nav_banned |= self.compute_banned(nav_policy)
            self._nav_banned_pids |= {int(str(p).lstrip("P"))
                                      for p in nav_policy.get("banned_pids", ())}
        # world-consistency invariant: the
        # per-direction listing cap must be >= the policy's max_degree, or
        # nodes in the (cap, max_degree] band are world-legal but not fully
        # enumerable — generator BFS and env would disagree about visibility.
        # Every legal node fits in one edges() call; stdout safety is the
        # harness's 10k print-truncation, not this cap.
        if edge_limit is None:
            edge_limit = (nav_policy or {}).get("max_degree") or 200
        self.edge_limit = edge_limit
        self.read_budget = read_budget
        self._spent = 0
        self._seen = set()
        self._log = []
        self._ranker = None       # lazy embedding ranker (rank())
        self._expanded_full = set()  # qids fully expanded (both, no pid)
        self._eindex = None          # lazy: entity part lower bounds
        self._eparts = None          # lazy: entity part descriptors
        self._records = {}           # qid -> parsed entity record
        self._search = None          # lazy: (index, searcher, fields)

    # ---------- world definition ----------

    def instances_of(self, class_q):
        """All nodes with a P31 edge to class_q (internal/world tooling —
        NOT exposed to the REPL). Uses the bwd index slice of class_q."""
        q = _qint(class_q)
        lo, hi = self._slice(self._b_dst, q)
        m = np.asarray(self._b_prop[lo:hi]) == 31
        return np.asarray(self._b_src[lo:hi])[m]

    def compute_banned(self, policy):
        """THE shared world-definition (T8b lesson): generator BFS, env edges()
        and the validator must all call exactly this. Policy dict:
          banned_types: {class_qid: note} — instances are banned nodes
          max_degree:   int — nodes with deg_out+deg_in above it are banned
          (banned_pids handled separately: they filter edges, not nodes)
        Returns a set of int QIDs."""
        banned = set()
        for cls in policy.get("banned_types", ()):
            banned.update(int(x) for x in self.instances_of(cls))
        md = policy.get("max_degree")
        if md:
            hot = self._qid[(self._deg_out.astype(np.int64)
                             + self._deg_in) > int(md)]
            banned.update(int(x) for x in hot)
        return banned

    # ---------- bookkeeping (same semantics as WDEnv) ----------

    # `degree` is not here because it is free unconditionally: charging it made
    # a generator's optimal read count depend on whether a QID had been seen,
    # and one family was budgeted at 16 reads for what the environment bills
    # as 2.
    _FREE_ON_SEEN = {"name", "has", "rank"}

    def _rec(self, fn, args, ids, free=False):
        self._log.append({"fn": fn, "args": args, "ids": sorted(set(ids))})
        new = [i for i in ids if i not in self._seen]
        self._seen.update(ids)
        charged = not free and fn != "reads_left" and not (
            fn in self._FREE_ON_SEEN and not new)
        if charged:
            self._spent += 1
        if self.read_budget is not None and self._spent > self.read_budget:
            raise BudgetExhausted(
                f"READ BUDGET EXHAUSTED ({self.read_budget} charged calls). No "
                f"further environment reads possible — produce your best answer "
                f"from what you have already gathered (FINAL/FINAL_VAR).")

    def reads_left(self):
        if self.read_budget is None:
            return None
        return max(0, self.read_budget - self._spent)

    # ---------- lookups ----------

    def _pos(self, q):
        i = np.searchsorted(self._qid, q)
        if i < len(self._qid) and self._qid[i] == q:
            return int(i)
        return None

    def _name_of(self, q):
        i = self._pos(q)
        return self._label[i].as_py() if i is not None else None

    def _slice(self, key_arr, q):
        # dtype-matched scalar is load-bearing: a Python-int key against an
        # int32 mmap makes numpy cast-copy the ENTIRE array per call (~1s)
        k = np.asarray(q, dtype=key_arr.dtype)
        lo = key_arr.searchsorted(k, side="left")
        hi = key_arr.searchsorted(k, side="right")
        return int(lo), int(hi)

    # ---------- REPL-facing ----------

    def has(self, qid):
        """Cheap existence check: is this id a node of the graph? (1 read;
        free once seen)."""
        try:
            q = _qint(qid)
        except KeyError:
            self._rec("has", [qid], [])
            return False
        out = self._pos(q) is not None
        self._rec("has", [qid], [f"Q{q}"] if out else [])
        return out

    def name(self, qid):
        """English label of a node QID or a PID. Free for ids already seen.
        None if unknown."""
        s = str(qid)
        if s[:1] == "P" and s[1:].isdigit():
            out = self._plabels.get(s)
            self._rec("name", [s], [s] if out is not None else [])
            return out
        q = _qint(qid)
        out = self._name_of(q)
        self._rec("name", [s], [f"Q{q}"] if out is not None else [])
        return out

    def count_edges(self, qid, direction="both", pid=None):
        """How many edges match, WITHOUT returning any of them. 1 read.

        `degree()` reports a node's total degree across every predicate, which
        is the wrong number for planning a filtered sweep and misleads badly:
        Berlin has 83,417 incoming edges, but only 245 of them are
        `filming location`. A model told 83,417 has no reason to attempt a
        sweep it could finish in two pages.

        Charged, not free. A pool-size field is an anti-sampling check, and a
        free exact count would let a trajectory report the pool without ever
        enumerating it. Charging keeps the intent visible in the read log.
        """
        q = _qint(qid)
        if self._pos(q) is None:
            raise KeyError(f"Q{q} is not in the graph.")
        if q in self._nav_banned:
            raise KeyError(
                f"Q{q} is not part of the traversable world for this task "
                f"(excluded hub).")
        if direction == "in_":
            direction = "in"
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', 'in', or 'in_'")
        want_pid = int(str(pid).lstrip("P")) if pid is not None else None

        def tally(srcarr, proparr, otherarr, key_is_src):
            lo, hi = self._slice(srcarr if key_is_src else otherarr, q)
            n = 0
            for j in range(lo, hi):
                p = int(proparr[j])
                o = int((otherarr if key_is_src else srcarr)[j])
                if o == q or p in self._nav_banned_pids \
                        or o in self._nav_banned:
                    continue
                if want_pid is not None and p != want_pid:
                    continue
                n += 1
            return n

        out = tally(self._f_src, self._f_prop, self._f_dst, True) \
            if direction in ("both", "out") else 0
        inn = tally(self._b_src, self._b_prop, self._b_dst, False) \
            if direction in ("both", "in") else 0
        self._rec("count_edges", [f"Q{q}", direction, str(pid)], [f"Q{q}"])
        return {"out": out, "in_": inn, "total": out + inn}

    def degree(self, qid):
        """(deg_out, deg_in) of a node — how expensive expanding it would be.
        Always free: it is planning metadata, not content."""
        q = _qint(qid)
        i = self._pos(q)
        if i is None:
            raise KeyError(f"Q{q} is not in the graph.")
        if q in self._nav_banned:
            raise KeyError(
                f"Q{q} is not part of the traversable world for this task "
                f"(excluded hub).")
        self._rec("degree", [f"Q{q}"], [f"Q{q}"], free=True)
        return int(self._deg_out[i]), int(self._deg_in[i])

    def edges(self, qid, direction="both", pid=None, limit=None, offset=0):
        """Graph edges of a node, both directions in ONE call (1 read).
        Returns {id, label, deg_out, deg_in, out: [...], in_: [...],
        truncated}. Each entry: (pid, pqidlabel, other_qid, other_label).
        Directions are capped at `limit` (default env cap) — check deg_* and
        `truncated`; use pid= or direction= to narrow a hub instead of paging
        blindly. Banned world elements never appear."""
        # The returned mapping names its incoming list ``in_`` because ``in``
        # is a Python keyword. Models naturally copied that key into the
        # direction argument, which previously returned an empty result and
        # silently consumed a read. Accept the alias and reject true typos.
        direction = "in" if direction == "in_" else direction
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', 'in', or 'in_'")
        q = _qint(qid)
        i = self._pos(q)
        if i is None:
            raise KeyError(
                f"Q{q} is not in the graph. Known ids only — do not guess "
                f"QIDs; use ids returned by edges().")
        # Banned nodes are non-traversable from BOTH directions. They never
        # appear in edges() output, and a QID recalled from model memory must
        # not open them either: expanding an excluded hub such as Q20 (Norway,
        # 418k incoming edges) would flood the model's intermediate maps.
        if q in self._nav_banned:
            raise KeyError(
                f"Q{q} is not part of the traversable world for this task "
                f"(excluded hub). Use the ids edges() returns.")
        lim = int(limit) if limit else self.edge_limit
        off = max(0, int(offset or 0))
        want_pid = int(str(pid).lstrip("P")) if pid is not None else None
        touched = [f"Q{q}"]
        out_entries, in_entries = [], []
        truncated = {"out": False, "in": False}

        def collect(srcarr, proparr, otherarr, key_is_src, acc, dkey):
            lo, hi = self._slice(srcarr if key_is_src else otherarr, q)
            n_kept, n_seen = 0, 0
            for j in range(lo, hi):
                p = int(proparr[j])
                o = int((otherarr if key_is_src else srcarr)[j])
                if o == q or p in self._nav_banned_pids \
                        or o in self._nav_banned:
                    continue
                if want_pid is not None and p != want_pid:
                    continue
                # offset counts MATCHES, not raw array positions: skipping by
                # index would drift as soon as a banned node is filtered out,
                # so page N+1 would silently omit rows page N never showed.
                n_seen += 1
                if n_seen <= off:
                    continue
                if n_kept >= lim:
                    truncated[dkey] = True
                    break
                pl = self._plabels.get(f"P{p}")
                acc.append((f"P{p}", pl, f"Q{o}", self._name_of(o)))
                touched.extend([f"P{p}", f"Q{o}"])
                n_kept += 1

        if direction in ("both", "out"):
            collect(self._f_src, self._f_prop, self._f_dst, True,
                    out_entries, "out")
        if direction in ("both", "in"):
            collect(self._b_src, self._b_prop, self._b_dst, False,
                    in_entries, "in")
        # cache economics: re-expanding a FULLY expanded node returns data
        # the caller already holds -> free (same contract as name()/labels)
        repeat = q in self._expanded_full
        if direction == "both" and pid is None:
            self._expanded_full.add(q)
        self._rec("edges", [f"Q{q}", direction, pid, lim], touched,
                  free=repeat)
        return Struct(
            id=f"Q{q}", label=self._name_of(q),
            deg_out=int(self._deg_out[i]), deg_in=int(self._deg_in[i]),
            out=out_entries, in_=in_entries,
            truncated=truncated,
        )

    # ---------- entity record (data/graph/entities) ----------

    def _entity_index(self):
        """Lower bounds of the entity parts — ~38 KB, loaded once on first use.

        Ranges are disjoint by construction (build_entities.py chooses them), so
        one searchsorted resolves a qid to exactly one part.
        """
        if self._eindex is None:
            path = self.dir / "entities_index.json"
            if not path.exists():
                raise FileNotFoundError(
                    "data/graph/entities/ is not built — run "
                    "scripts/build_entities.py")
            index = json.loads(path.read_text())
            self._eparts = index["ranges"]
            self._eindex = np.array([p["qid_min"] for p in self._eparts],
                                    dtype=np.int64)
        return self._eindex

    def _record(self, q):
        """The entity's full record, parsed and cached. One part read."""
        if q in self._records:
            return self._records[q]
        bounds = self._entity_index()
        position = int(np.searchsorted(bounds, q, side="right")) - 1
        if position < 0 or q > self._eparts[position]["qid_max"]:
            raise KeyError(f"Q{q} is not in the graph.")
        table = pq.read_table(self.dir / "entities" /
                              self._eparts[position]["file"])
        rows = table.filter(pc.equal(table.column("qid"), q))
        if rows.num_rows == 0:
            raise KeyError(f"Q{q} is not in the graph.")
        row = rows.to_pylist()[0]
        self._records[q] = row
        return row

    @staticmethod
    def _zip(row, prefix, fields):
        """Parallel list columns share one order — read them back as tuples."""
        columns = [row.get(f"{prefix}_{f}") or [] for f in fields]
        return list(zip(*columns)) if columns[0] else []

    @staticmethod
    def _keep_best_rank(claims, ranks):
        """Per property: preferred if any exists, else everything not deprecated.

        Wikidata's rule for "the current value". The corpus holds few deprecated
        statements with literal values — they were dropped upstream — so this
        arbitrates between the competing values that ARE present rather than
        guaranteeing every stale value has been seen and removed.
        """
        by_property = {}
        for claim in claims:
            by_property.setdefault(claim["property"], []).append(claim)
        out = []
        for statements in by_property.values():
            keep = [s for s in statements if s["rank"] == "preferred"] or \
                   [s for s in statements if s["rank"] != "deprecated"]
            out.extend(keep)
        return out

    def claims(self, qid, pid=None, all_ranks=False):
        """Everything asserted about an entity: properties, typed values and
        qualifiers — including the dates, quantities and identifiers that
        edges() cannot represent. Best-rank by default (preferred if any, else
        non-deprecated); all_ranks=True keeps every statement. Each entry
        carries property, value, value_label, value_type, rank and qualifiers.
        1 read; free afterwards for the same entity."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)

        ranks = {(p, v): r for p, v, r in
                 self._zip(record, "rank", ["property_id", "value_id", "rank"])}
        qualifiers = {}
        for p, v, qp, qv, ql in self._zip(
                record, "qual", ["property_id", "value_id",
                                 "qualifier_property_id", "qualifier_value_id",
                                 "qualifier_value_label"]):
            qualifiers.setdefault((p, v), []).append(
                {"property": qp, "property_label": self._plabels.get(qp),
                 "value": qv, "value_label": ql})

        out = [{"property": p, "property_label": self._plabels.get(p),
                "value": v, "value_label": vl, "value_type": vt,
                "rank": ranks.get((p, v), "normal"),
                "qualifiers": qualifiers.get((p, v), [])}
               for p, v, vl, vt in self._zip(
                   record, "claim", ["property_id", "value_id", "value_label",
                                     "value_type"])]
        if not all_ranks:
            out = self._keep_best_rank(out, ranks)
        if pid is not None:
            out = [c for c in out if c["property"] == str(pid)]
        touched = [f"Q{q}"] + [c["value"] for c in out
                               if c["value_type"] == "wikibase-entityid"]
        self._rec("claims", [f"Q{q}", pid, all_ranks], touched, free=cached)
        return out

    def references(self, qid, pid=None):
        """Which sources ground an entity's claims: [{property, value,
        sources: [{property, value}]}]. P248 'stated in' names a database,
        P143 'imported from' means a Wikipedia copy, P854 a URL. Free once the
        entity's record has been read."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        grouped = {}
        for p, v, sp, sv in self._zip(
                record, "ref", ["property_id", "value_id", "ref_property_id",
                                "ref_value_id"]):
            if pid is not None and p != str(pid):
                continue
            grouped.setdefault((p, v), []).append(
                {"property": sp, "property_label": self._plabels.get(sp),
                 "value": sv})
        out = [{"property": p, "property_label": self._plabels.get(p),
                "value": v, "sources": sources}
               for (p, v), sources in grouped.items()]
        self._rec("references", [f"Q{q}", pid], [f"Q{q}"], free=cached)
        return out

    def describe(self, qid, langs=None):
        """Labels, descriptions and aliases of an entity across the six target
        languages (en, fr, de, zh, ar, ru), plus its instance_of and sitelinks.
        Free once the entity's record has been read."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        wanted = list(langs) if langs else list(LANGS)
        out = Struct(id=f"Q{q}", instance_of=record.get("instance_of"),
                     languages={})
        for lang in wanted:
            entry = Struct((field, record.get(f"{field}_{lang}"))
                          for field in ("label", "description", "aliases",
                                        "sitelink"))
            if any(entry.values()):
                out["languages"][lang] = entry
        self._rec("describe", [f"Q{q}", wanted], [f"Q{q}"], free=cached)
        return out

    # ---------- archive parity: convenience views over one record ----------

    def entity(self, qid):
        """Overview of an entity: label, English description, and the list of
        property ids describing it. 1 read; free afterwards."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        properties = list(dict.fromkeys(record.get("claim_property_id") or []))
        self._rec("entity", [f"Q{q}"], [f"Q{q}"], free=cached)
        return Struct(id=f"Q{q}", label=record.get("label_en"),
                      description=record.get("description_en"),
                      instance_of=record.get("instance_of"),
                      properties=properties, n_properties=len(properties))

    def label(self, qid, lang="en"):
        """Label in ONE language. No fallback: None if that language is absent."""
        q = _qint(qid)
        cached = q in self._records
        value = self._record(q).get(f"label_{lang}")
        self._rec("label", [f"Q{q}", lang], [f"Q{q}"], free=cached)
        return value

    def labels(self, qid):
        """Labels in every target language that has one."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        self._rec("labels", [f"Q{q}"], [f"Q{q}"], free=cached)
        return {l: record.get(f"label_{l}") for l in LANGS
                if record.get(f"label_{l}")}

    def description(self, qid, lang="en"):
        """Description in ONE language. No fallback."""
        q = _qint(qid)
        cached = q in self._records
        value = self._record(q).get(f"description_{lang}")
        self._rec("description", [f"Q{q}", lang], [f"Q{q}"], free=cached)
        return value

    def descriptions(self, qid, _free=False):
        """Descriptions in every target language that has one."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        self._rec("descriptions", [f"Q{q}"], [f"Q{q}"],
                  free=cached or _free)
        return {l: record.get(f"description_{l}") for l in LANGS
                if record.get(f"description_{l}")}

    # ---------- name search (data/graph/label_index) ----------

    # Calibrated on 36 hand-labelled queries: recall@10 is 1.00 here, 0.86 at
    # depth 200, 0.78 at depth 60, and 0.50 with the notability weights at zero.
    # Depth is load-bearing because BM25 rewards short fields — Barack Obama
    # sits at raw rank 145 of 1689 for "obama", under dozens of entities whose
    # entire label is "Obama".
    SEARCH_DEPTH = 4000
    SEARCH_SITELINK_WEIGHT = 2.0
    SEARCH_DEGREE_WEIGHT = 4.0

    def _index(self):
        """The label index, opened once. Absent until build_label_index.py runs."""
        if self._search is None:
            import tantivy
            path = self.dir / "label_index"
            if not path.exists():
                raise RuntimeError(
                    f"no label index at {path}; run scripts/build_label_index.py")
            index = tantivy.Index.open(str(path))
            index.reload()
            fields = json.loads((path / "fields.json").read_text())["text_fields"]
            self._search = (index, index.searcher(), fields)
        return self._search

    def search_property(self, text, limit=10):
        """Find a PROPERTY by name across the six languages. Returns
        (total_matches, [{pid, label, lang}]).

        Separate from search_entity because the two indexes are nothing alike:
        13,097 properties fit in memory and need no ranking beyond exact
        before prefix before substring, while entities need a full-text index.
        Conflating them cost a run a read — search_entity("father") returned
        7,430 films and television series. 1 read."""
        key = str(text).strip().lower()
        if not key:
            return 0, []
        found = []
        for pid, labels in self._plabels_all.items():
            best = None
            for lang, label in labels.items():
                if not label:
                    continue
                low = label.lower()
                tier = (0 if low == key else 1 if low.startswith(key)
                        else 2 if key in low else None)
                if tier is not None and (best is None or tier < best[0]):
                    best = (tier, lang, label)
            if best:
                # lower PID numbers are the older, more fundamental properties
                found.append((best[0], int(pid[1:]), pid, best[2], best[1]))
        found.sort()
        out = [{"pid": pid, "label": label, "lang": lang}
               for _, _, pid, label, lang in found[:max(1, int(limit))]]
        self._rec("search_property", [key, limit], [d["pid"] for d in out])
        return len(found), out

    def search_entity(self, text, limit=10, lang="en"):
        """Find entities by name. Returns (total_matches, [dict, ...]) where
        each dict is {qid, label, description} — the label and description are
        what a reader needs to tell identical names apart, and 19% of this
        graph's entities share their label with another.

        Matching spans labels and aliases in all six languages plus a
        diacritic-folded form, so 'gerhard schroder' finds 'Gerhard Schröder'.
        All query words must match. Results are ordered by text relevance
        weighted by how connected and how widely documented the entity is.
        1 read."""
        import math
        index, searcher, fields = self._index()
        query = index.parse_query(str(text), fields, conjunction_by_default=True)
        found = searcher.search(query, self.SEARCH_DEPTH, count=True)
        scored = []
        for score, address in found.hits:
            doc = searcher.doc(address)
            weight = (1 + self.SEARCH_SITELINK_WEIGHT * doc["sl"][0]
                      + self.SEARCH_DEGREE_WEIGHT * math.log1p(doc["deg"][0]))
            scored.append((score * weight, int(doc["qid"][0])))
        scored.sort(key=lambda pair: -pair[0])
        out = []
        for _, q in scored[:max(1, int(limit))]:
            record = self._record(q)
            out.append({
                "qid": f"Q{q}",
                "label": next((record.get(f"label_{l}") for l in (lang,) + LANGS
                               if record.get(f"label_{l}")), None),
                "description": next(
                    (record.get(f"description_{l}") for l in (lang,) + LANGS
                     if record.get(f"description_{l}")), None),
            })
        if not out and SEARCH_HINTS:
            self._explain_empty_search(str(text), index, searcher, fields)
        self._rec("search_entity", [str(text), limit], [d["qid"] for d in out])
        return found.count, out

    def languages(self, qid):
        """Which of the six target languages describe this entity at all."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        self._rec("languages", [f"Q{q}"], [f"Q{q}"], free=cached)
        return [l for l in LANGS
                if record.get(f"label_{l}") or record.get(f"description_{l}")]

    def neighbors(self, qid, pid=None, limit=None):
        """(property, target_qid, target_label) for OUTGOING links, optionally
        restricted to one property. 1 read."""
        out = self.edges(qid, direction="out", pid=pid, limit=limit)["out"]
        return [(p, other, other_label) for p, _, other, other_label in out]

    def backlinks(self, qid, pid=None, limit=None):
        """(property, source_qid, source_label) for links pointing AT this
        entity, optionally restricted to one property. 1 read.

        These are invisible from the entity's own record — 40% of child/parent
        pairs are declared in one direction only.
        """
        incoming = self.edges(qid, direction="in", pid=pid, limit=limit)["in_"]
        return [(p, other, other_label)
                for p, _, other, other_label in incoming]

    def graph_stats(self):
        """What this graph is — orient before exploring."""
        self._rec("graph_stats", [], [])
        return Struct(nodes=self.meta["nodes"], edges=self.meta["edges"],
                      world=self.meta["world"])

    def _make_ranker(self):
        import os
        from embedder import Embedder
        model = os.environ.get("WD_RANK_MODEL", "BAAI/bge-m3")
        tag = model.split("/")[-1]
        return Embedder(model, cache_path=str(self.dir / f"embcache_{tag}.npz"))

    def scout(self, a, b, max_reads=10, gradient_min=0.50):
        """Cheap mechanical probe, SELF-ABORTING — always safe to call first.
        Expands the two endpoints (2 reads you need anyway; re-reads are
        free), checks for an immediate bridge, then MEASURES the similarity
        gradient (top frontier-label cosine vs the far endpoint's label,
        both directions). If the gradient is flat (< gradient_min) it aborts
        at 2 reads and reports the numbers instead of wasting budget; if
        sharp, it continues greedily up to max_reads. Calibration (5
        instances): gradient-driven wins measured 0.57–0.77;
        losses 0.43–0.45. Returns {found, path?, reads_used, gradient,
        aborted?, src_map, dst_map}. Pass gradient_min=0 to force a full
        probe. VERIFY any returned path yourself — the scout does no
        verification and no reasoning; it only follows similarity."""
        qa, qb = _qint(a), _qint(b)
        for q in (qa, qb):
            if self._pos(q) is None:
                raise KeyError(f"Q{q} is not in the graph.")
        if self._ranker is None:
            self._ranker = self._make_ranker()
        emb = self._ranker
        la = self._name_of(qa) or f"Q{qa}"
        lb = self._name_of(qb) or f"Q{qb}"
        src = {f"Q{qa}": None}
        dst = {f"Q{qb}": None}
        labels = {f"Q{qa}": la, f"Q{qb}": lb}
        expanded = set()
        used = 0

        def expand(qs, side_map):
            nonlocal used
            e = self.edges(qs)
            expanded.add(qs)
            used += 1
            for key, orient in (("out", "->"), ("in_", "<-")):
                for pid, pl, other, ol in e[key]:
                    labels.setdefault(other, ol)
                    if other not in side_map:
                        side_map[other] = (qs, pid, orient)

        def bridge():
            hit = set(src) & set(dst)
            return next(iter(hit)) if hit else None

        def choose(side_map, far_label):
            cands = [c for c in side_map if c not in expanded]
            if not cands:
                return None
            vecs = emb.get([labels.get(c) or c for c in cands])
            qv = emb.get([far_label])[0]
            return cands[int((vecs @ qv).argmax())]

        expand(f"Q{qa}", src)
        expand(f"Q{qb}", dst)
        hit = bridge()

        # gradient measurement (free): is greedy similarity trustworthy here?
        import numpy as _np
        gradient = {}
        for side_map, far_label, key in ((src, lb, "src->"), (dst, la, "dst->")):
            cands = [labels.get(c) or c for c in side_map
                     if c not in expanded]
            if cands:
                vecs = emb.get(cands)
                qv = emb.get([far_label])[0]
                s = vecs @ qv
                gradient[key] = {"top": round(float(s.max()), 3),
                                 "median": round(float(_np.median(s)), 3)}
        grad_top = max((g["top"] for g in gradient.values()), default=0.0)
        if hit is None and grad_top < gradient_min:
            emb.save()
            out = {"found": False, "aborted": "flat gradient",
                   "gradient": gradient, "reads_used": used,
                   "src_map": src, "dst_map": dst}
            self._rec("scout", [f"Q{qa}", f"Q{qb}", max_reads], [], free=True)
            return out

        while hit is None and used < max_reads:
            if self.read_budget is not None and self.reads_left() < 2:
                break
            side, m, far = (("src", src, lb)
                            if len([c for c in src if c not in expanded])
                            <= len([c for c in dst if c not in expanded])
                            else ("dst", dst, la))
            pick = choose(m, far)
            if pick is None:
                break
            expand(pick, m)
            hit = bridge()
        emb.save()

        out = {"found": hit is not None, "reads_used": used,
               "gradient": gradient, "src_map": src, "dst_map": dst}
        if hit:
            def chain(m, q):
                seq = []
                while m[q] is not None:
                    parent, pid, orient = m[q]
                    seq.append((parent, pid, q, orient))
                    q = parent
                return seq
            lines = []
            for parent, pid, child, orient in reversed(chain(src, hit)):
                x, y = (parent, child) if orient == "->" else (parent, child)
                arrow = f"-[{pid}]->" if orient == "->" else f"<-[{pid}]-"
                lines.append(f"{x} ({labels.get(x)}) {arrow} "
                             f"{y} ({labels.get(y)})")
            for parent, pid, child, orient in chain(dst, hit):
                arrow = f"-[{pid}]->" if orient == "<-" else f"<-[{pid}]-"
                lines.append(f"{child} ({labels.get(child)}) {arrow} "
                             f"{parent} ({labels.get(parent)})")
            out["path"] = lines
        self._rec("scout", [f"Q{qa}", f"Q{qb}", max_reads],
                  [hit] if hit else [], free=True)
        return out

    def rank(self, ids, query, top=20):
        """Rank ALREADY-SEEN entity ids by semantic similarity of their
        labels to a free-text query. Embedding-based (deterministic, no
        sub-LM), FREE — costs no reads, like llm_query. Use it to order a
        frontier ("which of these connect to Norwegian politics?") before
        spending reads on edges(). Only ids the environment has returned to
        you are accepted — it ranks what you hold, it cannot discover.
        Returns [(qid, label, score)] best-first, top N."""
        ids = list(ids)
        unseen = [i for i in ids if str(i) not in self._seen]
        if unseen:
            raise KeyError(
                f"rank() only orders ids you have already seen from the "
                f"environment; unseen: {unseen[:5]}")
        if self._ranker is None:
            self._ranker = self._make_ranker()
        labels = [self._name_of(_qint(i)) or str(i) for i in ids]
        vecs = self._ranker.get(labels)
        qv = self._ranker.get([query])[0]
        scores = vecs @ qv
        order = scores.argsort()[::-1][:int(top)]
        self._rec("rank", [str(query)[:80], len(ids)],
                  [str(ids[j]) for j in order])
        return [(str(ids[j]), labels[j], round(float(scores[j]), 4))
                for j in order]

    def _explain_empty_search(self, text, index, searcher, fields):
        """Say WHICH word matched nothing, when a search returns nothing.

        The index is conjunctive -- one unmatched word zeroes the whole query
        -- and an empty list cannot distinguish "this entity is absent" from
        "your third word is wrong". Without a hint the model rephrases
        blindly. The harness holds the information that lets the model
        correct itself, so it returns it. Costs nothing on the common path:
        this runs only when the result is already empty.
        """
        words = [w for w in str(text).split() if w][:8]
        if len(words) < 2:
            print(f"(no match for {text!r}. Matching spans labels and aliases "
                  f"in en/fr/de/zh/ar/ru; try a different name or spelling.)")
            return
        counts = []
        for word in words:
            try:
                q = index.parse_query(word, fields, conjunction_by_default=True)
                counts.append((word, searcher.search(q, 1, count=True).count))
            except Exception:
                counts.append((word, None))
        dead = [w for w, n in counts if n == 0]
        live = ", ".join(f"{w}={n}" for w, n in counts if n)
        note = (f"none of your words match nothing individually, but no entity "
                f"has them ALL" if not dead
                else f"these match nothing: {', '.join(dead)}")
        print(f"(no match for {text!r}. All query words must match, so one "
              f"bad word empties the result -- {note}. Words that do match: "
              f"{live or 'none'}. Search with fewer, more distinctive words.)")

    def stored_text_objection(self, value, keys=("evidence", "text",
                                                  "description",
                                                  "subject_evidence")):
        """Which quoted string is not a stored description of its entity.

        Returns a message for the model, or None. FREE and unlogged: this is
        the harness checking the model's own claim against the world, not the
        model reading the world.

        The line that must not be crossed: this compares the answer with the
        GRAPH, never with the gold. Refusing a quotation the graph does not
        contain tells the model something it could have checked itself.
        Refusing an answer because it differs from the reference would grade a
        trajectory into correctness instead of letting it earn one, and the
        corpus would record a conclusion the model never reached.
        """
        def walk(node, qid=None):
            if isinstance(node, dict):
                here = qid
                direct = node.get("qid") or node.get("id")
                if isinstance(direct, str) and direct[:1].upper() == "Q":
                    here = direct
                else:
                    nested = [v.get("qid") for v in node.values()
                              if isinstance(v, dict)
                              and isinstance(v.get("qid"), str)]
                    if len(nested) == 1:
                        here = nested[0]
                for key in keys:
                    text = node.get(key)
                    if isinstance(text, str) and text.strip() and here:
                        yield here, text.strip()
                for v in node.values():
                    yield from walk(v, here)
            elif isinstance(node, list):
                for v in node:
                    yield from walk(v, qid)

        for qid, text in walk(value):
            try:
                stored = self.descriptions(qid, _free=True) or {}
            except (KeyError, TypeError):
                continue
            if any(str(v) == text for v in stored.values()):
                continue
            # Six of the seven quotations that failed grounding on the low
            # run were not invented: five wrapped a real stored description in
            # prose ("Russian description: '...' (Taiwanese)"), and one
            # differed by a curly apostrophe. Saying "not stored" to a model
            # that HAS the right text and framed it wrongly is useless; naming
            # the framing is repairable in one turn.
            inside = [(code, str(v)) for code, v in stored.items()
                      if str(v) and str(v) in text]
            if inside:
                code, exact = max(inside, key=lambda pair: len(pair[1]))
                return (f"REJECTED: your evidence for {qid} contains the "
                        f"stored {code} description but adds text around it. "
                        f"Return the description alone, exactly: {exact!r}")
            have = "; ".join(f"{k}={v!r}" for k, v in sorted(stored.items()))
            return (f"REJECTED: the evidence you gave for {qid} is not one of "
                    f"its stored descriptions. You wrote {text!r}. The stored "
                    f"descriptions are: {have or '(none)'}. Quote one of them "
                    f"exactly, or drop {qid} from the answer.")
        return None

    # ---------- harness side ----------

    def repl_namespace(self, allowed=None):
        ns = {"edges": self.edges, "count_edges": self.count_edges,
              "name": self.name, "has": self.has,
              "degree": self.degree, "graph_stats": self.graph_stats,
              "rank": self.rank, "scout": self.scout,
              "reads_left": self.reads_left,
              "claims": self.claims, "references": self.references,
              "describe": self.describe, "entity": self.entity,
              "label": self.label, "labels": self.labels,
              "description": self.description,
              "descriptions": self.descriptions, "languages": self.languages,
              "neighbors": self.neighbors, "backlinks": self.backlinks,
              "search_entity": self.search_entity,
              "search_property": self.search_property}
        if allowed is not None:
            ns = {k: v for k, v in ns.items() if k in allowed}
        return ns

    @property
    def read_log(self):
        return list(self._log)

    def seen_ids(self):
        out = set()
        for e in self._log:
            out.update(e["ids"])
        return out

    def save_log(self, path):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._log, indent=1))


if __name__ == "__main__":
    import time
    t0 = time.time()
    env = WDGraphEnv()
    print(f"loaded in {time.time()-t0:.1f}s :: {env.graph_stats()}")
    t0 = time.time()
    e = env.edges("Q31")  # Belgium — a hub; exercises truncation
    print(f"edges(Q31) in {time.time()-t0:.2f}s: deg_out={e['deg_out']} "
          f"deg_in={e['deg_in']} out={len(e['out'])} in={len(e['in_'])} "
          f"truncated={e['truncated']}")
    print("sample out:", e["out"][:5])
    print("sample in:", e["in_"][:5])
    print("has(Q7186):", env.has("Q7186"), "| name(Q7186):", env.name("Q7186"))
    print(f"reads: {len(env.read_log)} | spent: {env._spent}")
