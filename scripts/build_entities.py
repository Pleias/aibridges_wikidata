"""build_entities.py — build data/graph/entities/, one complete record per entity.

Everything the agent can learn about an entity ends up on one row: its names in
six languages, its claims with qualifiers and ranks, and the sources backing
them. One lookup, one part read, no joins at runtime.

Prerequisites: data/raw/wikidata_formatted/ (claims; override with
    WIKIDATA_FORMATTED_SOURCE),
    data/graph/nodes.parquet (the language layer and the qid universe),
    data/graph/references/ and data/graph/ranks.parquet (built by
    build_annotations.py).
Outputs: data/graph/entities/part_NNNNN.parquet and data/graph/entities_index.json
    holding each part's inclusive [qid_min, qid_max].

Contract: parts cover contiguous, DISJOINT qid ranges chosen here, so a lookup
    is `searchsorted` over the index's lower bounds and resolves to exactly one
    part — the index is ~38 KB and belongs in memory. Repeated groups are stored
    as PARALLEL LIST COLUMNS sharing an order: claim_p[i] goes with claim_v[i],
    and qualifiers/references/ranks carry their own (property, value) so a
    reader joins them back by that pair. Entity identity is the int64 `qid`;
    only items are kept, and the Q prefix is checked rather than assumed,
    because "P15" stripped of its prefix is indistinguishable from "Q15".

Why: everything stays in Arrow. A first version grouped rows in Python and
    stored JSON strings; profiling put 89% of its time in to_pydict, the group
    loop and json.dumps — 0.28 s of Parquet read against 2.2 s of Python per
    batch — and projected to two hours. `group_by().aggregate([(col, "list")])`
    does the same grouping in C++. Lists are parallel rather than a list of
    structs because pyarrow 25 has no `hash_list` kernel for struct input.
    `property_label` is dropped: it is reconstructible from properties.json for
    every property observed, so carrying it is pure I/O.

Usage:
    uv run python scripts/build_entities.py
    uv run python scripts/build_entities.py --batches 40   # smoke test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CLAIMS_SRC = Path(os.environ.get(
    "WIKIDATA_FORMATTED_SOURCE", ROOT / "data/raw/wikidata_formatted"))
GRAPH = Path(os.environ.get("WIKIDATA_GRAPH_DIR", ROOT / "data/graph"))
OUT = GRAPH / "entities"
TMP = GRAPH / "_entities_tmp"

ENTITIES_PER_PART = 10_000
BUCKETS = 48
FLUSH_ROWS = 6_000_000

CLAIM_COLUMNS = ["subject_id", "property_id", "value_id", "value_label",
                 "value_type", "qualifier_property_id", "qualifier_value_id",
                 "qualifier_value_label"]
LANG_COLUMNS = [f"{field}_{lang}"
                for field in ("label", "description", "aliases", "sitelink")
                for lang in ("en", "fr", "de", "zh", "ar", "ru")]


def with_qid(table: pa.Table, column: str = "subject_id") -> pa.Table:
    """Add an int64 `qid` and drop every non-item subject.

    The Q prefix is checked, never assumed: "P15" stripped of its prefix is
    indistinguishable from "Q15", which files a property's rows under an
    unrelated entity.
    """
    ids = table.column(column)
    table = table.filter(pc.starts_with(ids, "Q"))
    numeric = pc.cast(pc.utf8_slice_codeunits(table.column(column), 1), "int64")
    return table.append_column("qid", numeric).drop_columns([column])


def collapse(table: pa.Table, columns: list[str], prefix: str) -> pa.Table:
    """Group by qid, turning each column into a per-entity list column."""
    grouped = table.group_by("qid").aggregate([(c, "list") for c in columns])
    return grouped.rename_columns(
        ["qid"] + [f"{prefix}_{c}" for c in columns])


def attach(base: pa.Table, extra: pa.Table | None) -> pa.Table:
    """Append `extra`'s columns to `base`, aligned on qid, null where absent.

    Arrow's join rejects list columns as non-key fields, and every column here
    is a list. Both sides are keyed by qid, so a searchsorted plus a masked
    take does the same work without materialising a join.
    """
    if extra is None or extra.num_rows == 0:
        return base
    extra = extra.sort_by("qid")
    keys = extra.column("qid").to_numpy()
    target = base.column("qid").to_numpy()
    position = np.clip(np.searchsorted(keys, target), 0, len(keys) - 1)
    present = keys[position] == target
    indices = pa.array(np.where(present, position, 0), type=pa.int64(),
                       mask=~present)
    aligned = extra.drop_columns(["qid"]).take(indices)
    for name in aligned.schema.names:
        base = base.append_column(name, aligned.column(name))
    return base


def nest_batch(table: pa.Table) -> pa.Table:
    """One batch as one row per entity: claim lists plus flat qualifier lists."""
    table = with_qid(table)
    claims = collapse(
        table.group_by(["qid", "property_id", "value_id", "value_label",
                        "value_type"]).aggregate([]),
        ["property_id", "value_id", "value_label", "value_type"], "claim")
    qualified = table.filter(pc.is_valid(table.column("qualifier_property_id")))
    quals = collapse(
        qualified, ["property_id", "value_id", "qualifier_property_id",
                    "qualifier_value_id", "qualifier_value_label"], "qual") \
        if qualified.num_rows else None
    return attach(claims.sort_by("qid"), quals)


def bucket_bounds(qids: np.ndarray) -> np.ndarray:
    step = max(1, len(qids) // BUCKETS)
    return np.array([int(qids[i]) for i in range(0, len(qids), step)][:BUCKETS],
                    dtype=np.int64)


class FragmentWriter:
    """Buffers per-entity rows per bucket, flushing them as fragment files."""

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds
        self.buffers: dict[int, list[pa.Table]] = defaultdict(list)
        self.buffered = 0
        self.flushes = 0
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)

    def add(self, table: pa.Table) -> None:
        buckets = np.searchsorted(self.bounds, table.column("qid").to_numpy(),
                                  side="right") - 1
        for bucket in np.unique(buckets):
            self.buffers[int(bucket)].append(
                table.filter(pa.array(buckets == bucket)))
        self.buffered += table.num_rows
        if self.buffered >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        for bucket, tables in self.buffers.items():
            directory = TMP / f"bucket_{bucket:03d}"
            directory.mkdir(exist_ok=True)
            pq.write_table(pa.concat_tables(tables, promote_options="default"),
                           directory / f"frag_{self.flushes:04d}.parquet",
                           compression="zstd")
        self.buffers.clear()
        self.buffered = 0
        self.flushes += 1


def annotations_for(lo: int, hi: int) -> pa.Table | None:
    """References of one bucket, as per-entity list columns."""
    index = json.loads((GRAPH / "annotations_index.json").read_text())
    tables = []
    for part in index["references"]["ranges"]:
        if part["qid_max"] < lo or part["qid_min"] >= hi:
            continue
        table = with_qid(pq.read_table(GRAPH / "references" / part["file"]))
        qids = table.column("qid").to_numpy()
        inside = (qids >= lo) & (qids < hi)
        if inside.any():
            tables.append(table.filter(pa.array(inside)))
    if not tables:
        return None
    return collapse(pa.concat_tables(tables),
                    ["property_id", "value_id", "ref_property_id",
                     "ref_value_id"], "ref")


def ranks_by_bucket(bounds: np.ndarray) -> dict[int, pa.Table]:
    """The whole ranks layer, pre-split per bucket. 3.3M rows, read once."""
    table = with_qid(pq.read_table(GRAPH / "ranks.parquet"))
    buckets = np.searchsorted(bounds, table.column("qid").to_numpy(),
                              side="right") - 1
    out = {}
    for bucket in np.unique(buckets):
        out[int(bucket)] = collapse(table.filter(pa.array(buckets == bucket)),
                                    ["property_id", "value_id", "rank"], "rank")
    return out


def compact_bucket(bucket: int, lo: int, hi: int, ranks, part_offset: int):
    """Join a bucket's claims with references, ranks and the language layer."""
    directory = TMP / f"bucket_{bucket:03d}"
    fragments = sorted(directory.glob("frag_*.parquet")) if directory.exists() \
        else []

    record = pq.read_table(
        GRAPH / "nodes.parquet", columns=["qid", "instance_of"] + LANG_COLUMNS,
        filters=[("qid", ">=", int(lo)), ("qid", "<", int(hi))]).sort_by("qid")
    claims = pa.concat_tables([pq.read_table(f) for f in fragments],
                              promote_options="default") if fragments else None
    if claims is not None:
        # attach() aligns one row per qid, so a second row for the same entity
        # would be dropped in silence. That can only happen if an entity's
        # claims span two source batches — the invariant this build rests on.
        seen = np.sort(claims.column("qid").to_numpy())
        if len(seen) > 1 and (seen[1:] == seen[:-1]).any():
            duplicate = seen[1:][seen[1:] == seen[:-1]][0]
            raise SystemExit(
                f"Q{duplicate} has claims in more than one source batch; "
                f"attach() would drop all but one. Check the Q-prefix filter "
                f"in with_qid() before trusting this build.")
    for extra in (claims, annotations_for(lo, hi), ranks.get(bucket)):
        record = attach(record, extra)

    written = []
    for offset in range(0, record.num_rows, ENTITIES_PER_PART):
        chunk = record.slice(offset, ENTITIES_PER_PART)
        qids = chunk.column("qid").to_numpy()
        number = part_offset + offset // ENTITIES_PER_PART
        path = OUT / f"part_{number:05d}.parquet"
        pq.write_table(chunk, path, compression="zstd")
        written.append({"file": path.name, "entities": chunk.num_rows,
                        "qid_min": int(qids[0]), "qid_max": int(qids[-1])})
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int,
                    help="read only the first N claim batches (smoke test)")
    args = ap.parse_args()

    global OUT, TMP
    batches = sorted(glob.glob(str(CLAIMS_SRC / "batch_*.parquet")))
    if args.batches:
        # a smoke test must never overwrite a real build: it rebuilds every
        # part from a fraction of the claims, so the output looks complete
        # while most entities have none
        batches = batches[:args.batches]
        OUT, TMP = GRAPH / "_entities_smoke", GRAPH / "_entities_smoke_tmp"
        print(f"SMOKE TEST: {len(batches)} batches -> {OUT.name}/", flush=True)
    qids = np.sort(pq.read_table(GRAPH / "nodes.parquet",
                                 columns=["qid"]).column("qid").to_numpy())
    bounds = bucket_bounds(qids)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print(f"{len(qids):,} entities | {len(batches)} batches | "
          f"{BUCKETS} buckets", flush=True)

    started = time.time()
    writer = FragmentWriter(bounds)
    for position, path in enumerate(batches):
        writer.add(nest_batch(pq.read_table(path, columns=CLAIM_COLUMNS)))
        if position % 200 == 0:
            print(f"  pass 1 [{position}/{len(batches)}] "
                  f"{time.time() - started:.0f}s", flush=True)
    writer.flush()
    print(f"  pass 1 done in {time.time() - started:.0f}s\n", flush=True)

    ranks = ranks_by_bucket(bounds)
    edges = list(bounds) + [int(qids[-1]) + 1]
    parts, offset = [], 0
    for bucket in range(len(bounds)):
        mark = time.time()
        written = compact_bucket(bucket, edges[bucket], edges[bucket + 1],
                                 ranks, offset)
        offset += len(written)
        parts.extend(written)
        print(f"  pass 2 [{bucket}/{len(bounds)}] {len(written)} parts "
              f"({time.time() - mark:.0f}s)", flush=True)

    for previous, current in zip(parts, parts[1:]):
        if current["qid_min"] <= previous["qid_max"]:
            raise SystemExit(f"ranges overlap at {current['file']}")
    index_name = ("entities_index.json" if not args.batches
                  else "_entities_smoke_index.json")
    (GRAPH / index_name).write_text(json.dumps({
        "parts": len(parts),
        "entities": sum(p["entities"] for p in parts),
        "ranges": parts}, indent=1))
    shutil.rmtree(TMP)
    print(f"\n{len(parts)} parts, {sum(p['entities'] for p in parts):,} "
          f"entities in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
