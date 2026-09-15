"""build_graph.py — build the whole navigation graph into data/graph/: edges,
nodes with their multilingual entity layer, and property labels.

Prerequisites: data/raw/wikidata_formatted/ (statement batches; override with
    WIKIDATA_FORMATTED_SOURCE) for the `edges` step, and
    data/raw/wikidata_extracted/{languages,instance_of}/ plus sitelinks.parquet
    and wikidata_property_translations.parquet for the `nodes` step (override
    with WIKIDATA_EXTRACTED_SOURCE).

Outputs: data/graph/ (override with WIKIDATA_GRAPH_DIR) — edges_fwd.arrow and edges_bwd.arrow (src, prop, dst as
    int32, memory-mappable, sorted by src and by dst respectively),
    nodes.parquet (qid, deg_out, deg_in, label_*, description_*, aliases_*,
    instance_of, sitelink_* for the six target languages), properties.json
    (PID to a per-language label map) and meta.json.

Contract: everything keys on integer `qid`; nodes.parquet is sorted by it, and
    row i of an edge file is one triple across all three columns. Edge files are
    written uncompressed so numpy can searchsorted them straight off the mmap —
    Parquet would force a 3 GB decompression first. Rows for entities outside
    the graph are dropped, never invented. Ranks and references are absent from
    every source and are not reconstructed.

Why: the two build steps live in one script because they produce one dataset,
    but `--only` exists because their costs differ by two orders of magnitude —
    `edges` rescans 32 GB, `nodes` takes about two minutes. The long-to-wide
    pivot is bucketed by qid range: the language layer is 11.7 GB in memory
    against 16 GB of RAM, so it is routed to per-range files on write and
    pivoted one range at a time. Passing a Python int to searchsorted against an
    int32 array silently copies the whole array — always cast the key first.

Usage:
    uv run python scripts/build_graph.py                  # everything
    uv run python scripts/build_graph.py --only nodes     # entity layer alone
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get(
    "WIKIDATA_FORMATTED_SOURCE", ROOT / "data/raw/wikidata_formatted"))
EXTRACTED = Path(os.environ.get(
    "WIKIDATA_EXTRACTED_SOURCE", ROOT / "data/raw/wikidata_extracted"))
OUT = Path(os.environ.get("WIKIDATA_GRAPH_DIR", ROOT / "data/graph"))

LANGUAGES = ("en", "fr", "de", "zh", "ar", "ru")
WIKIS = {lg: f"{lg}wiki" for lg in LANGUAGES}

# nodes per pivot bucket: keeps each in-memory pivot near 200 MB
BUCKET_NODES = 1_000_000

SCAN_COLS = ["subject_id", "subject_label", "statement_id",
             "property_id", "value_id", "value_type"]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def write_arrow(path: Path, table: pa.Table) -> None:
    """Write an uncompressed IPC file, memory-mappable and zero-copy on read."""
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_arrow(path: Path) -> pa.Table:
    """Memory-map an IPC file without materialising it."""
    return pa.ipc.open_file(pa.memory_map(str(path), "rb")).read_all()


def qid_to_int(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Parse entity ids to integers, mapping anything not an item to -1."""
    if pa.types.is_integer(column.type):
        return column.to_numpy(zero_copy_only=False).astype(np.int64)
    is_item = pc.starts_with(column, pattern="Q")
    numeric = pc.if_else(is_item, pc.utf8_slice_codeunits(column, start=1), "-1")
    return pc.fill_null(
        pc.cast(numeric, pa.int64(), safe=False), -1
    ).to_numpy(zero_copy_only=False)


def positions_in(qids: np.ndarray, graph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, positions) for qids present in the sorted graph node array."""
    index = np.clip(np.searchsorted(graph, qids), 0, len(graph) - 1)
    return (graph[index] == qids) & (qids >= 0), index


# --------------------------------------------------------------------------
# step 1 — edges
# --------------------------------------------------------------------------

def scan_one(path: str) -> tuple[int, int, int]:
    """Scan one statement batch into a resumable .npz of nodes and raw edges."""
    number = int(path.rsplit("_", 1)[1][:4])
    target = OUT / "tmp_scan" / f"b{number:04d}.npz"
    if target.exists():
        return number, -1, -1

    table = pq.ParquetFile(path).read(columns=SCAN_COLS)

    # rows are statement x qualifier; keep the first row of each statement
    seen: set[str] = set()
    first = np.zeros(table.num_rows, dtype=bool)
    for i, statement in enumerate(table.column("statement_id").to_pylist()):
        if statement not in seen:
            seen.add(statement)
            first[i] = True
    table = table.filter(pa.array(first))

    subjects = table.column("subject_id").to_pylist()
    labels = table.column("subject_label").to_pylist()
    types = table.column("value_type").to_pylist()
    values = table.column("value_id").to_pylist()
    properties = table.column("property_id").to_pylist()

    named: dict[str, str] = {}
    for i, subject in enumerate(subjects):
        if subject not in named and subject[0] == "Q" and subject[1:].isdigit():
            named[subject] = labels[i]

    src, prop, dst = [], [], []
    non_q_values = p_subjects = 0
    for i, value_type in enumerate(types):
        if value_type != "wikibase-entityid":
            continue
        subject, value, prop_id = subjects[i], values[i], properties[i]
        if not (subject[0] == "Q" and subject[1:].isdigit()):
            p_subjects += 1
            continue
        if not (value and value[0] == "Q" and value[1:].isdigit()):
            non_q_values += 1
            continue
        src.append(int(subject[1:]))
        prop.append(int(prop_id[1:]))
        dst.append(int(value[1:]))

    np.savez_compressed(
        target,
        qids=np.array([int(s[1:]) for s in named], dtype=np.int64),
        labels=np.array(list(named.values()), dtype=object),
        src=np.array(src, dtype=np.int32),
        prop=np.array(prop, dtype=np.int32),
        dst=np.array(dst, dtype=np.int32),
        stats=np.array([non_q_values, p_subjects], dtype=np.int64),
    )
    return number, len(src), len(named)


def build_edges(workers: int) -> None:
    """Scan every batch, then assemble deduplicated sorted edge files."""
    (OUT / "tmp_scan").mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/batch_*.parquet"))
    if not files:
        raise SystemExit(f"no batch found in {SRC}")

    print(f"### edges — scan {len(files)} batches ({workers} workers)")
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(scan_one, files, chunksize=4)):
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(files)}] {time.time() - started:.0f}s",
                      flush=True)

    print("### edges — assemble", flush=True)
    node_tables, all_src, all_prop, all_dst = [], [], [], []
    stats = np.zeros(2, dtype=np.int64)
    for path in sorted(glob.glob(f"{OUT}/tmp_scan/b*.npz")):
        data = np.load(path, allow_pickle=True)
        node_tables.append(pa.table({
            "qid": pa.array(data["qids"], type=pa.int64()),
            "label_en": pa.array(data["labels"].tolist(), type=pa.string()),
        }))
        all_src.append(data["src"])
        all_prop.append(data["prop"])
        all_dst.append(data["dst"])
        stats += data["stats"]

    nodes = pa.concat_tables(node_tables)
    del node_tables
    nodes = nodes.take(pc.sort_indices(nodes, sort_keys=[("qid", "ascending")]))
    qid = nodes.column("qid").to_numpy()
    src = np.concatenate(all_src)
    prop = np.concatenate(all_prop)
    dst = np.concatenate(all_dst)
    del all_src, all_prop, all_dst
    assert (np.diff(qid) > 0).all(), "subject duplicated across batches"
    print(f"  {len(qid):,} nodes, {len(src):,} raw edges", flush=True)

    keep, _ = positions_in(dst.astype(np.int64), qid)
    boundary = int((~keep).sum())
    src, prop, dst = src[keep], prop[keep], dst[keep]
    print(f"  edges to nodes outside the graph dropped: {boundary:,}", flush=True)

    order = np.lexsort((prop, src.astype(np.int64) << 32 | dst.astype(np.int64)))
    src, prop, dst = src[order], prop[order], dst[order]
    del order
    duplicate = np.zeros(len(src), dtype=bool)
    duplicate[1:] = ((src[1:] == src[:-1]) & (dst[1:] == dst[:-1])
                     & (prop[1:] == prop[:-1]))
    duplicates = int(duplicate.sum())
    keep = ~duplicate
    src, prop, dst = src[keep], prop[keep], dst[keep]
    del duplicate, keep
    print(f"  duplicate triples removed: {duplicates:,} | final {len(src):,}",
          flush=True)

    write_arrow(OUT / "edges_fwd.arrow",
                pa.table({"src": src, "prop": prop, "dst": dst}))
    order = np.lexsort((prop, dst.astype(np.int64) << 32 | src.astype(np.int64)))
    write_arrow(OUT / "edges_bwd.arrow",
                pa.table({"src": src[order], "prop": prop[order],
                          "dst": dst[order]}))
    del order
    print("  edges_fwd.arrow / edges_bwd.arrow written", flush=True)

    bounds = np.append(qid, qid[-1] + 1)
    deg_out = np.diff(np.searchsorted(src, bounds)).astype(np.int32)
    deg_in = np.diff(np.searchsorted(
        read_arrow(OUT / "edges_bwd.arrow").column("dst").to_numpy(), bounds
    )).astype(np.int32)
    nodes = nodes.append_column("deg_out", pa.array(deg_out))
    nodes = nodes.append_column("deg_in", pa.array(deg_in))
    pq.write_table(nodes, OUT / "nodes.parquet", compression="zstd")

    labels: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(f"{SRC}/batch_*.parquet"))[::60]:
        table = pq.ParquetFile(path).read(
            columns=["property_id", "property_label"])
        for pid, label in zip(table.column("property_id").to_pylist(),
                              table.column("property_label").to_pylist()):
            if label:
                labels.setdefault(pid, {"en": label})
    (OUT / "properties.json").write_text(json.dumps(labels))

    (OUT / "meta.json").write_text(json.dumps({
        "source": str(SRC),
        "built": time.strftime("%Y-%m-%d"),
        "world": "Q-subjects of the formatted dump (en-labeled); mainsnak Q->Q "
                 "edges between nodes; ranks invisible, deprecated statements "
                 "traversable",
        "nodes": int(len(qid)),
        "edges": int(len(src)),
        "dropped_boundary_dst": boundary,
        "dropped_dupe_triples": duplicates,
        "dropped_nonQ_values": int(stats[0]),
        "dropped_P_subject_edges": int(stats[1]),
    }, indent=1))
    print(f"  {len(qid):,} nodes, {len(src):,} edges, "
          f"{len(labels):,} properties", flush=True)


# --------------------------------------------------------------------------
# step 2 — nodes
# --------------------------------------------------------------------------

def bucket_bounds(graph: np.ndarray) -> list[tuple[int, int]]:
    """Split the sorted node array into contiguous ranges of node positions."""
    edges = list(range(0, len(graph), BUCKET_NODES)) + [len(graph)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def route_to_buckets(
    tables, id_column: str, graph: np.ndarray, buckets, directory: Path, name: str
) -> None:
    """Stream source tables into one file per node range, keyed by qid.

    This is what makes the pivot possible at all: each bucket is small enough to
    hold in memory, while the whole layer is not.
    """
    directory.mkdir(parents=True, exist_ok=True)
    starts = np.array([lo for lo, _ in buckets])
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    for table in tables:
        qids = qid_to_int(table.column(id_column).combine_chunks())
        mask, index = positions_in(qids, graph)
        if not mask.any():
            continue
        table = table.drop_columns([id_column]).filter(pa.array(mask))
        table = table.add_column(0, "position", pa.array(index[mask]))
        which = np.searchsorted(starts, index[mask], side="right") - 1
        for bucket in np.unique(which):
            part = table.filter(pa.array(which == bucket))
            if bucket not in writers:
                writers[bucket] = pq.ParquetWriter(
                    directory / f"{name}_{bucket:03d}.parquet", part.schema)
            writers[bucket].write_table(part)
            rows += part.num_rows
    for writer in writers.values():
        writer.close()
    print(f"  {name}: {rows:,} rows routed to {len(writers)} buckets",
          flush=True)


def build_nodes() -> None:
    """Pivot the entity layers into a wide nodes.parquet and translate properties."""
    base = pq.read_table(OUT / "nodes.parquet",
                         columns=["qid", "label_en", "deg_out", "deg_in"])
    graph = base.column("qid").to_numpy()
    buckets = bucket_bounds(graph)
    scratch = OUT / "tmp_pivot"
    print(f"### nodes — {len(graph):,} nodes, {len(buckets)} buckets")

    route_to_buckets(
        (t.filter(pc.is_in(t.column("language"), value_set=pa.array(LANGUAGES)))
         for t in (pq.read_table(p, columns=["id", "language", "label",
                                             "description", "aliases"])
                   for p in sorted((EXTRACTED / "languages").glob("batch_*.parquet")))),
        "id", graph, buckets, scratch, "lang")
    route_to_buckets(
        (pq.read_table(p, columns=["subject_id", "instance_of_id"])
         for p in sorted((EXTRACTED / "instance_of").glob("batch_*.parquet"))),
        "subject_id", graph, buckets, scratch, "inst")

    sitelinks = EXTRACTED / "sitelinks.parquet"
    if sitelinks.exists():
        parquet = pq.ParquetFile(sitelinks)
        wanted = pa.array(list(WIKIS.values()))
        route_to_buckets(
            (t.filter(pc.is_in(t.column("wiki"), value_set=wanted))
             for t in (parquet.read_row_group(g, columns=["qid", "wiki", "title"])
                       for g in range(parquet.metadata.num_row_groups))),
            "qid", graph, buckets, scratch, "site")

    writer = None
    for n, (lo, hi) in enumerate(buckets):
        size = hi - lo
        columns: dict[str, object] = {
            "qid": graph[lo:hi],
            "deg_out": base.column("deg_out").to_numpy()[lo:hi],
            "deg_in": base.column("deg_in").to_numpy()[lo:hi],
        }

        path = scratch / f"lang_{n:03d}.parquet"
        for field in ("label", "description", "aliases"):
            for language in LANGUAGES:
                columns[f"{field}_{language}"] = [None] * size
        if path.exists():
            part = pq.read_table(path).to_pydict()
            for pos, lang, lab, desc, alias in zip(
                part["position"], part["language"], part["label"],
                part["description"], part["aliases"]
            ):
                i = pos - lo
                columns[f"label_{lang}"][i] = lab
                columns[f"description_{lang}"][i] = desc
                columns[f"aliases_{lang}"][i] = alias

        instances: list[list[str]] = [[] for _ in range(size)]
        path = scratch / f"inst_{n:03d}.parquet"
        if path.exists():
            part = pq.read_table(path).to_pydict()
            for pos, value in zip(part["position"], part["instance_of_id"]):
                instances[pos - lo].append(value)
        columns["instance_of"] = instances

        for language in LANGUAGES:
            columns[f"sitelink_{language}"] = [None] * size
        path = scratch / f"site_{n:03d}.parquet"
        if path.exists():
            part = pq.read_table(path).to_pydict()
            reverse = {v: k for k, v in WIKIS.items()}
            for pos, wiki, title in zip(
                part["position"], part["wiki"], part["title"]
            ):
                columns[f"sitelink_{reverse[wiki]}"][pos - lo] = title

        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(
                OUT / "nodes_wide.parquet", table.schema, compression="zstd")
        writer.write_table(table)
        if n % 10 == 0:
            print(f"  bucket {n + 1}/{len(buckets)}", flush=True)
    if writer:
        writer.close()
    os.replace(OUT / "nodes_wide.parquet", OUT / "nodes.parquet")
    shutil.rmtree(scratch, ignore_errors=True)

    translations = EXTRACTED / "wikidata_property_translations.parquet"
    if translations.exists():
        labels = json.loads((OUT / "properties.json").read_text())
        table = pq.read_table(translations)
        table = table.filter(
            pc.is_in(table.column("language"), value_set=pa.array(LANGUAGES)))
        data = table.to_pydict()
        for pid, language, label in zip(
            data["property_id"], data["language"], data["translation"]
        ):
            labels.setdefault(pid, {})[language] = label
        (OUT / "properties.json").write_text(json.dumps(labels))
        print(f"  properties.json: {len(labels):,} properties translated")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("edges", "nodes"))
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("GRAPH_WORKERS", "8")))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # the node step always follows unless the caller asked for edges alone
    if args.only != "nodes":
        build_edges(args.workers)
    if args.only != "edges":
        build_nodes()

    print(f"\n{OUT} built in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
