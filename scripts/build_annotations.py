"""build_annotations.py — bring the Hugging Face ranks and references into
data/graph/, repartitioned by qid like everything else there.

data/hf_ranks/ (override with WIKIDATA_HF_RANKS_SOURCE) is raw download: 7,443 chunks cut on Hugging Face's own
boundaries, which align with nothing in this repo. This rewrites them into the
graph's convention — contiguous qid ranges we choose ourselves — so a reference
lookup is one small file read instead of a scan.

Prerequisites: data/hf_ranks/{ranks,references}/ and data/graph/nodes.parquet
    (its sorted qids define the range boundaries).
Outputs: data/graph/references/part_NNNN.parquet (one per qid range),
    data/graph/ranks.parquet (single file, 3.3M rows), and
    data/graph/annotations_index.json holding each part's inclusive
    [qid_min, qid_max] plus row counts.

Contract: unlike the HF partitioning, these ranges are disjoint BY
    CONSTRUCTION, so a qid maps to exactly one part and a range table is sound —
    the statements index needs a slice lookup, this one does not. Rows inside a
    part are NOT sorted; filter on subject_id after reading. Both files keep
    subject_id as the string "Q42", matching every other annotation layer and
    NOT nodes.parquet's int64 qid.

Why: values are strings — URLs, external identifiers, dates — so the
    uncompressed memory-mappable form used for edges would cost ~29 GB against
    3.7 GB of Parquet, and the win would be nil since a lookup reads one part
    whole. Parts are sized by entity count rather than qid width because QID
    density is wildly uneven: a fixed qid stride yields parts from empty to
    enormous. Buffers flush by accumulated rows, which stays cheap because the
    HF chunks arrive nearly in qid order, so few ranges are open at once.

Usage:
    uv run python scripts/build_annotations.py
    uv run python scripts/build_annotations.py --only ranks
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
HF = Path(os.environ.get("WIKIDATA_HF_RANKS_SOURCE", ROOT / "data/hf_ranks"))
GRAPH = Path(os.environ.get("WIKIDATA_GRAPH_DIR", ROOT / "data/graph"))
ENTITIES_PER_PART = 25_000
FLUSH_ROWS = 2_000_000


def range_boundaries() -> np.ndarray:
    """Lower qid bound of each part, taken every N-th entity of the graph."""
    qids = np.sort(pq.read_table(GRAPH / "nodes.parquet",
                                 columns=["qid"]).column("qid").to_numpy())
    return qids[::ENTITIES_PER_PART].copy()


def qid_column(table: pa.Table) -> np.ndarray:
    """Subject as int64, with -1 for anything that is not an item.

    The Q prefix must be checked, never assumed: "P15" stripped of its prefix
    is indistinguishable from "Q15", which silently files a property's rows
    under an unrelated entity.
    """
    column = table.column("subject_id")
    numeric = pc.cast(pc.utf8_slice_codeunits(column, 1), "int64")
    numeric = pc.if_else(pc.starts_with(column, "Q"), numeric, -1)
    return numeric.to_numpy(zero_copy_only=False)


class PartWriter:
    """Accumulates rows per qid range and flushes them as numbered parts."""

    def __init__(self, out_dir: Path, boundaries: np.ndarray):
        self.dir = out_dir
        self.boundaries = boundaries
        self.buffers: dict[int, list[pa.Table]] = {}
        self.buffered = 0
        self.written: dict[int, int] = {}
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    def add(self, table: pa.Table) -> None:
        parts = np.searchsorted(self.boundaries, qid_column(table),
                                side="right") - 1
        for part in np.unique(parts):
            rows = table.filter(pa.array(parts == part))
            self.buffers.setdefault(int(part), []).append(rows)
            self.buffered += rows.num_rows
        if self.buffered >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        for part, tables in self.buffers.items():
            combined = pa.concat_tables(tables)
            path = self.dir / f"part_{part:04d}.parquet"
            if path.exists():  # a later chunk reached back into this range
                combined = pa.concat_tables([pq.read_table(path), combined])
            pq.write_table(combined, path, compression="zstd")
            self.written[part] = combined.num_rows
        self.buffers.clear()
        self.buffered = 0


def build_references(boundaries: np.ndarray) -> dict:
    files = sorted(glob.glob(str(HF / "references/chunk_*.parquet")))
    print(f"references: {len(files)} chunks -> "
          f"{len(boundaries)} qid ranges ...")
    writer = PartWriter(GRAPH / "references", boundaries)
    rows = empty = 0
    started = time.time()
    for position, path in enumerate(files):
        table = pq.read_table(path)
        if table.num_rows == 0 or table.column("subject_id").null_count == \
                table.num_rows:
            empty += 1
            continue
        writer.add(table)
        rows += table.num_rows
        if position % 1000 == 0:
            print(f"  [{position}/{len(files)}] {rows:,} rows "
                  f"({time.time() - started:.0f}s)")
    writer.flush()
    print(f"  {rows:,} rows kept, {empty} empty chunks skipped, "
          f"{len(writer.written)} parts written")
    return describe_parts(GRAPH / "references")


def build_ranks() -> dict:
    files = sorted(glob.glob(str(HF / "ranks/chunk_*.parquet")))
    print(f"ranks: {len(files)} chunks -> one file ...")
    table = pq.read_table(files)
    order = pa.compute.sort_indices(table, sort_keys=[("subject_id", "ascending")])
    table = table.take(order)
    pq.write_table(table, GRAPH / "ranks.parquet", compression="zstd")
    counts = {kind: 0 for kind in set(table.column("rank").to_pylist())}
    for kind in table.column("rank").to_pylist():
        counts[kind] += 1
    size = (GRAPH / "ranks.parquet").stat().st_size
    print(f"  {table.num_rows:,} rows, {size / 2**20:.0f} MiB, {counts}")
    return {"rows": table.num_rows, "bytes": size, "by_rank": counts}


def describe_parts(directory: Path) -> dict:
    parts = []
    for path in sorted(directory.glob("part_*.parquet")):
        table = pq.read_table(path, columns=["subject_id"])
        numeric = qid_column(table)
        parts.append({"file": path.name, "rows": len(numeric),
                      "qid_min": int(numeric.min()),
                      "qid_max": int(numeric.max())})
    for previous, current in zip(parts, parts[1:]):
        if current["qid_min"] <= previous["qid_max"]:
            raise SystemExit(
                f"ranges overlap between {previous['file']} and "
                f"{current['file']} — the partitioning is broken")
    return {"parts": len(parts), "rows": sum(p["rows"] for p in parts),
            "ranges": parts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ranks", "references"])
    args = ap.parse_args()

    index_path = GRAPH / "annotations_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    if args.only != "ranks":
        index["references"] = build_references(range_boundaries())
    if args.only != "references":
        index["ranks"] = build_ranks()

    index_path.write_text(json.dumps(index, indent=1))
    print(f"\nwrote {index_path}")


if __name__ == "__main__":
    main()
