"""build_label_index.py — the searchable name index over data/graph/nodes.parquet.

Prerequisites: data/graph/nodes.parquet built by build_graph.py (the graph
    directory follows WIKIDATA_GRAPH_DIR when it is set).
Outputs:       data/graph/label_index/ — a tantivy index over labels and
    aliases in the six target languages.

Contract: labels and aliases live in SEPARATE fields, one pair per language,
    plus a diacritic-folded field spanning all of them. Neither split is
    cosmetic. BM25 normalises by field length, so any merge penalises exactly
    the entities that carry the most text — the notable ones. Merged into one
    field, Angela Merkel scored 27.7 on "merkel" against 59.3 for a Texas town,
    because her field also holds "Angela Dorothea Merkel; Angela Kasner; …";
    Ptolemy scored 19.2 against 91.2 for the given name. Recall@10 on the
    36-query calibration set went 1.00 -> 0.92 when merged, and back to 1.00
    when split — with a label boost of 1.0, so the separation is what mattered,
    not the weighting.

    `sl` (sitelinks) and `deg` (degree) are stored as fast fields because
    ranking needs them at query time. They are the whole reason search works:
    without the notability term, recall@10 on the calibration set falls from
    1.00 to 0.53. Retrieval depth is equally load-bearing — 4000 gives 1.00,
    200 gives 0.86, 60 gives 0.78. Barack Obama sits at raw-BM25 rank 145 of
    1689 for "obama", because BM25 rewards short fields and dozens of entities
    are labelled exactly "Obama".

Usage:
    uv run python scripts/build_label_index.py
    uv run python scripts/build_label_index.py --langs en fr
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tantivy

ROOT = Path(__file__).resolve().parent.parent
GRAPH = Path(os.environ.get("WIKIDATA_GRAPH_DIR", ROOT / "data" / "graph"))
NODES = GRAPH / "nodes.parquet"
OUT = GRAPH / "label_index"
LANGS = ("en", "fr", "de", "zh", "ar", "ru")
CHUNK = 500_000

# a translate table folds diacritics ~40x faster than per-string NFKD
FOLD = {}
for _cp in list(range(0xC0, 0x180)) + list(range(0x1E00, 0x1F00)):
    _plain = unicodedata.normalize("NFKD", chr(_cp)).encode("ascii", "ignore").decode()
    if _plain:
        FOLD[_cp] = _plain
FOLD[0xDF] = "ss"


def build_schema(langs: tuple[str, ...]) -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    for lang in langs:
        builder.add_text_field(f"l_{lang}", stored=False)   # label only: short
        builder.add_text_field(f"a_{lang}", stored=False)   # aliases
    builder.add_text_field("fold", stored=False)
    builder.add_unsigned_field("qid", stored=True, indexed=True, fast=True)
    builder.add_unsigned_field("sl", stored=True, indexed=True, fast=True)
    builder.add_unsigned_field("deg", stored=True, indexed=True, fast=True)
    return builder.build()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=list(LANGS))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    langs = tuple(args.langs)

    started = time.time()
    columns = ["qid", "deg_out", "deg_in"]
    columns += [f"label_{l}" for l in langs] + [f"aliases_{l}" for l in langs]
    columns += [f"sitelink_{l}" for l in LANGS]
    table = pq.read_table(NODES, columns=columns)
    qid = table.column("qid").to_numpy()
    deg = (table.column("deg_out").to_numpy().astype(np.int64)
           + table.column("deg_in").to_numpy().astype(np.int64))
    sitelinks = sum(
        table.column(f"sitelink_{l}").is_valid().to_numpy(zero_copy_only=False)
        .astype(np.int64) for l in LANGS)
    labels = {l: table.column(f"label_{l}").combine_chunks() for l in langs}
    aliases = {l: table.column(f"aliases_{l}").combine_chunks() for l in langs}
    print(f"load {time.time()-started:.1f}s | rows {table.num_rows:,} "
          f"| langs {' '.join(langs)}", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    index = tantivy.Index(build_schema(langs), path=str(OUT))
    writer = index.writer(heap_size=1_000_000_000, num_threads=args.threads)

    started = time.time()
    written = 0
    for start in range(0, table.num_rows, CHUNK):
        stop = min(start + CHUNK, table.num_rows)
        block = {l: (labels[l].slice(start, stop - start).to_pylist(),
                     aliases[l].slice(start, stop - start).to_pylist())
                 for l in langs}
        for offset in range(stop - start):
            row = start + offset
            parts = []
            document = tantivy.Document()
            for lang in langs:
                label, alias = block[lang][0][offset], block[lang][1][offset]
                if label:
                    document.add_text(f"l_{lang}", label)
                    parts.append(label)
                if alias:
                    document.add_text(f"a_{lang}", alias)
                    parts.append(alias)
            if not parts:
                continue                    # nothing searchable about this node
            document.add_text("fold", " ".join(parts).translate(FOLD))
            document.add_unsigned("qid", int(qid[row]))
            document.add_unsigned("sl", int(sitelinks[row]))
            document.add_unsigned("deg", int(deg[row]))
            writer.add_document(document)
            written += 1
        if start % (CHUNK * 10) == 0:
            print(f"  {stop:,} rows  {time.time()-started:.0f}s", flush=True)

    print(f"queued {written:,} documents in {time.time()-started:.0f}s "
          f"— committing", flush=True)
    writer.commit()
    index.reload()
    size = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e9
    # the tantivy Schema is opaque from Python, so the searchable fields are
    # declared here rather than discovered at query time
    (OUT / "fields.json").write_text(
        json.dumps({"text_fields": [f"{p}_{l}" for l in langs for p in "la"]
                                   + ["fold"],
                    "langs": list(langs), "documents": written}, indent=1))
    print(f"BUILT {written:,} documents in {(time.time()-started)/60:.1f} min "
          f"| {size:.2f} GB | {OUT}")


if __name__ == "__main__":
    main()
