"""Discover bidirectional graph hubs and recurring typed relations.

The script is deliberately observational: it discovers PIDs from the graph,
counts neighbours exactly for selected hubs, and annotates the result with
direct P31 values. It does not contain a semantic PID allowlist and does not
generate questions.

Pilot:
    uv run python scripts/taxonomy/discover_hubs.py \
        --max-hubs 50 --max-degree 5000 \
        --output-dir data/generation/taxonomy/hub_discovery_pilot

The edge files are memory-mapped and the node file is read one Parquet row
group at a time. Existing output is never overwritten unless --force is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
GRAPH = Path(os.environ.get("WIKIDATA_GRAPH_DIR") or ROOT / "data" / "graph")
DEFAULT_OUTPUT = ROOT / "data" / "generation" / "taxonomy" / "hub_discovery"
P31 = 31
UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def qid(value: int) -> str:
    return f"Q{int(value)}"


def pid(value: int) -> str:
    return f"P{int(value)}"


def stable_hash(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fast deterministic hash; unlike Python hash, stable across processes."""
    values = values.astype(np.uint64, copy=False)
    mixed = values ^ np.uint64(seed)
    return (mixed * np.uint64(11400714819323198485)) & UINT64_MASK


class NodeIndex:
    """Read node metadata in row-group-sized pieces and resolve selected QIDs."""

    def __init__(self, path: Path):
        self.path = path
        self.file = pq.ParquetFile(path)
        self.row_groups = []
        for i in range(self.file.metadata.num_row_groups):
            rg = self.file.metadata.row_group(i)
            qcol = rg.column(self.file.schema_arrow.get_field_index("qid"))
            stats = qcol.statistics
            if stats is None:
                raise RuntimeError("nodes.parquet needs qid row-group statistics")
            self.row_groups.append((int(stats.min), int(stats.max), i))

    def iter_degree_rows(self):
        for i in range(self.file.metadata.num_row_groups):
            table = self.file.read_row_group(
                i, columns=["qid", "deg_in", "deg_out"])
            yield (
                table["qid"].to_numpy(),
                table["deg_in"].to_numpy(),
                table["deg_out"].to_numpy(),
            )

    def lookup(self, wanted: set[int]) -> dict[int, dict]:
        """Resolve labels and direct instance_of values for selected QIDs."""
        if not wanted:
            return {}
        ordered = np.array(sorted(wanted), dtype=np.int64)
        result: dict[int, dict] = {}
        for lo, hi, row_group in self.row_groups:
            left = int(np.searchsorted(ordered, lo, side="left"))
            right = int(np.searchsorted(ordered, hi, side="right"))
            if left == right:
                continue
            table = self.file.read_row_group(row_group, columns=[
                "qid", "label_en", "instance_of",
                "description_en", "description_fr", "description_de",
                "description_zh", "description_ar", "description_ru",
            ])
            qids = table["qid"].to_numpy()
            qleft = int(np.searchsorted(qids, ordered[left], side="left"))
            qright = int(np.searchsorted(qids, ordered[right - 1], side="right"))
            for index in range(qleft, qright):
                value = int(qids[index])
                if value not in wanted:
                    continue
                raw_types = table["instance_of"][index].as_py() or []
                descriptions = {
                    language: bool(table[f"description_{language}"][index].as_py())
                    for language in ("en", "fr", "de", "zh", "ar", "ru")
                }
                result[value] = {
                    "label": table["label_en"][index].as_py() or "",
                    "types": tuple(sorted(set(raw_types))),
                    "has_rare_description": descriptions["ar"] or descriptions["ru"],
                    "has_only_rare_description": (
                        (descriptions["ar"] or descriptions["ru"])
                        and not any(descriptions[language]
                                    for language in ("en", "fr", "de", "zh"))
                    ),
                }
        return result


class EdgeIndex:
    """Memory-mapped forward and backward edge indexes."""

    def __init__(self, graph: Path):
        self.sources = []
        self.tables = []
        for filename, key in (("edges_fwd.arrow", "src"),
                              ("edges_bwd.arrow", "dst")):
            source = pa.memory_map(str(graph / filename), "r")
            table = ipc.open_file(source).read_all()
            self.sources.append(source)
            self.tables.append(table)
        self.f_src = self.tables[0]["src"].to_numpy()
        self.f_prop = self.tables[0]["prop"].to_numpy()
        self.f_dst = self.tables[0]["dst"].to_numpy()
        self.b_src = self.tables[1]["src"].to_numpy()
        self.b_prop = self.tables[1]["prop"].to_numpy()
        self.b_dst = self.tables[1]["dst"].to_numpy()

    def slice(self, hub: int, direction: str):
        if direction == "out":
            key, props, neighbours = self.f_src, self.f_prop, self.f_dst
        else:
            # edges_bwd is sorted by its ``dst`` column, but the linked entity
            # in an incoming relation is the row's ``src``.
            key, props, neighbours = self.b_dst, self.b_prop, self.b_src
        scalar = np.asarray(hub, dtype=key.dtype)
        start = int(key.searchsorted(scalar, side="left"))
        end = int(key.searchsorted(scalar, side="right"))
        return props[start:end], neighbours[start:end]


def load_property_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    return {
        key: (value.get("en") or next(iter(value.values()), ""))
        for key, value in raw.items()
        if key.startswith("P") and isinstance(value, dict)
    }


def top_from_group(qids, degrees, limit):
    if not len(qids):
        return []
    count = min(len(qids), max(limit * 2, limit))
    indices = np.argpartition(degrees, -count)[-count:]
    return [(int(qids[i]), int(degrees[i])) for i in indices]


def select_hubs(nodes: NodeIndex, args) -> dict[int, dict]:
    """Select top incoming/outgoing hubs plus a deterministic degree sample."""
    top_each = max(1, int(args.max_hubs * 0.30))
    stratified = max(0, args.max_hubs - 2 * top_each)
    top_in, top_out = [], []
    buckets: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    bounds = ((50, 99, "50-99"), (100, 999, "100-999"),
              (1000, 9999, "1000-9999"), (10000, args.max_degree, "10000+"))

    for qids, deg_in, deg_out in nodes.iter_degree_rows():
        eligible_in = (deg_in >= args.min_degree) & (deg_in <= args.max_degree)
        eligible_out = (deg_out >= args.min_degree) & (deg_out <= args.max_degree)
        eligible = eligible_in | eligible_out
        indices_in = np.flatnonzero(eligible_in)
        indices_out = np.flatnonzero(eligible_out)
        if len(indices_in):
            top_in.extend(top_from_group(qids[indices_in], deg_in[indices_in], top_each))
        if len(indices_out):
            top_out.extend(top_from_group(qids[indices_out], deg_out[indices_out], top_each))
        max_degree = np.maximum(deg_in, deg_out)
        for low, high, name in bounds:
            if high < low:
                continue
            mask = (max_degree >= low) & (max_degree <= high)
            mask &= eligible
            selected = np.flatnonzero(mask)
            if not len(selected):
                continue
            keep = min(len(selected), max(2, stratified // 4 * 3))
            hashes = stable_hash(qids[selected], args.seed)
            chosen = selected[np.argpartition(hashes, keep - 1)[:keep]]
            buckets[name].extend(
                (int(qids[i]), int(deg_in[i]), int(deg_out[i])) for i in chosen)

    selected: dict[int, dict] = {}
    for direction, rows in (("in", top_in), ("out", top_out)):
        rows.sort(key=lambda row: (-row[1], row[0]))
        for value, degree in rows[:top_each]:
            selected.setdefault(value, {"qid": value, "selected_by": set()})
            selected[value]["selected_by"].add(direction)

    per_bucket = max(1, stratified // max(1, len(bounds)))
    for name, rows in buckets.items():
        rows.sort(key=lambda row: (
            int(stable_hash(np.array([row[0]], dtype=np.uint64), args.seed)[0]),
            row[0],
        ))
        for value, _, _ in rows[:per_bucket]:
            selected.setdefault(value, {"qid": value, "selected_by": set()})
            selected[value]["selected_by"].add("stratified")

    return dict(sorted(selected.items()))


def collect_relations(edge_index: EdgeIndex, hubs: dict[int, dict], args):
    relations = []
    skipped = []
    linked: set[int] = set(hubs)
    for hub, metadata in hubs.items():
        for direction in ("in", "out"):
            props, neighbours = edge_index.slice(hub, direction)
            if len(props) > args.max_edge_scan:
                skipped.append({
                    "hub_qid": qid(hub), "direction": direction,
                    "edge_count": int(len(props)), "reason": "over_max_edge_scan",
                })
                continue
            unique_props, counts = np.unique(props, return_counts=True)
            for property_id, entity_count in zip(unique_props, counts):
                entity_count = int(entity_count)
                if entity_count < args.min_entities:
                    continue
                mask = props == property_id
                group_neighbours = np.asarray(neighbours[mask], dtype=np.int64)
                # Graph build deduplicates triples, so neighbours are distinct
                # for a fixed hub and PID. Keep the assertion local and cheap.
                if len(np.unique(group_neighbours)) != len(group_neighbours):
                    group_neighbours = np.unique(group_neighbours)
                    entity_count = len(group_neighbours)
                if entity_count < args.min_entities:
                    continue
                linked.update(int(value) for value in group_neighbours)
                relations.append({
                    "hub_qid": hub,
                    "direction": direction,
                    "pid": int(property_id),
                    "entity_count": entity_count,
                    "neighbours": group_neighbours,
                })
    return relations, skipped, linked


def annotate_relations(relations, node_records, type_labels, property_labels):
    """Build the auditable hub catalog and atomic typed observations."""
    relation_rows = []
    observations = []

    for relation in relations:
        hub = relation["hub_qid"]
        relation_key = (
            f'{qid(hub)}|{relation["direction"]}|{pid(relation["pid"])}')
        hub_record = node_records.get(hub, {})
        hub_types = hub_record.get("types", ()) or (None,)
        neighbour_types: dict[str | None, int] = defaultdict(int)
        example_rows = []
        rare_description_count = 0
        only_rare_description_count = 0
        for neighbour in relation["neighbours"]:
            record = node_records.get(int(neighbour), {})
            rare_description_count += int(record.get("has_rare_description", False))
            only_rare_description_count += int(
                record.get("has_only_rare_description", False))
            types = record.get("types", ()) or (None,)
            for value in types:
                neighbour_types[value] += 1
            if len(example_rows) < 5:
                example_rows.append({
                    "qid": qid(int(neighbour)),
                    "label": record.get("label", ""),
                    "types": list(types),
                })

        hub_type_values = list(hub_types)
        for linked_type, typed_count in sorted(
                neighbour_types.items(), key=lambda item: str(item[0])):
            if relation["direction"] == "out":
                source_types, target_types = hub_type_values, [linked_type]
            else:
                source_types, target_types = [linked_type], hub_type_values
            for source_type in source_types:
                for target_type in target_types:
                    observations.append({
                        "hub_qid": qid(hub),
                        "relation_key": relation_key,
                        "hub_direction": relation["direction"],
                        "source_type": source_type,
                        "pid": relation["pid"],
                        "target_type": target_type,
                        "typed_entity_count": int(typed_count),
                        "candidate_count": int(relation["entity_count"]),
                        "description_rare_lg_share": (
                            rare_description_count / relation["entity_count"]),
                        "only_description_rare_lg_share": (
                            only_rare_description_count / relation["entity_count"]),
                    })

        property_id = pid(relation["pid"])
        property_label = property_labels.get(property_id, "") or property_id
        hub_label = hub_record.get("label", "") or qid(hub)
        relation_text = (
            f"incoming --{property_label}--> {hub_label}"
            if relation["direction"] == "in"
            else f"{hub_label} --{property_label}--> outgoing"
        )
        relation_rows.append({
            "relation_key": relation_key,
            "hub_qid": qid(hub),
            "hub_label": hub_label,
            "relation": relation_text,
            "candidate_count": int(relation["entity_count"]),
            "description_rare_lg_share": (
                rare_description_count / relation["entity_count"]),
            "only_description_rare_lg_share": (
                only_rare_description_count / relation["entity_count"]),
            "examples": example_rows,
        })
    return relation_rows, observations


def percentile_ranks(values):
    """Return deterministic empirical percentile ranks in [0, 1]."""
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        rank = ((start + end - 1) / 2) / max(1, len(order) - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks.tolist()


def aggregate_atomic_patterns(observations, type_labels, property_labels,
                              volume_weight=0.3):
    """Aggregate exact ``source type × PID × target type`` signatures."""
    grouped = {}
    occurrence_frequency = {
        "source_type": defaultdict(int),
        "pid": defaultdict(int),
        "target_type": defaultdict(int),
    }
    entity_frequency = {
        "source_type": defaultdict(int),
        "pid": defaultdict(int),
        "target_type": defaultdict(int),
    }
    for observation in observations:
        key = (
            observation["source_type"],
            observation["pid"],
            observation["target_type"],
        )
        row = grouped.setdefault(key, {
            "source_type": observation["source_type"],
            "pid_number": observation["pid"],
            "target_type": observation["target_type"],
            "hub_qids": set(),
            "relation_keys": set(),
            "in_hub_qids": set(),
            "out_hub_qids": set(),
            "sizes": [],
            "typed_masses": [],
            "rare_description_shares": [],
            "only_rare_description_shares": [],
        })
        row["hub_qids"].add(observation["hub_qid"])
        row["relation_keys"].add(observation["relation_key"])
        row[f'{observation["hub_direction"]}_hub_qids'].add(
            observation["hub_qid"])
        row["sizes"].append(observation["candidate_count"])
        row["typed_masses"].append(observation["typed_entity_count"])
        row["rare_description_shares"].append(
            observation["description_rare_lg_share"])
        row["only_rare_description_shares"].append(
            observation["only_description_rare_lg_share"])
        for field in occurrence_frequency:
            value = observation[field]
            occurrence_frequency[field][value] += 1
            entity_frequency[field][value] += observation["typed_entity_count"]

    total_occurrences = max(1, len(observations))
    total_entity_mass = max(
        1, sum(observation["typed_entity_count"] for observation in observations))

    patterns = []
    for row in grouped.values():
        sizes = sorted(row["sizes"])
        source_type = row["source_type"]
        target_type = row["target_type"]
        property_number = row["pid_number"]
        source_label = type_labels.get(source_type) if source_type else None
        target_label = type_labels.get(target_type) if target_type else None
        property_id = pid(property_number)
        property_label = property_labels.get(property_id, "") or property_id
        component_rarity = []
        for field, value in (
            ("source_type", source_type),
            ("pid", property_number),
            ("target_type", target_type),
        ):
            occurrence_idf = math.log(
                (total_occurrences + 1)
                / (occurrence_frequency[field][value] + 1))
            entity_idf = math.log(
                (total_entity_mass + 1)
                / (entity_frequency[field][value] + 1))
            component_rarity.append((occurrence_idf + entity_idf) / 2)
        exact_occurrence_idf = math.log(
            (total_occurrences + 1) / (len(sizes) + 1))
        exact_typed_mass = sum(row["typed_masses"])
        exact_entity_idf = math.log(
            (total_entity_mass + 1) / (exact_typed_mass + 1))
        component_rarity.append(
            (exact_occurrence_idf + exact_entity_idf) / 2)
        patterns.append({
            "pattern_id": f"{source_type or 'NONE'}|{property_id}|{target_type or 'NONE'}",
            "pattern": (
                f"{source_label or source_type or 'untyped'} "
                f"--{property_label}--> "
                f"{target_label or target_type or 'untyped'}"
            ),
            "source_type": source_type,
            "source_type_label": source_label,
            "pid": property_id,
            "pid_label": property_label,
            "target_type": target_type,
            "target_type_label": target_label,
            "hub_count": len(row["hub_qids"]),
            "observation_count": len(sizes),
            "min_candidates_per_hub": sizes[0],
            "median_candidates_per_hub": float(np.median(sizes)),
            "max_candidates_per_hub": sizes[-1],
            "median_description_rare_lg_share": float(np.median(
                row["rare_description_shares"])),
            "median_only_description_rare_lg_share": float(np.median(
                row["only_rare_description_shares"])),
            "obscurity_raw": float(np.mean(component_rarity)),
            "relation_keys": sorted(row["relation_keys"]),
            "hub_qids": sorted(row["hub_qids"]),
            "in_hub_qids": sorted(row["in_hub_qids"]),
            "out_hub_qids": sorted(row["out_hub_qids"]),
        })
    volume_scores = percentile_ranks([
        math.log1p(row["median_candidates_per_hub"]) for row in patterns
    ])
    obscurity_scores = percentile_ranks([
        row["obscurity_raw"] for row in patterns
    ])
    for row, volume_score, obscurity_score in zip(
            patterns, volume_scores, obscurity_scores):
        row["volume_score"] = volume_score
        row["obscurity_score"] = obscurity_score
        row["selection_score"] = (
            volume_weight * volume_score
            + (1 - volume_weight) * obscurity_score
        )

    return sorted(patterns, key=lambda row: (
        -row["selection_score"], -row["hub_count"],
        str(row["source_type"]), row["pid"], str(row["target_type"])))


def write_table(path: Path, rows: list[dict]) -> None:
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    else:
        pq.write_table(pa.table({"empty": pa.array([], type=pa.string())}), path)


def write_outputs(output: Path, rows, atomic_patterns, selected_patterns,
                  skipped, metadata, force: bool):
    if output.exists() and not force:
        raise FileExistsError(f"output exists; use --force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_table(temporary / "hub_relations.parquet", rows)
        write_table(temporary / "atomic_relation_patterns.parquet", atomic_patterns)
        write_table(temporary / "relation_patterns.parquet", selected_patterns)
        (temporary / "skipped.json").write_text(
            json.dumps(skipped, ensure_ascii=False, indent=2))
        (temporary / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2))
        if output.exists():
            if not force:
                raise FileExistsError(f"output exists; use --force: {output}")
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-hubs", type=int, default=500)
    parser.add_argument("--min-degree", type=int, default=50)
    parser.add_argument("--max-degree", type=int, default=100_000)
    parser.add_argument("--max-edge-scan", type=int, default=100_000)
    parser.add_argument("--min-entities", type=int, default=30)
    parser.add_argument("--min-median-candidates", type=float, default=0)
    parser.add_argument("--max-median-candidates", type=float, default=float("inf"))
    parser.add_argument("--volume-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_hubs < 3 or args.min_degree < 1:
        raise SystemExit("max-hubs must be >= 3 and min-degree must be positive")
    if not 0 <= args.volume_weight <= 1:
        raise SystemExit("volume-weight must be between 0 and 1")
    started = time.monotonic()
    nodes = NodeIndex(args.graph_dir / "nodes.parquet")
    edges = EdgeIndex(args.graph_dir)
    property_labels = load_property_labels(args.graph_dir / "properties.json")

    hubs = select_hubs(nodes, args)
    relations, skipped, linked = collect_relations(edges, hubs, args)
    records = nodes.lookup(linked)
    type_ids = {
        value
        for record in records.values()
        for value in record.get("types", ())
    }
    type_records = nodes.lookup({int(value[1:]) for value in type_ids
                                 if value.startswith("Q") and value[1:].isdigit()})
    type_labels = {
        qid(value): record.get("label", "")
        for value, record in type_records.items()
    }
    relation_rows, observations = annotate_relations(
        relations, records, type_labels, property_labels)
    atomic_patterns = aggregate_atomic_patterns(
        observations, type_labels, property_labels, args.volume_weight)
    selected_patterns = [
        {
            key: row[key]
            for key in (
                "pattern_id", "pattern", "hub_count",
                "median_candidates_per_hub",
                "median_description_rare_lg_share",
                "median_only_description_rare_lg_share",
                "volume_score", "obscurity_score", "selection_score",
                "relation_keys",
            )
        }
        for row in atomic_patterns
        if args.min_median_candidates
        <= row["median_candidates_per_hub"]
        <= args.max_median_candidates
    ]
    metadata = {
        "max_hubs": args.max_hubs,
        "selected_hubs": len(hubs),
        "relation_groups": len(relation_rows),
        "atomic_patterns": len(atomic_patterns),
        "selected_patterns": len(selected_patterns),
        "min_median_candidates": args.min_median_candidates,
        "max_median_candidates": args.max_median_candidates,
        "volume_weight": args.volume_weight,
        "linked_qids": len(linked),
        "skipped": len(skipped),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "seed": args.seed,
        "min_degree": args.min_degree,
        "max_degree": args.max_degree,
        "max_edge_scan": args.max_edge_scan,
        "min_entities": args.min_entities,
    }
    write_outputs(args.output_dir, relation_rows, atomic_patterns,
                  selected_patterns, skipped, metadata, args.force)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
