"""Disk-backed retained graph compilation and path-subgraph extraction."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


@dataclass(frozen=True)
class CompiledGraphSummary:
    neurons: int
    edges: int
    synaptic_contacts: int
    minimum_weight: int


@dataclass(frozen=True)
class PathSubgraphSummary:
    neurons: int
    edges: int
    synaptic_contacts: int
    visual_inputs: int
    descending_outputs: int
    maximum_hops: int
    minimum_weight: int


def compile_retained_csr(
    weights_path: Path,
    retained_body_ids: Iterable[int],
    output_dir: Path,
    minimum_weight: int = 1,
) -> CompiledGraphSummary:
    """Compile a large body-ID edge table into forward and reverse CSR arrays."""
    if minimum_weight < 1:
        raise ValueError("minimum_weight must be at least 1")
    if output_dir.exists():
        raise FileExistsError(f"compiled graph already exists: {output_dir}")
    nodes = np.asarray(sorted(set(retained_body_ids)), dtype=np.int64)
    if not len(nodes):
        raise ValueError("retained neuron set cannot be empty")
    value_set = pa.array(nodes, type=pa.int64())

    edge_count = sum(
        len(pre)
        for pre, _, _ in _retained_batches(weights_path, value_set, minimum_weight)
    )
    if edge_count == 0:
        raise ValueError("retained graph has no edges")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-building-"
    ) as temp_name:
        build_dir = Path(temp_name)
        np.save(build_dir / "node_ids.npy", nodes)
        src = np.lib.format.open_memmap(
            build_dir / "source_index.npy", mode="w+", dtype=np.int32, shape=(edge_count,)
        )
        dst = np.lib.format.open_memmap(
            build_dir / "target_index.npy", mode="w+", dtype=np.int32, shape=(edge_count,)
        )
        weights = np.lib.format.open_memmap(
            build_dir / "edge_weight.npy", mode="w+", dtype=np.int32, shape=(edge_count,)
        )
        cursor = 0
        for pre, post, weight in _retained_batches(
            weights_path, value_set, minimum_weight
        ):
            size = len(pre)
            src[cursor : cursor + size] = np.searchsorted(nodes, pre).astype(np.int32)
            dst[cursor : cursor + size] = np.searchsorted(nodes, post).astype(np.int32)
            weights[cursor : cursor + size] = weight.astype(np.int32)
            cursor += size
        src.flush()
        dst.flush()
        weights.flush()
        contacts = int(np.sum(weights, dtype=np.int64))

        _write_csr(build_dir, "forward", src, dst, weights, len(nodes))
        _write_csr(build_dir, "reverse", dst, src, weights, len(nodes))
        del src, dst, weights
        for name in ("source_index.npy", "target_index.npy", "edge_weight.npy"):
            (build_dir / name).unlink()

        summary = CompiledGraphSummary(
            len(nodes), edge_count, contacts, minimum_weight
        )
        (build_dir / "manifest.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(build_dir, output_dir)
    return summary


def build_path_subgraph(
    compiled_dir: Path,
    visual_body_ids: Iterable[int],
    descending_body_ids: Iterable[int],
    output_dir: Path,
    maximum_hops: int = 8,
    minimum_weight: int = 5,
) -> PathSubgraphSummary:
    """Keep nodes lying on a visual-to-descending path within ``maximum_hops``."""
    if maximum_hops < 1 or minimum_weight < 1:
        raise ValueError("path limits must be positive")
    if output_dir.exists():
        raise FileExistsError(f"subgraph already exists: {output_dir}")
    nodes = np.load(compiled_dir / "node_ids.npy", mmap_mode="r")
    visual = _body_ids_to_indices(nodes, visual_body_ids)
    descending = _body_ids_to_indices(nodes, descending_body_ids)
    if not len(visual) or not len(descending):
        raise ValueError("visual inputs and descending outputs must both be present")

    f_offsets, f_targets, f_weights = _load_csr(compiled_dir, "forward")
    r_offsets, r_targets, r_weights = _load_csr(compiled_dir, "reverse")
    forward_distance = _shortest_distances(
        f_offsets, f_targets, f_weights, visual, maximum_hops, minimum_weight
    )
    reverse_distance = _shortest_distances(
        r_offsets, r_targets, r_weights, descending, maximum_hops, minimum_weight
    )
    selected = (
        (forward_distance >= 0)
        & (reverse_distance >= 0)
        & (forward_distance + reverse_distance <= maximum_hops)
    )
    selected_indices = np.flatnonzero(selected)
    if not len(selected_indices):
        raise ValueError("no visual-to-descending paths satisfy the configured limits")

    local_index = np.full(len(nodes), -1, dtype=np.int32)
    local_index[selected_indices] = np.arange(len(selected_indices), dtype=np.int32)
    source_parts = []
    target_parts = []
    weight_parts = []
    for source in selected_indices:
        start, end = int(f_offsets[source]), int(f_offsets[source + 1])
        targets = f_targets[start:end]
        weights = f_weights[start:end]
        keep = selected[targets] & (weights >= minimum_weight)
        kept_targets = targets[keep]
        if len(kept_targets):
            source_parts.append(
                np.full(len(kept_targets), local_index[source], dtype=np.int32)
            )
            target_parts.append(local_index[kept_targets])
            weight_parts.append(np.asarray(weights[keep], dtype=np.int32))
    if not source_parts:
        raise ValueError("selected subgraph contains no edges")
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    weights = np.concatenate(weight_parts)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-building-"
    ) as temp_name:
        build_dir = Path(temp_name)
        np.save(build_dir / "node_ids.npy", np.asarray(nodes[selected_indices]))
        np.save(build_dir / "source_index.npy", sources)
        np.save(build_dir / "target_index.npy", targets)
        np.save(build_dir / "edge_weight.npy", weights)
        summary = PathSubgraphSummary(
            len(selected_indices),
            len(sources),
            int(np.sum(weights, dtype=np.int64)),
            int(np.count_nonzero(selected[visual])),
            int(np.count_nonzero(selected[descending])),
            maximum_hops,
            minimum_weight,
        )
        (build_dir / "manifest.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(build_dir, output_dir)
    return summary


def _retained_batches(
    path: Path, value_set: pa.Array, minimum_weight: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            keep = pc.and_(
                pc.and_(
                    pc.is_in(batch["body_pre"], value_set=value_set),
                    pc.is_in(batch["body_post"], value_set=value_set),
                ),
                pc.greater_equal(batch["weight"], minimum_weight),
            )
            if not (pc.any(keep).as_py() or False):
                continue
            yield (
                pc.filter(batch["body_pre"], keep).to_numpy(zero_copy_only=False),
                pc.filter(batch["body_post"], keep).to_numpy(zero_copy_only=False),
                pc.filter(batch["weight"], keep).to_numpy(zero_copy_only=False),
            )


def _write_csr(
    directory: Path,
    prefix: str,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    neuron_count: int,
) -> None:
    order = np.argsort(sources, kind="stable")
    counts = np.bincount(sources, minlength=neuron_count)
    offsets = np.empty(neuron_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    np.save(directory / f"{prefix}_offsets.npy", offsets)
    ordered_targets = np.lib.format.open_memmap(
        directory / f"{prefix}_targets.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(order),),
    )
    ordered_weights = np.lib.format.open_memmap(
        directory / f"{prefix}_weights.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(order),),
    )
    ordered_targets[:] = targets[order]
    ordered_weights[:] = weights[order]
    ordered_targets.flush()
    ordered_weights.flush()


def _load_csr(
    directory: Path, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(directory / f"{prefix}_offsets.npy", mmap_mode="r"),
        np.load(directory / f"{prefix}_targets.npy", mmap_mode="r"),
        np.load(directory / f"{prefix}_weights.npy", mmap_mode="r"),
    )


def _body_ids_to_indices(nodes: np.ndarray, body_ids: Iterable[int]) -> np.ndarray:
    requested = np.asarray(sorted(set(body_ids)), dtype=np.int64)
    if not len(requested):
        return np.asarray([], dtype=np.int32)
    positions = np.searchsorted(nodes, requested)
    valid = positions < len(nodes)
    matched = np.zeros(len(requested), dtype=bool)
    matched[valid] = nodes[positions[valid]] == requested[valid]
    return positions[matched].astype(np.int32)


def _shortest_distances(
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    seeds: np.ndarray,
    maximum_hops: int,
    minimum_weight: int,
) -> np.ndarray:
    distance = np.full(len(offsets) - 1, -1, dtype=np.int16)
    frontier = np.unique(seeds)
    distance[frontier] = 0
    for depth in range(1, maximum_hops + 1):
        parts = []
        for source in frontier:
            start, end = int(offsets[source]), int(offsets[source + 1])
            if end > start:
                keep = weights[start:end] >= minimum_weight
                if np.any(keep):
                    parts.append(np.asarray(targets[start:end][keep]))
        if not parts:
            break
        neighbors = np.unique(np.concatenate(parts))
        frontier = neighbors[distance[neighbors] < 0]
        if not len(frontier):
            break
        distance[frontier] = depth
    return distance
