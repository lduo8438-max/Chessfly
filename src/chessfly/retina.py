"""Infer an explicit MaleCNS display projection from annotated optic columns."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image


LUMINANCE = 0
GREEN = 1
BLUE = 2


@dataclass(frozen=True)
class RetinaProjectionSummary:
    r1_r6_candidates: int
    r1_r6_mapped: int
    r1_r6_unmapped: int
    r8p_candidates: int
    r8p_mapped: int
    r8y_candidates: int
    r8y_mapped: int
    total_mapped: int
    confidence_median: float
    confidence_below_80_percent: int


def build_retina_projection(
    annotations: pa.Table, compiled_dir: Path, output_dir: Path
) -> RetinaProjectionSummary:
    """Map R1-R6, R8p and R8y bodies to normalized display coordinates."""
    if output_dir.exists():
        raise FileExistsError(f"retina projection already exists: {output_dir}")
    nodes = np.load(compiled_dir / "node_ids.npy", mmap_mode="r")
    offsets = np.load(compiled_dir / "forward_offsets.npy", mmap_mode="r")
    targets = np.load(compiled_dir / "forward_targets.npy", mmap_mode="r")
    weights = np.load(compiled_dir / "forward_weights.npy", mmap_mode="r")
    aligned = _align_annotations(annotations, nodes)
    cell_type = aligned["type"]
    sides = aligned["rootSide"]
    hex1 = aligned["assignedOlHex1"]
    hex2 = aligned["assignedOlHex2"]

    r1_candidates = np.flatnonzero(cell_type == "R1-R6")
    anchors = np.isin(cell_type, ("L1", "L2", "L3")) & np.isfinite(hex1) & np.isfinite(hex2)
    r1_index, r1_hex, r1_confidence = _modal_columns(
        r1_candidates, anchors, offsets, targets, weights, hex1, hex2, sides
    )
    r1_uv = _project_eye_coordinates(r1_hex, sides[r1_index])

    r8_candidates = np.flatnonzero(np.isin(cell_type, ("R8p", "R8y")))
    column_targets = np.isfinite(hex1) & np.isfinite(hex2)
    r8_index, r8_hex, r8_confidence = _modal_columns(
        r8_candidates,
        column_targets,
        offsets,
        targets,
        weights,
        hex1,
        hex2,
        sides,
    )
    r8_uv = _project_eye_coordinates(
        r8_hex,
        sides[r8_index],
        reference_hexes=r1_hex,
        reference_sides=sides[r1_index],
    )

    indices = np.concatenate((r1_index, r8_index)).astype(np.int32)
    uv = np.concatenate((r1_uv, r8_uv)).astype(np.float32)
    confidence = np.concatenate((r1_confidence, r8_confidence)).astype(np.float32)
    channels = np.concatenate(
        (
            np.full(len(r1_index), LUMINANCE, dtype=np.uint8),
            np.where(cell_type[r8_index] == "R8p", BLUE, GREEN).astype(np.uint8),
        )
    )
    r8p_candidates = int(np.count_nonzero(cell_type == "R8p"))
    r8y_candidates = int(np.count_nonzero(cell_type == "R8y"))
    summary = RetinaProjectionSummary(
        len(r1_candidates),
        len(r1_index),
        len(r1_candidates) - len(r1_index),
        r8p_candidates,
        int(np.count_nonzero(cell_type[r8_index] == "R8p")),
        r8y_candidates,
        int(np.count_nonzero(cell_type[r8_index] == "R8y")),
        len(indices),
        float(np.median(confidence)),
        int(np.count_nonzero(confidence < 0.8)),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-building-"
    ) as temp_name:
        build = Path(temp_name)
        np.save(build / "neuron_index.npy", indices)
        np.save(build / "body_id.npy", np.asarray(nodes[indices]))
        np.save(build / "uv.npy", uv)
        np.save(build / "channel.npy", channels)
        np.save(build / "confidence.npy", confidence)
        (build / "manifest.json").write_text(
            json.dumps(
                {
                    **asdict(summary),
                    "model": "modal-column-overlapping-viewports-v1",
                    "channels": {
                        "0": "linear-sRGB luminance proxy for R1-R6",
                        "1": "linear-sRGB green proxy for R8y",
                        "2": "linear-sRGB blue proxy for R8p",
                    },
                    "validated_physiology": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(build, output_dir)
    return summary


def load_retina_body_ids(projection_dir: Path) -> tuple[int, ...]:
    values = np.load(projection_dir / "body_id.npy", mmap_mode="r")
    return tuple(int(value) for value in values)


def sample_retina_stimulus(
    frame: Image.Image, projection_dir: Path
) -> np.ndarray:
    """Sample linear-sRGB luminance/green/blue values at mapped receptors."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    uv = np.load(projection_dir / "uv.npy", mmap_mode="r")
    channels = np.load(projection_dir / "channel.npy", mmap_mode="r")
    height, width = rgb.shape[:2]
    x = np.minimum((uv[:, 0] * (width - 1)).astype(int), width - 1)
    y = np.minimum((uv[:, 1] * (height - 1)).astype(int), height - 1)
    srgb = rgb[y, x].astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = (
        0.2126 * linear[:, 0] + 0.7152 * linear[:, 1] + 0.0722 * linear[:, 2]
    )
    values = luminance
    values = np.where(channels == GREEN, linear[:, 1], values)
    values = np.where(channels == BLUE, linear[:, 2], values)
    return np.asarray(values, dtype=np.float32)


def _align_annotations(annotations: pa.Table, node_ids: np.ndarray) -> dict[str, np.ndarray]:
    annotation_ids = np.asarray(annotations["bodyId"].to_numpy(), dtype=np.int64)
    order = np.argsort(annotation_ids)
    sorted_ids = annotation_ids[order]
    positions = np.searchsorted(sorted_ids, node_ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], node_ids
    ):
        raise ValueError("compiled node IDs are not fully represented in annotations")
    rows = order[positions]
    result = {}
    for name in ("type", "rootSide"):
        values = np.asarray(
            [value if value is not None else "" for value in annotations[name].to_pylist()],
            dtype=object,
        )
        result[name] = values[rows]
    for name in ("assignedOlHex1", "assignedOlHex2"):
        values = np.asarray(annotations[name].to_numpy(zero_copy_only=False), dtype=float)
        result[name] = values[rows]
    return result


def _modal_columns(
    candidates: np.ndarray,
    valid_targets: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    hex1: np.ndarray,
    hex2: np.ndarray,
    sides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped = []
    hexes = []
    confidence = []
    for source in candidates:
        if sides[source] not in ("L", "R"):
            continue
        start, end = int(offsets[source]), int(offsets[source + 1])
        edge_targets = targets[start:end]
        keep = valid_targets[edge_targets]
        if not np.any(keep):
            continue
        selected_targets = edge_targets[keep]
        selected_weights = weights[start:end][keep]
        pairs = np.column_stack((hex1[selected_targets], hex2[selected_targets]))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        votes = np.bincount(inverse, weights=selected_weights)
        winner = int(np.argmax(votes))
        mapped.append(source)
        hexes.append(unique[winner])
        confidence.append(float(votes[winner] / np.sum(votes)))
    return (
        np.asarray(mapped, dtype=np.int32),
        np.asarray(hexes, dtype=np.float64),
        np.asarray(confidence, dtype=np.float32),
    )


def _project_eye_coordinates(
    hexes: np.ndarray,
    sides: np.ndarray,
    reference_hexes: np.ndarray | None = None,
    reference_sides: np.ndarray | None = None,
) -> np.ndarray:
    if not len(hexes):
        return np.empty((0, 2), dtype=np.float32)
    xy = np.column_stack(
        (hexes[:, 0] - 0.5 * hexes[:, 1], np.sqrt(3) / 2 * hexes[:, 1])
    )
    reference_hexes = hexes if reference_hexes is None else reference_hexes
    reference_sides = sides if reference_sides is None else reference_sides
    reference_xy = np.column_stack(
        (
            reference_hexes[:, 0] - 0.5 * reference_hexes[:, 1],
            np.sqrt(3) / 2 * reference_hexes[:, 1],
        )
    )
    uv = np.empty_like(xy)
    for side in ("L", "R"):
        select = sides == side
        reference = reference_xy[reference_sides == side]
        if not len(reference):
            raise ValueError(f"retina projection has no {side} reference anchors")
        low = np.min(reference, axis=0)
        span = np.ptp(reference, axis=0)
        span[span == 0] = 1.0
        normalized = (xy[select] - low) / span
        if side == "L":
            uv[select, 0] = 0.60 * normalized[:, 0]
        else:
            uv[select, 0] = 0.40 + 0.60 * (1 - normalized[:, 0])
        uv[select, 1] = 1 - normalized[:, 1]
    return np.clip(uv, 0, 1).astype(np.float32)
