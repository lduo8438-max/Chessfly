"""Validated MaleCNS annotation access without pandas at runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .dataset import DATASET_FILES


ANNOTATION_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "superclass",
    "somaSide",
    "status",
)
TRANSMITTER_COLUMNS = (
    "body",
    "predicted_nt_confidence",
    "predicted_nt",
    "consensus_nt",
)


@dataclass(frozen=True)
class AnnotationSummary:
    neurons: int
    descending_neurons: int
    r1_r6_candidates: int
    r8_candidates: int
    transmitter_rows: int


def load_source_tables(data_dir: Path) -> tuple[pa.Table, pa.Table]:
    by_key = {item.key: item for item in DATASET_FILES}
    annotation_path = data_dir / "raw" / by_key["annotations"].filename
    transmitter_path = data_dir / "raw" / by_key["transmitters"].filename
    for path in (annotation_path, transmitter_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run chessfly prepare --files annotations "
                "transmitters --download"
            )
    annotations = feather.read_table(annotation_path, columns=ANNOTATION_COLUMNS)
    transmitters = feather.read_table(transmitter_path, columns=TRANSMITTER_COLUMNS)
    _require_columns(annotations, ANNOTATION_COLUMNS, "annotations")
    _require_columns(transmitters, TRANSMITTER_COLUMNS, "transmitters")
    return annotations, transmitters


def summarize_annotations(
    annotations: pa.Table, transmitters: pa.Table
) -> AnnotationSummary:
    superclass = annotations["superclass"]
    cell_type = pc.fill_null(annotations["type"], "")
    descending = pc.sum(
        pc.cast(pc.equal(superclass, "descending_neuron"), pa.int64())
    ).as_py()
    r1_r6 = pc.sum(pc.cast(pc.equal(cell_type, "R1-R6"), pa.int64())).as_py()
    r8_mask = pc.match_substring_regex(cell_type, r"^R8($|[a-zA-Z_])")
    r8 = pc.sum(pc.cast(r8_mask, pa.int64())).as_py()
    return AnnotationSummary(
        annotations.num_rows,
        int(descending or 0),
        int(r1_r6 or 0),
        int(r8 or 0),
        transmitters.num_rows,
    )


def descending_body_ids(annotations: pa.Table) -> tuple[int, ...]:
    mask = pc.equal(annotations["superclass"], "descending_neuron")
    selected = pc.filter(annotations["bodyId"], mask)
    return tuple(int(value) for value in selected.to_pylist())


def retained_neuron_body_ids(annotations: pa.Table) -> tuple[int, ...]:
    """Return every body with an assigned neuronal superclass."""
    mask = pc.is_valid(annotations["superclass"])
    selected = pc.filter(annotations["bodyId"], mask)
    return tuple(int(value) for value in selected.to_pylist())


def visual_candidate_body_ids(annotations: pa.Table) -> tuple[int, ...]:
    cell_type = pc.fill_null(annotations["type"], "")
    r1_r6 = pc.equal(cell_type, "R1-R6")
    r8 = pc.match_substring_regex(cell_type, r"^R8($|[a-zA-Z_])")
    selected = pc.filter(annotations["bodyId"], pc.or_(r1_r6, r8))
    return tuple(int(value) for value in selected.to_pylist())


def _require_columns(table: pa.Table, required: tuple[str, ...], label: str) -> None:
    missing = set(required) - set(table.column_names)
    if missing:
        raise ValueError(f"{label} schema is missing columns: {sorted(missing)}")
