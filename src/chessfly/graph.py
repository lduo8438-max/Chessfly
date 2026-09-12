"""Streaming validation for the large MaleCNS connection-weight table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


CONNECTION_COLUMNS = ("body_pre", "body_post", "weight")


@dataclass(frozen=True)
class ConnectionSummary:
    raw_edges: int
    retained_neurons: int
    retained_edges: int
    retained_synaptic_contacts: int
    record_batches: int


def summarize_connection_file(
    path: Path, retained_body_ids: Iterable[int]
) -> ConnectionSummary:
    retained = tuple(sorted(set(retained_body_ids)))
    if not retained:
        raise ValueError("retained neuron set cannot be empty")
    value_set = pa.array(retained, type=pa.int64())
    raw_edges = 0
    retained_edges = 0
    retained_contacts = 0

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        missing = set(CONNECTION_COLUMNS) - set(reader.schema.names)
        if missing:
            raise ValueError(f"connection schema is missing: {sorted(missing)}")
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            raw_edges += batch.num_rows
            keep = pc.and_(
                pc.is_in(batch["body_pre"], value_set=value_set),
                pc.is_in(batch["body_post"], value_set=value_set),
            )
            kept_count = pc.sum(pc.cast(keep, pa.int64())).as_py() or 0
            retained_edges += int(kept_count)
            kept_weights = pc.filter(batch["weight"], keep)
            contact_count = pc.sum(kept_weights).as_py() or 0
            retained_contacts += int(contact_count)
        batch_count = reader.num_record_batches

    return ConnectionSummary(
        raw_edges,
        len(retained),
        retained_edges,
        retained_contacts,
        batch_count,
    )

