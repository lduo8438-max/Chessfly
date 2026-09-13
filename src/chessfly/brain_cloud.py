"""Build a 3D point cloud of MaleCNS neurons at their recorded soma positions.

Every annotated body with a `somaLocation` (or, failing that, a
`tosomaLocation`) becomes one point, so the cloud has the real shape of the male
central nervous system.  Points are grouped the way FlyJack colours its brain.
The simulated reference subgraph is indexed into the cloud so recorded spikes
can light the exact cells that fired; simulated cells without a recorded
position -- mostly photoreceptors, whose somata lie outside the imaged volume --
are counted and reported, never placed by guesswork.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from .dataset import select_files


GROUPS = ("central", "mushroom body", "descending", "optic", "sensory", "other")
GROUP_COLOURS = {
    "central": "#3987e5",
    "mushroom body": "#d95926",
    "descending": "#199e70",
    "optic": "#c3c2b7",
    "sensory": "#c3c2b7",
    "other": "#c3c2b7",
}
MUSHROOM_BODY_PREFIXES = ("KC", "MBON", "PAM", "PPL", "APL", "DPM")
VOXEL_NM = 8.0
DEFAULT_SUBGRAPH = "mapped-retina-to-descending-h3-w5"


def group_of(superclass: str | None, cell_type: str | None) -> int:
    """Assign a body to one of the FlyJack-style display groups."""
    if cell_type and cell_type.startswith(MUSHROOM_BODY_PREFIXES):
        return GROUPS.index("mushroom body")
    name = superclass or ""
    if name.startswith("descending_neuron"):
        return GROUPS.index("descending")
    if "sensory" in name:
        return GROUPS.index("sensory")
    if name.startswith(("ol_", "visual_")):
        return GROUPS.index("optic")
    if name.startswith("cb_"):
        return GROUPS.index("central")
    return GROUPS.index("other")


def build_brain_cloud(
    data_dir: Path, subgraph_name: str = DEFAULT_SUBGRAPH
) -> dict[str, object]:
    data_dir = Path(data_dir)
    annotations_file = data_dir / "raw" / select_files(("annotations",))[0].filename
    table = feather.read_table(
        annotations_file,
        columns=["bodyId", "somaLocation", "tosomaLocation", "superclass", "type"],
    )
    body_ids = np.asarray(table["bodyId"].to_numpy(), dtype=np.int64)
    soma = table["somaLocation"].to_pylist()
    tosoma = table["tosomaLocation"].to_pylist()
    superclass = table["superclass"].to_pylist()
    cell_type = table["type"].to_pylist()

    keep, positions, groups, from_tosoma = [], [], [], []
    for row, (primary, fallback) in enumerate(zip(soma, tosoma)):
        location = primary if primary is not None else fallback
        if location is None or len(location) != 3:
            continue
        keep.append(row)
        positions.append(location)
        groups.append(group_of(superclass[row], cell_type[row]))
        from_tosoma.append(primary is None)
    if not keep:
        raise ValueError("no annotated body carries a soma position")

    cloud_body_ids = body_ids[np.asarray(keep)]
    order = np.argsort(cloud_body_ids)
    cloud_body_ids = cloud_body_ids[order]
    positions_nm = np.asarray(positions, dtype=np.float64)[order] * VOXEL_NM
    groups_array = np.asarray(groups, dtype=np.uint8)[order]
    tosoma_array = np.asarray(from_tosoma, dtype=bool)[order]

    node_ids = np.load(data_dir / "compiled" / subgraph_name / "node_ids.npy")
    slots = np.searchsorted(cloud_body_ids, node_ids)
    inside = slots < len(cloud_body_ids)
    matched = np.zeros(len(node_ids), dtype=bool)
    matched[inside] = cloud_body_ids[slots[inside]] == node_ids[inside]
    simulated_local = np.full(len(cloud_body_ids), -1, dtype=np.int32)
    simulated_local[slots[matched]] = np.flatnonzero(matched).astype(np.int32)

    target = data_dir / "compiled" / "brain-cloud"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "cloud.npz",
        body_ids=cloud_body_ids,
        positions_nm=positions_nm.astype(np.float32),
        groups=groups_array,
        from_tosoma=tosoma_array,
        simulated_local=simulated_local,
    )
    summary = {
        "schema_version": 1,
        "source": annotations_file.name,
        "subgraph": subgraph_name,
        "voxel_nm": VOXEL_NM,
        "points": int(len(cloud_body_ids)),
        "points_from_tosoma": int(tosoma_array.sum()),
        "groups": list(GROUPS),
        "group_colours": GROUP_COLOURS,
        "group_counts": {
            name: int(np.count_nonzero(groups_array == index))
            for index, name in enumerate(GROUPS)
        },
        "simulated_neurons": int(len(node_ids)),
        "simulated_with_position": int(matched.sum()),
        "simulated_without_position": int(len(node_ids) - matched.sum()),
        "bounds_nm": {
            "min": [float(value) for value in positions_nm.min(axis=0)],
            "max": [float(value) for value in positions_nm.max(axis=0)],
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
