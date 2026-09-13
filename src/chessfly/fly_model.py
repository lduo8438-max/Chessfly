"""Fetch the NeuroMechFly body model from a pinned FlyGym wheel.

The meshes, kinematic tree, standing pose and part colours come from FlyGym
1.1.0 (Apache-2.0, NeuroMechFly, Lobato-Rios et al. 2022; Wang-Chen et al.
2024).  Nothing is copied into the repository: the wheel is downloaded, checked
against a pinned SHA-256, and only the files the renderer needs are extracted.
The pose and appearance YAML are converted to JSON because neither this
project's environment nor Blender's bundled Python ships a YAML parser.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from .dataset import DatasetFile, download_file, sha256_file


FLYGYM_WHEEL = DatasetFile(
    "flygym-wheel",
    "flygym-1.1.0-py3-none-any.whl",
    "https://files.pythonhosted.org/packages/60/5b/"
    "6069ca27a019cb73cbdf849fdb065ca0317c1c8ba28aa879f28921ead95e/"
    "flygym-1.1.0-py3-none-any.whl",
    23_342_775,
    "NeuroMechFly meshes, kinematic tree, standing pose and colours (Apache-2.0)",
)
FLYGYM_SHA256 = "ca37e104dc0c6bd1a0f3504070b73b01c1e080837456ca98a2a4777a5ae947a4"

MJCF_MEMBER = "flygym/data/mjcf/neuromechfly_seqik_kinorder_ypr.xml"
POSE_MEMBER = "flygym/data/pose/pose_tripod.yaml"
CONFIG_MEMBER = "flygym/config.yaml"
LICENSE_MEMBER = "flygym-1.1.0.dist-info/LICENSE"
MESH_PREFIX = "flygym/data/mesh/"

_POSE_LINE = re.compile(r"^\s+(joint_\w+)\s*:\s*(-?[0-9.eE+-]+)\s*$")
_NUMBERS = r"\[\s*([^\]]*?)\s*\]"


def parse_pose(text: str) -> dict[str, float]:
    """Read joint angles, in degrees, from a FlyGym pose YAML."""
    joints = {}
    for line in text.splitlines():
        match = _POSE_LINE.match(line)
        if match:
            joints[match.group(1)] = float(match.group(2))
    if not joints:
        raise ValueError("pose file holds no joint angles")
    return joints


def parse_appearance(text: str) -> dict[str, dict[str, object]]:
    """Resolve each body part's RGBA from FlyGym's appearance block.

    Textured parts multiply a white material by the texture's `rgb1`, so their
    colour is `rgb1` with the material's alpha; untextured parts use `rgba`.
    """
    start = text.find("\nappearance:")
    if text.startswith("appearance:"):
        start = 0
    if start < 0:
        raise ValueError("config holds no appearance block")
    block = text[start:].lstrip("\n")
    lines = block.splitlines()[1:]
    body = []
    for line in lines:
        if line and not line.startswith(" "):
            break
        body.append(line)
    groups: dict[str, list[str]] = {}
    current = None
    for line in body:
        header = re.match(r"^  ([A-Za-z_]\w*):\s*$", line)
        if header:
            current = header.group(1)
            groups[current] = []
        elif current is not None:
            groups[current].append(line)

    parts: dict[str, dict[str, object]] = {}
    for name, group_lines in groups.items():
        chunk = "\n".join(group_lines)
        targets = re.search(r"apply_to:\s*" + _NUMBERS, chunk, re.S)
        rgba = re.search(r"rgba:\s*" + _NUMBERS, chunk)
        rgb1 = re.search(r"rgb1:\s*" + _NUMBERS, chunk)
        if not targets or not rgba:
            continue
        names = re.findall(r'"([^"]+)"', targets.group(1))
        material = [float(value) for value in rgba.group(1).split(",")]
        colour = (
            [float(value) for value in rgb1.group(1).split(",")]
            if rgb1
            else material[:3]
        )
        for part in names:
            parts[part] = {"group": name, "rgba": colour[:3] + [material[3]]}
    if not parts:
        raise ValueError("appearance block assigns no colours")
    return parts


def prepare_fly_model(data_dir: Path) -> dict[str, object]:
    """Download, verify and unpack the NeuroMechFly assets the renderer uses."""
    data_dir = Path(data_dir)
    record = download_file(FLYGYM_WHEEL, data_dir / "raw")
    if record["sha256"] != FLYGYM_SHA256:
        raise ValueError(
            f"FlyGym wheel hash {record['sha256']} does not match the pinned "
            f"{FLYGYM_SHA256}; refusing to use it"
        )
    target = data_dir / "assets" / "neuromechfly"
    target.parent.mkdir(parents=True, exist_ok=True)
    wheel = data_dir / "raw" / FLYGYM_WHEEL.filename

    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".neuromechfly-"))
    try:
        with zipfile.ZipFile(wheel) as archive:
            meshes = sorted(
                name
                for name in archive.namelist()
                if name.startswith(MESH_PREFIX) and name.lower().endswith(".stl")
            )
            if not meshes:
                raise ValueError("wheel contains no NeuroMechFly meshes")
            (staging / "mesh").mkdir()
            for member in meshes:
                (staging / "mesh" / Path(member).name).write_bytes(archive.read(member))
            (staging / "neuromechfly.xml").write_bytes(archive.read(MJCF_MEMBER))
            (staging / "LICENSE").write_bytes(archive.read(LICENSE_MEMBER))
            pose = parse_pose(archive.read(POSE_MEMBER).decode("utf-8"))
            appearance = parse_appearance(archive.read(CONFIG_MEMBER).decode("utf-8"))
        (staging / "pose.json").write_text(
            json.dumps({"units": "degrees", "joints": pose}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (staging / "appearance.json").write_text(
            json.dumps(appearance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "source": FLYGYM_WHEEL.url,
            "wheel_sha256": FLYGYM_SHA256,
            "license": "Apache-2.0",
            "attribution": (
                "NeuroMechFly body model via FlyGym 1.1.0 "
                "(Lobato-Rios et al. 2022; Wang-Chen et al. 2024)"
            ),
            "mjcf": Path(MJCF_MEMBER).name,
            "pose": Path(POSE_MEMBER).name,
            "meshes": len(meshes),
            "files": {
                str(path.relative_to(staging)): sha256_file(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"target": str(target.resolve()), "meshes": manifest["meshes"], **record}
