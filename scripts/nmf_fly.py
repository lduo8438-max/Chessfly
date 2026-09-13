"""Assemble the NeuroMechFly body in Blender from its MJCF kinematic tree.

Imported by the Blender scene scripts; it needs `bpy` and only the standard
library.  Assets come from `chessfly prepare-fly-model`.  Each MJCF <body> becomes
an empty carrying its offset, its orientation, and the rotations of its hinge
joints (applied in declaration order, as MuJoCo does); each mesh <geom> is an
imported STL parented to that empty.  Joint angles come from FlyGym's tripod
standing pose, and colours from FlyGym's appearance config.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


# FlyGym's mesh assets declare scale="1000 1000 1000": the STLs are in metres
# and the MJCF lengths are millimetres.
MESH_SCALE = 1000.0


def _vector(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(value) for value in text.split())


def _material(name: str, rgba: list[float]) -> bpy.types.Material:
    material = bpy.data.materials.new(f"nmf-{name}")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    red, green, blue, alpha = rgba
    shader.inputs["Base Color"].default_value = (red, green, blue, 1.0)
    shader.inputs["Roughness"].default_value = 0.55
    shader.inputs["Metallic"].default_value = 0.05
    material.diffuse_color = (red, green, blue, alpha)
    if alpha < 0.999:
        shader.inputs["Alpha"].default_value = alpha
        material.surface_render_method = "BLENDED"
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
    return material


def _import_stl(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=str(path))
    created = [obj for obj in bpy.data.objects if obj not in before]
    if len(created) != 1:
        raise RuntimeError(f"expected one object from {path.name}, got {len(created)}")
    return created[0]


def build_neuromechfly(
    assets: Path,
    location: tuple[float, float, float],
    heading: float,
    millimetre: float,
    smooth: bool = True,
) -> bpy.types.Object:
    """Place the posed fly with its feet at `location`, facing `heading` radians.

    NeuroMechFly faces +X with +Z up; `heading` turns it about world Z.  One
    millimetre of fly becomes `millimetre` scene units.
    """
    assets = Path(assets)
    tree = ElementTree.parse(assets / "neuromechfly.xml")
    mesh_files = {
        mesh.get("name"): assets / "mesh" / Path(mesh.get("file")).name
        for mesh in tree.getroot().iter("mesh")
    }
    pose = json.loads((assets / "pose.json").read_text())["joints"]
    appearance = json.loads((assets / "appearance.json").read_text())
    materials: dict[str, bpy.types.Material] = {}

    root = bpy.data.objects.new("NeuroMechFly", None)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    root.scale = (millimetre, millimetre, millimetre)
    bpy.context.collection.objects.link(root)

    def colour_for(part: str) -> bpy.types.Material:
        entry = appearance.get(part)
        if entry is None:
            # A few meshes (e.g. the antenna shells) carry no colour of their
            # own; borrow the antenna segment colour rather than inventing one.
            entry = appearance.get("LPedicel") or {"rgba": [0.59, 0.39, 0.12, 1.0]}
        key = ",".join(f"{value:.3f}" for value in entry["rgba"])
        if key not in materials:
            materials[key] = _material(part, entry["rgba"])
        return materials[key]

    def build(body: ElementTree.Element, parent: bpy.types.Object) -> None:
        name = body.get("name")
        empty = bpy.data.objects.new(f"nmf-{name}", None)
        empty.empty_display_size = 0.05
        bpy.context.collection.objects.link(empty)
        empty.parent = parent
        empty.location = _vector(body.get("pos"), (0.0, 0.0, 0.0))
        orientation = Quaternion(_vector(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") != "hinge":
                continue
            angle = math.radians(pose.get(joint.get("name"), 0.0))
            axis = Vector(_vector(joint.get("axis"), (0.0, 0.0, 1.0)))
            orientation = orientation @ Quaternion(axis, angle)
        empty.rotation_mode = "QUATERNION"
        empty.rotation_quaternion = orientation

        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if geom.get("type") != "mesh" or mesh_name not in mesh_files:
                continue
            obj = _import_stl(mesh_files[mesh_name])
            obj.name = f"nmf-geom-{geom.get('name')}"
            obj.parent = empty
            obj.location = _vector(geom.get("pos"), (0.0, 0.0, 0.0))
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = Quaternion(
                _vector(geom.get("quat"), (1.0, 0.0, 0.0, 0.0))
            )
            obj.scale = (MESH_SCALE, MESH_SCALE, MESH_SCALE)
            obj.data.materials.clear()
            obj.data.materials.append(colour_for(geom.get("name")))
            if smooth and callable(getattr(obj.data, "shade_smooth", None)):
                obj.data.shade_smooth()
        for child in body.findall("body"):
            build(child, empty)

    fly_body = tree.getroot().find(".//worldbody/body[@name='FlyBody']")
    if fly_body is None:
        fly_body = tree.getroot().find(".//body[@name='FlyBody']")
    if fly_body is None:
        raise RuntimeError("MJCF has no FlyBody")
    for child in fly_body.findall("body"):
        build(child, root)
    return root


def lowest_point(root: bpy.types.Object) -> float:
    """World-space height of the lowest vertex under `root` (for seating feet)."""
    bpy.context.view_layer.update()
    lowest = math.inf
    for obj in root.children_recursive:
        if obj.type != "MESH":
            continue
        matrix = obj.matrix_world
        for vertex in obj.data.vertices:
            lowest = min(lowest, (matrix @ vertex.co).z)
    return lowest
