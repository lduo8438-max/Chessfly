"""Build and render the original low-poly Chessfly cinematic in Blender.

Run with:
  Blender --background --python scripts/blender_chessfly.py -- \
    --run-dir runs/video-source-v1 --output runs/cinematic/preview.png --preview
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


CYAN = (0.05, 0.78, 1.0, 1.0)
AMBER = (1.0, 0.28, 0.08, 1.0)
VIOLET = (0.36, 0.05, 0.58, 1.0)
INK = (0.012, 0.025, 0.045, 1.0)
STEEL = (0.19, 0.25, 0.29, 1.0)
WHITE = (0.72, 0.78, 0.76, 1.0)


def arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--save-blend", type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        help="animation plan from `chessfly match-plan`; plays the whole game",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--preview-frame", type=int, default=220)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--yaw", type=float, default=-0.22)
    return parser.parse_args(raw)


def material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.45,
    emission: float = 0.0,
    alpha: float = 1.0,
) -> bpy.types.Material:
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color[:3], alpha)
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color[:3], alpha)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = color
        shader.inputs["Emission Strength"].default_value = emission
    if alpha < 1:
        shader.inputs["Alpha"].default_value = alpha
        value.surface_render_method = "DITHERED"
    return value


def cube(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
    bevel: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("soft-machined-edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    return obj


def ico(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
    subdivisions: int = 2,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions, radius=1, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj


def cylinder_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    mat: bpy.types.Material,
    vertices: int = 8,
) -> bpy.types.Object:
    a, b = Vector(start), Vector(end)
    delta = b - a
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=delta.length,
        location=(a + b) / 2,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    return obj


def text_object(
    body: str,
    location: tuple[float, float, float],
    size: float,
    mat: bpy.types.Material,
    *,
    extrude: float = 0.002,
    align: str = "LEFT",
) -> bpy.types.Object:
    bpy.ops.object.text_add(location=location, rotation=(math.pi / 2, 0, 0))
    obj = bpy.context.object
    obj.data.body = body
    obj.data.align_x = align
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.materials.append(mat)
    return obj


def add_screen(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float],
    frame_mat: bpy.types.Material,
    image_path: Path | None = None,
    emission_color: tuple[float, float, float, float] = CYAN,
) -> bpy.types.Object:
    x, y, z = center
    width, height = size
    cube(f"{name}-housing", (x, y + 0.05, z), (width / 2 + 0.11, 0.08, height / 2 + 0.11), frame_mat, 0.06)
    bpy.ops.mesh.primitive_plane_add(
        size=2,
        location=(x, y - 0.045, z),
        rotation=(math.pi / 2, 0, 0),
    )
    screen = bpy.context.object
    screen.name = f"{name}-display"
    screen.scale = (width / 2, height / 2, 1)
    if image_path is not None:
        image = bpy.data.images.load(str(image_path.resolve()), check_existing=True)
        mat = bpy.data.materials.new(f"{name}-image")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in list(nodes):
            nodes.remove(node)
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.8
        output = nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    else:
        mat = material(f"{name}-glow", emission_color, roughness=0.25, emission=1.2)
    screen.data.materials.append(mat)
    return screen


def make_raster_image(path: Path, run_dir: Path) -> None:
    artifact = run_dir / "neural" / "ply-003" / "spikes-10ms.npz"
    if not artifact.exists():
        artifact = sorted((run_dir / "neural").glob("ply-*/spikes-10ms.npz"))[0]
    with np.load(artifact) as values:
        counts = np.asarray(values["counts"], dtype=np.float32)
    ranked = np.argsort(counts.sum(axis=0))[::-1][:96]
    raster = counts[:, ranked].T
    canvas = np.zeros((192, 512, 4), dtype=np.float32)
    canvas[..., :3] = (0.008, 0.018, 0.03)
    canvas[..., 3] = 1
    for row in range(raster.shape[0]):
        for column in np.flatnonzero(raster[row] > 0):
            x0 = int(column / max(1, raster.shape[1] - 1) * 500) + 6
            y0 = int(row / max(1, raster.shape[0] - 1) * 178) + 7
            strength = min(1.0, raster[row, column] / max(1.0, raster.max()))
            canvas[y0 : y0 + 2, x0 : x0 + 3, :3] = (
                0.12 + 0.25 * strength,
                0.65 + 0.30 * strength,
                0.85 + 0.15 * strength,
            )
    image = bpy.data.images.new("real-spike-raster", width=512, height=192)
    image.pixels = canvas[::-1].reshape(-1)
    image.filepath_raw = str(path.resolve())
    image.file_format = "PNG"
    image.save()


def add_fly(materials: dict[str, bpy.types.Material]) -> bpy.types.Object:
    root = bpy.data.objects.new("Chessfly-low-poly", None)
    bpy.context.collection.objects.link(root)
    thorax = ico("thorax", (-1.45, -0.2, 1.83), (0.62, 0.46, 0.43), materials["fly"], 2)
    abdomen = ico("abdomen", (-2.05, -0.12, 1.82), (0.82, 0.38, 0.34), materials["abdomen"], 2)
    head = ico("head", (-0.91, -0.2, 1.88), (0.43, 0.42, 0.40), materials["head"], 2)
    for obj in (thorax, abdomen, head):
        obj.parent = root
    for side in (-1, 1):
        eye = ico(
            f"ruby-eye-{side}",
            (-0.69, -0.2 + side * 0.31, 1.94),
            (0.26, 0.11, 0.27),
            materials["eye"],
            2,
        )
        eye.parent = root
        wing = ico(
            f"wing-{side}",
            (-1.72, -0.2 + side * 0.48, 2.18),
            (1.05, 0.36, 0.045),
            materials["wing"],
            2,
        )
        wing.rotation_euler[2] = side * 0.16
        wing.parent = root
        for frame, angle in ((1, 0.12), (75, 0.19), (150, 0.10), (225, 0.18), (300, 0.12)):
            wing.rotation_euler[2] = side * angle
            wing.keyframe_insert("rotation_euler", frame=frame)
    for index, x in enumerate((-1.05, -1.45, -1.83)):
        for side in (-1, 1):
            hip = (x, -0.2 + side * 0.33, 1.75)
            knee = (x + 0.13 * (index - 1), -0.2 + side * 0.78, 1.48)
            foot = (x + 0.34 * (index - 1), -0.2 + side * 1.08, 1.29)
            for segment, start, end in (("upper", hip, knee), ("lower", knee, foot)):
                leg = cylinder_between(
                    f"leg-{index}-{side}-{segment}", start, end, 0.035, materials["leg"], 6
                )
                leg.parent = root
    for side in (-1, 1):
        antenna = cylinder_between(
            f"antenna-{side}",
            (-0.66, -0.2 + side * 0.13, 2.07),
            (-0.28, -0.2 + side * 0.25, 2.28),
            0.018,
            materials["leg"],
            6,
        )
        antenna.parent = root
    return root


def _piece_parts(kind: str, facing: float) -> list[tuple]:
    """Describe one Staunton-ish low-poly piece as primitives in local space.

    Each entry is ("cone"|"ball"|"box", parameters). Keeping the description
    declarative makes the six silhouettes readable and easy to tune.
    """
    parts: list[tuple] = [
        ("cone", 0.118, 0.106, 0.034, 0.017, 20),
        ("cone", 0.102, 0.074, 0.048, 0.058, 20),
    ]
    if kind == "pawn":
        parts += [
            ("cone", 0.050, 0.042, 0.098, 0.131, 16),
            ("cone", 0.064, 0.064, 0.015, 0.188, 16),
            ("ball", 0.059, 0.239, (1.0, 1.0, 0.94)),
        ]
    elif kind == "rook":
        parts += [
            ("cone", 0.064, 0.072, 0.132, 0.148, 16),
            ("cone", 0.088, 0.088, 0.044, 0.236, 16),
        ]
        for dx, dy in ((0.056, 0.0), (-0.056, 0.0), (0.0, 0.056), (0.0, -0.056)):
            parts.append(("box", 0.020, 0.020, 0.023, 0.270, dx, dy, 0.0))
    elif kind == "bishop":
        parts += [
            ("cone", 0.056, 0.040, 0.152, 0.158, 16),
            ("cone", 0.069, 0.069, 0.014, 0.243, 16),
            ("cone", 0.058, 0.009, 0.086, 0.293, 16),
            ("ball", 0.021, 0.345, (1.0, 1.0, 1.0)),
        ]
    elif kind == "knight":
        parts += [
            ("cone", 0.062, 0.056, 0.092, 0.128, 16),
            ("box", 0.052, 0.040, 0.062, 0.232, 0.0, 0.0, 0.0),
            ("box", 0.042, 0.056, 0.034, 0.288, 0.0, facing * 0.030, 0.0),
            ("box", 0.030, 0.040, 0.024, 0.318, 0.0, facing * 0.052, 0.0),
            ("box", 0.013, 0.013, 0.026, 0.330, facing * -0.022, facing * -0.026, 0.0),
            ("box", 0.013, 0.013, 0.026, 0.330, facing * 0.022, facing * -0.026, 0.0),
        ]
    elif kind == "queen":
        parts += [
            ("cone", 0.058, 0.044, 0.182, 0.173, 16),
            ("cone", 0.080, 0.080, 0.016, 0.272, 16),
            ("ball", 0.050, 0.314, (1.0, 1.0, 0.78)),
        ]
        for index in range(6):
            angle = index * math.tau / 6
            parts.append(
                (
                    "ball",
                    0.017,
                    0.350,
                    (1.0, 1.0, 1.0),
                    0.046 * math.cos(angle),
                    0.046 * math.sin(angle),
                )
            )
        parts.append(("ball", 0.020, 0.374, (1.0, 1.0, 1.0)))
    else:  # king
        parts += [
            ("cone", 0.058, 0.046, 0.198, 0.181, 16),
            ("cone", 0.082, 0.082, 0.016, 0.288, 16),
            ("ball", 0.049, 0.330, (1.0, 1.0, 0.82)),
            ("box", 0.011, 0.011, 0.040, 0.398, 0.0, 0.0, 0.0),
            ("box", 0.031, 0.011, 0.011, 0.390, 0.0, 0.0, 0.0),
        ]
    return parts


def _build_body(
    name: str,
    kind: str,
    facing: float,
    mat: bpy.types.Material,
    root: bpy.types.Object,
) -> bpy.types.Object:
    """Build one piece body under its own empty so promotion can swap bodies."""
    body = bpy.data.objects.new(name, None)
    body.location = (0, 0, 0)
    bpy.context.collection.objects.link(body)
    body.parent = root
    for part in _piece_parts(kind, facing):
        if part[0] == "cone":
            _, radius1, radius2, depth, zc, vertices = part
            bpy.ops.mesh.primitive_cone_add(
                vertices=vertices,
                radius1=radius1,
                radius2=radius2,
                depth=depth,
                location=(0, 0, 0),
            )
            obj = bpy.context.object
            obj.location = (0, 0, zc)
        elif part[0] == "ball":
            radius, zc, scale = part[1], part[2], part[3]
            dx = part[4] if len(part) > 4 else 0.0
            dy = part[5] if len(part) > 5 else 0.0
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
            obj = bpy.context.object
            obj.location = (dx, dy, zc)
            obj.scale = scale
        else:
            _, sx, sy, sz, zc, dx, dy, rot = part
            bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
            obj = bpy.context.object
            obj.location = (dx, dy, zc)
            obj.scale = (sx, sy, sz)
            obj.rotation_euler[2] = rot
        obj.name = f"{name}-{part[0]}"
        obj.data.materials.append(mat)
        obj.parent = body
    return body


def square_location(square: str) -> tuple[float, float, float]:
    file_index = ord(square[0]) - ord("a")
    rank_index = int(square[1]) - 1
    return (0.78 + (file_index - 3.5) * 0.36, -0.45 + (rank_index - 3.5) * 0.36, 1.38)


def add_piece(
    piece: dict,
    promotes_to: str | None,
    mats: dict[str, bpy.types.Material],
) -> dict:
    """Place one piece, with a second hidden body when it later promotes."""
    root = bpy.data.objects.new(f"piece-{piece['id']}", None)
    root.location = square_location(piece["square"])
    bpy.context.collection.objects.link(root)
    mat = mats["piece-white"] if piece["color"] == "white" else mats["piece-black"]
    facing = 1.0 if piece["color"] == "white" else -1.0
    bodies = {
        "main": _build_body(f"{piece['id']}-{piece['kind']}", piece["kind"], facing, mat, root)
    }
    if promotes_to:
        promoted = _build_body(
            f"{piece['id']}-{promotes_to}", promotes_to, facing, mat, root
        )
        promoted.scale = (0.001, 0.001, 0.001)
        promoted.keyframe_insert("scale", frame=1)
        bodies["promoted"] = promoted
    return {"root": root, "bodies": bodies}


def add_board(mats: dict[str, bpy.types.Material]) -> None:
    for rank in range(8):
        for file_index in range(8):
            x, y, z = square_location(f"{'abcdefgh'[file_index]}{rank + 1}")
            mat = mats["board-light"] if (file_index + rank) % 2 else mats["board-dark"]
            cube(f"tile-{file_index}-{rank}", (x, y, z - 0.055), (0.175, 0.175, 0.055), mat, 0.018)
    cube("board-base", (0.78, -0.45, 1.275), (1.52, 1.52, 0.055), mats["frame"], 0.08)


def add_chessboard(plan: dict, mats: dict[str, bpy.types.Material]) -> None:
    """Build the opening position and key the whole recorded game onto it."""
    add_board(mats)
    promotions = {
        event["piece"]: event["to"]
        for event in plan["events"]
        if event["kind"] == "promote"
    }
    pieces = {
        piece["id"]: add_piece(piece, promotions.get(piece["id"]), mats)
        for piece in plan["pieces"]
    }
    for event in plan["events"]:
        entry = pieces[event["piece"]]
        if event["kind"] == "move":
            _key_move(entry["root"], event)
        elif event["kind"] == "capture":
            _key_capture(entry["root"], event["frame"])
        elif event["kind"] == "promote":
            _key_promotion(entry["bodies"], event["frame"])


def _key_move(root: bpy.types.Object, event: dict) -> None:
    start = Vector(square_location(event["from"]))
    end = Vector(square_location(event["to"]))
    lift = max(1, int(event["lift_frame"]))
    land = max(lift + 2, int(event["land_frame"]))
    apex = (start + end) / 2 + Vector((0, 0, 0.30))
    for frame, location in (
        (lift, start),
        ((lift + land) // 2, apex),
        (land, end),
    ):
        root.location = location
        root.keyframe_insert("location", frame=frame)


def _key_capture(root: bpy.types.Object, frame: int) -> None:
    for offset, scale in ((-5, 1.0), (3, 0.001)):
        root.scale = (scale, scale, scale)
        root.keyframe_insert("scale", frame=max(1, frame + offset))


def _key_promotion(bodies: dict, frame: int) -> None:
    promoted = bodies.get("promoted")
    if promoted is None:
        return
    for offset, main_scale, promoted_scale in ((-3, 1.0, 0.001), (4, 0.001, 1.0)):
        bodies["main"].scale = (main_scale,) * 3
        bodies["main"].keyframe_insert("scale", frame=max(1, frame + offset))
        promoted.scale = (promoted_scale,) * 3
        promoted.keyframe_insert("scale", frame=max(1, frame + offset))


def make_cyclic(obj: bpy.types.Object) -> None:
    """Let a short idle animation repeat for however long the match runs."""
    animation = obj.animation_data
    if animation is None or animation.action is None:
        return
    for curve in _action_fcurves(animation.action):
        if not any(modifier.type == "CYCLES" for modifier in curve.modifiers):
            curve.modifiers.new("CYCLES")


def _action_fcurves(action) -> list:
    """Read curves from both legacy actions and Blender's slotted actions."""
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    return curves


def configure_scene(args: argparse.Namespace) -> None:
    print("[chessfly] clear scene", flush=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)

    plan = None
    if args.plan is not None:
        plan = json.loads(args.plan.read_text())
    total_frames = int(plan["total_frames"]) if plan else 300

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.fps = int(plan["fps"]) if plan else 30
    scene.frame_start = args.frame_start or 1
    scene.frame_end = args.frame_end or total_frames
    for attribute, value in (
        ("taa_render_samples", args.samples),
        ("use_raytracing", True),
    ):
        if hasattr(scene.eevee, attribute):
            setattr(scene.eevee, attribute, value)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.output.resolve())
    scene.render.image_settings.compression = 45
    scene.world.color = (0.002, 0.005, 0.012)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    mats = {
        "floor": material("floor", INK, metallic=0.48, roughness=0.28),
        "frame": material("monitor-frame", (0.025, 0.045, 0.065, 1), metallic=0.75, roughness=0.22),
        "fly": material("fly-thorax", STEEL, metallic=0.16, roughness=0.62),
        "abdomen": material("fly-abdomen", (0.10, 0.15, 0.17, 1), metallic=0.1, roughness=0.62),
        "head": material("fly-head", (0.16, 0.19, 0.19, 1), metallic=0.1, roughness=0.55),
        "eye": material("ruby-compound-eyes", (0.35, 0.002, 0.018, 1), metallic=0.15, roughness=0.32, emission=0.35),
        "wing": material("silver-wings", (0.38, 0.55, 0.62, 1), metallic=0.12, roughness=0.24, alpha=0.42),
        "leg": material("fly-legs", (0.055, 0.075, 0.078, 1), metallic=0.1, roughness=0.6),
        "board-light": material("board-ivory", (0.47, 0.55, 0.52, 1), roughness=0.42),
        "board-dark": material("board-teal", (0.035, 0.18, 0.21, 1), metallic=0.1, roughness=0.35),
        "piece-white": material("pieces-white", (0.73, 0.77, 0.72, 1), metallic=0.18, roughness=0.3),
        "piece-black": material("pieces-black", (0.018, 0.028, 0.034, 1), metallic=0.65, roughness=0.22),
        "cyan": material("cyan-emission", CYAN, emission=5.5, roughness=0.2),
        "amber": material("amber-emission", AMBER, emission=4.5, roughness=0.2),
        "muted": material("muted-text", (0.24, 0.38, 0.42, 1), emission=1.2),
    }
    print("[chessfly] materials ready", flush=True)

    cube("laboratory-floor", (0, 0, 0), (7.2, 6.2, 0.1), mats["floor"], 0.08)
    cube("desk", (0, -0.25, 1.05), (4.4, 2.25, 0.18), mats["frame"], 0.12)
    for x in (-4.0, 4.0):
        cube(f"desk-leg-{x}", (x, -0.25, 0.5), (0.13, 1.75, 0.55), mats["frame"], 0.05)
    print("[chessfly] desk ready", flush=True)
    for x in np.linspace(-5.8, 5.8, 18):
        cube(f"data-column-{x:.2f}", (float(x), 3.8, 2.9), (0.018, 0.025, 2.5), mats["cyan"] if int(abs(x) * 10) % 3 else mats["amber"])
    print("[chessfly] data columns ready", flush=True)
    print("[chessfly] laboratory ready", flush=True)

    run_dir = args.run_dir.resolve()
    stimulus = run_dir / "neural" / "ply-003" / "stimulus.png"
    if not stimulus.exists():
        stimulus = sorted((run_dir / "neural").glob("ply-*/stimulus.png"))[0]
    raster_path = args.output.parent.resolve() / "real-spike-raster.png"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    make_raster_image(raster_path, run_dir)
    print("[chessfly] raster ready", flush=True)
    add_screen("chess-monitor", (1.55, 2.28, 3.25), (3.75, 2.12), mats["frame"], stimulus)
    add_screen("neural-monitor", (-2.65, 2.02, 2.95), (2.35, 1.52), mats["frame"], raster_path)
    # In full-match mode the composite owns the HUD, so the set carries no text;
    # baking a second wordmark into the plate only collides with the overlay.
    if plan is None:
        text_object("CHESSFLY", (-3.70, 1.90, 4.00), 0.24, mats["cyan"])
        text_object("NEURAL REPLAY  /  500 ms", (-3.70, 1.89, 1.98), 0.095, mats["muted"])
        text_object(
            "100,595 SPIKES  /  2 DN READOUT", (-3.70, 1.88, 1.78), 0.085, mats["cyan"]
        )
        text_object("STOCKFISH 1320", (0.05, 2.12, 4.53), 0.12, mats["amber"])
    print("[chessfly] monitors ready", flush=True)

    if plan is not None:
        add_chessboard(plan, mats)
    else:
        add_board(mats)
    print("[chessfly] chessboard ready", flush=True)
    fly = add_fly(mats)
    print("[chessfly] fly ready", flush=True)
    for frame, z in ((1, 0.0), (80, 0.018), (160, -0.008), (240, 0.015), (300, 0.0)):
        fly.location.z = z
        fly.keyframe_insert("location", frame=frame)
    make_cyclic(fly)
    for child in bpy.data.objects:
        if child.name.startswith("wing-"):
            make_cyclic(child)

    for location, energy, color, size in (
        ((-3.5, -2.8, 4.8), 900, (0.10, 0.64, 1.0), 5.5),
        ((3.8, -0.5, 4.1), 700, (1.0, 0.24, 0.10), 4.5),
        ((0.0, 2.0, 5.8), 1150, (0.26, 0.40, 1.0), 5.5),
        # A soft key over the board so the pieces read as solid, not silhouettes.
        ((0.4, -3.2, 3.5), 620, (0.78, 0.86, 1.0), 3.2),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.color = color
        light.data.shape = "DISK"
        light.data.size = size
        look_at(light, (0, 0, 1.5))
    print("[chessfly] lights ready", flush=True)

    bpy.ops.object.camera_add(location=(6.9, -9.4, 5.0))
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 45
    camera.data.sensor_width = 36
    focus = bpy.data.objects.new("camera-focus", None)
    focus.location = (-0.15, -0.05, 1.85)
    bpy.context.collection.objects.link(focus)
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = focus
    camera.data.dof.aperture_fstop = 2.8
    # One slow orbit across the whole match, yawed so the subject sits in the
    # left two thirds and the overlay panel never covers it.
    centre = Vector((0.10, -0.35, 1.72))
    steps = max(2, total_frames // 60)
    for step in range(steps + 1):
        frame = 1 + round(step * (total_frames - 1) / steps)
        phase = step / steps
        angle = math.radians(-114) + phase * math.radians(40)
        radius = 11.9 - 0.9 * math.sin(phase * math.pi)
        height = 6.7 - 0.7 * math.sin(phase * math.pi)
        camera.location = (
            centre.x + radius * math.cos(angle),
            centre.y + radius * math.sin(angle),
            height,
        )
        target = centre + Vector((0.0, 0.0, 0.10 * math.sin(phase * math.tau)))
        look_at(camera, tuple(target))
        camera.rotation_euler.rotate_axis("Z", args.yaw)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        focus.location = tuple(target)
        focus.keyframe_insert("location", frame=frame)
    print("[chessfly] camera ready", flush=True)

    scene.render.image_settings.color_mode = "RGB"
    if args.preview:
        scene.frame_set(min(args.preview_frame, scene.frame_end))
        scene.render.filepath = str(args.output.resolve())
    if args.save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_scene(args)
    print("[chessfly] scene configured", flush=True)
    if args.preview:
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
