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


def parse_fen(fen: str) -> dict[str, str]:
    rows = fen.split()[0].split("/")
    pieces = {}
    for row_index, row in enumerate(rows):
        file_index = 0
        for value in row:
            if value.isdigit():
                file_index += int(value)
            else:
                square = f"{'abcdefgh'[file_index]}{8 - row_index}"
                pieces[square] = value
                file_index += 1
    return pieces


def square_location(square: str) -> tuple[float, float, float]:
    file_index = ord(square[0]) - ord("a")
    rank_index = int(square[1]) - 1
    return (0.78 + (file_index - 3.5) * 0.36, -0.45 + (rank_index - 3.5) * 0.36, 1.38)


def add_piece(
    square: str,
    symbol: str,
    mats: dict[str, bpy.types.Material],
) -> bpy.types.Object:
    x, y, z = square_location(square)
    root = bpy.data.objects.new(f"piece-{square}", None)
    root.location = (0, 0, 0)
    bpy.context.collection.objects.link(root)
    color = mats["piece-white"] if symbol.isupper() else mats["piece-black"]
    kind = symbol.lower()
    heights = {"p": 0.28, "n": 0.39, "b": 0.43, "r": 0.38, "q": 0.52, "k": 0.56}
    height = heights[kind]
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.105, depth=0.07, location=(x, y, z + 0.035))
    base = bpy.context.object
    base.data.materials.append(color)
    base.parent = root
    if kind == "n":
        body = ico(f"knight-{square}", (x + 0.025, y, z + 0.22), (0.11, 0.09, 0.22), color, 1)
        body.rotation_euler[1] = -0.35
    elif kind in ("q", "k"):
        bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.11, radius2=0.065, depth=height * 0.62, location=(x, y, z + height * 0.36))
        body = bpy.context.object
        body.data.materials.append(color)
        crown = ico(f"crown-{square}", (x, y, z + height * 0.74), (0.09, 0.09, 0.08), color, 1)
        crown.parent = root
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.09, radius2=0.055, depth=height * 0.58, location=(x, y, z + height * 0.34))
        body = bpy.context.object
        body.data.materials.append(color)
        cap = ico(f"cap-{square}", (x, y, z + height * 0.68), (0.075, 0.075, 0.075), color, 1)
        cap.parent = root
    body.parent = root
    return root


def add_chessboard(run_dir: Path, mats: dict[str, bpy.types.Material]) -> None:
    records = [json.loads(line) for line in (run_dir / "moves.jsonl").read_text().splitlines()]
    move = next(record for record in records if record["actor"] == "chessfly" and record["ply"] > 1)
    pieces = parse_fen(move["fen_before"])
    for rank in range(8):
        for file_index in range(8):
            x, y, z = square_location(f"{'abcdefgh'[file_index]}{rank + 1}")
            mat = mats["board-light"] if (file_index + rank) % 2 else mats["board-dark"]
            cube(f"tile-{file_index}-{rank}", (x, y, z - 0.055), (0.175, 0.175, 0.055), mat, 0.018)
    cube("board-base", (0.78, -0.45, 1.275), (1.52, 1.52, 0.055), mats["frame"], 0.08)
    roots = {square: add_piece(square, symbol, mats) for square, symbol in pieces.items()}
    uci = str(move["move_uci"])
    moving = roots.get(uci[:2])
    if moving:
        start = Vector(square_location(uci[:2]))
        end = Vector(square_location(uci[2:4]))
        delta = end - start
        for frame, location in (
            (115, Vector((0, 0, 0))),
            (155, delta / 2 + Vector((0, 0, 0.42))),
            (195, delta),
        ):
            moving.location = location
            moving.keyframe_insert("location", frame=frame)


def configure_scene(args: argparse.Namespace) -> None:
    print("[chessfly] clear scene", flush=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = 640 if args.preview else 960
    scene.render.resolution_y = 360 if args.preview else 540
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 300
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
    text_object("CHESSFLY", (-3.70, 1.90, 4.00), 0.24, mats["cyan"])
    text_object("NEURAL REPLAY  /  500 ms", (-3.70, 1.89, 1.98), 0.095, mats["muted"])
    text_object("100,595 SPIKES  /  2 DN READOUT", (-3.70, 1.88, 1.78), 0.085, mats["cyan"])
    text_object("STOCKFISH 1320", (0.05, 2.12, 4.53), 0.12, mats["amber"])
    print("[chessfly] monitors ready", flush=True)

    add_chessboard(run_dir, mats)
    print("[chessfly] chessboard ready", flush=True)
    fly = add_fly(mats)
    print("[chessfly] fly ready", flush=True)
    for frame, z in ((1, 0.0), (80, 0.018), (160, -0.008), (240, 0.015), (300, 0.0)):
        fly.location.z = z
        fly.keyframe_insert("location", frame=frame)

    for location, energy, color, size in (
        ((-3.5, -2.8, 4.8), 850, (0.08, 0.62, 1.0), 5.0),
        ((3.8, -0.5, 4.1), 1050, (1.0, 0.10, 0.04), 4.0),
        ((0.0, 2.0, 5.8), 1250, (0.22, 0.36, 1.0), 5.0),
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
    camera.data.lens = 48
    camera.data.sensor_width = 36
    focus = bpy.data.objects.new("camera-focus", None)
    focus.location = (-0.15, -0.05, 1.85)
    bpy.context.collection.objects.link(focus)
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = focus
    camera.data.dof.aperture_fstop = 2.8
    keyframes = (
        (1, (6.9, -9.4, 5.0), (-0.15, -0.02, 1.85)),
        (105, (5.8, -8.3, 4.45), (-0.35, -0.02, 1.82)),
        (205, (4.25, -7.25, 3.85), (-0.1, -0.12, 1.72)),
        (300, (3.25, -6.6, 3.55), (0.35, -0.08, 1.68)),
    )
    for frame, location, target in keyframes:
        camera.location = location
        look_at(camera, target)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        focus.location = target
        focus.keyframe_insert("location", frame=frame)
    print("[chessfly] camera ready", flush=True)

    scene.render.image_settings.color_mode = "RGB"
    if args.preview:
        scene.frame_set(220)
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
