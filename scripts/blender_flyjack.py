"""Render the Chessfly match as a FlyJack-style scene: one table under one lamp.

Run with:
  Blender --background --python scripts/blender_flyjack.py -- \
    --run-dir runs/full-game-skill0-v1 --plan runs/full-game-skill0-v1/plan.json \
    --output runs/flyjack/frames/frame-

The visual grammar follows FlyJack (fanpu.io/games/flyjack): a round felt table
in darkness under a single warm spotlight with soft shadows and haze, the
NeuroMechFly body at the table, and camera shots that cut between an overview,
the board, and the brain.  The brain itself is not rendered here: this script
writes every frame's camera matrices so the compositor can draw the recorded
spikes at the neurons' real soma positions, frame-accurately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_chessfly as base  # noqa: E402  (Blender adds scripts at runtime)
import nmf_fly  # noqa: E402


TABLE_TOP = 1.22
TABLE_CENTRE = (0.78, -1.05)
TABLE_RADIUS = 4.5
FLY_LOCATION = (0.78, -3.50)
FLY_MILLIMETRE = 0.50
BRAIN_ANCHOR = (0.78, -3.10, 3.40)
BRAIN_WIDTH = 2.4
LAMP_HEIGHT = 10.5
LAMP_ENERGY = 5200.0
THINK_LAMP_SHARE = 0.42
SERIF = Path("/System/Library/Fonts/Supplemental/Georgia.ttf")

# Camera poses per shot: (position, target).  The compositor's panel covers the
# right third, so every target is later panned to keep the subject left.
SHOTS = {
    "overview": ((8.4, -13.8, 10.6), (0.8, -1.5, 1.3)),
    "board": ((6.4, -11.2, 7.4), (0.9, -1.3, 1.4)),
    "think": ((5.6, -11.0, 3.6), (0.8, -3.2, 2.85)),
    "result": ((5.0, -9.4, 5.8), (0.9, -0.9, 1.4)),
}
DOLLY = 0.05
TRANSITION_FRAMES = 22


def arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fly-assets", type=Path, default=Path("data/assets/neuromechfly")
    )
    parser.add_argument("--camera-out", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--pan", type=float, default=1.25)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=300)
    parser.add_argument("--no-haze", action="store_true")
    return parser.parse_args(raw)


def _hex(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """sRGB hex to the linear colour Blender's shaders expect."""
    def channel(text: str) -> float:
        c = int(text, 16) / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (channel(value[1:3]), channel(value[3:5]), channel(value[5:7]), alpha)


def felt_material() -> bpy.types.Material:
    """Grey-white baize: a radial falloff from lit centre to a greyer rim, with fibre noise."""
    material = bpy.data.materials.new("felt")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 0.95
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / TABLE_RADIUS,) * 3
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "SPHERICAL"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = _hex("#a19e97")
    ramp.color_ramp.elements[1].color = _hex("#dcdad4")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 180.0
    noise.inputs["Detail"].default_value = 6.0
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.12
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["A"])
    links.new(noise.outputs["Color"], mix.inputs["B"])
    links.new(mix.outputs["Result"], shader.inputs["Base Color"])
    return material


def add_table(mats: dict) -> None:
    cx, cy = TABLE_CENTRE
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=160, radius=TABLE_RADIUS, depth=0.12, location=(cx, cy, TABLE_TOP - 0.06)
    )
    top = bpy.context.object
    top.name = "felt"
    top.data.materials.append(mats["felt"])
    bpy.ops.mesh.primitive_torus_add(
        major_segments=160,
        minor_segments=24,
        major_radius=TABLE_RADIUS + 0.16,
        minor_radius=0.24,
        location=(cx, cy, TABLE_TOP - 0.02),
    )
    rim = bpy.context.object
    rim.name = "wood-rim"
    rim.data.materials.append(mats["rim"])
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=120,
        radius=TABLE_RADIUS + 0.35,
        depth=TABLE_TOP - 0.1,
        location=(cx, cy, (TABLE_TOP - 0.1) / 2),
    )
    stand = bpy.context.object
    stand.name = "table-base"
    stand.data.materials.append(mats["stand"])
    bpy.ops.mesh.primitive_plane_add(size=80, location=(cx, cy, 0.0))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(mats["floor"])


def add_lettering(mats: dict, plan: dict) -> None:
    """Serif lettering printed on the far side of the felt, FlyJack-style."""
    font = bpy.data.fonts.load(str(SERIF)) if SERIF.exists() else None
    cx, cy = TABLE_CENTRE
    lines = (
        ("C H E S S F L Y", 3.55, 0.36),
        (_opponent_line(plan), 3.05, 0.17),
    )
    for body, radius, size in lines:
        bpy.ops.curve.primitive_bezier_circle_add(
            radius=radius, location=(cx, cy, TABLE_TOP + 0.002)
        )
        path = bpy.context.object
        path.name = f"arc-{radius}"
        # Start the text at the far side so it reads toward the camera.
        path.rotation_euler[2] = math.radians(90)
        path.scale = (-1.0, 1.0, 1.0)
        bpy.ops.object.text_add(location=(cx, cy, TABLE_TOP + 0.004))
        text = bpy.context.object
        text.data.body = body
        text.data.size = size
        text.data.extrude = 0.001
        text.data.align_x = "CENTER"
        if font is not None:
            text.data.font = font
        text.data.follow_curve = path
        text.data.offset_x = math.pi * radius / 2
        text.data.materials.append(mats["gold"])
        path.hide_render = True


def _opponent_line(plan: dict) -> str:
    options = plan.get("stockfish_options") or {}
    if "Skill Level" in options:
        opponent = f"STOCKFISH SKILL {options['Skill Level']}"
    else:
        opponent = f"STOCKFISH {plan['stockfish_elo']}"
    return f"FROZEN MALECNS  ·  {opponent}  ·  NO PLASTICITY"


def add_lamp(total_frames: int, shots: list[dict]) -> bpy.types.Object:
    cx, cy = TABLE_CENTRE
    bpy.ops.object.light_add(type="SPOT", location=(cx, cy, LAMP_HEIGHT))
    lamp = bpy.context.object
    lamp.name = "table-lamp"
    lamp.data.color = (1.0, 0.92, 0.83)
    lamp.data.spot_size = math.radians(62)
    lamp.data.spot_blend = 0.75
    lamp.data.shadow_soft_size = 0.7
    lamp.data.energy = LAMP_ENERGY
    lamp.rotation_euler = (0.0, 0.0, 0.0)
    # FlyJack dims the table lamp while the brain is on screen.
    for shot in shots:
        settle = min(shot["end_frame"], shot["start_frame"] + TRANSITION_FRAMES)
        share = THINK_LAMP_SHARE if shot["name"] == "think" else 1.0
        lamp.data.energy = LAMP_ENERGY * share
        lamp.data.keyframe_insert("energy", frame=max(1, settle))
        lamp.data.keyframe_insert("energy", frame=max(1, shot["end_frame"]))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, location=(cx, cy, LAMP_HEIGHT + 0.4))
    glow = bpy.context.object
    glow.name = "lamp-glow"
    glow_material = bpy.data.materials.new("lamp-glow")
    glow_material.use_nodes = True
    shader = glow_material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (1.0, 0.72, 0.42, 1.0)
    shader.inputs["Emission Color"].default_value = (1.0, 0.72, 0.42, 1.0)
    shader.inputs["Emission Strength"].default_value = 6.0
    glow.data.materials.append(glow_material)
    glow.visible_shadow = False

    bpy.ops.object.light_add(type="SUN", location=(-8, 7, 12))
    rim = bpy.context.object
    rim.name = "cool-rim"
    rim.data.color = (0.56, 0.70, 1.0)
    rim.data.energy = 0.35
    rim.rotation_euler = (math.radians(55), 0.0, math.radians(-140))
    return lamp


def configure_world(scene: bpy.types.Scene, haze: bool) -> None:
    world = scene.world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes.get("Background")
    background.inputs["Color"].default_value = _hex("#050607")
    background.inputs["Strength"].default_value = 1.0
    if haze:
        volume = nodes.new("ShaderNodeVolumePrincipled")
        volume.inputs["Density"].default_value = 0.010
        volume.inputs["Anisotropy"].default_value = 0.35
        output = nodes.get("World Output")
        links.new(volume.outputs["Volume"], output.inputs["Volume"])


def key_camera(
    camera: bpy.types.Object, focus: bpy.types.Object, shots: list[dict], pan: float
) -> None:
    for index, shot in enumerate(shots):
        position, target = (Vector(value) for value in SHOTS[shot["name"]])
        start, end = shot["start_frame"], shot["end_frame"]
        settle = start + min(TRANSITION_FRAMES, max(1, (end - start) // 2))
        if index == 0:
            settle = 1
        dollied = position + (target - position) * DOLLY
        for frame, eye in ((settle, position), (end, dollied)):
            camera.location = eye
            _aim(camera, eye, target, pan)
            camera.keyframe_insert("location", frame=max(1, frame))
            camera.keyframe_insert("rotation_euler", frame=max(1, frame))
            focus.location = target
            focus.keyframe_insert("location", frame=max(1, frame))


def _aim(camera: bpy.types.Object, eye: Vector, target: Vector, pan: float) -> None:
    """Aim beside the subject in world space; rolling the camera would tilt the frame."""
    sight = target - eye
    sideways = Vector((sight.y, -sight.x, 0.0))
    if sideways.length > 1e-6:
        sideways.normalize()
    aim = target + sideways * pan
    camera.rotation_euler = (aim - eye).to_track_quat("-Z", "Y").to_euler()


def _brain_facing() -> list[float]:
    eye = Vector(SHOTS["think"][0])
    facing = Vector((eye.x - BRAIN_ANCHOR[0], eye.y - BRAIN_ANCHOR[1], 0.0))
    facing.normalize()
    return [facing.x, facing.y, 0.0]


def export_cameras(
    scene: bpy.types.Scene, camera: bpy.types.Object, path: Path, total_frames: int
) -> None:
    """Write per-frame world and projection matrices for the brain compositor."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    width, height = scene.render.resolution_x, scene.render.resolution_y
    frames = {}
    for frame in range(1, total_frames + 1):
        scene.frame_set(frame)
        view = camera.matrix_world.inverted()
        projection = camera.calc_matrix_camera(depsgraph, x=width, y=height)
        frames[str(frame)] = {
            "view": [list(row) for row in view],
            "projection": [list(row) for row in projection],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "width": width,
                "height": height,
                "brain_anchor": list(BRAIN_ANCHOR),
                # The brain hologram faces the think camera so its front view reads.
                "brain_facing": _brain_facing(),
                "brain_width": BRAIN_WIDTH,
                "fly_millimetre": FLY_MILLIMETRE,
                "frames": frames,
            }
        )
    )


def build(args: argparse.Namespace) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)

    plan = json.loads(args.plan.read_text())
    manifest = json.loads((args.run_dir / "run.json").read_text())
    plan.setdefault("stockfish_options", manifest.get("stockfish_options"))
    total_frames = int(plan["total_frames"])
    shots = plan["shots"]

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = int(plan["fps"])
    scene.frame_start = args.frame_start or 1
    scene.frame_end = args.frame_end or total_frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(args.output.resolve())
    for attribute, value in (
        ("taa_render_samples", args.samples),
        ("use_shadows", True),
        ("use_volumetric_shadows", True),
        ("volumetric_tile_size", "4"),
    ):
        if hasattr(scene.eevee, attribute):
            try:
                setattr(scene.eevee, attribute, value)
            except TypeError:
                pass
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    configure_world(scene, haze=not args.no_haze)

    mats = {
        "felt": felt_material(),
        "rim": base.material("wood-rim", _hex("#3b2314"), roughness=0.45, metallic=0.05),
        "stand": base.material("table-base", _hex("#1c110a"), roughness=0.7),
        "floor": base.material("floor", _hex("#0a0b0c"), roughness=0.92),
        # Printed ink: charcoal reads on grey-white felt where gold would wash out.
        "gold": base.material("felt-ink", _hex("#3a3834"), roughness=0.7),
        "board-light": base.material("maple", _hex("#c9a878"), roughness=0.5),
        "board-dark": base.material("walnut", _hex("#4a2e1a"), roughness=0.5),
        "frame": base.material("board-frame", _hex("#24160d"), roughness=0.4, metallic=0.05),
        "piece-white": base.material("pieces-white", (0.73, 0.77, 0.72, 1), metallic=0.18, roughness=0.3),
        "piece-black": base.material("pieces-black", (0.018, 0.028, 0.034, 1), metallic=0.65, roughness=0.22),
    }
    add_table(mats)
    add_lettering(mats, plan)
    base.add_chessboard(plan, mats)

    fly = nmf_fly.build_neuromechfly(
        args.fly_assets,
        (FLY_LOCATION[0], FLY_LOCATION[1], TABLE_TOP),
        heading=math.radians(90),
        millimetre=FLY_MILLIMETRE,
    )
    fly.location.z += TABLE_TOP - nmf_fly.lowest_point(fly)

    add_lamp(total_frames, shots)

    bpy.ops.object.camera_add(location=SHOTS["overview"][0])
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 40
    camera.data.sensor_width = 36
    camera.data.clip_end = 300
    focus = bpy.data.objects.new("camera-focus", None)
    bpy.context.collection.objects.link(focus)
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = focus
    camera.data.dof.aperture_fstop = 4.0
    key_camera(camera, focus, shots, args.pan)

    camera_out = args.camera_out or args.output.parent / "camera.json"
    export_cameras(scene, camera, camera_out, total_frames)
    print(f"[flyjack] cameras written to {camera_out}", flush=True)


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args)
    if args.preview:
        bpy.context.scene.frame_set(args.preview_frame)
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
