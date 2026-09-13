"""Composite the FlyJack-style match: lamp-lit plates, a spiking brain, a panel.

The plates come from `scripts/blender_flyjack.py`, which also writes each frame's
camera matrices.  Here the MaleCNS brain is drawn as a point cloud at the
neurons' recorded soma positions, floating over the fly: every annotated brain
cell is a dim point, and the simulated cells flash when the recorded
`spikes-10ms.npz` says they fired, decaying the way FlyJack's shader does.  The
right third keeps the live-board panel, restyled in FlyJack's palette.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import pyarrow.feather as feather
from PIL import Image, ImageDraw, ImageFilter

from .board_art import render_display_board
from .brain_cloud import group_of
from .dataset import select_files
from .match_video import (
    BOARD_ORIGIN,
    BOARD_SIZE,
    BRAIN_BOX,
    PANEL,
    _brain_layout,
    _load_plates,
)
from .social_video import VideoRun, _evaluation, _font, _outcome_lines, load_video_run


MASTER = (1920, 1080)
# FlyJack's colour tokens.
PAGE = (5, 6, 7)
GLASS = (18, 18, 17)
INK = (255, 255, 255)
INK2 = (195, 194, 183)
MUTED = (137, 135, 129)
GRID = (44, 44, 42)
AXIS = (56, 56, 53)
RING = (58, 58, 56)
BLUE = (57, 135, 229)
ORANGE = (217, 89, 38)
GREEN = (25, 158, 112)
AMBER = (201, 133, 0)
CRITICAL = (208, 59, 59)
WARM = (241, 194, 125)
GROUP_RGB = np.array([BLUE, ORANGE, GREEN, INK2, INK2, INK2], dtype=np.float32) / 255.0
IDLE_RGB = np.array([0.16, 0.19, 0.24], dtype=np.float32)

NECK_NM = 400_000.0
TAU_MS = 20.0
WINDOW_MS = 500.0
FADE_FRAMES = 12
IDLE_ENVELOPE = 0.10
CONTEXT_GAIN = 0.32
SPIKE_GAIN = 3.0
HALO_GAIN = 2.6
EVAL_CLAMP_CP = 1000
SPLAT = ((0, 0, 1.0), (-1, 0, 0.45), (1, 0, 0.45), (0, -1, 0.45), (0, 1, 0.45))


def render_flyjack_video(
    run_dir: Path,
    plan_path: Path,
    frames_dir: Path,
    camera_path: Path,
    output: Path,
    data_dir: Path = Path("data"),
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to render the MP4")
    run_dir, data_dir, output = Path(run_dir), Path(data_dir), Path(output)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    cameras = json.loads(Path(camera_path).read_text(encoding="utf-8"))
    run = load_video_run(run_dir)
    plates = _load_plates(frames_dir)
    total = int(plan["total_frames"])
    if len(plates) < total:
        raise ValueError(f"{frames_dir} holds {len(plates)} plates; the plan needs {total}")
    missing = [n for n in range(1, total + 1) if str(n) not in cameras["frames"]]
    if missing:
        raise ValueError(f"camera export lacks {len(missing)} frames, first {missing[0]}")

    hologram = BrainHologram(data_dir, run_dir, cameras)
    panel_brain = _brain_layout(run_dir, data_dir, run)
    panel_groups = _subgraph_groups(data_dir)
    fps = int(plan["fps"])
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".chessfly-flyjack-") as tmp:
        audio = Path(tmp) / "telemetry.wav"
        _write_audio(audio, plan)
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{MASTER[0]}x{MASTER[1]}", "-r", str(fps), "-i", "-",
            "-i", str(audio),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for frame in range(total):
                image = render_frame(
                    frame, plan, run, plates, cameras, hologram, panel_brain, panel_groups
                )
                process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed while encoding the FlyJack-style video")

    metadata = {
        "schema_version": 1,
        "look": "flyjack",
        "source_run": str(run_dir.resolve()),
        "plan": str(Path(plan_path).resolve()),
        "output": str(output.resolve()),
        "fps": fps,
        "frames": total,
        "resolution": list(MASTER),
        "brain": hologram.describe(),
        "telemetry_only": True,
        "plasticity_shown": False,
        "fly_model": "NeuroMechFly via FlyGym 1.1.0 (Apache-2.0)",
        "board_art": "display-only Staunton glyphs; neural stimulus unchanged",
        "scientific_boundary": "Wiring-constrained simulation; not a biological chess brain.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


class BrainHologram:
    """The MaleCNS brain as points at recorded soma positions, lit by recorded spikes."""

    def __init__(self, data_dir: Path, run_dir: Path, cameras: dict) -> None:
        cloud = np.load(Path(data_dir) / "compiled" / "brain-cloud" / "cloud.npz")
        positions = cloud["positions_nm"].astype(np.float64)
        simulated = cloud["simulated_local"]
        in_brain = positions[:, 2] < NECK_NM
        self.placed_total = int(np.count_nonzero(simulated >= 0))
        positions, groups, simulated = (
            positions[in_brain],
            cloud["groups"][in_brain],
            simulated[in_brain],
        )
        centre = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
        local = positions - centre
        width_nm = float(local[:, 0].max() - local[:, 0].min())
        width = float(cameras["brain_width"])
        facing = np.asarray(cameras["brain_facing"], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(facing, up)
        # MaleCNS x runs left to right, y dorsal to ventral, z anterior to posterior.
        basis = np.column_stack([right, -up, -facing])
        if np.linalg.det(basis) < 0:
            basis[:, 0] *= -1.0
        anchor = np.asarray(cameras["brain_anchor"], dtype=np.float64)
        with np.errstate(all="ignore"):
            world = anchor + (local * (width / width_nm)) @ basis.T
        if not np.isfinite(world).all():
            raise ValueError("brain hologram placement produced non-finite positions")
        self.points = np.c_[world, np.ones(len(world))].astype(np.float64)
        self.groups = groups
        lit = simulated >= 0
        self.lit_index = np.flatnonzero(lit)
        self.lit_local = simulated[lit].astype(np.int64)
        self.run_dir = Path(run_dir)
        self.anchor = anchor
        self.width_nm = width_nm
        self.magnification = width / (width_nm / 1e6 * float(cameras["fly_millimetre"]))

    def describe(self) -> dict:
        return {
            "points_drawn": int(len(self.points)),
            "simulated_cells_lit": int(len(self.lit_index)),
            "simulated_cells_with_position": self.placed_total,
            "region": f"brain only (soma z < {NECK_NM / 1000:.0f} um)",
            "positions": "MaleCNS v1.0 somaLocation, tosomaLocation where soma missing",
            "decay_tau_ms": TAU_MS,
            "magnification_vs_fly": round(self.magnification, 1),
        }

    @lru_cache(maxsize=4)
    def last_spike_ms(self, ply: int) -> Optional[np.ndarray]:
        """For each 10 ms bin, the time of each lit cell's latest spike so far."""
        path = self.run_dir / "neural" / f"ply-{ply:03d}" / "spikes-10ms.npz"
        if not path.exists():
            return None
        with np.load(path) as values:
            counts = np.asarray(values["counts"])
            bin_ms = float(np.asarray(values["bin_ms"]))
        fired = counts[:, self.lit_local] > 0
        bins = np.arange(fired.shape[0], dtype=np.float64)[:, None]
        stamped = np.where(fired, bins * bin_ms + bin_ms / 2.0, -np.inf)
        return np.maximum.accumulate(stamped, axis=0)

    def draw(
        self,
        plate: np.ndarray,
        camera: dict,
        envelope: float,
        ply: Optional[int],
        sim_ms: Optional[float],
    ) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
        height, width = plate.shape[:2]
        view = np.asarray(camera["view"], dtype=np.float64)
        projection = np.asarray(camera["projection"], dtype=np.float64)
        with np.errstate(all="ignore"):
            clip = self.points @ (projection @ view).T
        w = clip[:, 3]
        visible = w > 1e-6
        x = np.full(len(w), -1, dtype=np.int64)
        y = np.full(len(w), -1, dtype=np.int64)
        x[visible] = ((clip[visible, 0] / w[visible] + 1.0) * 0.5 * width).astype(np.int64)
        y[visible] = ((1.0 - clip[visible, 1] / w[visible]) * 0.5 * height).astype(np.int64)
        inside = visible & (x >= 0) & (x < PANEL[0]) & (y >= 0) & (y < height)

        glow = np.zeros_like(plate)
        np.add.at(glow, (y[inside], x[inside]), IDLE_RGB * CONTEXT_GAIN * envelope)

        halo = np.zeros((height // 2, width // 2, 3), dtype=np.float32)
        stamps = self.last_spike_ms(ply) if ply is not None else None
        if stamps is not None and sim_ms is not None and envelope > 0:
            bin_index = min(int(sim_ms // 10.0), stamps.shape[0] - 1)
            since = sim_ms - stamps[bin_index]
            with np.errstate(over="ignore", invalid="ignore"):
                bright = np.where(np.isfinite(since), np.exp(-since / TAU_MS), 0.0)
            hot = bright > 0.02
            chosen = self.lit_index[hot]
            keep = inside[chosen]
            chosen, level = chosen[keep], bright[hot][keep].astype(np.float32)
            colour = GROUP_RGB[self.groups[chosen]] * (level * envelope)[:, None]
            for dy, dx, weight in SPLAT:
                ys = np.clip(y[chosen] + dy, 0, height - 1)
                xs = np.clip(x[chosen] + dx, 0, PANEL[0] - 1)
                np.add.at(glow, (ys, xs), colour * (SPIKE_GAIN * weight))
            np.add.at(halo, (y[chosen] // 2, x[chosen] // 2), colour)

        if halo.any():
            halo_image = Image.fromarray(np.clip(halo * 255.0, 0, 255).astype(np.uint8))
            blurred = halo_image.filter(ImageFilter.GaussianBlur(3.0)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
            glow += np.asarray(blurred, dtype=np.float32) / 255.0 * HALO_GAIN

        with np.errstate(all="ignore"):
            label_clip = projection @ view @ np.r_[self.anchor + [0.0, 0.0, 0.95], 1.0]
        label = None
        if label_clip[3] > 1e-6:
            label = (
                (label_clip[0] / label_clip[3] + 1.0) * 0.5 * width,
                (1.0 - label_clip[1] / label_clip[3]) * 0.5 * height,
            )
        return np.minimum(plate + glow, 1.0), label


def render_frame(
    frame: int,
    plan: dict,
    run: VideoRun,
    plates,
    cameras: dict,
    hologram: BrainHologram,
    panel_brain: dict,
    panel_groups: np.ndarray,
) -> Image.Image:
    shot = _span(plan["shots"], frame)
    ply_entry = _span(plan["plies"], frame)
    index = min(int(ply_entry["ply"]) - 1, len(run.moves) - 1)
    record = run.moves[index]

    envelope, sim_ms, slow = IDLE_ENVELOPE, None, None
    decision_ply = _latest_decision(run, index)
    if shot["name"] == "think":
        start, end = int(shot["start_frame"]), int(shot["end_frame"])
        fade_in = min(1.0, (frame - start) / FADE_FRAMES)
        fade_out = min(1.0, (end - 1 - frame) / FADE_FRAMES)
        envelope = IDLE_ENVELOPE + (1.0 - IDLE_ENVELOPE) * max(0.0, min(fade_in, fade_out))
        play_start, play_end = start + FADE_FRAMES, max(start + FADE_FRAMES + 1, end - FADE_FRAMES)
        progress = min(1.0, max(0.0, (frame - play_start) / (play_end - play_start)))
        sim_ms = progress * WINDOW_MS
        decision_ply = int(shot["ply"])
        slow = (play_end - play_start) / float(plan["fps"]) / (WINDOW_MS / 1000.0)

    with Image.open(plates[frame]) as handle:
        plate = np.asarray(handle.convert("RGB").resize(MASTER), dtype=np.float32) / 255.0
    lit, label = hologram.draw(
        plate, cameras["frames"][str(frame + 1)], envelope, decision_ply, sim_ms
    )
    image = Image.fromarray((lit * 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")

    _scene_overlay(image, draw, plan, run, hologram, decision_ply, envelope, label, slow)
    _panel(image, draw, run, index, record, panel_brain, panel_groups, decision_ply)
    return image


def _span(spans: list[dict], frame: int) -> dict:
    for span in spans:
        if int(span["start_frame"]) <= frame < int(span["end_frame"]):
            return span
    return spans[-1]


def _latest_decision(run: VideoRun, index: int) -> Optional[int]:
    for position in range(index, -1, -1):
        if run.moves[position]["actor"] == "chessfly":
            return int(run.moves[position]["ply"])
    return None


def _scene_overlay(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    plan: dict,
    run: VideoRun,
    hologram: BrainHologram,
    decision_ply: Optional[int],
    envelope: float,
    label: Optional[tuple[float, float]],
    slow: Optional[float],
) -> None:
    # Header wordmark, set like FlyJack's "Fly<span>Jack</span>".
    top = Image.new("RGBA", (PANEL[0], 150), (0, 0, 0, 0))
    shade = ImageDraw.Draw(top)
    for row in range(150):
        shade.line((0, row, PANEL[0], row), fill=(5, 6, 7, int(235 * (1 - row / 150))))
    image.paste(top, (0, 0), top)
    draw.text((44, 34), "Chess", font=_font(42, True), fill=INK)
    offset = draw.textlength("Chess", font=_font(42, True))
    draw.text((44 + offset, 34), "fly", font=_font(42, True), fill=WARM)
    draw.text(
        (46, 88),
        f"frozen MaleCNS  ·  {_opponent(run)}  ·  plasticity off",
        font=_font(18),
        fill=INK2,
    )

    if label is not None and envelope > 0.55:
        alpha = int(255 * min(1.0, (envelope - 0.55) / 0.3))
        text = f"brain, magnified ≈{hologram.magnification:.0f}×"
        font = _font(17, True)
        width = draw.textlength(text, font=font)
        x, y = label
        x = min(max(x, width / 2 + 24), PANEL[0] - width / 2 - 24)
        box = (x - width / 2 - 12, y - 16, x + width / 2 + 12, y + 16)
        draw.rounded_rectangle(box, radius=16, fill=(10, 10, 10, int(alpha * 0.72)), outline=RING + (alpha,), width=1)
        draw.text((x, y), text, font=font, fill=INK2 + (alpha,), anchor="mm")

    # Bottom-left HUD: what the network was shown, and what it did with it.
    hud = (36, 870, 420, 1044)
    draw.rounded_rectangle(hud, radius=12, fill=GLASS + (200,), outline=RING, width=1)
    phase = "THINKING" if envelope > 0.55 else "WATCHING THE BOARD"
    draw.text((52, 884), phase, font=_font(15, True), fill=WARM)
    if decision_ply is not None:
        draw.text((404, 884), f"decision · ply {decision_ply}", font=_font(13), fill=MUTED, anchor="ra")
        stimulus = run.root / "neural" / f"ply-{decision_ply:03d}" / "stimulus.png"
        if stimulus.exists():
            with Image.open(stimulus) as handle:
                thumb = handle.convert("RGB").resize((192, 108), Image.Resampling.NEAREST)
            image.paste(thumb, (52, 910))
            draw.rectangle((51, 909, 244, 1018), outline=AXIS, width=1)
            draw.text((52, 1024), "network input (real stimulus)", font=_font(12), fill=MUTED)
        stats = run.runtime[decision_ply - 1] if decision_ply - 1 < len(run.runtime) else None
        if stats is not None:
            draw.text((260, 910), f"{int(stats['total_spikes']):,}", font=_font(22, True), fill=INK)
            draw.text((260, 938), "spikes / 500 ms", font=_font(12), fill=MUTED)
            draw.text((260, 958), f"{int(stats['readout_spikes'])}", font=_font(22, True), fill=GREEN)
            draw.text((260, 986), "descending readout", font=_font(12), fill=MUTED)
            if run.moves[decision_ply - 1]["tie_break"]:
                draw.rounded_rectangle((260, 1008, 364, 1032), radius=12, fill=AMBER + (46,), outline=AMBER, width=1)
                draw.text((312, 1020), "tie-broken", font=_font(12, True), fill=INK, anchor="mm")

    info = hologram.describe()
    note = (
        f"Neurons at MaleCNS soma positions; {info['simulated_cells_lit']:,} of "
        f"{run.manifest['brain']['neurons']:,} simulated cells placed."
    )
    detail = (
        f"Activity is the recorded 500 ms decision"
        + (f", slowed {slow:.0f}×." if slow else ".")
        + "  Not a biological chess brain."
    )
    # The felt is light, so the disclaimer needs its own dark backing to read.
    font = _font(13)
    width = max(draw.textlength(note, font=font), draw.textlength(detail, font=font))
    draw.rounded_rectangle(
        (PANEL[0] - 38 - width, 992, PANEL[0] - 12, 1042),
        radius=10,
        fill=GLASS + (200,),
        outline=RING,
        width=1,
    )
    draw.text((PANEL[0] - 24, 1004), note, font=font, fill=INK2, anchor="ra")
    draw.text((PANEL[0] - 24, 1024), detail, font=font, fill=INK2, anchor="ra")


def _opponent(run: VideoRun) -> str:
    options = run.manifest.get("stockfish_options") or {}
    if "Skill Level" in options:
        return f"Stockfish skill {options['Skill Level']}"
    return f"Stockfish {run.manifest['stockfish']['elo']}"


def _panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    run: VideoRun,
    index: int,
    record: dict,
    brain: dict,
    groups: np.ndarray,
    decision_ply: Optional[int],
) -> None:
    draw.rounded_rectangle(PANEL, radius=12, fill=GLASS, outline=RING, width=2)
    draw.text((1316, 72), "LIVE BOARD", font=_font(18, True), fill=INK2, anchor="lm")
    draw.text((1868, 72), f"ply {record['ply']} / {len(run.moves)}", font=_font(18), fill=MUTED, anchor="rm")

    art = render_display_board(
        chess.Board(str(record["fen_after"])),
        size=BOARD_SIZE,
        last_move=chess.Move.from_uci(str(record["move_uci"])),
        light_square=(208, 198, 176),
        dark_square=(104, 92, 76),
        border_colour=AXIS,
        highlight_colour=WARM,
    )
    image.paste(art, BOARD_ORIGIN)

    actor = "Chessfly" if record["actor"] == "chessfly" else "Stockfish"
    draw.text((1332, 648), actor, font=_font(20, True), fill=BLUE if actor == "Chessfly" else ORANGE)
    draw.text((1852, 642), str(record["move_san"]), font=_font(36, True), fill=INK, anchor="ra")

    evaluation = int(record["evaluation_after_cp"])
    draw.text((1332, 698), "engine eval", font=_font(15), fill=MUTED)
    track = (1432, 700, 1780, 714)
    draw.rounded_rectangle(track, radius=3, fill=GRID)
    middle = (track[0] + track[2]) / 2
    draw.line((middle, track[1] - 3, middle, track[3] + 3), fill=AXIS, width=1)
    share = max(-1.0, min(1.0, evaluation / EVAL_CLAMP_CP))
    reach = share * (track[2] - track[0]) / 2
    if abs(reach) >= 1:
        bar = (middle, track[1] + 2, middle + reach, track[3] - 2) if reach > 0 else (middle + reach, track[1] + 2, middle, track[3] - 2)
        draw.rounded_rectangle(bar, radius=3, fill=BLUE if reach > 0 else ORANGE)
    draw.text((1852, 707), _evaluation(evaluation), font=_font(18, True), fill=INK, anchor="rm")

    if index == len(run.moves) - 1 and str(run.manifest["result"]) != "*":
        headline, subtitle = _outcome_lines(run)
        draw.rounded_rectangle((1332, 732, 1852, 770), radius=19, fill=CRITICAL + (46,), outline=CRITICAL, width=1)
        draw.text((1592, 751), f"{subtitle.lower()}  ·  {headline.lower()}", font=_font(16, True), fill=INK, anchor="mm")
    elif record["actor"] == "chessfly" and record["tie_break"]:
        draw.rounded_rectangle((1332, 736, 1462, 764), radius=14, fill=AMBER + (46,), outline=AMBER, width=1)
        draw.text((1397, 750), "tie-broken", font=_font(15, True), fill=INK, anchor="mm")

    draw.text((1332, 786), "BRAIN ACTIVITY", font=_font(16, True), fill=INK2)
    draw.text((1852, 788), "latest decision", font=_font(14), fill=MUTED, anchor="ra")
    _panel_brain(draw, run, brain, groups, decision_ply)


def _panel_brain(
    draw: ImageDraw.ImageDraw,
    run: VideoRun,
    brain: dict,
    groups: np.ndarray,
    decision_ply: Optional[int],
) -> None:
    x0, y0, x1, y1 = BRAIN_BOX
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=(26, 26, 25))
    activity = run.activity[decision_ply - 1] if decision_ply else None
    positions = brain["positions"]
    for source, target in brain["edges"]:
        draw.line((*positions[source], *positions[target]), fill=(60, 60, 58, 150), width=1)
    for _, members in brain["columns"]:
        if not len(members):
            continue
        peak = float(np.max(activity[members])) if activity is not None else 0.0
        for neuron in members:
            share = 0.0 if not peak else float(activity[neuron]) / peak
            level = share ** 0.6
            colour = tuple(int(c * 255) for c in GROUP_RGB[groups[neuron]])
            base = tuple(int(0.22 * c) + 12 for c in colour)
            fill = tuple(int(b + (c - b) * level) for b, c in zip(base, colour))
            radius = 2.5 + 4.0 * level
            x, y = positions[neuron]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    for label, members in brain["columns"]:
        if len(members):
            x = float(np.mean(positions[members][:, 0]))
            draw.text((x, y1 + 12), label.lower(), font=_font(13), fill=MUTED, anchor="mt")
    stats = run.runtime[decision_ply - 1] if decision_ply else None
    if stats is not None:
        draw.text((1332, 1006), f"{int(stats['total_spikes']):,} spikes / 500 ms", font=_font(18, True), fill=INK)
        draw.text((1852, 1006), f"{int(stats['readout_spikes'])} descending readout", font=_font(18, True), fill=GREEN, anchor="ra")


def _subgraph_groups(data_dir: Path) -> np.ndarray:
    """Display group for every simulated neuron, placed or not."""
    table = feather.read_table(
        Path(data_dir) / "raw" / select_files(("annotations",))[0].filename,
        columns=["bodyId", "superclass", "type"],
    )
    body = np.asarray(table["bodyId"].to_numpy(), dtype=np.int64)
    order = np.argsort(body)
    node_ids = np.load(
        Path(data_dir) / "compiled" / "mapped-retina-to-descending-h3-w5" / "node_ids.npy"
    )
    rows = order[np.searchsorted(body[order], node_ids)]
    superclass = table["superclass"].take(rows).to_pylist()
    cell_type = table["type"].take(rows).to_pylist()
    return np.asarray(
        [group_of(s, t) for s, t in zip(superclass, cell_type)], dtype=np.uint8
    )


def _write_audio(path: Path, plan: dict, sample_rate: int = 48000) -> None:
    fps = float(plan["fps"])
    samples = np.zeros(int(plan["total_frames"] / fps * sample_rate) + 1, dtype=np.float32)
    rng = np.random.default_rng(20260912)
    for ply in plan["plies"]:
        start = int(ply["land_frame"] / fps * sample_rate)
        length = int(0.11 * sample_rate)
        t = np.arange(length) / sample_rate
        envelope = np.exp(-t * 34)
        frequency = 720 if ply["actor"] == "chessfly" else 310
        tone = 0.16 * np.sin(2 * math.pi * frequency * t) * envelope
        if ply["actor"] == "chessfly":
            tone += 0.035 * rng.standard_normal(length) * envelope
        end = min(len(samples), start + length)
        if end > start:
            samples[start:end] += tone[: end - start]
    time = np.arange(len(samples)) / sample_rate
    samples += 0.010 * np.sin(2 * math.pi * 55 * time)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(samples, -0.95, 0.95) * 32767).astype("<i2").tobytes())
