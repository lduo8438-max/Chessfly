"""Render a 16:9 full-match cut over the original cinematic plates.

The left two thirds keep the rendered laboratory scene.  The right third is a
dedicated instrument screen: a readable board above, and a fly-brain activity
diagram below driven by the recorded spike artifacts.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Sequence

import chess
import numpy as np
from PIL import Image, ImageDraw

from .board_art import render_display_board
from .social_video import (
    VideoRun,
    _evaluation,
    _font,
    _outcome_lines,
    load_video_run,
)


MASTER = (1920, 1080)
BG = (4, 8, 14)
INK = (229, 242, 242)
MUTED = (118, 139, 146)
CYAN = (82, 229, 255)
AMBER = (255, 174, 81)
VIOLET = (156, 111, 255)

PANEL = (1288, 32, 1896, 1048)
BOARD_SIZE = 520
BOARD_ORIGIN = (1332, 108)
BRAIN_BOX = (1332, 808, 1852, 948)
HOLD_PLIES = 4
HOLD_WEIGHT = 5.0
THINK_WEIGHT = 8.0


def render_match_video(
    run_dir: Path,
    frames_dir: Path,
    output: Path,
    data_dir: Path = Path("data"),
    duration_seconds: float = 60.0,
    fps: int = 30,
) -> Path:
    """Composite the recorded match onto the cinematic plates as a 16:9 master."""
    if duration_seconds < 12 or fps < 12:
        raise ValueError("match video requires at least 12 seconds and 12 fps")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to render the MP4")
    run = load_video_run(run_dir)
    plates = _load_plates(frames_dir)
    brain = _brain_layout(run_dir, data_dir, run)
    weights = _pacing(len(run.moves))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".chessfly-match-") as tmp:
        audio_path = Path(tmp) / "telemetry.wav"
        _write_audio(audio_path, run, weights, duration_seconds)
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{MASTER[0]}x{MASTER[1]}", "-r", str(fps), "-i", "-",
            "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        total = round(duration_seconds * fps)
        try:
            for number in range(total):
                frame = _render_frame(
                    run, plates, brain, weights, number / fps, duration_seconds
                )
                process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed while encoding the match video")

    metadata = {
        "schema_version": 1,
        "source_run": str(Path(run_dir).resolve()),
        "plates": str(Path(frames_dir).resolve()),
        "output": str(output.resolve()),
        "duration_seconds": duration_seconds,
        "fps": fps,
        "resolution": list(MASTER),
        "telemetry_only": True,
        "plasticity_shown": False,
        "board_art": "display-only Staunton glyphs; neural stimulus unchanged",
        "brain_diagram": "node brightness normalized per group; counters are raw counts",
        "scientific_boundary": "Wiring-constrained simulation; not a biological chess brain.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _load_plates(frames_dir: Path) -> tuple[Path, ...]:
    """List the plates. A full-length 1920x1080 sequence is gigabytes decoded,
    so frames are opened one at a time while encoding rather than held in RAM."""
    paths = sorted(Path(frames_dir).glob("frame-*.png"))
    if not paths:
        raise FileNotFoundError(f"no cinematic plates found in {frames_dir}")
    return tuple(paths)


def _open_plate(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return image if image.size == MASTER else image.resize(MASTER, Image.Resampling.LANCZOS)


def _pacing(count: int, highlights: Sequence[int] = ()) -> np.ndarray:
    """Weight plies so highlights and the closing moves hold instead of flashing past.

    `highlights` are zero-based ply indices that get a full "think" beat, long
    enough for the recorded brain activity to be seen.
    """
    if count <= 0:
        raise ValueError("a match video needs at least one recorded ply")
    weights = np.ones(count, dtype=np.float64)
    weights[0] = 2.0
    for offset in range(1, min(HOLD_PLIES, count) + 1):
        weights[-offset] = HOLD_WEIGHT
    for index in highlights:
        if not 0 <= index < count:
            raise ValueError(f"highlight ply index {index} is outside the game")
        weights[index] = max(weights[index], THINK_WEIGHT)
    return np.cumsum(weights) / float(weights.sum())


def _ply_index(weights: np.ndarray, fraction: float) -> int:
    return int(np.searchsorted(weights, min(max(fraction, 0.0), 0.999999), side="right"))


def _plate(plates: Sequence[Image.Image], timestamp: float, fps_hint: float = 30.0):
    """Ping-pong the finite plate sequence so a long match never cuts abruptly."""
    if len(plates) == 1:
        return plates[0]
    position = int(timestamp * fps_hint) % (2 * len(plates) - 2)
    if position >= len(plates):
        position = 2 * len(plates) - 2 - position
    return plates[position]


def _brain_layout(run_dir: Path, data_dir: Path, run: VideoRun) -> dict:
    """Place retina, core, and descending neurons in three real, labelled columns."""
    run_dir = Path(run_dir)
    artifacts = sorted((run_dir / "neural").glob("ply-*"))
    if not artifacts:
        raise FileNotFoundError(f"{run_dir} holds no neural artifacts to visualize")
    with np.load(artifacts[0] / "spikes-10ms.npz") as values:
        readout = np.asarray(values["readout_indices"], dtype=np.int64)
        neurons = int(np.asarray(values["counts"]).shape[1])
    with np.load(artifacts[0] / "retina.npz") as values:
        retina = np.asarray(values["local_indices"], dtype=np.int64)

    total = np.zeros(neurons, dtype=np.float64)
    for activity in run.activity:
        if activity is not None:
            total += activity

    is_retina = np.zeros(neurons, dtype=bool)
    is_retina[retina] = True
    is_readout = np.zeros(neurons, dtype=bool)
    is_readout[readout] = True
    core_mask = ~is_retina & ~is_readout

    columns = (
        ("RETINA", _pick(total, is_retina, 34)),
        ("SNN CORE", _pick(total, core_mask, 96)),
        ("DESCENDING", _pick(total, is_readout, 32)),
    )
    x0, y0, x1, y1 = BRAIN_BOX
    top, bottom = y0 + 14, y1 - 14
    boxes = (
        (x0 + 18, x0 + 132),
        (x0 + 156, x0 + 380),
        (x0 + 404, x0 + 502),
    )
    positions = np.zeros((neurons, 2), dtype=np.float64)
    selected: list[int] = []
    for (left, right), (_, members) in zip(boxes, columns):
        for neuron, point in zip(members, _grid(len(members), left, right, top, bottom)):
            positions[neuron] = point
            selected.append(int(neuron))

    chosen = np.asarray(selected, dtype=np.int64)
    edges = _edges(data_dir, chosen, neurons)
    held: list[Optional[np.ndarray]] = []
    previous: Optional[np.ndarray] = None
    for activity in run.activity:
        if activity is not None:
            previous = activity
        held.append(previous)
    return {
        "held_activity": tuple(held),
        "columns": tuple((label, members) for label, members in columns),
        "positions": positions,
        "selected": chosen,
        "edges": edges,
        "is_retina": is_retina,
        "is_readout": is_readout,
    }


def _grid(count: int, left: float, right: float, top: float, bottom: float):
    """Lay a group out as a tidy block that fits its column, whatever its size."""
    if count <= 0:
        return []
    width, height = right - left, bottom - top
    columns = max(1, min(count, round(math.sqrt(count * width / max(height, 1e-6)))))
    rows = math.ceil(count / columns)
    step_x = width / columns
    step_y = height / rows
    return [
        (
            left + step_x * (index % columns + 0.5),
            top + step_y * (index // columns + 0.5),
        )
        for index in range(count)
    ]


def _pick(total: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        return candidates
    order = candidates[np.argsort(total[candidates])[::-1]]
    return np.sort(order[:count])


def _edges(data_dir: Path, selected: np.ndarray, neurons: int) -> np.ndarray:
    """Keep only real compiled edges that join two drawn neurons."""
    root = Path(data_dir) / "compiled" / "mapped-retina-to-descending-h3-w5"
    sources = np.load(root / "source_index.npy", mmap_mode="r")
    targets = np.load(root / "target_index.npy", mmap_mode="r")
    drawn = np.zeros(neurons, dtype=bool)
    drawn[selected] = True
    rows = np.flatnonzero(drawn[np.asarray(sources)] & drawn[np.asarray(targets)])
    rows = rows[:420]
    return np.column_stack(
        (np.asarray(sources)[rows], np.asarray(targets)[rows])
    ).astype(np.int64)


def _render_frame(
    run: VideoRun,
    plates: Sequence[Image.Image],
    brain: dict,
    weights: np.ndarray,
    timestamp: float,
    duration: float,
) -> Image.Image:
    image = _open_plate(_plate(plates, timestamp))
    draw = ImageDraw.Draw(image, "RGBA")
    index = _ply_index(weights, timestamp / duration)
    index = min(index, len(run.moves) - 1)
    record = run.moves[index]

    _scene_chrome(draw, run, timestamp)
    _panel(image, draw, run, brain, index, record)
    return image


def _scene_chrome(draw: ImageDraw.ImageDraw, run: VideoRun, timestamp: float) -> None:
    draw.text((48, 40), "CHESSFLY", font=_font(44, True), fill=INK)
    draw.text((50, 96), "MALECNS v1.0  /  FROZEN BASELINE", font=_font(18, True), fill=CYAN)
    draw.text(
        (48, 962),
        f"STOCKFISH {int(run.manifest['stockfish']['elo'])}  /  "
        f"{int(run.manifest['stockfish']['movetime_ms'])} MS",
        font=_font(18, True),
        fill=AMBER,
    )
    draw.text(
        (48, 996),
        "WIRING-CONSTRAINED SIMULATION.  NOT A BIOLOGICAL CHESS BRAIN.  PLASTICITY OFF.",
        font=_font(16, True),
        fill=MUTED,
    )
    draw.text(
        (1256, 996), f"T+{timestamp:05.1f}", font=_font(16), fill=MUTED, anchor="ra"
    )


def _panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    run: VideoRun,
    brain: dict,
    index: int,
    record: dict,
) -> None:
    draw.rounded_rectangle(PANEL, radius=16, fill=(3, 8, 13), outline=(46, 83, 93), width=2)
    actor = str(record["actor"]).upper()
    accent = CYAN if actor == "CHESSFLY" else AMBER
    draw.text((1316, 72), "LIVE BOARD", font=_font(20, True), fill=CYAN, anchor="lm")
    draw.text(
        (1868, 72),
        f"PLY {record['ply']:02d} / {len(run.moves)}",
        font=_font(20, True),
        fill=MUTED,
        anchor="rm",
    )

    board = chess.Board(str(record["fen_after"]))
    art = render_display_board(
        board,
        size=BOARD_SIZE,
        last_move=chess.Move.from_uci(str(record["move_uci"])),
    )
    image.paste(art, BOARD_ORIGIN)

    draw.text((1332, 664), actor, font=_font(20, True), fill=accent)
    draw.text((1852, 664), str(record["move_san"]), font=_font(34, True), fill=INK, anchor="ra")
    draw.text((1332, 706), "ENGINE EVAL", font=_font(15, True), fill=MUTED)
    draw.text(
        (1332, 728),
        _evaluation(int(record["evaluation_after_cp"])),
        font=_font(30, True),
        fill=INK,
    )
    if index == len(run.moves) - 1 and str(run.manifest["result"]) != "*":
        headline, subtitle = _outcome_lines(run)
        draw.text((1852, 700), subtitle, font=_font(17, True), fill=AMBER, anchor="ra")
        draw.text((1852, 726), headline, font=_font(22, True), fill=INK, anchor="ra")
    elif record["actor"] == "chessfly" and record["tie_break"]:
        draw.text((1852, 728), "TIE-BROKEN", font=_font(18, True), fill=AMBER, anchor="ra")

    draw.text((1332, 786), "FLY BRAIN ACTIVITY", font=_font(18, True), fill=VIOLET)
    _brain(draw, run, brain, index, accent)

    stats = run.runtime[index]
    if stats is None:
        draw.text(
            (1332, 1006),
            "STOCKFISH RESPONSE  /  NEURAL STATE HELD",
            font=_font(19, True),
            fill=AMBER,
        )
    else:
        draw.text(
            (1332, 1006),
            f"{int(stats['total_spikes']):,} SPIKES / 500 MS",
            font=_font(19, True),
            fill=INK,
        )
        draw.text(
            (1852, 1006),
            f"{int(stats['readout_spikes'])} DN READOUT",
            font=_font(19, True),
            fill=CYAN,
            anchor="ra",
        )


def _brain(
    draw: ImageDraw.ImageDraw,
    run: VideoRun,
    brain: dict,
    index: int,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = BRAIN_BOX
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=(6, 13, 19, 210))
    positions = brain["positions"]
    live = run.activity[index] is not None
    activity = brain["held_activity"][index]
    if not live:
        draw.text((x1 - 10, y0 + 10), "HELD", font=_font(12, True), fill=MUTED, anchor="rt")

    for source, target in brain["edges"]:
        draw.line(
            (
                positions[source][0], positions[source][1],
                positions[target][0], positions[target][1],
            ),
            fill=(30, 58, 68, 150),
            width=1,
        )

    scale = 1.0 if live else 0.45
    for _, members in brain["columns"]:
        if not len(members):
            continue
        # Normalize within each group: the retina always outruns the core, and a
        # single global peak would render the core as a dead field.
        group_peak = (
            float(np.max(activity[members])) if activity is not None else 0.0
        )
        for neuron in members:
            x, y = positions[neuron]
            share = 0.0 if not group_peak else float(activity[neuron]) / group_peak
            value = (share ** 0.6) * scale
            if brain["is_readout"][neuron]:
                colour = CYAN if value > 0 else (44, 78, 88)
                radius = 4 + 5 * value
            elif brain["is_retina"][neuron]:
                colour = VIOLET if value > 0 else (52, 44, 82)
                radius = 3 + 4 * value
            else:
                colour = accent if value > 0.2 else (48, 72, 80)
                radius = 2.5 + 4 * value
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)

    for label, members in brain["columns"]:
        if not len(members):
            continue
        x = float(np.mean(positions[members][:, 0]))
        draw.text((x, y1 + 10), label, font=_font(13, True), fill=MUTED, anchor="mt")
        draw.text(
            (x, y1 + 28), f"{len(members)} SHOWN", font=_font(11), fill=(70, 92, 98), anchor="mt"
        )


def _write_audio(
    path: Path,
    run: VideoRun,
    weights: np.ndarray,
    duration: float,
    sample_rate: int = 48000,
) -> None:
    """Place one synthesized tick per ply on the same pacing curve as the picture."""
    samples = np.zeros(round(duration * sample_rate), dtype=np.float32)
    rng = np.random.default_rng(20260912)
    starts = np.concatenate(([0.0], weights[:-1]))
    for index, record in enumerate(run.moves):
        event = float(starts[index]) * duration
        start = int(event * sample_rate)
        length = int(0.11 * sample_rate)
        t = np.arange(length) / sample_rate
        frequency = 720 if record["actor"] == "chessfly" else 310
        envelope = np.exp(-t * 34)
        tone = 0.16 * np.sin(2 * math.pi * frequency * t) * envelope
        if record["actor"] == "chessfly":
            tone += 0.035 * rng.standard_normal(length) * envelope
        end = min(len(samples), start + length)
        if end > start:
            samples[start:end] += tone[: end - start]
    time = np.arange(len(samples)) / sample_rate
    samples += 0.012 * np.sin(2 * math.pi * 55 * time)
    samples = np.clip(samples, -0.95, 0.95)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
