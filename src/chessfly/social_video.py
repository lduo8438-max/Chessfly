"""Render an original telemetry-driven vertical Chessfly social video."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .vision import BOARD_ORIGIN, BOARD_PIXELS, render_board_stimulus


CANVAS = (540, 960)
MASTER = (1080, 1920)
BG = (4, 8, 14)
INK = (229, 242, 242)
MUTED = (118, 139, 146)
CYAN = (82, 229, 255)
AMBER = (255, 174, 81)
VIOLET = (156, 111, 255)


@dataclass(frozen=True)
class VideoRun:
    root: Path
    manifest: dict[str, object]
    moves: tuple[dict[str, object], ...]
    boards: tuple[Image.Image, ...]
    activity: tuple[np.ndarray | None, ...]
    runtime: tuple[dict[str, object] | None, ...]


def load_video_run(run_dir: Path) -> VideoRun:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    moves = tuple(
        json.loads(line)
        for line in (run_dir / "moves.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not moves:
        raise ValueError("video source run contains no moves")
    boards = []
    activity = []
    runtime = []
    x0, y0 = BOARD_ORIGIN
    for record in moves:
        board = chess.Board(str(record["fen_after"]))
        last_move = chess.Move.from_uci(str(record["move_uci"]))
        stimulus = render_board_stimulus(board, chess.WHITE, last_move)
        board_image = stimulus.crop(
            (x0, y0, x0 + BOARD_PIXELS, y0 + BOARD_PIXELS)
        ).resize((430, 430), Image.Resampling.LANCZOS)
        boards.append(board_image)
        artifact = run_dir / "neural" / f"ply-{int(record['ply']):03d}"
        if artifact.exists():
            with np.load(artifact / "spikes-10ms.npz") as raster:
                activity.append(np.asarray(raster["counts"].sum(axis=0)))
            runtime.append(
                json.loads((artifact / "runtime.json").read_text(encoding="utf-8"))
            )
        else:
            activity.append(None)
            runtime.append(None)
    return VideoRun(run_dir, manifest, moves, tuple(boards), tuple(activity), tuple(runtime))


def render_social_video(
    run_dir: Path,
    output: Path,
    data_dir: Path = Path("data"),
    duration_seconds: float = 30.0,
    fps: int = 30,
) -> Path:
    if duration_seconds < 12 or fps < 12:
        raise ValueError("social video requires at least 12 seconds and 12 fps")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to render the MP4")
    run = load_video_run(run_dir)
    graph = _video_graph(data_dir, run)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".chessfly-video-") as tmp:
        audio_path = Path(tmp) / "telemetry.wav"
        _write_audio(audio_path, run, duration_seconds)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{CANVAS[0]}x{CANVAS[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(audio_path),
            "-vf",
            f"scale={MASTER[0]}:{MASTER[1]}:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        frames = round(duration_seconds * fps)
        try:
            for frame_number in range(frames):
                timestamp = frame_number / fps
                image = _render_frame(run, graph, timestamp, duration_seconds)
                process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        finally:
            process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    metadata = {
        "schema_version": 1,
        "source_run": str(Path(run_dir).resolve()),
        "output": str(output.resolve()),
        "duration_seconds": duration_seconds,
        "fps": fps,
        "resolution": list(MASTER),
        "telemetry_only": True,
        "plasticity_shown": False,
        "scientific_boundary": "Wiring-constrained simulation; not a biological chess brain.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _video_graph(data_dir: Path, run: VideoRun) -> dict[str, np.ndarray]:
    root = Path(data_dir) / "compiled" / "mapped-retina-to-descending-h3-w5"
    sources = np.load(root / "source_index.npy", mmap_mode="r")
    targets = np.load(root / "target_index.npy", mmap_mode="r")
    summed = np.zeros(int(run.manifest["brain"]["neurons"]), dtype=np.float64)
    for values in run.activity:
        if values is not None:
            summed += values
    ranked = np.argsort(summed)[::-1]
    selected = ranked[summed[ranked] > 0][:140]
    if len(selected) < 80:
        selected = ranked[:140]
    lookup = np.full(len(summed), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected), dtype=np.int32)
    edge_mask = (lookup[sources] >= 0) & (lookup[targets] >= 0)
    edge_rows = np.flatnonzero(edge_mask)[:360]
    rng = np.random.default_rng(8598)
    theta = np.linspace(0, 2 * math.pi, len(selected), endpoint=False)
    radius = 0.54 + 0.35 * rng.random(len(selected))
    positions = np.column_stack(
        (
            270 + 215 * radius * np.cos(theta),
            738 + 112 * radius * np.sin(theta),
        )
    ).astype(np.int32)
    return {
        "selected": selected,
        "positions": positions,
        "edge_source": lookup[sources[edge_rows]],
        "edge_target": lookup[targets[edge_rows]],
        "total_activity": summed[selected],
    }


def _render_frame(
    run: VideoRun,
    graph: dict[str, np.ndarray],
    timestamp: float,
    duration: float,
) -> Image.Image:
    image = _background()
    draw = ImageDraw.Draw(image)
    _chrome(draw, timestamp)
    intro_end = duration * 0.13
    mechanism_end = duration * 0.29
    match_end = duration * 0.84
    if timestamp < intro_end:
        _intro(draw, graph, timestamp, intro_end)
    elif timestamp < mechanism_end:
        _mechanism(draw, timestamp - intro_end, mechanism_end - intro_end)
    elif timestamp < match_end:
        _match(image, draw, run, graph, timestamp, mechanism_end, match_end)
    else:
        _result(draw, run, timestamp - match_end, duration - match_end)
    return image


def _background() -> Image.Image:
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)
    for y in range(CANVAS[1]):
        glow = int(10 * max(0, 1 - abs(y - 570) / 480))
        draw.line((0, y, CANVAS[0], y), fill=(4, 8 + glow // 2, 14 + glow))
    for x in range(0, CANVAS[0], 36):
        draw.line((x, 0, x, CANVAS[1]), fill=(10, 20, 27))
    for y in range(0, CANVAS[1], 36):
        draw.line((0, y, CANVAS[0], y), fill=(10, 20, 27))
    return image


def _chrome(draw: ImageDraw.ImageDraw, timestamp: float) -> None:
    draw.text((34, 28), "CHESSFLY", font=_font(22, True), fill=INK)
    draw.text((34, 55), "MALECNS v1.0  /  FROZEN BASELINE", font=_font(10), fill=CYAN)
    draw.text((505, 32), f"T+{timestamp:05.1f}", font=_font(10), fill=MUTED, anchor="ra")
    draw.line((34, 78, 506, 78), fill=(36, 62, 70), width=1)


def _intro(
    draw: ImageDraw.ImageDraw,
    graph: dict[str, np.ndarray],
    timestamp: float,
    end: float,
) -> None:
    pulse = 0.5 + 0.5 * math.sin(timestamp * math.pi * 2)
    _network(draw, graph, None, pulse, center_shift=-90)
    draw.text((34, 555), "A FLY CONNECTOME", font=_font(36, True), fill=INK)
    draw.text((34, 598), "SIMULATION", font=_font(36, True), fill=CYAN)
    draw.text((34, 642), "PLAYS STOCKFISH", font=_font(36, True), fill=INK)
    draw.text(
        (36, 706),
        "8,598 neurons. 70,308 connections.\nNo chess evaluation enters the neural controller.",
        font=_font(15),
        fill=MUTED,
        spacing=7,
    )
    progress = min(1.0, timestamp / max(end, 0.001))
    draw.line((34, 874, 34 + int(472 * progress), 874), fill=CYAN, width=3)


def _mechanism(draw: ImageDraw.ImageDraw, local: float, span: float) -> None:
    draw.text((34, 118), "THE SIGNAL PATH", font=_font(30, True), fill=INK)
    labels = (
        ("01", "RGB BOARD", "320 × 180 rendered pixels"),
        ("02", "2,147 RETINA INPUTS", "R1–R6 luminance / R8 color"),
        ("03", "8,598-NEURON SNN", "0.1 ms leaky integrate-and-fire"),
        ("04", "960 DESCENDING CELLS", "32 channels score legal moves"),
    )
    active = min(len(labels) - 1, int(local / span * len(labels)))
    for index, (number, title, body) in enumerate(labels):
        y = 220 + index * 140
        color = CYAN if index <= active else (54, 76, 83)
        draw.ellipse((34, y, 72, y + 38), outline=color, width=2)
        draw.text((53, y + 19), number, font=_font(11, True), fill=color, anchor="mm")
        draw.line((53, y + 38, 53, y + 122), fill=color, width=2)
        draw.text((94, y), title, font=_font(21, True), fill=INK if index <= active else MUTED)
        draw.text((94, y + 34), body, font=_font(13), fill=MUTED)
    draw.text(
        (34, 830),
        "ANATOMY IS DATA.  DYNAMICS + CHESS READOUT ARE MODEL CHOICES.",
        font=_font(10, True),
        fill=AMBER,
    )


def _match(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    run: VideoRun,
    graph: dict[str, np.ndarray],
    timestamp: float,
    start: float,
    end: float,
) -> None:
    fraction = min(0.999999, max(0.0, (timestamp - start) / (end - start)))
    index = min(len(run.moves) - 1, int(fraction * len(run.moves)))
    record = run.moves[index]
    image.paste(run.boards[index], (55, 128))
    actor = str(record["actor"]).upper()
    actor_color = CYAN if actor == "CHESSFLY" else AMBER
    draw.rectangle((55, 96, 485, 122), fill=(9, 19, 27))
    draw.text((67, 109), f"PLY {record['ply']:02d}  /  {actor}", font=_font(11, True), fill=actor_color, anchor="lm")
    draw.text((475, 109), str(record["move_san"]), font=_font(17, True), fill=INK, anchor="rm")
    repeats = _repeat_count(run, index)
    if repeats >= 2:
        draw.text(
            (313, 110),
            f"REPEATED {repeats}x  /  TIE-BROKEN",
            font=_font(9, True),
            fill=AMBER,
            anchor="mm",
        )

    evaluation = int(record["evaluation_after_cp"])
    reward = float(record["shaped_reward"])
    draw.text((55, 582), "ENGINE EVAL", font=_font(10, True), fill=MUTED)
    draw.text((55, 604), _evaluation(evaluation), font=_font(26, True), fill=INK)
    draw.text((220, 582), "RECORDED REWARD", font=_font(10, True), fill=MUTED)
    reward_color = CYAN if reward > 0 else AMBER if reward < 0 else MUTED
    draw.text((220, 604), f"{reward:+.3f}", font=_font(26, True), fill=reward_color)
    draw.text((392, 582), "MOVE", font=_font(10, True), fill=MUTED)
    draw.text((392, 604), str(record["move_uci"]), font=_font(18, True), fill=INK)

    activity = run.activity[index]
    phase = (fraction * len(run.moves)) % 1.0
    _network(draw, graph, activity, phase)
    stats = run.runtime[index]
    if stats is None:
        status = "STOCKFISH RESPONSE  /  NEURAL STATE HELD"
    else:
        status = (
            f"500 ms  /  {int(stats['total_spikes']):,} SPIKES  /  "
            f"{int(stats['readout_spikes'])} DN READOUT"
        )
    draw.text((34, 890), status, font=_font(11, True), fill=actor_color)
    draw.text((34, 916), "REAL RUN TELEMETRY  /  LAYOUT ILLUSTRATIVE", font=_font(9), fill=MUTED)


def _network(
    draw: ImageDraw.ImageDraw,
    graph: dict[str, np.ndarray],
    activity: np.ndarray | None,
    phase: float,
    center_shift: int = 0,
) -> None:
    positions = graph["positions"].copy()
    positions[:, 1] += center_shift
    for source, target in zip(graph["edge_source"], graph["edge_target"]):
        a, b = positions[source], positions[target]
        draw.line((int(a[0]), int(a[1]), int(b[0]), int(b[1])), fill=(20, 48, 58), width=1)
    if activity is None:
        node_activity = graph["total_activity"]
    else:
        node_activity = activity[graph["selected"]]
    maximum = max(1.0, float(np.max(node_activity)))
    wave = 0.65 + 0.35 * math.sin(phase * math.pi * 2)
    for point, value in zip(positions, node_activity):
        intensity = math.sqrt(float(value) / maximum) if value > 0 else 0.0
        radius = 1 + int(3 * intensity * wave)
        color = (
            int(28 + intensity * (CYAN[0] - 28)),
            int(57 + intensity * (CYAN[1] - 57)),
            int(65 + intensity * (CYAN[2] - 65)),
        )
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _outcome_lines(run: VideoRun) -> tuple[str, str]:
    """Name the recorded ending without softening a loss or inflating a stop."""
    plies = len(run.moves)
    termination = str(run.manifest["termination"])
    result = str(run.manifest["result"])
    if termination == "MAX_PLIES" or result == "*":
        return f"{plies}-PLY DEMO", "STOPPED AT THE PLY CAP"
    readable = termination.replace("_", " ")
    move_number = (plies + 1) // 2
    if result == "1/2-1/2":
        return f"{plies}-PLY FULL GAME", f"DRAWN BY {readable} ON MOVE {move_number}"
    chessfly_white = str(run.manifest["chessfly_color"]) == "white"
    won = result == ("1-0" if chessfly_white else "0-1")
    return (
        f"{plies}-PLY FULL GAME",
        f"{'WON' if won else 'LOST'} BY {readable} ON MOVE {move_number}",
    )


def _decision_summary(run: VideoRun) -> tuple[int, float, float]:
    """Report how many decisions were decoded, and how weakly they were driven."""
    decisions = [record for record in run.moves if record["actor"] == "chessfly"]
    if not decisions:
        return 0, 0.0, 0.0
    tie_broken = sum(1 for record in decisions if record["tie_break"])
    spikes = sum(len(record["output_spikes"]) for record in decisions)
    return len(decisions), 100.0 * tie_broken / len(decisions), spikes / len(decisions)


def _repeat_count(run: VideoRun, index: int) -> int:
    """Count how often a tie-broken Chessfly move has already been replayed."""
    record = run.moves[index]
    if record["actor"] != "chessfly" or not record["tie_break"]:
        return 0
    return sum(
        1
        for earlier in run.moves[: index + 1]
        if earlier["actor"] == "chessfly"
        and earlier["move_uci"] == record["move_uci"]
    )


def _result(
    draw: ImageDraw.ImageDraw, run: VideoRun, local: float, span: float
) -> None:
    headline, subtitle = _outcome_lines(run)
    draw.text((34, 120), "WHAT HAPPENED", font=_font(31, True), fill=INK)
    draw.text((34, 173), headline, font=_font(44, True), fill=CYAN)
    draw.text((34, 224), subtitle, font=_font(16, True), fill=MUTED)
    rows = (
        ("MALECNS FROZEN", -276.6, CYAN),
        ("FIRST LEGAL", -84.9, MUTED),
        ("RANDOM", -114.7, AMBER),
    )
    draw.text((34, 320), "10-POSITION CHECK  /  MEAN Δ CENTIPAWNS", font=_font(11, True), fill=MUTED)
    for index, (label, value, color) in enumerate(rows):
        y = 370 + index * 82
        draw.text((34, y), label, font=_font(14, True), fill=INK)
        width = int(min(420, abs(value) / 300 * 420))
        draw.rectangle((34, y + 31, 34 + width, y + 43), fill=color)
        draw.text((495, y + 36), f"{value:+.1f}", font=_font(13, True), fill=color, anchor="rm")
    decisions, tie_percent, mean_spikes = _decision_summary(run)
    draw.text(
        (34, 598),
        f"THIS GAME  /  {decisions} NEURAL DECISIONS",
        font=_font(11, True),
        fill=MUTED,
    )
    draw.text((34, 620), f"{tie_percent:.0f}% TIE-BROKEN", font=_font(15, True), fill=AMBER)
    draw.text(
        (250, 620),
        f"{mean_spikes:.1f} DN SPIKES / DECISION",
        font=_font(15, True),
        fill=CYAN,
    )
    draw.line((34, 648, 506, 648), fill=(43, 73, 80), width=1)
    draw.text((34, 690), "THE HONEST RESULT", font=_font(12, True), fill=AMBER)
    draw.text((34, 725), "IT RUNS.\nIT DOES NOT YET PLAY WELL.", font=_font(29, True), fill=INK, spacing=7)
    draw.text(
        (34, 835),
        "WIRING-CONSTRAINED SIMULATION.\nNOT A BIOLOGICAL CHESS BRAIN.  PLASTICITY OFF.",
        font=_font(12, True),
        fill=MUTED,
        spacing=6,
    )


def _write_audio(path: Path, run: VideoRun, duration: float, sample_rate: int = 48000) -> None:
    samples = np.zeros(round(duration * sample_rate), dtype=np.float32)
    rng = np.random.default_rng(20260912)
    match_start, match_end = duration * 0.29, duration * 0.84
    for index, record in enumerate(run.moves):
        event = match_start + (index + 0.2) / len(run.moves) * (match_end - match_start)
        start = int(event * sample_rate)
        length = int(0.11 * sample_rate)
        t = np.arange(length) / sample_rate
        frequency = 720 if record["actor"] == "chessfly" else 310
        envelope = np.exp(-t * 34)
        tone = 0.16 * np.sin(2 * math.pi * frequency * t) * envelope
        if record["actor"] == "chessfly":
            tone += 0.035 * rng.standard_normal(length) * envelope
        end = min(len(samples), start + length)
        samples[start:end] += tone[: end - start]
    # A quiet synthetic laboratory bed; generated locally, no third-party music.
    time = np.arange(len(samples)) / sample_rate
    samples += 0.012 * np.sin(2 * math.pi * 55 * time)
    samples = np.clip(samples, -0.95, 0.95)
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _evaluation(value: int) -> str:
    if abs(value) >= 100000:
        return "MATE"
    return f"{value / 100:+.2f}"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFCompact.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default()
