"""Encode Blender frames with a minimal real-telemetry HUD and synthesized audio."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (960, 540)
CYAN = (74, 228, 255, 255)
AMBER = (255, 119, 73, 255)
INK = (228, 241, 242, 255)
MUTED = (117, 144, 150, 255)


def font(size: int, bold: bool = False):
    for path in (
        "/System/Library/Fonts/SFCompact.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size, index=1 if bold else 0)
        except OSError:
            pass
    return ImageFont.load_default()


def neural_artifact(run_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    artifact = sorted((run_dir / "neural").glob("ply-*"))[0]
    with np.load(artifact / "spikes-10ms.npz") as values:
        bins = np.asarray(values["counts"].sum(axis=1), dtype=np.float64)
    runtime = json.loads((artifact / "runtime.json").read_text(encoding="utf-8"))
    return bins, runtime


def make_overlay(path: Path, run_dir: Path) -> dict[str, object]:
    bins, runtime = neural_artifact(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    image = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((22, 18), "CHESSFLY", font=font(22, True), fill=INK)
    draw.text((22, 44), "MALECNS v1.0", font=font(9, True), fill=CYAN)
    draw.text((938, 23), "STOCKFISH 1320", font=font(10, True), fill=AMBER, anchor="ra")
    draw.text((938, 43), "b1  >  c3", font=font(14, True), fill=INK, anchor="ra")

    box = (20, 397, 397, 516)
    draw.rounded_rectangle(box, radius=6, fill=(2, 8, 14, 218), outline=(46, 83, 93, 235), width=1)
    draw.text((34, 410), "NEURAL REPLAY", font=font(9, True), fill=CYAN)
    draw.text((382, 410), "500 ms", font=font(9, True), fill=MUTED, anchor="ra")
    draw.text(
        (34, 431),
        f"{int(runtime['total_spikes']):,}",
        font=font(25, True),
        fill=INK,
    )
    draw.text((150, 443), "SPIKES", font=font(8, True), fill=MUTED)
    x0, x1, y0, y1 = 206, 381, 431, 458
    maximum = max(1.0, float(np.max(bins)))
    for index, value in enumerate(bins):
        x = x0 + int(index / max(1, len(bins) - 1) * (x1 - x0))
        height = max(1, int(value / maximum * (y1 - y0)))
        draw.line((x, y1, x, y1 - height), fill=CYAN, width=2)
    draw.line((34, 470, 382, 470), fill=(42, 72, 80, 255), width=1)
    draw.text((34, 484), "8,598 NEURONS", font=font(8, True), fill=MUTED)
    draw.text((137, 484), "70,308 EDGES", font=font(8, True), fill=MUTED)
    draw.text(
        (244, 484),
        f"{int(runtime['readout_spikes'])} DN READOUT",
        font=font(8, True),
        fill=CYAN,
    )
    draw.text((938, 510), "WIRING-CONSTRAINED SIMULATION", font=font(8, True), fill=MUTED, anchor="ra")
    image.save(path)
    return {
        "runtime": runtime,
        "source_run": str(run_dir.resolve()),
        "overlay": "real first-decision 10 ms spike-bin telemetry",
        "scene_geometry": "illustrative original low-poly Blender scene",
    }


def make_audio(path: Path, bins: np.ndarray, seconds: float = 10.0, rate: int = 48000) -> None:
    count = round(seconds * rate)
    t = np.arange(count, dtype=np.float64) / rate
    rng = np.random.default_rng(20260912)
    envelope = np.linspace(0.75, 1.0, count)
    audio = envelope * (
        0.018 * np.sin(2 * math.pi * 48 * t)
        + 0.011 * np.sin(2 * math.pi * 73 * t)
        + 0.004 * rng.standard_normal(count)
    )
    maximum = max(1.0, float(np.max(bins)))
    for index, value in enumerate(bins):
        event = 0.7 + index / len(bins) * 8.3
        start = int(event * rate)
        length = int(0.018 * rate)
        local = np.arange(length) / rate
        click = (
            0.025
            * (value / maximum)
            * np.sin(2 * math.pi * (700 + 900 * value / maximum) * local)
            * np.exp(-local * 160)
        )
        end = min(count, start + length)
        audio[start:end] += click[: end - start]
    # Knight movement at frames 115–195, with a soft landing near 6.5 seconds.
    start = int(5.1 * rate)
    length = int(1.2 * rate)
    local = np.arange(length) / rate
    whoosh = 0.020 * rng.standard_normal(length) * np.sin(math.pi * local / 1.2) ** 2
    audio[start : start + length] += whoosh
    start = int(6.5 * rate)
    length = int(0.16 * rate)
    local = np.arange(length) / rate
    audio[start : start + length] += 0.10 * np.sin(2 * math.pi * 165 * local) * np.exp(-local * 30)
    audio = np.clip(audio, -0.95, 0.95)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())


def encode(frames: Path, run_dir: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    bins, _ = neural_artifact(run_dir)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".cinematic-post-") as temp:
        temp_dir = Path(temp)
        overlay = temp_dir / "hud.png"
        audio = temp_dir / "sound.wav"
        metadata = make_overlay(overlay, run_dir)
        make_audio(audio, bins)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "30",
            "-start_number",
            "1",
            "-i",
            str(frames / "frame-%04d.png"),
            "-loop",
            "1",
            "-i",
            str(overlay),
            "-i",
            str(audio),
            "-filter_complex",
            "[0:v][1:v]overlay=0:0,scale=1920:1080:flags=lanczos[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-t",
            "10",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)
    metadata.update(
        {
            "output": str(output.resolve()),
            "duration_seconds": 10,
            "fps": 30,
            "resolution": [1920, 1080],
            "audio": "locally synthesized; no third-party music",
        }
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def encode_instagram(master: Path, output: Path) -> None:
    """Create a 9:16 delivery file while preserving the complete 16:9 scene."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=28,eq=brightness=-0.18[background];"
        "[0:v]scale=1080:-2[foreground];"
        "[background][foreground]overlay=(W-w)/2:(H-h)/2[video]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(master),
            "-filter_complex",
            filter_graph,
            "-map",
            "[video]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instagram-output", type=Path)
    args = parser.parse_args()
    encode(args.frames, args.run_dir, args.output)
    if args.instagram_output:
        encode_instagram(args.output, args.instagram_output)
    print(args.output.resolve())
    if args.instagram_output:
        print(args.instagram_output.resolve())


if __name__ == "__main__":
    main()
