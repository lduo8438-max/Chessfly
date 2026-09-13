"""Chessfly command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import chess

from .annotations import (
    descending_body_ids,
    load_source_tables,
    retained_neuron_body_ids,
    summarize_annotations,
)
from .benchmark import POSITION_LINES, run_fixed_benchmark, run_seed_sweep
from .brain_cloud import build_brain_cloud
from .compiled_graph import build_path_subgraph, compile_retained_csr
from .dataset import download_file, inventory, select_files, write_manifest
from .fly_model import prepare_fly_model
from .flyjack_video import render_flyjack_video
from .game import play_game
from .graph import summarize_connection_file
from .male_cns_brain import MaleCNSSubgraphBrain
from .match_plan import write_plan
from .match_video import render_match_video
from .retina import build_retina_projection, load_retina_body_ids
from .social_video import render_social_video
from .stockfish import StockfishConfig, StockfishOpponent
from .toy_brain import ToyChessBrain
from .vision import save_board_stimulus


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"smoke-{stamp}"


def _report(result) -> None:
    print(f"result={result.result} termination={result.termination}")
    print(f"plies={result.plies}")
    if result.resumed_from_ply:
        print(f"resumed_from_ply={result.resumed_from_ply}")
    if result.engine_restarts:
        print(f"stockfish_restarts={result.engine_restarts}")
    print(f"run_dir={result.run_dir.resolve()}")


def run_smoke(args: argparse.Namespace) -> int:
    config = StockfishConfig(
        path=args.stockfish_path,
        elo=args.stockfish_elo,
        movetime_ms=args.movetime_ms,
    )
    color = chess.WHITE if args.color == "white" else chess.BLACK
    with StockfishOpponent(config) as stockfish:
        result = play_game(
            ToyChessBrain(),
            stockfish,
            args.run_dir or _default_run_dir(),
            chessfly_color=color,
            max_plies=args.max_plies,
            progress=print,
        )
    print(f"mode=toy-not-male-cns")
    _report(result)
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    selected = select_files(args.files)
    if not args.download:
        selected_keys = {item.key for item in selected}
        print(
            json.dumps(
                [item for item in inventory() if item["key"] in selected_keys],
                indent=2,
                sort_keys=True,
            )
        )
        print("dry_run=true; pass --download to fetch these public files")
        return 0
    raw_dir = args.data_dir / "raw"
    records = []
    for item in selected:
        print(f"preparing {item.key}: {item.url}")
        record = download_file(item, raw_dir)
        records.append(record)
        print(
            f"{record['state']} {record['filename']} "
            f"bytes={record['bytes']} sha256={record['sha256']}"
        )
    write_manifest(records, args.data_dir / "manifest.json")
    return 0


def run_prepare_fly_model(args: argparse.Namespace) -> int:
    summary = prepare_fly_model(args.data_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_build_brain_cloud(args: argparse.Namespace) -> int:
    print(json.dumps(build_brain_cloud(args.data_dir), indent=2, sort_keys=True))
    return 0


def run_inspect_data(args: argparse.Namespace) -> int:
    annotations, transmitters = load_source_tables(args.data_dir)
    summary = summarize_annotations(annotations, transmitters)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def run_inspect_graph(args: argparse.Namespace) -> int:
    annotations, _ = load_source_tables(args.data_dir)
    retained = retained_neuron_body_ids(annotations)
    weights = next(
        item for item in select_files(("weights",)) if item.key == "weights"
    )
    summary = summarize_connection_file(
        args.data_dir / "raw" / weights.filename, retained
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _weights_path(data_dir: Path) -> Path:
    item = select_files(("weights",))[0]
    return data_dir / "raw" / item.filename


def run_compile_graph(args: argparse.Namespace) -> int:
    annotations, _ = load_source_tables(args.data_dir)
    summary = compile_retained_csr(
        _weights_path(args.data_dir),
        retained_neuron_body_ids(annotations),
        args.data_dir / "compiled" / "retained",
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def run_build_subgraph(args: argparse.Namespace) -> int:
    annotations, _ = load_source_tables(args.data_dir)
    summary = build_path_subgraph(
        args.data_dir / "compiled" / "retained",
        load_retina_body_ids(args.data_dir / "compiled" / "retina"),
        descending_body_ids(annotations),
        args.data_dir
        / "compiled"
        / f"mapped-retina-to-descending-h{args.maximum_hops}-w{args.minimum_weight}",
        maximum_hops=args.maximum_hops,
        minimum_weight=args.minimum_weight,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def run_render_board(args: argparse.Namespace) -> int:
    board = chess.Board(args.fen)
    perspective = chess.WHITE if args.perspective == "white" else chess.BLACK
    last_move = chess.Move.from_uci(args.last_move) if args.last_move else None
    save_board_stimulus(board, args.output, perspective, last_move)
    print(f"output={args.output.resolve()}")
    return 0


def run_build_retina(args: argparse.Namespace) -> int:
    annotations, _ = load_source_tables(args.data_dir)
    summary = build_retina_projection(
        annotations,
        args.data_dir / "compiled" / "retained",
        args.data_dir / "compiled" / "retina",
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def run_male_cns_decision(args: argparse.Namespace) -> int:
    board = chess.Board(args.fen)
    perspective = chess.WHITE if args.perspective == "white" else chess.BLACK
    brain = MaleCNSSubgraphBrain(
        args.data_dir,
        perspective=perspective,
        window_ms=args.window_ms,
        episode_seed=args.episode_seed,
    )
    decision, _ = brain.decide(board)
    result = {
        "brain": brain.describe(),
        "fen": board.fen(),
        "selected_uci": decision.selected_uci,
        "silent": decision.silent,
        "tie_break": decision.tie_break,
        "selected_score": (
            decision.move_scores.get(decision.selected_uci)
            if decision.selected_uci is not None
            else None
        ),
        "channel_rates_hz": decision.channel_rates_hz,
        "runtime": asdict(brain.last_stats) if brain.last_stats else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_match(args: argparse.Namespace) -> int:
    config = StockfishConfig(
        path=args.stockfish_path,
        elo=args.stockfish_elo,
        movetime_ms=args.movetime_ms,
        skill_level=args.stockfish_skill,
    )
    color = chess.WHITE if args.color == "white" else chess.BLACK
    if args.network == "male-cns":
        brain = MaleCNSSubgraphBrain(
            args.data_dir,
            perspective=color,
            window_ms=args.window_ms,
            episode_seed=args.episode_seed,
        )
    else:
        brain = ToyChessBrain()
    run_dir = args.run_dir or _default_run_dir()
    with StockfishOpponent(config) as stockfish:
        result = play_game(
            brain,
            stockfish,
            run_dir,
            chessfly_color=color,
            max_plies=args.max_plies,
            resume=args.resume,
            progress=print,
            engine_retries=args.engine_retries,
        )
    print(f"mode={brain.mode}")
    _report(result)
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    config = StockfishConfig(
        path=args.stockfish_path,
        elo=args.stockfish_elo,
        movetime_ms=args.movetime_ms,
    )
    frozen = MaleCNSSubgraphBrain(
        args.data_dir, window_ms=args.window_ms, episode_seed=args.episode_seed
    )
    shuffled = MaleCNSSubgraphBrain(
        args.data_dir,
        window_ms=args.window_ms,
        shuffle_seed=args.shuffle_seed,
        episode_seed=args.episode_seed,
    )
    with StockfishOpponent(config) as stockfish:
        summary = run_fixed_benchmark(
            stockfish,
            frozen,
            shuffled,
            args.run_dir,
            position_limit=args.positions,
            random_seed=args.random_seed,
            evaluation_depth=args.evaluation_depth,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_dir={args.run_dir.resolve()}")
    return 0


def run_readout_seed_sweep(args: argparse.Namespace) -> int:
    board = chess.Board(args.fen)
    brain = MaleCNSSubgraphBrain(args.data_dir, window_ms=args.window_ms)
    summary = run_seed_sweep(
        brain,
        board,
        args.run_dir,
        start_seed=args.start_seed,
        count=args.count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_dir={args.run_dir.resolve()}")
    return 0


def run_match_plan(args: argparse.Namespace) -> int:
    highlights = [int(value) for value in args.highlight_plies.split(",") if value.strip()]
    output = write_plan(
        args.run_dir,
        args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        highlight_plies=highlights,
    )
    print(f"plan={output.resolve()}")
    return 0


def run_render_flyjack_video(args: argparse.Namespace) -> int:
    output = render_flyjack_video(
        args.run_dir,
        args.plan,
        args.frames,
        args.camera,
        args.output,
        data_dir=args.data_dir,
    )
    print(f"output={output.resolve()}")
    print(f"metadata={output.with_suffix('.json').resolve()}")
    return 0


def run_render_match_video(args: argparse.Namespace) -> int:
    output = render_match_video(
        args.run_dir,
        args.frames,
        args.output,
        data_dir=args.data_dir,
        duration_seconds=args.duration,
        fps=args.fps,
    )
    print(f"output={output.resolve()}")
    print(f"metadata={output.with_suffix('.json').resolve()}")
    return 0


def run_render_video(args: argparse.Namespace) -> int:
    output = render_social_video(
        args.run_dir,
        args.output,
        data_dir=args.data_dir,
        duration_seconds=args.duration,
        fps=args.fps,
    )
    print(f"output={output.resolve()}")
    print(f"metadata={output.with_suffix('.json').resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chessfly",
        description="A wiring-constrained fly-connectome chess experiment",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="run the transparent toy SNN against Stockfish"
    )
    smoke.add_argument("--stockfish-path")
    smoke.add_argument("--stockfish-elo", type=int, default=1320)
    smoke.add_argument("--movetime-ms", type=int, default=50)
    smoke.add_argument("--color", choices=("white", "black"), default="white")
    smoke.add_argument("--max-plies", type=int, default=20)
    smoke.add_argument("--run-dir", type=Path)
    smoke.set_defaults(handler=run_smoke)

    prepare = commands.add_parser(
        "prepare", help="inspect or download public MaleCNS v1.0 source tables"
    )
    prepare.add_argument(
        "--files",
        nargs="+",
        choices=("annotations", "transmitters", "weights", "all"),
        default=("annotations", "transmitters", "weights"),
    )
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--download", action="store_true")
    prepare.set_defaults(handler=run_prepare)

    inspect_data = commands.add_parser(
        "inspect-data", help="validate and summarize downloaded MaleCNS tables"
    )
    inspect_data.add_argument("--data-dir", type=Path, default=Path("data"))
    inspect_data.set_defaults(handler=run_inspect_data)

    inspect_graph = commands.add_parser(
        "inspect-graph",
        help="stream the full weights table and verify the retained neuronal graph",
    )
    inspect_graph.add_argument("--data-dir", type=Path, default=Path("data"))
    inspect_graph.set_defaults(handler=run_inspect_graph)

    compile_graph = commands.add_parser(
        "compile-graph", help="compile the full retained graph into disk-backed CSR"
    )
    compile_graph.add_argument("--data-dir", type=Path, default=Path("data"))
    compile_graph.set_defaults(handler=run_compile_graph)

    subgraph = commands.add_parser(
        "build-subgraph", help="extract visual-to-descending paths from retained CSR"
    )
    subgraph.add_argument("--data-dir", type=Path, default=Path("data"))
    subgraph.add_argument("--maximum-hops", type=int, default=3)
    subgraph.add_argument("--minimum-weight", type=int, default=5)
    subgraph.set_defaults(handler=run_build_subgraph)

    render_board = commands.add_parser(
        "render-board", help="render a deterministic 320x180 neural stimulus"
    )
    render_board.add_argument("--fen", default=chess.STARTING_FEN)
    render_board.add_argument("--perspective", choices=("white", "black"), default="white")
    render_board.add_argument("--last-move")
    render_board.add_argument("--output", type=Path, default=Path("runs/board-stimulus.png"))
    render_board.set_defaults(handler=run_render_board)

    retina = commands.add_parser(
        "build-retina", help="infer R1-R6 and R8 display coordinates from anatomy"
    )
    retina.add_argument("--data-dir", type=Path, default=Path("data"))
    retina.set_defaults(handler=run_build_retina)

    decision = commands.add_parser(
        "male-cns-decision",
        help="run one frozen MaleCNS reference-subgraph chess decision",
    )
    decision.add_argument("--data-dir", type=Path, default=Path("data"))
    decision.add_argument("--fen", default=chess.STARTING_FEN)
    decision.add_argument(
        "--perspective", choices=("white", "black"), default="white"
    )
    decision.add_argument("--window-ms", type=float, default=500.0)
    decision.add_argument("--episode-seed", type=int, default=20260912)
    decision.set_defaults(handler=run_male_cns_decision)

    fly_model = commands.add_parser(
        "prepare-fly-model",
        help="download and verify the NeuroMechFly body model used by the renderer",
    )
    fly_model.add_argument("--data-dir", type=Path, default=Path("data"))
    fly_model.set_defaults(handler=run_prepare_fly_model)

    cloud = commands.add_parser(
        "build-brain-cloud",
        help="place MaleCNS neurons at their recorded soma positions for rendering",
    )
    cloud.add_argument("--data-dir", type=Path, default=Path("data"))
    cloud.set_defaults(handler=run_build_brain_cloud)

    match = commands.add_parser(
        "match", help="play the toy or frozen MaleCNS controller against Stockfish"
    )
    match.add_argument("--network", choices=("male-cns", "toy"), default="male-cns")
    match.add_argument("--data-dir", type=Path, default=Path("data"))
    match.add_argument("--stockfish-path")
    match.add_argument("--stockfish-elo", type=int, default=1320)
    match.add_argument(
        "--stockfish-skill",
        type=int,
        help=(
            "play by Skill Level 0-20 instead of Elo; the engine's own Elo floor "
            "is 1320, and it ignores Skill Level while limiting strength by Elo"
        ),
    )
    match.add_argument("--movetime-ms", type=int, default=50)
    match.add_argument("--color", choices=("white", "black"), default="white")
    match.add_argument("--window-ms", type=float, default=500.0)
    match.add_argument("--episode-seed", type=int, default=20260912)
    match.add_argument("--max-plies", type=int, default=20)
    match.add_argument("--run-dir", type=Path)
    match.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from its checkpoint instead of starting over",
    )
    match.add_argument(
        "--engine-retries",
        type=int,
        default=3,
        help="how many times to restart Stockfish after a failed call",
    )
    match.set_defaults(handler=run_match)

    benchmark = commands.add_parser(
        "benchmark", help="compare frozen MaleCNS against fixed, random, and shuffled controls"
    )
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))
    benchmark.add_argument("--stockfish-path")
    benchmark.add_argument("--stockfish-elo", type=int, default=1320)
    benchmark.add_argument("--movetime-ms", type=int, default=50)
    benchmark.add_argument("--window-ms", type=float, default=500.0)
    benchmark.add_argument(
        "--positions", type=int, default=len(POSITION_LINES)
    )
    benchmark.add_argument("--random-seed", type=int, default=20260912)
    benchmark.add_argument("--shuffle-seed", type=int, default=20260912)
    benchmark.add_argument("--episode-seed", type=int, default=20260912)
    benchmark.add_argument("--evaluation-depth", type=int, default=10)
    benchmark.add_argument(
        "--run-dir", type=Path, default=Path("runs/fixed-benchmark-v1")
    )
    benchmark.set_defaults(handler=run_benchmark)

    seed_sweep = commands.add_parser(
        "seed-sweep", help="replay one MaleCNS response through fixed readout seeds"
    )
    seed_sweep.add_argument("--data-dir", type=Path, default=Path("data"))
    seed_sweep.add_argument("--fen", default=chess.STARTING_FEN)
    seed_sweep.add_argument("--window-ms", type=float, default=500.0)
    seed_sweep.add_argument("--start-seed", type=int, default=0)
    seed_sweep.add_argument("--count", type=int, default=100)
    seed_sweep.add_argument(
        "--run-dir", type=Path, default=Path("runs/readout-seed-sweep-v1")
    )
    seed_sweep.set_defaults(handler=run_readout_seed_sweep)

    video = commands.add_parser(
        "render-video", help="render a telemetry-driven 9:16 social MP4"
    )
    video.add_argument("--run-dir", type=Path, required=True)
    video.add_argument("--data-dir", type=Path, default=Path("data"))
    video.add_argument("--output", type=Path, default=Path("runs/chessfly-social.mp4"))
    video.add_argument("--duration", type=float, default=30.0)
    video.add_argument("--fps", type=int, default=30)
    video.set_defaults(handler=run_render_video)

    match_video = commands.add_parser(
        "render-match-video",
        help="render a 16:9 full-match cut over the original cinematic plates",
    )
    match_video.add_argument("--run-dir", type=Path, required=True)
    match_video.add_argument(
        "--frames", type=Path, default=Path("runs/cinematic/frames")
    )
    match_video.add_argument("--data-dir", type=Path, default=Path("data"))
    match_video.add_argument(
        "--output", type=Path, default=Path("runs/chessfly-match.mp4")
    )
    match_video.add_argument("--duration", type=float, default=60.0)
    match_video.add_argument("--fps", type=int, default=30)
    match_video.set_defaults(handler=run_render_match_video)

    flyjack = commands.add_parser(
        "render-flyjack-video",
        help="composite the FlyJack-style match with the spiking brain hologram",
    )
    flyjack.add_argument("--run-dir", type=Path, required=True)
    flyjack.add_argument("--plan", type=Path, required=True)
    flyjack.add_argument("--frames", type=Path, required=True)
    flyjack.add_argument("--camera", type=Path, required=True)
    flyjack.add_argument("--data-dir", type=Path, default=Path("data"))
    flyjack.add_argument("--output", type=Path, required=True)
    flyjack.set_defaults(handler=run_render_flyjack_video)

    plan = commands.add_parser(
        "match-plan",
        help="emit the frame-accurate 3D animation plan for a recorded match",
    )
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--duration", type=float, default=60.0)
    plan.add_argument("--fps", type=int, default=30)
    plan.add_argument(
        "--highlight-plies",
        default="",
        help="comma-separated Chessfly plies that get a brain 'think' shot",
    )
    plan.set_defaults(handler=run_match_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
