"""Chessfly command-line interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import chess

from .annotations import (
    descending_body_ids,
    load_source_tables,
    retained_neuron_body_ids,
    summarize_annotations,
)
from .compiled_graph import build_path_subgraph, compile_retained_csr
from .dataset import download_file, inventory, select_files, write_manifest
from .game import play_game
from .graph import summarize_connection_file
from .retina import build_retina_projection, load_retina_body_ids
from .stockfish import StockfishConfig, StockfishOpponent
from .toy_brain import ToyChessBrain
from .vision import save_board_stimulus


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"smoke-{stamp}"


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
        )
    print(f"mode=toy-not-male-cns")
    print(f"result={result.result} termination={result.termination}")
    print(f"plies={result.plies}")
    print(f"run_dir={result.run_dir.resolve()}")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
