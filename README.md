# Chessfly

**What happens when a fly-connectome simulation has to play chess?**

Chessfly is a wiring-constrained spiking-network experiment built toward the
[MaleCNS v1.0](https://male-cns.janelia.org/) connectome. A rendered chessboard
becomes visual stimulation, descending-neuron population activity scores legal
moves, and Stockfish supplies an opponent and an explicitly engineered learning
signal.

This repository does **not** claim that a fly understands chess. MaleCNS supplies
anatomy; neuron dynamics, display mapping, move decoding, and reinforcement are
model choices that must be tested against controls.

## Current milestone

The frozen MaleCNS reference baseline is live:

- deterministic LIF reference engine;
- a 32-channel `from file / from rank / to file / to rank` move decoder;
- legal-move masking with explicit silence and tie-break reporting;
- a transparent toy SNN for end-to-end testing before MaleCNS is downloaded;
- adjustable-strength Stockfish UCI integration;
- reproducible `run.json`, `moves.jsonl`, and `game.pgn` artifacts, flushed
  after every ply so a long game survives an interruption;
- verified MaleCNS downloads with local SHA-256 provenance;
- streamed reconstruction of the retained graph: 166,700 neurons, 25,582,938
  directed connections, and 124,177,617 synaptic contacts;
- a defined social-video direction based only on real run telemetry.
- a vectorized 8,598-neuron / 70,308-edge MaleCNS LIF runtime;
- mapped R1–R6/R8 board input and 960-neuron descending readout;
- 10 ms spike rasters, exact stimuli, retina samples, move logs and PGN;
- fixed-position comparisons against fixed, random and shuffled-topology controls.

The toy mode is deliberately labeled `toy-not-male-cns` in every output.

## Run the smoke match

Python 3.9+ and Stockfish are required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/chessfly smoke --stockfish-elo 1320 --max-plies 20
.venv/bin/chessfly inspect-data
.venv/bin/chessfly inspect-graph
.venv/bin/chessfly compile-graph
.venv/bin/chessfly build-retina
.venv/bin/chessfly build-subgraph
.venv/bin/chessfly male-cns-decision
.venv/bin/chessfly match --network male-cns --max-plies 20
.venv/bin/chessfly benchmark --positions 10
.venv/bin/chessfly seed-sweep --count 100
.venv/bin/chessfly render-video \
  --run-dir runs/video-source-v1 \
  --output runs/chessfly-social-master.mp4
```

Specify a non-standard Stockfish location with `--stockfish-path`. Each run gets
its own directory under `runs/` and never overwrites an earlier result.

## Making the opponent weaker than Elo 1320

1320 is Stockfish's own floor for `UCI_Elo`, not a limit this project adds. To
go below it, leave Elo mode and set a skill level instead:

```bash
.venv/bin/chessfly match --network male-cns --stockfish-skill 0 --max-plies 400
```

`--stockfish-skill` takes 0-20. The engine ignores `Skill Level` while
`UCI_LimitStrength` is on, so the two are mutually exclusive: passing a skill
level turns Elo limiting off and leaves `--stockfish-elo` unused. `run.json`
records both the requested config and, under `stockfish_options`, the options
actually sent to the engine.

The same engine also produces the `evaluation_*_cp` columns, so a weakened
opponent is also a weaker analyst; those evaluations are indicative, not a
reference score.

## Long games, checkpoints and resuming

`match` writes durable state after every ply, so an interrupted long game is not
lost:

| File | Written | Holds |
|---|---|---|
| `moves.jsonl` | appended and flushed per ply | one record per ply |
| `game.pgn`, `run.json` | rewritten per ply | the game so far |
| `progress.json` | rewritten per ply, last | FEN, move stack, ply count, termination |
| `checkpoint.npz` | rewritten per ply | membrane voltages, synaptic currents, refractory counters, delay queue, retina adaptation |
| `neural/ply-NNN/decision.json` | per Chessfly ply | channel rates, every legal-move score, tie and silence flags |

`progress.json` is written last, so it is the authority; on resume any move
records written past it are truncated and that ply is recomputed.

```bash
# continue an interrupted run in place
.venv/bin/chessfly match --network male-cns --run-dir runs/full-game-v1 --resume

# continue a run that stopped at its ply cap
.venv/bin/chessfly match --network male-cns --run-dir runs/full-game-v1 \
  --max-plies 600 --resume
```

Resuming restores the simulated network state rather than only the position, so
the continued game is identical to an uninterrupted one. A run that reached a
rule termination (checkmate, stalemate, repetition, fifty-move) cannot be
resumed. `--engine-retries` (default 3) controls how many times a failed
Stockfish call restarts the engine before the run stops.

## Render the cinematic social video

The 10-second cinematic uses the real first-decision telemetry from a frozen
MaleCNS run. Blender creates an original low-poly laboratory, fly, monitors,
board, pieces, camera move, and the real `b1 > c3` knight animation. The post
step adds the recorded spike count/raster and locally synthesized sound.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python scripts/blender_chessfly.py -- \
  --run-dir runs/video-source-v1 \
  --output runs/cinematic/frames/frame- \
  --save-blend runs/cinematic/chessfly-cinematic.blend

.venv/bin/python scripts/cinematic_post.py \
  --frames runs/cinematic/frames \
  --run-dir runs/video-source-v1 \
  --output runs/cinematic/chessfly-cinematic-x.mp4 \
  --instagram-output runs/cinematic/chessfly-cinematic-instagram.mp4
```

Deliverables are H.264/AAC at 30 fps: a 1920×1080 X master and a 1080×1920
Instagram version. The scene is illustrative; the HUD is measured telemetry.

## Render the 16:9 full-match cut

The match cut keeps the cinematic plates on the left two thirds and adds a
dedicated instrument screen on the right third: a readable board above, and a
fly-brain activity diagram below, both driven by one recorded run.

The 3D board plays the recorded game. `match-plan` resolves every ply into
piece moves, captures, castling, en passant and promotions with exact frame
numbers, and the Blender script only applies those keyframes — so the plates and
the overlay panel step on one pacing curve and cannot drift apart.

```bash
.venv/bin/chessfly match-plan \
  --run-dir runs/full-game-v1 \
  --output runs/full-game-v1/plan.json \
  --duration 60 --fps 30

/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python scripts/blender_chessfly.py -- \
  --run-dir runs/full-game-v1 \
  --plan runs/full-game-v1/plan.json \
  --output runs/cinematic-full-game/frames/frame- \
  --width 1920 --height 1080

.venv/bin/chessfly render-match-video \
  --run-dir runs/full-game-v1 \
  --frames runs/cinematic-full-game/frames \
  --duration 60 \
  --output runs/chessfly-match-v1.mp4
```

Without `--plan` the Blender script still builds the static 10-second cinematic.
Plates are streamed from disk one frame at a time while encoding; a full-length
1920×1080 sequence is gigabytes once decoded.

Closing plies are held longer than the middlegame so a checkmate does not flash
past, Stockfish replies hold the last neural frame and label it `HELD`, and node
brightness is normalized per group so the core does not read as a dead field —
the printed counters stay raw. Board pieces are Staunton glyphs from a system
Unicode font, drawn for display only: `chessfly.vision` still renders the
letter-based 320×180 stimulus the network actually sees, and no video change is
allowed to alter it.

## How a move is selected

```text
rendered board pixels
        ↓
visual input population
        ↓
spiking connectome
        ↓
32 normalized readout rates
        ↓
from-file + from-rank + to-file + to-rank
        ↓
score every legal move
        ↓
highest score becomes Chessfly's move
```

The legal-move mask prevents an invalid command but never replaces Chessfly's
highest-scoring legal proposal with a stronger chess move. Stockfish evaluation
is recorded after the move; it is not provided to the frozen move decoder.

## Roadmap

1. Expand the frozen benchmark across readout seeds and full games.
2. Add the native event-driven backend required for the full retained graph.
3. Add plasticity only after frozen and shuffled controls are stable.
4. Extend the released 10-second cinematic into a full-match scientific cut.

See [the architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md),
[model card](docs/MODEL.md), [source context](docs/CONTEXT.md), and
[video direction](docs/VIDEO_STYLE.md).

## Data and licensing

MaleCNS data is CC BY 4.0 and is downloaded separately. Chessfly will preserve
source attribution and dataset hashes. The Python chess rules/UCI layer uses
`python-chess`, which is GPL-3.0-or-later; Chessfly will be released under the
same license. Stockfish is detected as an external executable and is not bundled.
