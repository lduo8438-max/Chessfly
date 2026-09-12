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

The first implementation milestone is live:

- deterministic LIF reference engine;
- a 32-channel `from file / from rank / to file / to rank` move decoder;
- legal-move masking with explicit silence and tie-break reporting;
- a transparent toy SNN for end-to-end testing before MaleCNS is downloaded;
- adjustable-strength Stockfish UCI integration;
- reproducible `run.json`, `moves.jsonl`, and `game.pgn` artifacts;
- verified MaleCNS downloads with local SHA-256 provenance;
- streamed reconstruction of the retained graph: 166,700 neurons, 25,582,938
  directed connections, and 124,177,617 synaptic contacts;
- a defined social-video direction based only on real run telemetry.

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
```

Specify a non-standard Stockfish location with `--stockfish-path`. Each run gets
its own directory under `runs/` and never overwrites an earlier result.

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

1. Replace the toy network with the compiled 8,598-neuron reference subgraph.
2. Add the native event-driven backend required for the full retained graph.
3. Run frozen, random, and shuffled controls before enabling plasticity.
4. Render an English 59-second scientific-cinematic video and a full match cut.

See [the architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md),
[source context](docs/CONTEXT.md), and [video direction](docs/VIDEO_STYLE.md).

## Data and licensing

MaleCNS data is CC BY 4.0 and is downloaded separately. Chessfly will preserve
source attribution and dataset hashes. The Python chess rules/UCI layer uses
`python-chess`, which is GPL-3.0-or-later; Chessfly will be released under the
same license. Stockfish is detected as an external executable and is not bundled.
