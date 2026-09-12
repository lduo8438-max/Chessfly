# Chessfly video direction

Reference: [Alex Wormuth's Stonkfly post](https://x.com/nftechie_/status/2098012107652391357)

## Creative thesis

- **Visual thesis:** a dark neural chess-control room where the board is the
  brightest object and every luminous trace corresponds to recorded activity.
- **Content plan:** one hook, one mechanism, one real match, one measured result,
  and one explicit scientific boundary.
- **Motion thesis:** energy travels from board to brain, the neural camera gains
  depth during each decision, and HUD values switch only when telemetry changes.

The public X page exposes the poster without login: a dark cinematic lab, a
central fly, luminous monitors, and a live telemetry HUD. Chessfly will use the
same broad storytelling grammar while creating original assets and a distinct
chess control-room identity.

## Visual language

- Near-black laboratory with restrained cyan, electric violet, and warm amber.
- A stylized fly is the physical protagonist; a projected chessboard replaces
  trading screens.
- Neural activity, output populations, move score, Stockfish evaluation, and
  elapsed neural time are real values read from a run artifact.
- Close, low-angle camera movement and shallow depth of field provide drama;
  clean scientific overlays prevent the scene from becoming generic cyberpunk.
- No fake spikes, fake evaluations, copied footage, or claims that the fly
  understands chess.

## Released 10-second cinematic

1. **0–3.8 s — Establish:** slow dolly across an original low-poly fly, physical
   chessboard, and monitors displaying the exact stimulus and spike raster.
2. **3.8–6.5 s — Decision:** the recorded `b1 > c3` move lifts and travels across
   the board as the neural HUD remains visible.
3. **6.5–10 s — Resolve:** the knight lands, camera closes in, and the scientific
   boundary remains on screen.

The HUD is based on a real frozen-controller artifact: 100,595 spikes in 500 ms,
8,598 mapped neurons, 70,308 retained edges, and two descending-neuron readout
spikes. Geometry is illustrative and the video makes no biological chess claim.

Outputs are a 1920×1080 X master and a 1080×1920 Instagram delivery file, both
30 fps, H.264/AAC, and 10 seconds. Spike events drive original synthesized
sound; no third-party music or Stonkfly media is used.

## 16:9 full-match cut

The match cut reuses the cinematic plates and adds a single instrument screen so
a whole game stays readable:

- **Left two thirds** — the rendered laboratory plate. The 3D board plays the
  same recorded game as the panel: pieces lift, travel and land, captures shrink
  away, and a promotion swaps the pawn body for a queen. A plate sequence
  shorter than the cut ping-pongs rather than cutting abruptly.
- **Pieces** — low-poly Staunton silhouettes built from primitives (base, stem,
  collar, and a head per type), lit by a soft key over the board so they read as
  solid rather than as dark cones.
- **Right third** — an opaque screen. A 520 px board sits at the top, the
  fly-brain activity diagram below it, and the raw counters at the foot.
- **Brain diagram** — three real groups (retina inputs, SNN core, descending
  readout) drawn from the recorded `spikes-10ms.npz`, joined only by edges that
  exist in the compiled subgraph. Brightness is normalized within each group,
  because one global peak renders the core as a dead field; the printed spike
  and readout counts are raw.
- **Honesty rules carried over** — a held neural frame is labelled `HELD`, a
  tie-broken move says so, and the ending is named from the manifest rather than
  described as a demo.

### Board art

Pieces are Staunton glyphs taken from a system Unicode font. No third-party
piece artwork is copied. This is display art only: `chessfly.vision` keeps
rendering the letter-based 320×180 stimulus that the retina actually samples,
and changing it to suit a render would invalidate every recorded run.

The referenced Stonkfly repository contains image-generation direction but no
video source, scene, model, or rendering pipeline. Chessfly therefore follows
the public post's broad cinematic grammar without copying its assets.
