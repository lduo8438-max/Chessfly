# Chessfly Model Card

## Scientific boundary

Chessfly uses MaleCNS anatomy as a wiring constraint. The pixels, dynamics,
neurotransmitter sign proxy, display projection, chess readout and reward are
engineered adapters. The experiment does not show that a fruit fly understands
chess, and the current frozen baseline does not learn.

## Frozen reference model

- Source: public `male-cns:v1.0` annotation, neurotransmitter and connection
  weights tables, with URLs and SHA-256 hashes in `data/manifest.json`.
- Graph: 8,598 neurons and 70,308 directed connections on mapped-retina to
  descending-neuron paths of at most three hops; minimum connection weight five.
- Input: a deterministic 320×180 RGB chessboard. 2,147 mapped R1–R6/R8 cells
  present in the reference graph sample linear-sRGB luminance, green or blue.
- Visual adapter: 10 ms low-pass, saturating retinal current capped at 60
  mV-equivalent units, plus a 12-unit tonic L1/L2/L3/L5 bias.
- Dynamics: NumPy LIF, 0.1 ms steps, 20 ms membrane and 5 ms synaptic time
  constants, −52 mV rest/reset, −45 mV threshold, 1.8 ms delay and 2.2 ms
  refractory period. Contact count is multiplied by 0.275.
- Fast sign proxy: ACh positive; GABA, glutamate and histamine negative;
  dopamine, serotonin and octopamine excluded from generic fast transmission;
  unresolved consensus labels use a declared positive fallback.
- Output: 960 descending neurons are deterministically balanced across 32
  from-file, from-rank, to-file and to-rank channels. The highest-scoring legal
  move is selected; legal masking is the only chess rule supplied to the brain.
- State: neural and retinal state persists between decisions in a game. A fixed
  position benchmark resets the state before each position.

These values are model assumptions, not fitted Drosophila physiology.

## Initial engineering baseline

The tracked 10-position opening suite at Stockfish depth 10 produced only two
descending spikes per 500 ms decision. With episode seed 20260912, frozen
MaleCNS averaged −276.6 cp per selected move, compared with −84.9 cp for the
lexicographic first-legal control and −114.7 cp for one fixed random control.
The shuffled-target graph was completely silent and therefore fell back to the
same deterministic first-legal tie-break.

This is weak chess behavior and a deliberately unflattering result. Ten opening
positions are sufficient to validate the experiment plumbing, not to estimate
playing strength or connectome advantage. See
`results/m2-fixed-benchmark-v1-summary.json` for the machine-readable record.

## Required controls before a learning claim

- larger held-out FEN suites and multiple readout seeds;
- full games from both colors;
- fixed, random, shuffled-target, shuffled-sign and no-retina controls;
- frozen-weight versus learned-weight comparisons from identical checkpoints;
- retention and reset tests, with confidence intervals rather than a best run.
