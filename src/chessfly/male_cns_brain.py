"""MaleCNS reference-subgraph controller for chess decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .annotations import descending_body_ids, load_source_tables
from .array_snn import ArrayLIFNetwork, ArrayLIFParameters, transmitter_signs
from .chess_control import ChessMoveDecoder, NeuralDecision, assign_readout_populations
from .retina import sample_retina_stimulus
from .vision import render_board_stimulus


DEFAULT_SUBGRAPH = "mapped-retina-to-descending-h3-w5"


@dataclass(frozen=True)
class MaleCNSDecisionStats:
    steps: int
    simulated_ms: float
    retina_inputs: int
    retina_mean: float
    total_spikes: int
    readout_spikes: int
    active_neurons: int
    peak_step_spikes: int


class MaleCNSSubgraphBrain:
    """Frozen MaleCNS wiring with explicit engineered dynamics and readout."""

    mode = "male-cns-subgraph-v1"
    display_name = "Chessfly (MaleCNS subgraph)"

    def __init__(
        self,
        data_dir: Path = Path("data"),
        perspective: chess.Color = chess.WHITE,
        window_ms: float = 500.0,
        retina_current_max: float = 60.0,
        retina_half_saturation: float = 0.02,
        lamina_bias: float = 12.0,
        parameters: ArrayLIFParameters | None = None,
        subgraph_name: str = DEFAULT_SUBGRAPH,
        shuffle_seed: int | None = None,
        episode_seed: int = 20260912,
    ) -> None:
        if (
            window_ms <= 0
            or retina_current_max <= 0
            or retina_half_saturation <= 0
            or lamina_bias < 0
        ):
            raise ValueError("window and retina current parameters must be positive")
        self.data_dir = Path(data_dir)
        self.perspective = perspective
        self.window_ms = float(window_ms)
        self.retina_current_max = float(retina_current_max)
        self.retina_half_saturation = float(retina_half_saturation)
        self.lamina_bias = float(lamina_bias)
        self.subgraph_dir = self.data_dir / "compiled" / subgraph_name
        self.projection_dir = self.data_dir / "compiled" / "retina"
        self.node_ids = np.load(self.subgraph_dir / "node_ids.npy", mmap_mode="r")
        sources = np.load(self.subgraph_dir / "source_index.npy", mmap_mode="r")
        targets = np.load(self.subgraph_dir / "target_index.npy", mmap_mode="r")
        weights = np.load(self.subgraph_dir / "edge_weight.npy", mmap_mode="r")
        if shuffle_seed is not None:
            targets = np.random.default_rng(shuffle_seed).permutation(targets)
            self.mode = f"male-cns-shuffled-targets-seed-{shuffle_seed}"
            self.display_name = "Chessfly (shuffled MaleCNS control)"
        else:
            self.mode = type(self).mode
            self.display_name = type(self).display_name
        self.shuffle_seed = shuffle_seed
        self.episode_seed = int(episode_seed)
        annotations, transmitters = load_source_tables(self.data_dir)
        signs = transmitter_signs(transmitters, self.node_ids)
        self.network = ArrayLIFNetwork(
            len(self.node_ids), sources, targets, weights, signs, parameters
        )

        retina_body_ids = np.load(self.projection_dir / "body_id.npy", mmap_mode="r")
        self._retina_projection_rows, self._retina_indices = _intersect_body_ids(
            retina_body_ids, self.node_ids
        )
        lamina_mask = pc.is_in(
            pc.fill_null(annotations["type"], ""),
            value_set=pa.array(["L1", "L2", "L3", "L5"]),
        )
        lamina_body_ids = np.asarray(
            pc.filter(annotations["bodyId"], lamina_mask).to_numpy(), dtype=np.int64
        )
        _, self._lamina_indices = _intersect_body_ids(lamina_body_ids, self.node_ids)
        descending = np.asarray(descending_body_ids(annotations), dtype=np.int64)
        _, output_indices = _intersect_body_ids(descending, self.node_ids)
        if len(output_indices) < 32:
            raise ValueError("reference subgraph contains fewer than 32 outputs")
        self._output_indices = np.sort(output_indices)
        self.output_neuron_indices = frozenset(
            int(value) for value in self._output_indices
        )
        populations = assign_readout_populations(
            (int(value) for value in output_indices),
            seed=f"chessfly-male-cns-v1:{self.episode_seed}",
        )
        self.decoder = ChessMoveDecoder(populations)
        self._retina_light = np.zeros(len(self._retina_indices), dtype=np.float32)
        self.last_stats: MaleCNSDecisionStats | None = None
        self.last_raster: np.ndarray | None = None
        self.last_stimulus = None
        self.last_retina_values: np.ndarray | None = None

    @property
    def parameters(self) -> ArrayLIFParameters:
        return self.network.parameters

    def reset(self) -> None:
        self.network.reset()
        self._retina_light.fill(0)
        self.last_stats = None
        self.last_raster = None
        self.last_stimulus = None
        self.last_retina_values = None

    def state_dict(self) -> dict:
        """Return the full continuous runtime state for mid-game checkpoints."""
        state = {f"network.{key}": value for key, value in self.network.state_dict().items()}
        state["retina_light"] = self._retina_light.copy()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore state saved by :meth:`state_dict` so a resumed game continues exactly."""
        network_state = {
            key[len("network.") :]: value
            for key, value in state.items()
            if key.startswith("network.")
        }
        self.network.load_state_dict(network_state)
        if "retina_light" not in state:
            raise ValueError("checkpoint is missing the retina adaptation state")
        light = np.asarray(state["retina_light"], dtype=np.float32)
        if light.shape != self._retina_light.shape:
            raise ValueError("checkpoint retina input count does not match this brain")
        self._retina_light[:] = light

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "neurons": self.network.neuron_count,
            "edges": len(self.network.sources),
            "retina_inputs": len(self._retina_indices),
            "descending_outputs": len(self.output_neuron_indices),
            "window_ms": self.window_ms,
            "retina_current_max": self.retina_current_max,
            "retina_half_saturation": self.retina_half_saturation,
            "lamina_bias": self.lamina_bias,
            "lamina_neurons": len(self._lamina_indices),
            "dynamics": asdict(self.parameters),
            "validated_physiology": False,
            "plasticity": False,
            "topology_control": (
                None
                if self.shuffle_seed is None
                else {
                    "method": "global target permutation",
                    "seed": self.shuffle_seed,
                    "preserves": ["source out-degree", "target in-degree multiset"],
                }
            ),
            "episode_seed": self.episode_seed,
        }

    def decide(
        self, board: chess.Board
    ) -> tuple[NeuralDecision, list[tuple[int, ...]]]:
        last_move = board.peek() if board.move_stack else None
        frame = render_board_stimulus(board, self.perspective, last_move)
        sampled = sample_retina_stimulus(frame, self.projection_dir)
        values = sampled[self._retina_projection_rows]
        drive = np.zeros(self.network.neuron_count, dtype=np.float32)
        drive[self._lamina_indices] = self.lamina_bias

        steps = max(1, round(self.window_ms / self.parameters.dt_ms))
        bin_steps = max(1, round(10.0 / self.parameters.dt_ms))
        raster = np.zeros(
            ((steps + bin_steps - 1) // bin_steps, self.network.neuron_count),
            dtype=np.uint16,
        )
        frames: list[tuple[int, ...]] = []
        active = np.zeros(self.network.neuron_count, dtype=bool)
        total_spikes = 0
        readout_spikes = 0
        peak = 0
        output_mask = np.zeros(self.network.neuron_count, dtype=bool)
        output_mask[self._output_indices] = True
        light_alpha = np.float32(1 - np.exp(-self.parameters.dt_ms / 10.0))
        for step in range(steps):
            self._retina_light += light_alpha * (values - self._retina_light)
            drive[self._retina_indices] = (
                self.retina_current_max
                * self._retina_light
                / (self.retina_half_saturation + self._retina_light)
            )
            spikes = self.network.step(drive)
            if len(spikes):
                raster[step // bin_steps, spikes] += 1
            total_spikes += len(spikes)
            peak = max(peak, len(spikes))
            active[spikes] = True
            readout = spikes[output_mask[spikes]]
            readout_spikes += len(readout)
            frames.append(tuple(int(value) for value in readout))

        self.last_stats = MaleCNSDecisionStats(
            steps=steps,
            simulated_ms=steps * self.parameters.dt_ms,
            retina_inputs=len(self._retina_indices),
            retina_mean=float(np.mean(values)),
            total_spikes=total_spikes,
            readout_spikes=readout_spikes,
            active_neurons=int(np.count_nonzero(active)),
            peak_step_spikes=peak,
        )
        self.last_raster = raster
        self.last_stimulus = frame
        self.last_retina_values = values.copy()
        return self.decoder.decode(board, frames, self.last_stats.simulated_ms), frames

    def export_decision_artifact(self, run_dir: Path, ply: int) -> None:
        """Persist real visual input, binned spikes, and runtime telemetry."""
        if (
            self.last_stats is None
            or self.last_raster is None
            or self.last_stimulus is None
            or self.last_retina_values is None
        ):
            raise RuntimeError("no completed MaleCNS decision to export")
        target = Path(run_dir) / "neural" / f"ply-{ply:03d}"
        target.mkdir(parents=True, exist_ok=False)
        self.last_stimulus.save(target / "stimulus.png", format="PNG")
        np.savez_compressed(
            target / "spikes-10ms.npz",
            counts=self.last_raster,
            node_ids=np.asarray(self.node_ids),
            readout_indices=self._output_indices,
            bin_ms=np.asarray(10.0, dtype=np.float32),
        )
        np.savez_compressed(
            target / "retina.npz",
            local_indices=self._retina_indices,
            body_ids=np.asarray(self.node_ids[self._retina_indices]),
            sampled=self.last_retina_values,
        )
        (target / "runtime.json").write_text(
            json.dumps(asdict(self.last_stats), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _intersect_body_ids(
    requested_body_ids: np.ndarray, sorted_node_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return requested-row and compiled-node indices for matching body IDs."""
    requested = np.asarray(requested_body_ids, dtype=np.int64)
    positions = np.searchsorted(sorted_node_ids, requested)
    valid = positions < len(sorted_node_ids)
    matched = np.zeros(len(requested), dtype=bool)
    matched[valid] = sorted_node_ids[positions[valid]] == requested[valid]
    return np.flatnonzero(matched).astype(np.int32), positions[matched].astype(np.int32)
