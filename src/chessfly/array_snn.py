"""Vectorized LIF runtime for compiled MaleCNS subgraphs.

The dynamics are an engineered approximation over connectome anatomy.  They are
not a claim about calibrated Drosophila membrane or synaptic physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp
from typing import Mapping

import numpy as np
import pyarrow as pa


FAST_TRANSMITTER_SIGN: Mapping[str, float] = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    # Neuromodulators are deliberately excluded from generic fast transmission.
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    # The public table leaves many bodies unresolved.  A positive fallback is
    # explicit so controls can later replace it with zero or shuffled signs.
    "unclear": 1.0,
}


@dataclass(frozen=True)
class ArrayLIFParameters:
    dt_ms: float = 0.1
    tau_membrane_ms: float = 20.0
    tau_synaptic_ms: float = 5.0
    resting_mv: float = -52.0
    threshold_mv: float = -45.0
    reset_mv: float = -52.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    contact_scale: float = 0.275

    def __post_init__(self) -> None:
        positive = (
            self.dt_ms,
            self.tau_membrane_ms,
            self.tau_synaptic_ms,
            self.delay_ms,
            self.contact_scale,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("time constants, delay, dt, and contact scale must be positive")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms cannot be negative")
        if self.reset_mv >= self.threshold_mv:
            raise ValueError("reset_mv must be below threshold_mv")


def transmitter_signs(transmitters: pa.Table, node_ids: np.ndarray) -> np.ndarray:
    """Align MaleCNS consensus transmitter labels to sorted compiled body IDs."""
    bodies = np.asarray(transmitters["body"].to_numpy(), dtype=np.int64)
    order = np.argsort(bodies, kind="stable")
    sorted_bodies = bodies[order]
    positions = np.searchsorted(sorted_bodies, node_ids)
    matched = positions < len(sorted_bodies)
    if not np.all(matched):
        raise ValueError("compiled node IDs are not fully represented in transmitter data")
    if not np.array_equal(sorted_bodies[positions], np.asarray(node_ids)):
        raise ValueError("compiled node IDs are not fully represented in transmitter data")
    labels = np.asarray(
        [
            value or "unclear"
            for value in transmitters["consensus_nt"]
            .take(pa.array(order[positions]))
            .to_pylist()
        ],
        dtype=object,
    )
    unknown = sorted(set(labels) - set(FAST_TRANSMITTER_SIGN))
    if unknown:
        raise ValueError(f"unsupported consensus transmitters: {unknown}")
    return np.asarray([FAST_TRANSMITTER_SIGN[label] for label in labels], dtype=np.float32)


class ArrayLIFNetwork:
    """Synchronous NumPy LIF network with a fixed axonal delay."""

    def __init__(
        self,
        neuron_count: int,
        sources: np.ndarray,
        targets: np.ndarray,
        contact_weights: np.ndarray,
        source_signs: np.ndarray | None = None,
        parameters: ArrayLIFParameters | None = None,
    ) -> None:
        if neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        sources = np.asarray(sources, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.int32)
        contacts = np.asarray(contact_weights, dtype=np.float32)
        if not (sources.ndim == targets.ndim == contacts.ndim == 1):
            raise ValueError("edge arrays must be one-dimensional")
        if not (len(sources) == len(targets) == len(contacts)):
            raise ValueError("edge arrays must have equal length")
        if len(sources) and (
            np.min(sources) < 0
            or np.min(targets) < 0
            or np.max(sources) >= neuron_count
            or np.max(targets) >= neuron_count
        ):
            raise ValueError("edge index exceeds neuron_count")
        if np.any(contacts < 0):
            raise ValueError("contact weights cannot be negative")
        signs = (
            np.ones(neuron_count, dtype=np.float32)
            if source_signs is None
            else np.asarray(source_signs, dtype=np.float32)
        )
        if signs.shape != (neuron_count,):
            raise ValueError("source_signs must have one value per neuron")

        self.neuron_count = neuron_count
        self.parameters = parameters or ArrayLIFParameters()
        order = np.argsort(sources, kind="stable")
        self.sources = sources[order]
        self.targets = targets[order]
        self.weights = (
            contacts[order]
            * signs[self.sources]
            * np.float32(self.parameters.contact_scale)
        )
        self._delay_steps = max(1, round(self.parameters.delay_ms / self.parameters.dt_ms))
        self._refractory_steps = ceil(
            self.parameters.refractory_ms / self.parameters.dt_ms
        )
        self._queue = np.zeros(
            (self._delay_steps + 1, neuron_count), dtype=np.float32
        )
        self._queue_index = 0
        self.voltage_mv = np.full(
            neuron_count, self.parameters.resting_mv, dtype=np.float32
        )
        self.synaptic_current = np.zeros(neuron_count, dtype=np.float32)
        self._refractory_left = np.zeros(neuron_count, dtype=np.int16)
        self.step_index = 0

    def reset(self) -> None:
        self.voltage_mv.fill(self.parameters.resting_mv)
        self.synaptic_current.fill(0)
        self._refractory_left.fill(0)
        self._queue.fill(0)
        self._queue_index = 0
        self.step_index = 0

    def step(self, external_current: np.ndarray | None = None) -> np.ndarray:
        p = self.parameters
        if external_current is None:
            drive = 0.0
        else:
            drive = np.asarray(external_current, dtype=np.float32)
            if drive.shape != (self.neuron_count,):
                raise ValueError("external_current must have one value per neuron")

        due = self._queue[self._queue_index]
        self.synaptic_current *= np.float32(exp(-p.dt_ms / p.tau_synaptic_ms))
        self.synaptic_current += due
        due.fill(0)

        refractory = self._refractory_left > 0
        self._refractory_left[refractory] -= 1
        self.voltage_mv[refractory] = p.reset_mv
        active = ~refractory
        total_current = self.synaptic_current + drive
        self.voltage_mv[active] += np.float32(p.dt_ms / p.tau_membrane_ms) * (
            p.resting_mv - self.voltage_mv[active] + total_current[active]
        )
        spiked_mask = active & (self.voltage_mv >= p.threshold_mv)
        spikes = np.flatnonzero(spiked_mask).astype(np.int32)
        if len(spikes):
            self.voltage_mv[spikes] = p.reset_mv
            self._refractory_left[spikes] = self._refractory_steps
            active_edges = spiked_mask[self.sources]
            if np.any(active_edges):
                delivery = np.bincount(
                    self.targets[active_edges],
                    weights=self.weights[active_edges],
                    minlength=self.neuron_count,
                ).astype(np.float32)
                bucket = (self._queue_index + self._delay_steps) % len(self._queue)
                self._queue[bucket] += delivery

        self._queue_index = (self._queue_index + 1) % len(self._queue)
        self.step_index += 1
        return spikes
