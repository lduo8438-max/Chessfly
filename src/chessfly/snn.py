"""Small deterministic LIF reference engine.

The full MaleCNS backend will implement the same step semantics in native code.
Inputs and weights are explicit voltage increments, not calibrated physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Synapse:
    pre: int
    post: int
    weight: float
    delay_steps: int = 1

    def __post_init__(self) -> None:
        if self.pre < 0 or self.post < 0:
            raise ValueError("neuron indices must be non-negative")
        if self.delay_steps < 1:
            raise ValueError("delay_steps must be at least 1")


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float = 0.1
    tau_membrane_ms: float = 20.0
    resting_mv: float = -65.0
    threshold_mv: float = -50.0
    reset_mv: float = -65.0
    refractory_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.dt_ms <= 0 or self.tau_membrane_ms <= 0:
            raise ValueError("dt_ms and tau_membrane_ms must be positive")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms cannot be negative")
        if self.reset_mv >= self.threshold_mv:
            raise ValueError("reset_mv must be below threshold_mv")


class LIFNetwork:
    """Synchronous sparse LIF network with deterministic delayed delivery."""

    def __init__(
        self,
        neuron_count: int,
        synapses: Iterable[Synapse] = (),
        parameters: LIFParameters | None = None,
    ) -> None:
        if neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        self.neuron_count = neuron_count
        self.parameters = parameters or LIFParameters()
        self.synapses = tuple(synapses)

        outgoing: list[list[Synapse]] = [[] for _ in range(neuron_count)]
        max_delay = 1
        for synapse in self.synapses:
            if synapse.pre >= neuron_count or synapse.post >= neuron_count:
                raise ValueError("synapse index exceeds neuron_count")
            outgoing[synapse.pre].append(synapse)
            max_delay = max(max_delay, synapse.delay_steps)
        self._outgoing = tuple(tuple(items) for items in outgoing)
        self._queue: list[list[float]] = [
            [0.0] * neuron_count for _ in range(max_delay + 1)
        ]
        self._queue_index = 0
        self._refractory_steps = ceil(
            self.parameters.refractory_ms / self.parameters.dt_ms
        )
        self.voltage_mv = [self.parameters.resting_mv] * neuron_count
        self._refractory_left = [0] * neuron_count
        self.step_index = 0

    def reset(self) -> None:
        self.voltage_mv[:] = [self.parameters.resting_mv] * self.neuron_count
        self._refractory_left[:] = [0] * self.neuron_count
        for bucket in self._queue:
            bucket[:] = [0.0] * self.neuron_count
        self._queue_index = 0
        self.step_index = 0

    def step(
        self,
        external_mv: Mapping[int, float] | Sequence[float] | None = None,
    ) -> tuple[int, ...]:
        drive = self._coerce_drive(external_mv)
        due = self._queue[self._queue_index]
        spikes: list[int] = []
        p = self.parameters
        leak_scale = p.dt_ms / p.tau_membrane_ms

        for neuron in range(self.neuron_count):
            if self._refractory_left[neuron] > 0:
                self._refractory_left[neuron] -= 1
                self.voltage_mv[neuron] = p.reset_mv
                continue
            voltage = self.voltage_mv[neuron]
            voltage += (p.resting_mv - voltage) * leak_scale
            voltage += due[neuron] + drive[neuron]
            if voltage >= p.threshold_mv:
                spikes.append(neuron)
                self.voltage_mv[neuron] = p.reset_mv
                self._refractory_left[neuron] = self._refractory_steps
            else:
                self.voltage_mv[neuron] = voltage

        due[:] = [0.0] * self.neuron_count
        for pre in spikes:
            for synapse in self._outgoing[pre]:
                bucket = (self._queue_index + synapse.delay_steps) % len(self._queue)
                self._queue[bucket][synapse.post] += synapse.weight
        self._queue_index = (self._queue_index + 1) % len(self._queue)
        self.step_index += 1
        return tuple(spikes)

    def _coerce_drive(
        self,
        external_mv: Mapping[int, float] | Sequence[float] | None,
    ) -> list[float]:
        drive = [0.0] * self.neuron_count
        if external_mv is None:
            return drive
        if isinstance(external_mv, Mapping):
            for neuron, value in external_mv.items():
                if neuron < 0 or neuron >= self.neuron_count:
                    raise IndexError(f"external input neuron out of range: {neuron}")
                drive[neuron] = float(value)
            return drive
        if len(external_mv) != self.neuron_count:
            raise ValueError("external input sequence has wrong length")
        return [float(value) for value in external_mv]

