"""Optional semi-empirical battery ageing models for :py:class:`citylearn.energy_model.Battery`.

The only model is BLAST-Lite (https://github.com/NatLabRockies/BLAST-Lite), installed with
``pip install citylearn[degradation]``. The battery records one SOC and temperature sample per step and
passes the batch to the BLAST-Lite state update every ``update_interval_steps`` steps. Without a
``degradation`` setting the battery keeps its linear capacity-loss model.
"""

from typing import Any, Callable, Mapping, Optional

import numpy as np

DEFAULT_CELL = 'Nmc111_Gr_Kokam75Ah_Battery'
DEFAULT_UPDATE_INTERVAL_STEPS = 24
DEFAULT_TEMPERATURE_C = 25.0
OUTDOOR_DRY_BULB = 'outdoor_dry_bulb'
_KEYS = {'model', 'cell', 'update_interval_steps', 'temperature', 'degradation_scalar'}


def available_cells() -> list:
    """BLAST-Lite cell model ids, for the `cell` setting."""

    return list(_blast_models().available_models())


def _blast_models():
    try:
        from blast import models
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Battery degradation model 'blast_lite' needs BLAST-Lite: pip install citylearn[degradation]"
        ) from error
    return models


class BlastLiteDegradation:
    """Capacity fade of one battery from a BLAST-Lite cell model.

    Parameters
    ----------
    cell : str, default: 'Nmc111_Gr_Kokam75Ah_Battery'
        Class name in ``blast.models``, e.g. ``Lfp_Gr_250AhPrismatic`` (LFP) or
        ``Nmc111_Gr_Kokam75Ah_Battery`` (NMC). See :py:func:`available_cells`.
    update_interval_steps : int, default: 24
        Steps between state updates. Larger values cost less and resolve less of the SOC trajectory.
    temperature : float or 'outdoor_dry_bulb', default: 25.0
        Cell temperature in [C], constant or the building weather dry bulb temperature.
    degradation_scalar : float, default: 1.0
        Multiplier on the model's degradation rate.

    Notes
    -----
    SOC is passed as a fraction of the current capacity, as BLAST-Lite expects. Resistance growth is not
    modelled by CityLearn and is not applied. Steps after the last completed interval of an episode are not
    included in the capacity.
    """

    def __init__(
        self, cell: str = None, update_interval_steps: int = None, temperature: Any = None,
        degradation_scalar: float = None
    ):
        self.cell = DEFAULT_CELL if cell is None else str(cell)
        self.update_interval_steps = DEFAULT_UPDATE_INTERVAL_STEPS if update_interval_steps is None else int(update_interval_steps)
        self.degradation_scalar = 1.0 if degradation_scalar is None else float(degradation_scalar)
        temperature = DEFAULT_TEMPERATURE_C if temperature is None else temperature
        if self.update_interval_steps < 1:
            raise ValueError('update_interval_steps must be >= 1.')
        if self.degradation_scalar < 0.0:
            raise ValueError('degradation_scalar must be >= 0.')
        if temperature == OUTDOOR_DRY_BULB:
            self.temperature = temperature
        else:
            self.temperature = float(temperature)
        models = _blast_models()
        if self.cell not in models.available_models():
            raise ValueError(f"Unknown BLAST-Lite cell '{self.cell}'. Available: {', '.join(models.available_models())}.")
        self._constructor = getattr(models, self.cell)
        self.temperature_provider: Optional[Callable[[int], float]] = None
        self.reset(None)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> 'BlastLiteDegradation':
        """Build from a schema ``degradation`` mapping."""

        unknown = set(config) - _KEYS
        if unknown:
            raise ValueError(f"Unknown degradation keys {sorted(unknown)}; valid keys are {sorted(_KEYS)}.")
        if config.get('model', 'blast_lite') != 'blast_lite':
            raise ValueError("degradation.model must be 'blast_lite'.")
        return cls(**{k: v for k, v in config.items() if k != 'model'})

    @property
    def needs_weather(self) -> bool:
        return self.temperature == OUTDOOR_DRY_BULB

    @property
    def q(self) -> float:
        """Capacity as a fraction of the beginning-of-life capacity."""

        return self._q

    def reset(self, soc0: Optional[float]):
        """Start a new episode at relative SOC `soc0` and time zero."""

        self._model = self._constructor(degradation_scalar=self.degradation_scalar)
        self._q = 1.0
        self._t = []
        self._soc = []
        self._steps = []
        self._steps_since_update = 0
        if soc0 is not None:
            self._append(0.0, soc0, 0)
            self._steps_since_update = 0

    def _temperature_c(self, time_step: int) -> float:
        if self.temperature == OUTDOOR_DRY_BULB:
            if self.temperature_provider is None:
                raise ValueError("degradation.temperature 'outdoor_dry_bulb' needs a building weather source.")
            return float(self.temperature_provider(time_step))
        return self.temperature

    def _append(self, t_seconds: float, soc: float, time_step: int):
        self._t.append(float(t_seconds))
        self._soc.append(float(min(max(soc, 0.0), 1.0)))
        self._steps.append(int(time_step))

    def record(self, time_step: int, seconds_per_time_step: float, soc: float):
        """Record the SOC at the end of `time_step` (relative to current capacity); update on the interval.

        A second call in the same step replaces that step's sample.
        """

        t = (time_step + 1) * float(seconds_per_time_step)
        if self._t and self._t[-1] == t:
            self._soc[-1] = float(min(max(soc, 0.0), 1.0))
            return self._q
        self._append(t, soc, time_step)
        self._steps_since_update += 1

        if self._steps_since_update >= self.update_interval_steps:
            self._update()

        return self._q

    def _update(self):
        t = np.asarray(self._t)
        temperature = np.fromiter((self._temperature_c(i) for i in self._steps), float, len(self._steps))
        self._model.update_battery_state(t, np.asarray(self._soc), temperature)
        self._q = float(self._model.outputs['q'][-1])
        # The last sample starts the next batch so no interval is skipped.
        self._t, self._soc, self._steps = self._t[-1:], self._soc[-1:], self._steps[-1:]
        self._steps_since_update = 0
