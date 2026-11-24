r"""Compare CityLearn thermal dynamics with a BOPTEST (Modelica) building emulator.

Evaluation only: this module is not imported by :py:class:`citylearn.citylearn.CityLearnEnv`.

The comparison has three parts.

1. :py:class:`BOPTESTClient` drives a BOPTEST web service (version 2 REST API) and records a
   trajectory. :py:func:`load_reference_trajectory` reads recorded BOPTEST results (CSV) instead.
2. :py:class:`CityLearnThermalEmulator` replays the CityLearn LSTM temperature model
   (:py:class:`citylearn.dynamics.LSTMDynamics`) with the same input pipeline as
   :py:class:`citylearn.building.LSTMDynamicsBuilding`, and
   :py:class:`citylearn.energy_model.HeatPump` converts delivered heat to electricity.
3. :py:func:`compare` reports temperature RMSE and bias, electric energy difference and comfort
   (thermal discomfort in Kh, BOPTEST ``tdis_tot`` definition) for both models.

Units: BOPTEST reports SI (K, W); CityLearn uses degrees C and kWh per time step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

HOUR_S = 3600.0
# BOPTEST calendar: zero time is 1 January 2019 (a Tuesday); weekday index 0 = Monday.
_FIRST_WEEKDAY = 1


# --------------------------------------------------------------------------------------------
# BOPTEST REST client
# --------------------------------------------------------------------------------------------

class BOPTESTError(RuntimeError):
    """The BOPTEST service returned an error status or an unexpected payload."""


class BOPTESTClient:
    """Minimal client for the BOPTEST version 2 REST API (``POST /testcases/<name>/select``).

    Parameters
    ----------
    base_url : str
        Service root, e.g. ``http://127.0.0.1:8000``.
    session : requests.Session, optional
        Injected for tests; a new session (connection reuse) is created otherwise.
    timeout : float, default: 600.0
        Request timeout in seconds. Advancing a stiff model can take minutes.
    """

    def __init__(self, base_url: str, session: Any = None, timeout: float = 600.0):
        if session is None:
            import requests
            session = requests.Session()

        self.base_url = base_url.rstrip('/')
        self.session = session
        self.timeout = timeout
        self.testid: Optional[str] = None

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, f'{self.base_url}{path}', timeout=self.timeout, **kwargs)

        if response.status_code >= 400:
            raise BOPTESTError(f'{method} {path} failed with HTTP {response.status_code}: {response.text[:200]}')

        body = response.json()
        status = body.get('status', 200) if isinstance(body, dict) else 200

        if status >= 400:
            raise BOPTESTError(f'{method} {path} failed with status {status}: {body.get("message")}')

        return body.get('payload', body) if isinstance(body, dict) else body

    def select(self, testcase: str) -> str:
        """Start a new test of `testcase` and return its test id."""

        body = self._request('POST', f'/testcases/{testcase}/select', json={})
        self.testid = body['testid']
        return self.testid

    def initialize(self, start_time: float, warmup_period: float = 0.0) -> Mapping[str, float]:
        return self._request('PUT', f'/initialize/{self.testid}', json={'start_time': start_time, 'warmup_period': warmup_period})

    def set_step(self, step: float) -> Any:
        return self._request('PUT', f'/step/{self.testid}', json={'step': step})

    def advance(self, **inputs: float) -> Mapping[str, float]:
        """Advance one control step; `inputs` use BOPTEST names (``oveHeaPumY_u``, ``oveHeaPumY_activate``)."""

        return self._request('POST', f'/advance/{self.testid}', json=inputs)

    def results(self, point_names: Iterable[str], start_time: float, final_time: float) -> Mapping[str, list]:
        return self._request(
            'PUT', f'/results/{self.testid}',
            json={'point_names': list(point_names), 'start_time': start_time, 'final_time': final_time},
        )

    def kpi(self) -> Mapping[str, float]:
        return self._request('GET', f'/kpi/{self.testid}')

    def stop(self) -> None:
        if self.testid is not None:
            self._request('PUT', f'/stop/{self.testid}')
            self.testid = None


HEAT_PUMP_POINTS = ('reaTZon_y', 'reaQFloHea_y', 'reaPHeaPum_y', 'reaPFan_y', 'reaPPumEmi_y')


def record_open_loop(
    client: BOPTESTClient, testcase: str, start_time: float, length: float, step: float,
    schedule: Iterable[float], warmup_period: float = 7 * 86400.0, control_input: str = 'oveHeaPumY',
) -> pd.DataFrame:
    """Drive a BOPTEST test case with a modulation `schedule` (one value in [0, 1] per step) and return the raw results.

    The frame has the columns of the BOPTEST results files (``time`` and :py:data:`HEAT_PUMP_POINTS`), so
    it can be written to CSV and passed to :py:func:`load_reference_trajectory`.
    """

    schedule = list(schedule)
    n_steps = int(round(length / step))

    if len(schedule) != n_steps:
        raise ValueError(f'schedule has {len(schedule)} values, expected {n_steps}.')

    client.select(testcase)

    try:
        client.initialize(start_time, warmup_period)
        client.set_step(step)

        for value in schedule:
            client.advance(**{f'{control_input}_u': float(value), f'{control_input}_activate': 1})

        points = client.results(('time', *HEAT_PUMP_POINTS), start_time, start_time + length)
    finally:
        client.stop()

    frame = pd.DataFrame(points)
    frame['time'] = frame['time'].astype(float)
    return frame


# --------------------------------------------------------------------------------------------
# Reference data and calendar
# --------------------------------------------------------------------------------------------

def occupants(time_s: np.ndarray) -> np.ndarray:
    """Occupant count of ``bestest_hydronic_heat_pump``: 5 before 07:00, from 20:00 and all weekend, else 0."""

    time_s = np.asarray(time_s, dtype=float)
    hour = (time_s % 86400.0) // HOUR_S
    weekday = (np.floor(time_s / 86400.0).astype(int) + _FIRST_WEEKDAY) % 7
    return np.where((hour < 7) | (hour > 19) | (weekday > 4), 5.0, 0.0)


def comfort_band(time_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Comfort band in degrees C: 21 to 24 when occupied, 15 to 30 when unoccupied."""

    occupied = occupants(time_s) > 0
    return np.where(occupied, 21.0, 15.0), np.where(occupied, 24.0, 30.0)


def load_reference_trajectory(results_csv: Path | str, weather_csv: Path | str, step: float = HOUR_S) -> pd.DataFrame:
    """Read a BOPTEST results file and resample it to a uniform `step` (seconds).

    Power signals are converted to energy per step by integrating the recorded power (trapezoid rule) and
    interpolating the cumulative energy, so the resampled energy matches the recorded energy. Temperatures
    and weather are interpolated linearly.

    Returns
    -------
    pandas.DataFrame
        Columns: ``time_s`` (end of the step), ``t_zone_c``, ``t_out_c``, ``solar_direct``, ``solar_diffuse``,
        ``heat_kwh``, ``electric_kwh`` (energies per step) and ``occupant_count``.
    """

    raw = pd.read_csv(results_csv).sort_values('time')
    weather = pd.read_csv(weather_csv)
    raw_time = raw['time'].to_numpy(float)
    edges = np.arange(np.ceil(raw_time[0] / step) * step, raw_time[-1] + 1e-6, step)

    def energy_kwh(power_w: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (power_w[1:] + power_w[:-1]) * np.diff(raw_time))])
        return np.diff(np.interp(edges, raw_time, cumulative)) / 3.6e6

    electric_w = sum(raw[c].to_numpy(float) for c in ('reaPHeaPum_y', 'reaPFan_y', 'reaPPumEmi_y'))
    time_s = edges[1:]
    mid = time_s - step / 2

    return pd.DataFrame({
        'time_s': time_s,
        't_zone_c': np.interp(time_s, raw_time, raw['reaTZon_y'].to_numpy(float)) - 273.15,
        't_out_c': np.interp(mid, weather['time'], weather['TDryBul']) - 273.15,
        'solar_direct': np.interp(mid, weather['time'], weather['HDirNor']),
        'solar_diffuse': np.interp(mid, weather['time'], weather['HDifHor']),
        'heat_kwh': energy_kwh(raw['reaQFloHea_y'].to_numpy(float)),
        'electric_kwh': energy_kwh(electric_w),
        'occupant_count': occupants(mid),
    })


# --------------------------------------------------------------------------------------------
# CityLearn side
# --------------------------------------------------------------------------------------------

@dataclass
class LSTMSpec:
    """LSTM dynamics and heat pump parameters taken from a CityLearn dataset schema."""

    dynamics: Dict[str, Any]
    heat_pump: Dict[str, Any]
    directory: Path


def load_schema_spec(schema_path: Path | str, building: str) -> LSTMSpec:
    schema_path = Path(schema_path)
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    b = schema['buildings'][building]
    return LSTMSpec(dynamics=dict(b['dynamics']['attributes']), heat_pump=dict(b['heating_device']['attributes']), directory=schema_path.parent)


class CityLearnThermalEmulator:
    """Standalone replay of the CityLearn LSTM temperature model.

    Reuses :py:class:`citylearn.dynamics.LSTMDynamics` (weights, hidden state, normalization limits) and
    mirrors the input assembly of :py:meth:`citylearn.building.LSTMDynamicsBuilding.update_indoor_dry_bulb_temperature`:
    inputs are min-max normalized, the temperature input is shifted by one step, and the LSTM hidden state
    carries over between steps. Requires PyTorch.
    """

    def __init__(self, spec: LSTMSpec):
        from citylearn.dynamics import LSTMDynamics
        from citylearn.energy_model import HeatPump
        from citylearn.preprocessing import PeriodicNormalization

        attrs = {k: v for k, v in spec.dynamics.items() if k != 'filename'}
        self.dynamics = LSTMDynamics(filepath=spec.directory / spec.dynamics['filename'], **attrs)
        self.dynamics.reset()
        self.heat_pump = HeatPump(**spec.heat_pump)
        self._pn = {'month': PeriodicNormalization(12), 'hour': PeriodicNormalization(24), 'day_type': PeriodicNormalization(7)}
        self._names = list(self.dynamics.input_observation_names)
        self._lo = np.asarray(self.dynamics.input_normalization_minimum, dtype=float)
        self._hi = np.asarray(self.dynamics.input_normalization_maximum, dtype=float)

    def _observations(self, row: Mapping[str, float]) -> np.ndarray:
        """CityLearn observations for one step. ``time_s`` is the step end; CityLearn hour 1 is 00:00 to 01:00."""

        t = row['time_s'] - HOUR_S
        hour = int((t % 86400.0) // HOUR_S) + 1
        weekday = (int(t // 86400.0) + _FIRST_WEEKDAY) % 7 + 1  # 1 = Monday
        month = (pd.Timestamp('2019-01-01') + pd.Timedelta(seconds=t)).month
        values = {
            'direct_solar_irradiance': row['solar_direct'], 'diffuse_solar_irradiance': row['solar_diffuse'],
            'outdoor_dry_bulb_temperature': row['t_out_c'], 'occupant_count': row['occupant_count'],
            'heating_demand': row['heat_kwh'], 'cooling_demand': 0.0,
        }

        for key, raw in (('month', month), ('hour', hour), ('day_type', weekday)):
            values[f'{key}_sin'], values[f'{key}_cos'] = self._pn[key] * raw

        return np.asarray([values.get(n, np.nan) for n in self._names], dtype=float)

    def simulate(self, trajectory: pd.DataFrame, free_run: bool = True) -> np.ndarray:
        """Predict zone temperature in degrees C for every row of `trajectory`.

        The first ``lookback`` rows copy the recorded temperature (warm-up, as CityLearn does with dataset values).
        ``free_run=True`` feeds predictions back (closed-loop rollout, the CityLearn runtime mode).
        ``free_run=False`` feeds the recorded temperature (one-step-ahead accuracy).
        """

        import torch

        lookback = self.dynamics.lookback
        ix = self._names.index('indoor_dry_bulb_temperature')
        span = self._hi[ix] - self._lo[ix]
        measured = trajectory['t_zone_c'].to_numpy(float)
        predicted = np.full(len(trajectory), np.nan)
        window = np.full((len(self._names), lookback + 1), np.nan)
        hidden = self.dynamics.init_hidden(1)

        for k, row in enumerate(trajectory.to_dict('records')):
            obs = (self._observations(row) - self._lo) / (self._hi - self._lo)
            window = np.concatenate([window[:, 1:], obs[:, None]], axis=1)

            if k < lookback:
                predicted[k] = measured[k]
                window[ix, -1] = (measured[k] - self._lo[ix]) / span
                continue

            x = window[:, 1:].copy()
            x[ix] = window[ix, :-1]

            with torch.no_grad():
                out, hidden = self.dynamics(torch.tensor(x.T[None], dtype=torch.float32), hidden)

            norm = float(out.item())
            predicted[k] = norm * span + self._lo[ix]
            window[ix, -1] = norm if free_run else (measured[k] - self._lo[ix]) / span

        return predicted

    def electric_kwh(self, trajectory: pd.DataFrame) -> np.ndarray:
        """Heat pump input energy per step for the delivered heat in `trajectory`."""

        return np.asarray(self.heat_pump.get_input_power(
            trajectory['heat_kwh'].to_numpy(float), trajectory['t_out_c'].to_numpy(float), heating=True), dtype=float)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def discomfort_kh(temperature_c: np.ndarray, time_s: np.ndarray, step: float = HOUR_S) -> float:
    """Thermal discomfort in Kh: integral of the distance outside the BOPTEST comfort band."""

    low, high = comfort_band(np.asarray(time_s) - step / 2)
    t = np.asarray(temperature_c, dtype=float)
    return float((np.clip(low - t, 0, None) + np.clip(t - high, 0, None)).sum() * step / HOUR_S)


def temperature_metrics(reference: np.ndarray, model: np.ndarray) -> Dict[str, float]:
    err = np.asarray(model, dtype=float) - np.asarray(reference, dtype=float)
    return {'rmse_k': float(np.sqrt(np.mean(err ** 2))), 'bias_k': float(err.mean()), 'mae_k': float(np.abs(err).mean())}


def compare(trajectory: pd.DataFrame, emulator: CityLearnThermalEmulator, warmup_steps: Optional[int] = None) -> Dict[str, Any]:
    """Compare CityLearn against the BOPTEST `trajectory` on the steps after the LSTM warm-up.

    A constant-temperature baseline (the BOPTEST mean) bounds what a model without dynamics scores.
    """

    k0 = emulator.dynamics.lookback if warmup_steps is None else warmup_steps
    ref = trajectory.iloc[k0:]
    ref_t = ref['t_zone_c'].to_numpy()
    out: Dict[str, Any] = {'steps': int(len(ref)), 'warmup_steps': int(k0), 'step_s': HOUR_S}
    candidates = {
        'free_run': emulator.simulate(trajectory, free_run=True)[k0:],
        'one_step': emulator.simulate(trajectory, free_run=False)[k0:],
        'constant_mean': np.full_like(ref_t, ref_t.mean()),
    }

    for label, pred in candidates.items():
        out[label] = {**temperature_metrics(ref_t, pred), 'discomfort_kh': discomfort_kh(pred, ref['time_s'])}

    citylearn_electric = float(emulator.electric_kwh(trajectory)[k0:].sum())
    boptest_electric = float(ref['electric_kwh'].sum())
    out['boptest'] = {
        'discomfort_kh': discomfort_kh(ref_t, ref['time_s']), 'mean_temperature_c': float(ref_t.mean()),
        'electric_kwh': boptest_electric, 'heat_kwh': float(ref['heat_kwh'].sum()),
    }
    out['citylearn_electric_kwh'] = citylearn_electric
    out['electric_difference_pct'] = 100.0 * (citylearn_electric - boptest_electric) / boptest_electric
    return out


def format_report(name: str, result: Mapping[str, Any]) -> str:
    rows = [
        f'| {k} | {result[k]["rmse_k"]:.2f} | {result[k]["bias_k"]:+.2f} | {result[k]["mae_k"]:.2f} | {result[k]["discomfort_kh"]:.1f} |'
        for k in ('free_run', 'one_step', 'constant_mean')
    ]
    b = result['boptest']
    return '\n'.join([
        f'### {name} ({result["steps"]} hourly steps after {result["warmup_steps"]} warm-up steps)', '',
        '| CityLearn model | RMSE [K] | bias [K] | MAE [K] | discomfort [Kh] |', '|---|---|---|---|---|', *rows, '',
        f'BOPTEST: mean zone temperature {b["mean_temperature_c"]:.2f} C, discomfort {b["discomfort_kh"]:.1f} Kh, '
        f'heat {b["heat_kwh"]:.0f} kWh, electricity {b["electric_kwh"]:.0f} kWh.',
        f'CityLearn HeatPump electricity for the same heat: {result["citylearn_electric_kwh"]:.0f} kWh ({result["electric_difference_pct"]:+.1f} %).', '',
    ])
