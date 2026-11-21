"""Opt-in peak-window flexibility KPIs separating BAU load reduction from EV V2X contribution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from citylearn.internal.units import seconds_to_hours

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv
    from citylearn.internal.kpi import CityLearnKPIService


DEFAULT_FLEXIBILITY_PEAK_HOURS: Tuple[int, int] = (16, 19)


def validate_flexibility_peak_hours(peak_hours: Optional[Sequence[int]]) -> Tuple[float, float]:
    """Validate ``(start_hour, end_hour)`` in 24-hour clock ``[0, 24]``."""
    if peak_hours is None:
        return (16.0, 19.0)

    if isinstance(peak_hours, (str, bytes, Mapping)):
        raise ValueError('flexibility_peak_hours must be a 2-tuple (start_hour, end_hour) in [0, 24].')

    try:
        values = list(peak_hours)
    except TypeError as exc:
        raise ValueError('flexibility_peak_hours must be a 2-tuple (start_hour, end_hour) in [0, 24].') from exc

    if len(values) != 2 or any(isinstance(v, bool) for v in values):
        raise ValueError('flexibility_peak_hours must have exactly two numeric bounds (start_hour, end_hour).')

    try:
        start_h = float(values[0])
        end_h = float(values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError('flexibility_peak_hours bounds must be numeric.') from exc

    if not (math.isfinite(start_h) and math.isfinite(end_h)):
        raise ValueError('flexibility_peak_hours bounds must be finite.')
    if not (0.0 <= start_h <= 24.0 and 0.0 <= end_h <= 24.0):
        raise ValueError('flexibility_peak_hours bounds must lie within [0, 24].')
    if abs(start_h - end_h) <= 1.0e-12:
        raise ValueError('flexibility_peak_hours start_hour and end_hour must differ.')

    return (start_h, end_h)


def _in_peak_window(hours_raw: np.ndarray, start_h: float, end_h: float) -> np.ndarray:
    """Boolean mask of steps whose 24-hour clock hour falls in ``[start_h, end_h)``."""
    finite = np.isfinite(hours_raw)
    hours_24 = np.where(finite, np.mod(hours_raw, 24.0), np.nan)
    if start_h < end_h:
        return finite & (hours_24 >= start_h) & (hours_24 < end_h)
    return finite & ((hours_24 >= start_h) | (hours_24 < end_h))


def _window_stats(series_kw: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(mean_kw, p90_firm_kw)`` over finite peak-window samples (10th percentile = P90 firm)."""
    finite = series_kw[np.isfinite(series_kw)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.percentile(finite, 10))


def compute_peak_flexibility_audit_records(
    kpi: "CityLearnKPIService",
    *,
    kpi_buildings: Sequence,
    building_windows: Mapping[str, Tuple[int, int]],
    bau_env: Optional["CityLearnEnv"],
    flexibility_peak_hours: Optional[Sequence[int]] = DEFAULT_FLEXIBILITY_PEAK_HOURS,
) -> List[Dict[str, object]]:
    """Compute opt-in peak-window BAU load reduction and EV V2X discharge KPI rows."""
    start_h, end_h = validate_flexibility_peak_hours(flexibility_peak_hours)
    env = kpi.env
    step_hours = max(seconds_to_hours(float(getattr(env, 'seconds_per_time_step', 3600.0) or 3600.0)), 1.0e-12)

    executed_steps = min(
        len(getattr(env, 'net_electricity_consumption', []) or []),
        max(int(getattr(env, 'time_step', 0) or 0), 0),
    )

    bau_building_by_name: Dict[str, object] = {}
    if bau_env is not None:
        bau_buildings = list(getattr(bau_env, '_all_buildings', None) or getattr(bau_env, 'buildings', []) or [])
        bau_building_by_name = {str(b.name): b for b in bau_buildings}

    rows: List[Dict[str, object]] = []

    def add(level: str, name: str, cost_function: str, value) -> None:
        rows.append({
            'cost_function': cost_function,
            'value': value,
            'name': name,
            'level': level,
        })

    district_charger_net_kwh = np.zeros(executed_steps, dtype='float64')
    reference_hours: Optional[np.ndarray] = None

    for building in kpi_buildings:
        b_name = str(building.name)
        t_start, t_end_raw = building_windows.get(b_name, (0, -1))
        b_net_all = np.asarray(getattr(building, 'net_electricity_consumption', []), dtype='float64')
        hours_all = np.asarray(getattr(building.energy_simulation, 'hour', []), dtype='float64')
        if reference_hours is None and hours_all.size >= executed_steps and executed_steps > 0:
            reference_hours = hours_all[:executed_steps]

        start = max(int(t_start), 0)
        end = min(int(t_end_raw), executed_steps - 1, len(b_net_all) - 1, len(hours_all) - 1)
        if executed_steps <= 0 or end < start:
            add('building', b_name, 'building_peak_window_mean_load_reduction_to_business_as_usual_kw', None)
            add('building', b_name, 'building_peak_window_p90_load_reduction_to_business_as_usual_kw', None)
            add('building', b_name, 'building_ev_peak_window_net_export_kwh', 0.0)
            add('building', b_name, 'building_ev_peak_window_mean_net_discharge_kw', None)
            add('building', b_name, 'building_ev_peak_window_p90_net_discharge_kw', None)
            continue

        window_len = end - start + 1
        charger_net = np.zeros(window_len, dtype='float64')
        for charger in kpi._chargers_for_metrics(building):
            c_series = np.asarray(getattr(charger, 'electricity_consumption', []), dtype='float64')
            c_slice = c_series[start:end + 1]
            if c_slice.size > 0:
                valid_c = np.where(np.isfinite(c_slice), c_slice, 0.0)
                charger_net[:valid_c.size] += valid_c

        district_charger_net_kwh[start:end + 1] += charger_net

        peak_mask = _in_peak_window(hours_all[start:end + 1], start_h, end_h)
        if not np.any(peak_mask):
            add('building', b_name, 'building_peak_window_mean_load_reduction_to_business_as_usual_kw', None)
            add('building', b_name, 'building_peak_window_p90_load_reduction_to_business_as_usual_kw', None)
            add('building', b_name, 'building_ev_peak_window_net_export_kwh', 0.0)
            add('building', b_name, 'building_ev_peak_window_mean_net_discharge_kw', None)
            add('building', b_name, 'building_ev_peak_window_p90_net_discharge_kw', None)
            continue

        # 1) EV / V2X contribution: strictly from EV charger electricity, independent of BAU.
        ev_net_export_kwh = float(max(0.0, float(np.sum(-charger_net[peak_mask]))))
        ev_net_discharge_kw = -charger_net[peak_mask] / step_hours
        ev_mean_kw, ev_p90_kw = _window_stats(ev_net_discharge_kw)

        # 2) Building load reduction vs BAU: undefined (None -> NaN) when BAU is unavailable.
        bau_mean_kw, bau_p90_kw = None, None
        b_bau = bau_building_by_name.get(b_name) if bau_env is not None else None
        if b_bau is not None:
            bau_net_all = np.asarray(getattr(b_bau, 'net_electricity_consumption', []), dtype='float64')
            if len(bau_net_all) > end:
                ctrl_kw = b_net_all[start:end + 1][peak_mask] / step_hours
                bau_kw = bau_net_all[start:end + 1][peak_mask] / step_hours
                bau_mean_kw, bau_p90_kw = _window_stats(bau_kw - ctrl_kw)

        add('building', b_name, 'building_peak_window_mean_load_reduction_to_business_as_usual_kw', bau_mean_kw)
        add('building', b_name, 'building_peak_window_p90_load_reduction_to_business_as_usual_kw', bau_p90_kw)
        add('building', b_name, 'building_ev_peak_window_net_export_kwh', ev_net_export_kwh)
        add('building', b_name, 'building_ev_peak_window_mean_net_discharge_kw', ev_mean_kw)
        add('building', b_name, 'building_ev_peak_window_p90_net_discharge_kw', ev_p90_kw)

    if reference_hours is None and executed_steps > 0 and kpi_buildings:
        first_hours = np.asarray(getattr(kpi_buildings[0].energy_simulation, 'hour', []), dtype='float64')
        if first_hours.size > 0:
            reference_hours = first_hours[:min(executed_steps, first_hours.size)]

    if executed_steps <= 0 or reference_hours is None or reference_hours.size == 0:
        add('district', 'District', 'district_peak_window_mean_load_reduction_to_business_as_usual_kw', None)
        add('district', 'District', 'district_peak_window_p90_load_reduction_to_business_as_usual_kw', None)
        add('district', 'District', 'district_ev_peak_window_net_export_kwh', 0.0)
        add('district', 'District', 'district_ev_peak_window_mean_net_discharge_kw', None)
        add('district', 'District', 'district_ev_peak_window_p90_net_discharge_kw', None)
        return rows

    n_ref = reference_hours.size
    d_peak_mask = _in_peak_window(reference_hours, start_h, end_h)
    if not np.any(d_peak_mask):
        add('district', 'District', 'district_peak_window_mean_load_reduction_to_business_as_usual_kw', None)
        add('district', 'District', 'district_peak_window_p90_load_reduction_to_business_as_usual_kw', None)
        add('district', 'District', 'district_ev_peak_window_net_export_kwh', 0.0)
        add('district', 'District', 'district_ev_peak_window_mean_net_discharge_kw', None)
        add('district', 'District', 'district_ev_peak_window_p90_net_discharge_kw', None)
        return rows

    d_ev_net_export_kwh = float(max(0.0, float(np.sum(-district_charger_net_kwh[:n_ref][d_peak_mask]))))
    d_ev_net_discharge_kw = -district_charger_net_kwh[:n_ref][d_peak_mask] / step_hours
    d_ev_mean_kw, d_ev_p90_kw = _window_stats(d_ev_net_discharge_kw)

    d_bau_mean_kw, d_bau_p90_kw = None, None
    if bau_env is not None:
        bau_dist_all = np.asarray(getattr(bau_env, 'net_electricity_consumption', []), dtype='float64')
        if bau_dist_all.size >= n_ref:
            ctrl_dist_kw = np.asarray(env.net_electricity_consumption[:n_ref], dtype='float64')[d_peak_mask] / step_hours
            bau_dist_kw = bau_dist_all[:n_ref][d_peak_mask] / step_hours
            d_bau_mean_kw, d_bau_p90_kw = _window_stats(bau_dist_kw - ctrl_dist_kw)

    add('district', 'District', 'district_peak_window_mean_load_reduction_to_business_as_usual_kw', d_bau_mean_kw)
    add('district', 'District', 'district_peak_window_p90_load_reduction_to_business_as_usual_kw', d_bau_p90_kw)
    add('district', 'District', 'district_ev_peak_window_net_export_kwh', d_ev_net_export_kwh)
    add('district', 'District', 'district_ev_peak_window_mean_net_discharge_kw', d_ev_mean_kw)
    add('district', 'District', 'district_ev_peak_window_p90_net_discharge_kw', d_ev_p90_kw)

    return rows
