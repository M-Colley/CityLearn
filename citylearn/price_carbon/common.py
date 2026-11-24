"""Shared helpers: resampling to the CityLearn grid, predicted columns and CSV output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PRICING_COLUMNS = (
    'electricity_pricing',
    'electricity_pricing_predicted_1',
    'electricity_pricing_predicted_2',
    'electricity_pricing_predicted_3',
)
CARBON_COLUMN = 'carbon_intensity'
# Longest native period that may be held to cover a target step (hourly day-ahead prices).
MAX_NATIVE_PERIOD = pd.Timedelta(hours=1)


def parse_time(value, tz: str = 'UTC') -> pd.Timestamp:
    """Parse an ISO string as a time in `tz` unless it carries its own offset."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(tz) if ts.tzinfo is None else ts


def build_grid(start, end, resolution_minutes: int, tz: str = 'UTC') -> pd.DatetimeIndex:
    """Contiguous grid on absolute time, `[start, end)`. A local day at a DST change keeps its steps contiguous in UTC."""
    if resolution_minutes < 1:
        raise ValueError('resolution_minutes must be a positive integer.')
    start, end = parse_time(start, tz).tz_convert('UTC'), parse_time(end, tz).tz_convert('UTC')
    if end <= start:
        raise ValueError('end must be after start.')
    return pd.date_range(start, end, freq=f'{int(resolution_minutes)}min', inclusive='left')


def to_grid(
    series: pd.Series, grid: pd.DatetimeIndex, resolution_minutes: int, max_gap_steps: int = 3,
    max_native_period: pd.Timedelta = MAX_NATIVE_PERIOD,
) -> pd.Series:
    """Resample an irregular, coarser or finer series onto `grid`.

    Each native value covers the period up to the next value (at most `max_native_period`).
    A target step averages the minutes it covers. Steps without data are linearly interpolated
    up to `max_gap_steps` consecutive steps; a longer gap raises `ValueError`.
    """
    s = series.dropna()
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.index = s.index.tz_convert('UTC')
    if s.empty:
        raise ValueError('No data in the requested range.')
    step = pd.Timedelta(minutes=resolution_minutes)
    minutes = pd.date_range(grid[0], grid[-1] + step, freq='1min', inclusive='left')
    union = minutes.union(s.index)
    fine = s.reindex(union).ffill()
    # A native value covers its own period: the smaller of the spacing to the previous and to the next value,
    # at most `max_native_period`. A missing value therefore leaves a gap instead of a held value.
    spacing = s.index.to_series().diff()
    duration = pd.concat([spacing, spacing.shift(-1)], axis=1).min(axis=1).fillna(max_native_period).clip(upper=max_native_period)
    stamp = pd.Series(union.where(union.isin(s.index)), index=union).ffill()
    span = duration.reindex(union).ffill()
    age = union.to_series() - stamp
    fine = fine.where(age < span).reindex(minutes)
    out = fine.resample(step, label='left', closed='left').mean().reindex(grid)
    return fill_gaps(out, max_gap_steps)


def fill_gaps(series: pd.Series, max_gap_steps: int) -> pd.Series:
    missing = series.isna()
    if not missing.any():
        return series
    runs = (missing != missing.shift()).cumsum()
    longest = int(missing.groupby(runs).sum().max())
    if longest > max_gap_steps:
        first = series.index[int(missing.to_numpy().argmax())]
        raise ValueError(
            f'{int(missing.sum())} of {len(series)} steps have no data, the longest gap is {longest} steps '
            f'(limit {max_gap_steps}), first missing step {first}.')
    LOGGER.warning('Interpolated %d missing step(s).', int(missing.sum()))
    return series.interpolate(limit_area='inside').ffill().bfill()


def add_predicted(values: np.ndarray, resolution_minutes: int, horizons_hours: Sequence[float] = (1, 2, 3)) -> list:
    """Perfect-foresight predictions: step `t` holds the value at `t + horizon`; the tail repeats the last value."""
    out = []
    n = len(values)
    for hours in horizons_hours:
        shift = int(round(hours * 60 / resolution_minutes))
        if shift < 1:
            raise ValueError(f'Horizon {hours} h is shorter than one step of {resolution_minutes} min.')
        shifted = np.full(n, values[-1], dtype=float)
        if shift < n:
            shifted[:n - shift] = values[shift:]
        out.append(shifted)
    return out


def pricing_frame(price_per_kwh: pd.Series, resolution_minutes: int, horizons_hours: Sequence[float] = (1, 2, 3)) -> pd.DataFrame:
    values = price_per_kwh.to_numpy(dtype=float)
    predicted = add_predicted(values, resolution_minutes, horizons_hours)
    return pd.DataFrame(dict(zip(PRICING_COLUMNS, [values, *predicted])))


def carbon_frame(kg_per_kwh: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({CARBON_COLUMN: kg_per_kwh.to_numpy(dtype=float)})


def write_outputs(
    out_dir, pricing: Optional[pd.DataFrame], carbon: Optional[pd.DataFrame], metadata: Mapping,
    index: Optional[pd.DatetimeIndex] = None,
) -> list:
    """Write `pricing.csv`, `carbon_intensity.csv` and `price_carbon_metadata.json` (never a secret)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (('pricing.csv', pricing), ('carbon_intensity.csv', carbon)):
        if frame is not None:
            path = out_dir / name
            frame.to_csv(path, index=False)
            written.append(path)
    meta = dict(metadata)
    if index is not None:
        meta.update(first_step_utc=str(index[0]), last_step_utc=str(index[-1]), time_steps=len(index))
    path = out_dir / 'price_carbon_metadata.json'
    path.write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return written + [path]
