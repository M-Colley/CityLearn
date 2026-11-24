"""Great Britain carbon intensity from the public Carbon Intensity API (no key)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from citylearn.price_carbon.common import build_grid, carbon_frame, to_grid

API_URL = 'https://api.carbonintensity.org.uk'
SOURCE = 'National Energy System Operator (NESO) Carbon Intensity API, https://carbonintensity.org.uk/'
# The API accepts at most 14 days per request.
MAX_DAYS = 13


def parse_intensity(payload: dict, field: str = 'actual') -> pd.Series:
    """Parse an `/intensity/{from}/{to}` response to gCO2/kWh indexed by period start (UTC).

    `actual` is null for the most recent periods; it then falls back to `forecast`.
    """
    rows = {}
    for item in payload['data']:
        intensity = item['intensity']
        value = intensity.get(field)
        if value is None:
            value = intensity.get('forecast')
        if value is not None:
            rows[pd.Timestamp(item['from'])] = float(value)
    return pd.Series(rows, dtype=float).sort_index()


def fetch_intensity(start, end, field: str = 'actual', session=None, timeout: float = 60) -> pd.Series:
    """Download gCO2/kWh for `[start, end)` in chunks of 13 days. `start` and `end` are UTC timestamps."""
    import requests

    http = session or requests.Session()
    parts = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=MAX_DAYS), end)
        url = f"{API_URL}/intensity/{cursor.strftime('%Y-%m-%dT%H:%MZ')}/{chunk_end.strftime('%Y-%m-%dT%H:%MZ')}"
        response = http.get(url, headers={'Accept': 'application/json'}, timeout=timeout)
        response.raise_for_status()
        parts.append(parse_intensity(response.json(), field))
        cursor = chunk_end
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep='last')].sort_index()


def gb_carbon(start, end, resolution_minutes: int = 60, tz: str = 'UTC', field: str = 'actual', max_gap_steps: int = 3,
              raw: Optional[pd.Series] = None, session=None):
    """Return `(carbon_frame, grid, metadata)` in kgCO2/kWh. `raw` (gCO2/kWh) skips the download."""
    grid = build_grid(start, end, resolution_minutes, tz)
    if raw is None:
        # The API returns the 30 min period that ends at the query start, so request from 30 min earlier.
        raw = fetch_intensity(grid[0] - pd.Timedelta(minutes=30), grid[-1] + pd.Timedelta(minutes=resolution_minutes), field, session)
    grams = to_grid(raw, grid, resolution_minutes, max_gap_steps, max_native_period=pd.Timedelta(minutes=30))
    meta = {'source': SOURCE, 'api': API_URL, 'field': field, 'unit_in': 'gCO2/kWh', 'unit_out': 'kgCO2/kWh',
            'basis': 'national average, generation-based (not marginal)', 'resolution_minutes': resolution_minutes}
    return carbon_frame(grams / 1000.0), grid, meta
