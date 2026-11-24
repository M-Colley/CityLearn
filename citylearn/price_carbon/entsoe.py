"""ENTSO-E day-ahead prices and generation-mix carbon intensity (needs the `price_carbon` extra).

The API key is read only from the `ENTSOE_API_KEY` environment variable. It is never logged,
written to a file or kept in the metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional, Tuple

import pandas as pd

from citylearn.price_carbon.common import build_grid, carbon_frame, pricing_frame, to_grid

LOGGER = logging.getLogger(__name__)

API_KEY_VARIABLE = 'ENTSOE_API_KEY'

# gCO2eq/kWh, lifecycle medians of IPCC AR5 WG3 Annex III, Table A.III.2 (Schlomer et al., 2014).
AR5_MEDIANS = {
    'coal': 820.0, 'gas': 490.0, 'biomass': 230.0, 'geothermal': 38.0, 'hydropower': 24.0, 'nuclear': 12.0,
    'solar_pv_utility': 48.0, 'wind_onshore': 11.0, 'wind_offshore': 12.0,
}
# ENTSO-E production types mapped to AR5 technologies. Pumped-storage output is counted as hydropower;
# the emissions of the pumping energy are not traced. Types missing here (oil, waste, peat, marine,
# 'Other', 'Other renewable', 'Energy storage') are dropped from the mix and reported in the metadata.
PSR_TO_AR5 = {
    'Biomass': 'biomass',
    'Fossil Brown coal/Lignite': 'coal',
    'Fossil Hard coal': 'coal',
    'Fossil Gas': 'gas',
    'Geothermal': 'geothermal',
    'Hydro Pumped Storage': 'hydropower',
    'Hydro Run-of-river and poundage': 'hydropower',
    'Hydro Water Reservoir': 'hydropower',
    'Nuclear': 'nuclear',
    'Solar': 'solar_pv_utility',
    'Wind Offshore': 'wind_offshore',
    'Wind Onshore': 'wind_onshore',
}
FACTOR_SOURCE = ('IPCC AR5 WG3 Annex III Table A.III.2 lifecycle medians (gCO2eq/kWh); production-based, '
                 'cross-border imports are not included')


def get_client(api_key: Optional[str] = None):
    """Create an `EntsoePandasClient` from `ENTSOE_API_KEY`."""
    key = api_key or os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise RuntimeError(f'Set the {API_KEY_VARIABLE} environment variable (free key from transparency.entsoe.eu).')
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError('Install the optional dependencies: pip install "citylearn[price_carbon]".') from exc
    return EntsoePandasClient(api_key=key), key


def mask(text: str, key: Optional[str]) -> str:
    return text.replace(key, '***') if key else text


def flatten_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two-level columns that entsoe-py returns without `nett=True` to net generation per type."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = {}
    for name in df.columns.get_level_values(0).unique():
        sub = df[name]
        if isinstance(sub, pd.Series):
            out[name] = sub
            continue
        net = sub['Actual Aggregated'].fillna(0) if 'Actual Aggregated' in sub else 0
        if 'Actual Consumption' in sub:
            net = net - sub['Actual Consumption'].fillna(0)
        out[name] = net
    return pd.DataFrame(out)


def carbon_from_mix(generation: pd.DataFrame, factors: Optional[Mapping[str, float]] = None, mapping: Optional[Mapping[str, str]] = None) -> Tuple[pd.Series, dict]:
    """Average intensity in gCO2/kWh per timestamp: sum(generation * factor) / sum(generation) over mapped types.

    Negative values (net consumption, e.g. pumped storage) are clipped to zero.
    """
    factors = dict(AR5_MEDIANS if factors is None else factors)
    mapping = dict(PSR_TO_AR5 if mapping is None else mapping)
    gen = flatten_generation(generation).clip(lower=0).fillna(0)
    used = [c for c in gen.columns if mapping.get(c) in factors]
    dropped = [c for c in gen.columns if c not in used]
    if not used:
        raise ValueError(f'No generation type has an emission factor: {list(gen.columns)}')
    weights = pd.Series({c: factors[mapping[c]] for c in used})
    total = gen[used].sum(axis=1)
    intensity = gen[used].mul(weights, axis=1).sum(axis=1) / total.where(total > 0)
    all_total = gen.sum(axis=1).sum()
    info = {'types_used': used, 'types_dropped': dropped,
            'dropped_energy_share': float(gen[dropped].sum().sum() / all_total) if all_total > 0 and dropped else 0.0}
    return intensity, info


def entsoe_datasets(
    zone: str, start, end, resolution_minutes: int = 60, tz: Optional[str] = None, horizons_hours=(1, 2, 3),
    max_gap_steps: int = 3, prices: Optional[pd.Series] = None, generation: Optional[pd.DataFrame] = None,
    factors: Optional[Mapping[str, float]] = None, max_dropped_share: float = 0.1,
):
    """Return `(pricing, carbon, grid, metadata)`. Prices are converted from currency/MWh to currency/kWh.

    `prices` and `generation` skip the download (used by the tests and for cached data).
    """
    key = None
    if prices is None or generation is None:
        client, key = get_client()
        zone_tz = tz or 'Europe/Brussels'
        grid = build_grid(start, end, resolution_minutes, zone_tz)
        try:
            # Pad one period on each side so every grid step is covered.
            lo, hi = grid[0] - pd.Timedelta(hours=1), grid[-1] + pd.Timedelta(minutes=resolution_minutes) + pd.Timedelta(hours=1)
            if prices is None:
                prices = client.query_day_ahead_prices(zone, start=lo, end=hi)
            if generation is None:
                generation = client.query_generation(zone, start=lo, end=hi, nett=True)
        except Exception as exc:
            raise RuntimeError(mask(f'{type(exc).__name__}: {exc}', key)) from None
    else:
        grid = build_grid(start, end, resolution_minutes, tz or 'UTC')

    price = to_grid(prices, grid, resolution_minutes, max_gap_steps) / 1000.0
    intensity, info = carbon_from_mix(generation, factors)
    if info['dropped_energy_share'] > max_dropped_share:
        raise ValueError(f"{info['dropped_energy_share']:.1%} of the generation has no emission factor "
                         f"({info['types_dropped']}); pass factors/mapping that cover it.")
    carbon = to_grid(intensity.dropna(), grid, resolution_minutes, max_gap_steps) / 1000.0
    meta = {'source': 'ENTSO-E Transparency Platform (entsoe-py)', 'zone': zone, 'price_unit': 'currency/kWh (EUR/MWh / 1000)',
            'carbon_unit': 'kgCO2/kWh', 'emission_factors_g_per_kwh': dict(AR5_MEDIANS if factors is None else factors),
            'emission_factor_source': FACTOR_SOURCE, 'resolution_minutes': resolution_minutes, 'horizons_hours': list(horizons_hours),
            'predicted_columns': 'perfect foresight (value at t + horizon)', **info}
    return pricing_frame(price, resolution_minutes, horizons_hours), carbon_frame(carbon), grid, meta
