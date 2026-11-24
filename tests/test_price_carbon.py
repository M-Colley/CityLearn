"""Price and carbon dataset tools. No network access: GB responses are recorded, ENTSO-E data is synthetic."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citylearn.price_carbon import common
from citylearn.price_carbon.gb import gb_carbon, parse_intensity

DATA = Path(__file__).parent / 'data' / 'price_carbon'
ROOT = Path(__file__).resolve().parents[1]


def test_not_imported_by_environment():
    code = "import sys, citylearn.citylearn; assert 'citylearn.price_carbon' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)


def test_to_grid_downsample_15min_to_hourly_and_mixed_resolution():
    idx = pd.date_range('2025-09-30 22:00', periods=8, freq='15min', tz='UTC')
    series = pd.Series([10, 20, 30, 40, 1, 2, 3, 4], index=idx, dtype=float)
    grid = common.build_grid('2025-09-30T22:00Z', '2025-10-01T00:00Z', 60)
    assert common.to_grid(series, grid, 60).tolist() == [25.0, 2.5]


def test_to_grid_upsample_hourly_to_15min():
    series = pd.Series([1.0, 2.0], index=pd.date_range('2025-01-01', periods=2, freq='h', tz='UTC'))
    grid = common.build_grid('2025-01-01T00:00Z', '2025-01-01T02:00Z', 15)
    assert common.to_grid(series, grid, 15).tolist() == [1.0] * 4 + [2.0] * 4


def test_to_grid_dst_day_stays_contiguous():
    # 2024-03-31 has 23 local hours in Europe/Berlin; the UTC grid has 23 steps for that local day.
    grid = common.build_grid('2024-03-31', '2024-04-01', 60, 'Europe/Berlin')
    assert len(grid) == 23
    series = pd.Series(np.arange(len(grid), dtype=float), index=grid.tz_convert('Europe/Berlin'))
    assert common.to_grid(series, grid, 60).tolist() == list(range(23))


def test_to_grid_short_gap_interpolated_long_gap_raises():
    idx = pd.date_range('2025-01-01', periods=10, freq='h', tz='UTC')
    series = pd.Series(np.arange(10, dtype=float), index=idx).drop(idx[[4, 5]])
    grid = common.build_grid('2025-01-01T00:00Z', '2025-01-01T10:00Z', 60)
    assert common.to_grid(series, grid, 60, max_gap_steps=2).tolist() == list(range(10))
    with pytest.raises(ValueError, match='longest gap is 2'):
        common.to_grid(series, grid, 60, max_gap_steps=1)


def test_predicted_columns_shift_and_pad():
    values = np.arange(6, dtype=float)
    p = common.add_predicted(values, 60, (1, 2, 3))
    assert p[0].tolist() == [1, 2, 3, 4, 5, 5]
    assert p[2].tolist() == [3, 4, 5, 5, 5, 5]
    assert common.add_predicted(values, 30, (1,))[0].tolist() == [2, 3, 4, 5, 5, 5]
    with pytest.raises(ValueError):
        common.add_predicted(values, 120, (1,))


def test_gb_fixture_recorded_response():
    payload = json.loads((DATA / 'gb_intensity.json').read_text())
    raw = parse_intensity(payload)
    assert len(raw) == 96 and raw.iloc[0] == 96.0
    frame, grid, meta = gb_carbon('2024-03-30T00:00Z', '2024-04-01T00:00Z', 60, raw=raw)
    assert list(frame.columns) == ['carbon_intensity'] and len(frame) == 48 == len(grid)
    # first hour averages the periods starting 00:00 (96) and 00:30 (95) g/kWh
    assert frame['carbon_intensity'].iloc[0] == pytest.approx(0.0955)
    assert meta['unit_out'] == 'kgCO2/kWh'


def test_gb_actual_null_falls_back_to_forecast():
    payload = {'data': [{'from': '2024-01-01T00:00Z', 'to': '2024-01-01T00:30Z', 'intensity': {'forecast': 100, 'actual': None}}]}
    assert parse_intensity(payload).iloc[0] == 100.0


def _synthetic_entsoe(periods=96):
    idx = pd.date_range('2025-01-01', periods=periods, freq='h', tz='Europe/Berlin')
    rng = np.random.default_rng(0)
    prices = pd.Series(rng.uniform(20, 120, periods), index=idx)
    gen = pd.DataFrame({'Fossil Gas': 100.0, 'Wind Onshore': 100.0, 'Nuclear': 200.0, 'Waste': 5.0}, index=idx)
    return prices, gen


def test_entsoe_datasets_units_and_carbon(monkeypatch):
    from citylearn.price_carbon.entsoe import entsoe_datasets
    prices, gen = _synthetic_entsoe()
    pricing, carbon, grid, meta = entsoe_datasets('DE_LU', '2025-01-01', '2025-01-04', 60, 'Europe/Berlin', prices=prices, generation=gen)
    assert len(pricing) == len(carbon) == 72
    assert pricing['electricity_pricing'].iloc[0] == pytest.approx(prices.iloc[0] / 1000)
    assert pricing['electricity_pricing_predicted_1'].iloc[0] == pytest.approx(prices.iloc[1] / 1000)
    assert carbon['carbon_intensity'].iloc[0] == pytest.approx((490 * 100 + 11 * 100 + 12 * 200) / 400 / 1000)
    assert meta['types_dropped'] == ['Waste'] and 'IPCC AR5' in meta['emission_factor_source']


def test_entsoe_multiindex_generation_is_netted():
    from citylearn.price_carbon.entsoe import carbon_from_mix
    idx = pd.date_range('2025-01-01', periods=2, freq='h', tz='UTC')
    cols = pd.MultiIndex.from_tuples([('Fossil Gas', 'Actual Aggregated'), ('Hydro Pumped Storage', 'Actual Aggregated'), ('Hydro Pumped Storage', 'Actual Consumption')])
    gen = pd.DataFrame([[100, 10, 40], [100, 50, 0]], index=idx, columns=cols, dtype=float)
    intensity, _ = carbon_from_mix(gen)
    assert intensity.iloc[0] == 490.0  # pumping (net negative) is clipped to zero
    assert intensity.iloc[1] == pytest.approx((490 * 100 + 24 * 50) / 150)


def test_entsoe_unmapped_mix_raises():
    from citylearn.price_carbon.entsoe import entsoe_datasets
    prices, gen = _synthetic_entsoe()
    gen = pd.DataFrame({'Waste': 100.0, 'Wind Onshore': 1.0}, index=gen.index)
    with pytest.raises(ValueError, match='no emission factor'):
        entsoe_datasets('DE_LU', '2025-01-01', '2025-01-04', 60, 'Europe/Berlin', prices=prices, generation=gen)


def test_entsoe_key_only_from_environment_and_masked(monkeypatch):
    from citylearn.price_carbon import entsoe
    monkeypatch.delenv('ENTSOE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='ENTSOE_API_KEY'):
        entsoe.get_client()
    assert entsoe.mask('GET ?securityToken=abc123', 'abc123') == 'GET ?securityToken=***'


def test_scenario_prices_and_emissions(tmp_path):
    from citylearn.price_carbon.scenario import scenario_datasets
    config = {
        'snapshots': {'start': '2035-01-01', 'end': '2035-01-02', 'freq': '1h'}, 'buses': ['a'],
        'load': {'bus': 'a', 'constant': 100},
        'generators': [
            {'name': 'wind', 'bus': 'a', 'p_nom': 60, 'marginal_cost': 0, 'emission_factor': 10},
            {'name': 'gas', 'bus': 'a', 'p_nom': 100, 'marginal_cost': 80, 'emission_factor': 500},
        ],
    }
    pricing, carbon, index, meta = scenario_datasets(config)
    assert len(pricing) == 24 and index[0] == pd.Timestamp('2035-01-01', tz='UTC')
    assert np.allclose(pricing['electricity_pricing'], 0.08)  # gas is marginal
    assert np.allclose(carbon['carbon_intensity'], (60 * 10 + 40 * 500) / 100 / 1000)
    assert meta['shed_share'] == 0


def test_scenario_two_nodes_with_congested_link():
    from citylearn.price_carbon.scenario import build_network, extract, solve
    config = {
        'snapshots': {'start': '2035-01-01', 'end': '2035-01-01 06:00', 'freq': '1h'}, 'buses': ['a', 'b'],
        'load': [{'bus': 'a', 'constant': 50}, {'bus': 'b', 'constant': 50}],
        'generators': [{'name': 'cheap', 'bus': 'a', 'p_nom': 200, 'marginal_cost': 10, 'emission_factor': 100},
                       {'name': 'dear', 'bus': 'b', 'p_nom': 200, 'marginal_cost': 90, 'emission_factor': 800}],
        'links': [{'bus0': 'a', 'bus1': 'b', 'p_nom': 20}],
    }
    n = build_network(config)
    solve(n)
    pa, _, _ = extract(n, 'a')
    pb, _, _ = extract(n, 'b')
    assert np.allclose(pa, 10) and np.allclose(pb, 90)


def test_cli_scenario_writes_citylearn_files(tmp_path):
    out = tmp_path / 'o'
    cmd = [sys.executable, '-m', 'citylearn.price_carbon', 'scenario', '--config', str(ROOT / 'examples/price_carbon/scenario.yaml'), '--out', str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    assert list(pd.read_csv(out / 'pricing.csv').columns) == list(common.PRICING_COLUMNS)
    assert list(pd.read_csv(out / 'carbon_intensity.csv').columns) == ['carbon_intensity']
    meta = json.loads((out / 'price_carbon_metadata.json').read_text())
    assert meta['time_steps'] == 168
