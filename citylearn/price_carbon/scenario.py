"""Future-scenario prices and carbon intensity from a small PyPSA dispatch model (needs the `price_carbon` extra).

A YAML file describes snapshots, buses, load, generators, storage units and links. HiGHS solves the
least-cost dispatch. The marginal price of the bus balance is the electricity price and the average
emission intensity is the dispatched emissions divided by the dispatched energy.

Example (`examples/price_carbon/scenario.yaml`)::

    snapshots: {start: "2035-01-01", end: "2035-01-08", freq: "1h", tz: "UTC"}
    buses: [DE]
    price_bus: DE
    load: {bus: DE, daily_shape: [0.7, 0.65, 0.6, 0.6, 0.65, 0.75, 0.9, 1.0, 1.0, 0.95, 0.95, 0.95,
                                  0.95, 0.95, 0.95, 0.95, 1.0, 1.1, 1.15, 1.1, 1.0, 0.9, 0.8, 0.75], scale_mw: 1000}
    generators:
      - {name: wind, bus: DE, p_nom: 900, marginal_cost: 0, emission_factor: 11, profile: {seed: 1, mean: 0.35, sd: 0.2}}
      - {name: gas, bus: DE, p_nom: 1200, marginal_cost: 85, emission_factor: 490}
    storage_units:
      - {name: battery, bus: DE, p_nom: 200, max_hours: 4, efficiency_store: 0.95, efficiency_dispatch: 0.95}
    load_shedding: {marginal_cost: 3000}

Units: MW, currency/MWh, gCO2/kWh (`emission_factor`). Output: currency/kWh and kgCO2/kWh.
A `profile` is a `file` (CSV with one value per snapshot, column `column`, relative to the YAML),
a list `values` repeated over the snapshots, or a synthetic `{seed, mean, sd}` bounded in [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from citylearn.price_carbon.common import carbon_frame, pricing_frame

LOGGER = logging.getLogger(__name__)
SHEDDING = 'load_shedding'


def load_config(path) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config['_base'] = str(path.parent)
    return config


def _snapshots(spec: dict) -> pd.DatetimeIndex:
    tz = spec.get('tz', 'UTC')
    idx = pd.date_range(spec['start'], spec['end'], freq=spec.get('freq', '1h'), inclusive='left', tz=tz)
    return idx.tz_convert('UTC').tz_localize(None)


def _series(spec, snapshots: pd.DatetimeIndex, base: str, scale: float = 1.0) -> np.ndarray:
    n = len(snapshots)
    if isinstance(spec, (int, float)):
        return np.full(n, float(spec))
    if 'file' in spec:
        frame = pd.read_csv(Path(base) / spec['file'])
        values = frame[spec['column']].to_numpy(dtype=float)
        if len(values) != n:
            raise ValueError(f"{spec['file']} has {len(values)} rows, expected {n} snapshots.")
        return values
    if 'values' in spec:
        return np.resize(np.asarray(spec['values'], dtype=float), n)
    if 'daily_shape' in spec:
        shape = np.asarray(spec['daily_shape'], dtype=float)
        hours = snapshots.hour.to_numpy()
        return shape[hours % len(shape)] * float(spec.get('scale_mw', 1.0))
    if 'seed' in spec:
        rng = np.random.default_rng(spec['seed'])
        return np.clip(rng.normal(spec.get('mean', 0.3), spec.get('sd', 0.2), n), 0.0, 1.0)
    if 'constant' in spec:
        return np.full(n, float(spec['constant']))
    raise ValueError(f'Unknown series specification: {spec}')


def build_network(config: dict):
    import pypsa

    base = config.get('_base', '.')
    snapshots = _snapshots(config['snapshots'])
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    buses = config.get('buses') or ['node']
    for bus in buses:
        n.add('Bus', bus)

    loads = config['load'] if isinstance(config['load'], list) else [config['load']]
    for i, load in enumerate(loads):
        n.add('Load', f'load_{i}', bus=load.get('bus', buses[0]), p_set=_series(load, snapshots, base))

    for gen in config['generators']:
        kwargs = {'bus': gen.get('bus', buses[0]), 'p_nom': gen['p_nom'], 'marginal_cost': gen.get('marginal_cost', 0.0)}
        if 'profile' in gen:
            kwargs['p_max_pu'] = _series(gen['profile'], snapshots, base)
        n.add('Generator', gen['name'], carrier=gen.get('carrier', gen['name']), **kwargs)
    n.generators['emission_factor'] = {g['name']: float(g.get('emission_factor', 0.0)) for g in config['generators']}

    for st in config.get('storage_units', []):
        n.add('StorageUnit', st['name'], bus=st.get('bus', buses[0]), p_nom=st['p_nom'], max_hours=st.get('max_hours', 4),
              efficiency_store=st.get('efficiency_store', 0.95), efficiency_dispatch=st.get('efficiency_dispatch', 0.95),
              cyclic_state_of_charge=st.get('cyclic', True))
    for i, link in enumerate(config.get('links', [])):
        n.add('Link', link.get('name', f'link_{i}'), bus0=link['bus0'], bus1=link['bus1'], p_nom=link['p_nom'],
              efficiency=link.get('efficiency', 1.0), p_min_pu=-1.0 if link.get('bidirectional', True) else 0.0)

    shed = config.get('load_shedding', {'marginal_cost': 3000.0})
    for bus in buses:
        n.add('Generator', f'{SHEDDING}_{bus}', bus=bus, p_nom=1e6, marginal_cost=shed['marginal_cost'], carrier=SHEDDING)
    n.generators.loc[n.generators.carrier == SHEDDING, 'emission_factor'] = 0.0
    return n


def solve(n, solver_name: str = 'highs') -> None:
    status, condition = n.optimize(solver_name=solver_name, include_objective_constant=False, log_to_console=False)
    if status != 'ok':
        raise RuntimeError(f'Dispatch did not solve: {status}, {condition}')


def extract(n, price_bus: str, intensity_basis: str = 'system') -> tuple:
    """Return `(price currency/MWh, intensity gCO2/kWh, shed_share)` per snapshot.

    `system` divides all emissions by all generation; `bus` uses only generators at `price_bus`
    (imports over links are not traced).
    """
    price = n.buses_t.marginal_price[price_bus]
    p = n.generators_t.p
    gens = n.generators
    real = gens.index[gens.carrier != SHEDDING]
    if intensity_basis == 'bus':
        real = [g for g in real if gens.at[g, 'bus'] == price_bus]
    elif intensity_basis != 'system':
        raise ValueError("intensity_basis must be 'system' or 'bus'.")
    energy = p[real].sum(axis=1)
    emissions = p[real].mul(gens.loc[real, 'emission_factor'], axis=1).sum(axis=1)
    intensity = emissions / energy.where(energy > 1e-9)
    shed = p[[g for g in gens.index if gens.at[g, 'carrier'] == SHEDDING]].sum().sum()
    demand = n.loads_t.p_set.sum().sum()
    return price, intensity.ffill().bfill().fillna(0.0), float(shed / demand) if demand else 0.0


def scenario_datasets(config: dict, resolution_minutes: Optional[int] = None, horizons_hours=(1, 2, 3), solver_name: str = 'highs'):
    """Solve the scenario and return `(pricing, carbon, index, metadata)` at the snapshot resolution."""
    n = build_network(config)
    step = int((n.snapshots[1] - n.snapshots[0]).total_seconds() // 60) if len(n.snapshots) > 1 else 60
    if resolution_minutes not in (None, step):
        raise ValueError(f'Snapshot step is {step} min; set snapshots.freq instead of resolution_minutes={resolution_minutes}.')
    solve(n, solver_name)
    price_bus = config.get('price_bus', n.buses.index[0])
    price, intensity, shed_share = extract(n, price_bus, config.get('intensity_basis', 'system'))
    if shed_share > 0:
        LOGGER.warning('Load shedding covers %.2f%% of the demand; prices reach the shedding cost.', 100 * shed_share)
    index = pd.DatetimeIndex(n.snapshots).tz_localize('UTC')
    pricing = pricing_frame(pd.Series(price.to_numpy() / 1000.0), step, horizons_hours)
    carbon = carbon_frame(pd.Series(intensity.to_numpy() / 1000.0))
    meta = {'source': 'PyPSA dispatch (HiGHS)', 'name': config.get('name', 'scenario'), 'price_bus': price_bus, 'resolution_minutes': step,
            'price_unit': 'currency/kWh (marginal price of the bus balance / 1000)', 'carbon_unit': 'kgCO2/kWh',
            'carbon_basis': 'average dispatched emissions / dispatched energy, ' + config.get('intensity_basis', 'system'),
            'shed_share': shed_share, 'horizons_hours': list(horizons_hours), 'predicted_columns': 'perfect foresight (value at t + horizon)',
            'config': {k: v for k, v in config.items() if k != '_base'}}
    return pricing, carbon, index, meta
