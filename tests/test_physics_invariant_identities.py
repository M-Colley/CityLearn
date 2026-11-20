"""Physics invariant identities that do not rebuild the net consumption from its own components (#277).

The net-balance check recomputes the net from the arrays `Building.update_variables()` summed, so a wrong
device value passed it. Each identity below compares records written by different code paths. The tests
tamper with one record after a normal step and check that the checker names the identity.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytest.importorskip('gymnasium')

from citylearn.citylearn import CityLearnEnv
from citylearn.internal.physics_invariants import PhysicsInvariantError

ROOT = Path(__file__).resolve().parents[1]
MINUTE_DEMO = ROOT / 'tests' / 'data' / 'minute_ev_demo'
THERMAL_DATASET = ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1'
OUTAGE_ROWS = slice(4, 12)


def _env(schema: Path, **kwargs) -> CityLearnEnv:
    options = {'physics_invariant_checks': True, 'random_seed': 0, 'export_kpis_on_episode_end': False,
               'central_agent': True, 'interface': 'flat', 'offline': True}
    options.update(kwargs)
    return CityLearnEnv(str(schema), **options)


def _step_once(env: CityLearnEnv, completed: bool = True):
    """Step once; return the building, the checked time step and the checker.

    The checker runs on the step that just finished (`time_step - 1`); the current time step is not yet simulated.
    """

    env.reset(seed=0)
    env.step([[0.0]*len(env.action_names[0])])
    return env.buildings[0], env.time_step - int(completed), env._physics_invariant_service


def _random_actions(env: CityLearnEnv, rng: np.random.Generator, scale: float = 1.0) -> list:
    """Seeded actions inside the action space; a device action below its lower bound would book a negative demand."""

    space = env.action_space[0]
    return [(space.low + scale*rng.uniform(0.0, 1.0, space.shape)*(space.high - space.low)).astype('float32').tolist()]


def _copy(source: Path, target: Path, mutator=None) -> Path:
    shutil.copytree(source, target)
    path = target / 'schema.json'
    schema = json.loads(path.read_text(encoding='utf-8'))
    schema['root_directory'] = str(target)
    if mutator is not None:
        mutator(schema, target)
    path.write_text(json.dumps(schema), encoding='utf-8')
    return path


def _add_outage(schema: dict, root: Path, rows: slice = OUTAGE_ROWS):
    """Mark `rows` of every building's energy simulation as an outage and enable the simulation."""

    for building in schema['buildings'].values():
        path = root / building['energy_simulation']
        frame = pd.read_csv(path)
        signal = np.zeros(len(frame), dtype=int)
        signal[rows] = 1
        frame['power_outage'] = signal
        frame.to_csv(path, index=False)
        building['power_outage'] = {'simulate_power_outage': True}


@pytest.fixture
def minute_env():
    env = _env(MINUTE_DEMO / 'schema.json')
    try:
        yield env
    finally:
        env.close()


def test_issue_277_component_error_in_a_device_is_detected(minute_env):
    building, t, service = _step_once(minute_env, completed=False)
    service.assert_step_invariants(t)

    # The net balance recomputes the net from the same arrays, so it passes; the load is far above the dataset's.
    building.non_shiftable_load_device.update_electricity_consumption(100.0)
    building.update_variables()
    assert building.net_electricity_consumption[t] == pytest.approx(100.0)

    with pytest.raises(PhysicsInvariantError, match='non_shiftable_served_le_demand'):
        service.assert_step_invariants(t)


def test_served_non_shiftable_load_names_building_step_and_values(minute_env):
    building, t, service = _step_once(minute_env)
    building.non_shiftable_load_device.set_electricity_consumption(5.0, time_step=t)

    with pytest.raises(PhysicsInvariantError) as error:
        service._assert_served_within_demand(building, t)

    message = str(error.value)
    assert f"'{building.name}'" in message and f't={t}' in message and 'served=5.000000' in message


@pytest.fixture
def thermal_env(tmp_path):
    env = _env(_copy(THERMAL_DATASET, tmp_path / 'thermal'), simulation_start_time_step=0, simulation_end_time_step=23,
              episode_time_steps=24)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize('end_use', ['cooling', 'heating', 'dhw'])
def test_thermal_energy_identity_detects_extra_electricity(thermal_env, end_use):
    building, t, service = _step_once(thermal_env)
    service.assert_step_invariants(t)
    device = getattr(building, f'{end_use}_device')
    device.set_electricity_consumption(float(device.electricity_consumption[t]) + 3.0, time_step=t)

    with pytest.raises(PhysicsInvariantError, match=f'{end_use}_heat_le_output_plus_charge'):
        service._assert_thermal_energy_identity(building, t)


@pytest.mark.parametrize('end_use', ['cooling', 'dhw'])
def test_thermal_energy_identity_detects_missing_electricity(thermal_env, end_use):
    env = thermal_env
    env.reset(seed=0)
    service = env._physics_invariant_service
    building = env.buildings[0]
    # A step in which the end use runs; skip to one with demand.
    for _ in range(env.episode_time_steps - 1):
        env.step([[0.0]*len(env.action_names[0])])
        t = env.time_step - 1
        device = getattr(building, f'{end_use}_device')
        if float(device.electricity_consumption[t]) > 0.0:
            device.set_electricity_consumption(0.0, time_step=t)
            with pytest.raises(PhysicsInvariantError, match=f'{end_use}_heat_ge_output_plus_charge'):
                service._assert_thermal_energy_identity(building, t)
            return

    pytest.skip(f'no {end_use} demand in the first day')


@pytest.mark.parametrize('end_use', ['cooling', 'heating', 'dhw'])
def test_served_energy_may_not_exceed_the_demand(thermal_env, end_use):
    building, t, service = _step_once(thermal_env)
    storage = getattr(building, f'{end_use}_storage')
    # A storage discharge of 50 kWh that no demand takes inflates the heat served.
    storage._StorageDevice__energy_balance[t] = -50.0

    with pytest.raises(PhysicsInvariantError, match=f'{end_use}_served_le_demand'):
        service._assert_served_within_demand(building, t)


def test_battery_soc_below_the_depth_of_discharge_reserve_is_detected(thermal_env):
    building, t, service = _step_once(thermal_env)
    battery = building.electrical_storage
    battery.depth_of_discharge = 0.5
    battery._StorageDevice__soc[t - 1] = 0.9
    battery._StorageDevice__soc[t] = 0.5
    service._assert_battery_reserve(building, t)
    battery._StorageDevice__soc[t] = 0.1

    with pytest.raises(PhysicsInvariantError, match='battery_soc_ge_reserve'):
        service._assert_battery_reserve(building, t)


def test_battery_idle_at_the_reserve_may_drift_down_by_the_standby_loss_only(thermal_env):
    building, t, service = _step_once(thermal_env)
    battery = building.electrical_storage
    battery.depth_of_discharge = 0.5
    battery._StorageDevice__soc[t - 1] = 0.5
    battery._StorageDevice__soc[t] = 0.5*(1.0 - float(battery.loss_coefficient))
    service._assert_battery_reserve(building, t)


@pytest.fixture
def outage_env(tmp_path):
    env = _env(_copy(THERMAL_DATASET, tmp_path / 'outage', _add_outage), simulation_start_time_step=0,
               simulation_end_time_step=15, episode_time_steps=16, physics_version=2)
    try:
        yield env
    finally:
        env.close()


def test_outage_discharge_that_leaves_pv_curtailed_is_a_spill(outage_env):
    env = outage_env
    env.reset(seed=0)
    service = env._physics_invariant_service
    # Step into the outage window, then force a discharge that no local load needs.
    while env.time_step < OUTAGE_ROWS.start + 1:
        env.step([[0.0]*len(env.action_names[0])])

    building = env.buildings[0]
    t = env.time_step
    assert building.power_outage_signal[t] == 1
    building.electrical_storage.update_electricity_consumption(-50.0, enforce_polarity=False)
    building.update_variables()

    with pytest.raises(PhysicsInvariantError, match='outage_discharge_le_load_after_pv'):
        service._assert_outage_no_storage_spill(building, t)


def test_outage_rollout_with_random_actions_satisfies_every_identity(outage_env):
    env = outage_env
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    while not env.terminated:
        env.step(_random_actions(env, rng))

    residuals = env._physics_invariant_service.max_residuals
    assert 'outage_discharge_le_load_after_pv' in residuals
    assert max(residuals.values()) <= env._physics_invariant_service.ENERGY_EPS


# --- property-based: random actions and schemas across sub-hourly steps, with the market and outages ----------


@pytest.fixture(scope='module')
def market_outage_schema(tmp_path_factory):
    def mutate(schema, root):
        _add_outage(schema, root, slice(6, 30))
        schema['community_market'] = {'enabled': True, 'intra_community_sell_ratio': 0.8, 'grid_export_price': 0.05}

    return _copy(THERMAL_DATASET, tmp_path_factory.mktemp('property') / 'market_outage', mutate)


@settings(max_examples=40, deadline=None, derandomize=True, database=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seconds=st.sampled_from([3600, 1800, 900, 300]), start=st.integers(0, 12), seed=st.integers(0, 2**16),
       scale=st.sampled_from([0.1, 0.5, 1.0]), physics_version=st.sampled_from([1, 2]))
def test_random_rollouts_satisfy_the_invariants_with_market_and_outages(
        market_outage_schema, seconds, start, seed, scale, physics_version):
    kwargs = {'simulation_start_time_step': start, 'simulation_end_time_step': start + 23, 'episode_time_steps': 24,
              'seconds_per_time_step': seconds, 'time_step_ratio_mode': 'compress', 'physics_version': physics_version,
              'random_seed': seed}
    env = _env(market_outage_schema, **kwargs)
    rng = np.random.default_rng(seed)

    try:
        env.reset(seed=seed)

        while not env.terminated:
            env.step(_random_actions(env, rng, scale))

        assert env.community_market_enabled
    finally:
        env.close()
