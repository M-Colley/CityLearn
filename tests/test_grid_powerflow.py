"""Optional low-voltage grid power flow (schema ``grid`` block): config, parity, observations, KPIs, no-op default."""

import copy

import numpy as np
import pandas as pd
import pytest

from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.internal.grid_powerflow import parse_grid_config

DATASET = 'citylearn_challenge_2022_phase_1'
STEPS = 30
CODE = '1-LV-rural1--0-sw'


def _schema(grid=None):
    schema = DataSet().get_schema(DATASET)

    if grid is not None:
        schema['grid'] = grid

    return schema


def _env(grid=None, **kwargs):
    kwargs.setdefault('central_agent', True)
    kwargs.setdefault('interface', 'flat')
    env = CityLearnEnv(
        _schema(grid), simulation_start_time_step=0, simulation_end_time_step=STEPS, random_seed=0, **kwargs,
    )
    env.reset()
    return env


def _run(env, seed=0):
    rng = np.random.default_rng(seed)
    low, high = env.action_space[0].low, env.action_space[0].high
    observations = []

    while not env.terminated:
        obs, *_ = env.step([rng.uniform(low, high).tolist()])
        observations.append(np.asarray(obs[0], dtype=float))

    return observations


@pytest.fixture(scope='module')
def network_file(tmp_path_factory):
    pytest.importorskip('pandapower')
    pytest.importorskip('simbench')
    pytest.importorskip('lightsim2grid')
    import pandapower as pp
    from citylearn.internal.grid_powerflow import load_network

    path = tmp_path_factory.mktemp('grid') / 'net.json'
    pp.to_json(load_network({'network_file': None, 'simbench_code': CODE}), str(path))
    return str(path)


def test_config_validation():
    with pytest.raises(ValueError, match='Unknown schema "grid" keys'):
        parse_grid_config({'simbench_code': CODE, 'bogus': 1})

    with pytest.raises(ValueError, match='simbench_code'):
        parse_grid_config({})

    with pytest.raises(ValueError, match='backend'):
        parse_grid_config({'simbench_code': CODE, 'backend': 'x'})

    with pytest.raises(ValueError, match='observations'):
        parse_grid_config({'simbench_code': CODE, 'observations': ['nope']})

    with pytest.raises(ValueError, match='voltage_limits_pu'):
        parse_grid_config({'simbench_code': CODE, 'voltage_limits_pu': [1.1, 0.9]})

    cfg = parse_grid_config({'simbench_code': CODE})
    assert cfg['voltage_limits_pu'] == (0.9, 1.1) and cfg['mode'] == 'ac' and cfg['backend'] == 'lightsim2grid'


def test_absent_block_is_a_no_op():
    env = _env()
    assert env.grid is None
    assert 'grid' not in env.get_metadata()

    with pytest.raises(ValueError, match='no "grid" block'):
        env.grid_kpis()


def test_disabled_block_is_a_no_op():
    assert _env({'enabled': False, 'simbench_code': CODE}).grid is None


def test_results_are_bit_identical_with_and_without_grid(network_file):
    plain, with_grid = _env(), _env({'network_file': network_file})
    obs_plain, obs_grid = _run(plain), _run(with_grid)

    for a, b in zip(obs_plain, obs_grid):
        np.testing.assert_array_equal(a, b)

    assert plain.rewards == with_grid.rewards
    np.testing.assert_array_equal(plain.net_electricity_consumption, with_grid.net_electricity_consumption)
    kpi_plain = plain.evaluate_v2(include_business_as_usual=False)
    kpi_grid = with_grid.evaluate_v2(include_business_as_usual=False)
    pd.testing.assert_frame_equal(kpi_plain, kpi_grid)


def test_lightsim2grid_matches_pandapower(network_file):
    runs = {}

    for backend in ('pandapower', 'lightsim2grid'):
        env = _env({'network_file': network_file, 'backend': backend, 'load_scale': 3.0})
        _run(env)
        runs[backend] = env.grid.series

    for key in ('vm_min', 'vm_max', 'line_max', 'trafo_max', 'losses_kw'):
        np.testing.assert_allclose(runs['lightsim2grid'][key][:-1], runs['pandapower'][key][:-1], rtol=1e-4, atol=1e-4)


def test_linear_mode_error_and_reanchoring(network_file):
    base = {'network_file': network_file, 'load_scale': 3.0}
    ac, lin, anchored, every_step = (
        _env({**base, 'mode': 'ac'}), _env({**base, 'mode': 'linear'}),
        _env({**base, 'mode': 'linear', 'ac_every_n_steps': 5}), _env({**base, 'mode': 'linear', 'ac_every_n_steps': 1}),
    )

    for env in (ac, lin, anchored, every_step):
        _run(env)

    ref = ac.grid.series['vm_min'][:-1]
    assert np.nanmax(np.abs(lin.grid.series['vm_min'][:-1] - ref)) < 0.02
    assert np.nanmax(np.abs(anchored.grid.series['vm_min'][:-1] - ref)) <= np.nanmax(np.abs(lin.grid.series['vm_min'][:-1] - ref))
    np.testing.assert_allclose(every_step.grid.series['vm_min'][:-1], ref, atol=1e-9)
    assert every_step.grid._n_ac == int(np.sum(~np.isnan(every_step.grid.series['vm_min'])))


def test_observations_are_appended(network_file):
    names = ['voltage_min_pu', 'line_loading_max_percent', 'building_voltage_pu']
    plain = _env()
    env = _env({'network_file': network_file, 'observations': names})
    extra = 2 + len(env.buildings)
    assert len(env.observation_names[0]) == len(plain.observation_names[0]) + extra
    assert env.observation_names[0][-extra] == 'grid_voltage_min_pu'
    assert env.observation_space[0].shape == (len(env.observation_names[0]),)
    obs, *_ = env.step([np.zeros(env.action_space[0].shape).tolist()])
    assert len(obs[0]) == env.observation_space[0].shape[0]
    assert 0.5 < obs[0][-extra] < 1.5
    assert env.observation_space[0].contains(np.asarray(obs[0], dtype=np.float32))


def test_observations_require_flat_central_agent(network_file):
    with pytest.raises(ValueError, match='central_agent'):
        _env({'network_file': network_file, 'observations': ['voltage_min_pu']}, central_agent=False)


def test_kpis_and_violations(network_file):
    env = _env({'network_file': network_file, 'load_scale': 40.0})
    _run(env)
    kpis = env.grid_kpis().set_index('cost_function')['value']
    assert kpis['grid_voltage_minimum_pu'] < 0.9
    assert kpis['grid_voltage_violation_time_total_hours'] > 0
    assert kpis['grid_overload_time_total_hours'] > 0
    assert kpis['grid_overload_energy_total_kvah'] > 0
    assert kpis['grid_losses_total_kwh'] > 0
    quiet = _env({'network_file': network_file, 'load_scale': 0.01})
    _run(quiet)
    q = quiet.grid_kpis().set_index('cost_function')['value']
    assert q['grid_voltage_violation_time_total_hours'] == 0
    assert q['grid_overload_energy_total_kvah'] == 0


def test_building_load_mapping_and_validation(network_file):
    env = _env({'network_file': network_file})
    assert len(env.grid.building_load) == len(env.buildings)

    with pytest.raises(ValueError, match='unknown buildings'):
        _env({'network_file': network_file, 'building_loads': {'nope': 0}})


def test_env_deepcopy_continues_identically(network_file):
    env = _env({'network_file': network_file, 'load_scale': 3.0})
    rng = np.random.default_rng(1)
    low, high = env.action_space[0].low, env.action_space[0].high

    for _ in range(5):
        env.step([rng.uniform(low, high).tolist()])

    clone = copy.deepcopy(env)
    actions = [rng.uniform(low, high).tolist() for _ in range(5)]

    for a in actions:
        env.step([a])
        clone.step([a])

    np.testing.assert_allclose(env.grid.series['vm_min'][:10], clone.grid.series['vm_min'][:10], atol=1e-9)


def test_simbench_object_dtype_columns_are_coerced():
    pytest.importorskip('pandapower')
    pytest.importorskip('simbench')
    from citylearn.internal.grid_powerflow import load_network

    net = load_network({'network_file': None, 'simbench_code': CODE})
    assert net.sgen.min_p_mw.dtype == float and net.load.max_p_mw.dtype == float
