from __future__ import annotations

import numpy as np
import pytest

from citylearn.base import EpisodeTracker
from citylearn.energy_model import Battery

pytest.importorskip('blast')

LFP = 'Lfp_Gr_250AhPrismatic'
NMC = 'Nmc111_Gr_Kokam75Ah_Battery'


def _battery(degradation=None, steps=24 * 60, capacity=100.0, **kwargs) -> Battery:
    tracker = EpisodeTracker(0, steps - 1)
    tracker.next_episode(steps, rolling_episode_split=False, random_episode_split=False, random_seed=0)
    battery = Battery(
        capacity=capacity, nominal_power=50.0, initial_soc=0.5, efficiency=0.9, loss_coefficient=0.0,
        capacity_loss_coefficient=1e-4, power_efficiency_curve=[[0.0, 0.9], [1.0, 0.9]],
        capacity_power_curve=[[0.0, 1.0], [1.0, 1.0]], depth_of_discharge=1.0, seconds_per_time_step=3600,
        time_step_ratio=1.0, episode_tracker=tracker, random_seed=0, degradation=degradation, **kwargs
    )
    battery.reset()
    return battery


def _cycle(battery: Battery, days: int):
    for step in range(days * 24):
        battery.charge(30.0 if (step % 24) < 6 else -10.0 if (step % 24) < 18 else 0.0)
        if step < days * 24 - 1:
            battery.next_time_step()


def test_default_battery_has_no_degradation_model():
    battery = _battery()
    assert battery._degradation is None
    _cycle(battery, 2)
    assert battery.degraded_capacity < battery.capacity


@pytest.mark.parametrize('cell', [LFP, NMC])
def test_blast_lite_fades_capacity_monotonically(cell):
    battery = _battery({'model': 'blast_lite', 'cell': cell, 'update_interval_steps': 24, 'temperature': 25.0})
    _cycle(battery, 30)
    history = np.asarray(battery.capacity_history)
    assert history[-1] < battery.capacity
    assert history[-1] > 0.9*battery.capacity
    assert np.all(np.diff(history) <= 1e-12)


def test_capacity_is_constant_between_updates():
    battery = _battery({'model': 'blast_lite', 'cell': NMC, 'update_interval_steps': 24, 'temperature': 25.0})
    _cycle(battery, 3)
    history = np.asarray(battery.capacity_history)
    assert len(set(history[1:24].tolist())) == 1
    assert history[24] < history[23]


def test_higher_temperature_fades_faster():
    cold = _battery({'model': 'blast_lite', 'cell': NMC, 'temperature': 10.0})
    hot = _battery({'model': 'blast_lite', 'cell': NMC, 'temperature': 45.0})
    _cycle(cold, 30)
    _cycle(hot, 30)
    assert hot.degraded_capacity < cold.degraded_capacity


def test_reset_restarts_the_model():
    battery = _battery({'model': 'blast_lite', 'cell': NMC})
    _cycle(battery, 5)
    battery.reset()
    assert battery.degraded_capacity == battery.capacity
    _cycle(battery, 5)
    first = battery.degraded_capacity
    battery.reset()
    _cycle(battery, 5)
    assert battery.degraded_capacity == first


@pytest.mark.parametrize('config, message', [
    ({'model': 'other'}, 'blast_lite'),
    ({'model': 'blast_lite', 'cell': 'NoSuchCell'}, 'Unknown BLAST-Lite cell'),
    ({'model': 'blast_lite', 'update_interval_steps': 0}, 'update_interval_steps'),
    ({'model': 'blast_lite', 'interval': 5}, 'Unknown degradation keys'),
])
def test_invalid_configuration_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        _battery(config)


def _schema(degradation, ev_degradation=None):
    from citylearn.data import DataSet

    schema = DataSet().get_schema('citylearn_challenge_2022_phase_all_plus_evs')
    schema['simulation_end_time_step'] = 71
    for building in schema['buildings'].values():
        storage = building.get('electrical_storage')
        if storage and storage.get('type'):
            storage.setdefault('attributes', {})['degradation'] = dict(degradation)
    for ev in schema['electric_vehicles_def'].values():
        ev['battery']['attributes']['degradation'] = dict(ev_degradation or {**degradation, 'temperature': 25.0})
    return schema


def test_schema_enables_degradation_for_building_and_ev_batteries():
    from citylearn.citylearn import CityLearnEnv

    env = CityLearnEnv(_schema({'model': 'blast_lite', 'cell': NMC, 'update_interval_steps': 12, 'temperature': 'outdoor_dry_bulb'}), central_agent=True)
    env.reset()
    rng = np.random.default_rng(0)
    while not env.terminated:
        env.step([rng.uniform(-1, 1, size=len(env.action_space[0].low)).astype('float32').tolist()])
    batteries = [b.electrical_storage for b in env.buildings if b.electrical_storage.capacity] + [e.battery for e in env.electric_vehicles]
    assert batteries and all(b._degradation is not None for b in batteries)
    assert any(b.degraded_capacity < b.capacity for b in batteries)


def test_ev_battery_rejects_weather_temperature():
    from citylearn.citylearn import CityLearnEnv

    config = {'model': 'blast_lite', 'temperature': 'outdoor_dry_bulb'}
    with pytest.raises(ValueError, match='constant'):
        CityLearnEnv(_schema(config, config), central_agent=True)
