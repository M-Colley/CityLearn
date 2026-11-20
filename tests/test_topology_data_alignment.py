"""Regressions for dynamic topology and data alignment (#246, #247, #234 and terminal-step events)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.base import EpisodeTracker
from citylearn.citylearn import CityLearnEnv
from citylearn.data import EnergySimulation

ROOT = Path(__file__).resolve().parents[1]
MINUTE_EV_DEMO = ROOT / "tests/data/minute_ev_demo"
TOPOLOGY_DEMO = ROOT / "data/datasets/citylearn_three_phase_dynamic_topology_demo"
CHALLENGE_2023 = ROOT / "data/datasets/citylearn_challenge_2023_phase_1"
CHALLENGE_2022_EVS = ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs"


def _schema(root: Path, **updates):
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    schema["root_directory"] = str(root)
    schema.update(updates)
    return schema


def _zero_flat_actions(env):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _zero_entity_actions(env):
    return {
        "tables": {
            name: np.zeros(space.shape, dtype="float32")
            for name, space in env.action_space["tables"].items()
            if name in {"building", "charger", "deferrable_appliance"}
        }
    }


def _run_episode(env, actions):
    executed = []
    start = env.episode_tracker.episode_start_time_step

    while not env.terminated:
        executed.append(start + env.time_step)
        env.step(actions(env))

    return executed


# --- #247: consecutive episode splits -------------------------------------------------------------


def test_consecutive_splits_simulate_the_boundary_interval():
    env = CityLearnEnv(str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, episode_time_steps=4, random_seed=0)
    splits, executed = [], []

    for _ in range(2):
        env.reset()
        splits.append((env.episode_tracker.episode_start_time_step, env.episode_tracker.episode_end_time_step))
        executed += _run_episode(env, _zero_flat_actions)

    # 3.6.0 split the 8 rows into (0, 3) and (4, 7) and never simulated intervals 3 and 7.
    assert env.episode_split_boundary == "shared"
    assert splits == [(0, 3), (3, 6)]
    assert executed == [0, 1, 2, 3, 4, 5]


def test_disjoint_split_boundary_keeps_the_3_6_0_splits():
    env = CityLearnEnv(
        str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, episode_time_steps=4, random_seed=0,
        episode_split_boundary="disjoint",
    )
    splits = []

    for _ in range(2):
        env.reset()
        splits.append((env.episode_tracker.episode_start_time_step, env.episode_tracker.episode_end_time_step))

    assert splits == [(0, 3), (4, 7)]


def test_episode_split_boundary_schema_key_and_validation():
    env = CityLearnEnv(
        _schema(MINUTE_EV_DEMO, episode_split_boundary="disjoint"), central_agent=True, episode_time_steps=4,
    )
    assert env.episode_split_boundary == "disjoint"

    with pytest.raises(ValueError, match="episode_split_boundary"):
        CityLearnEnv(str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, episode_split_boundary="overlap")


def test_one_row_consecutive_episodes_are_rejected():
    tracker = EpisodeTracker(0, 9)

    with pytest.raises(ValueError, match="must be >= 2"):
        tracker.next_episode(1, False, False, 0)

    disjoint = EpisodeTracker(0, 9)
    disjoint.next_episode(1, False, False, 0, episode_split_boundary="disjoint")
    assert (disjoint.episode_start_time_step, disjoint.episode_end_time_step) == (0, 0)

    # A one-row simulation window has a single, whole-window episode.
    single = EpisodeTracker(5, 5)
    single.next_episode(1, False, False, 0)
    assert (single.episode_start_time_step, single.episode_end_time_step) == (5, 5)


def test_rolling_and_whole_window_splits_are_unchanged():
    tracker = EpisodeTracker(0, 9)
    tracker.next_episode(4, True, False, 0)
    tracker.next_episode(4, True, False, 0)
    assert (tracker.episode_start_time_step, tracker.episode_end_time_step) == (1, 4)

    whole = EpisodeTracker(0, 9)
    whole.next_episode(10, False, False, 0)
    assert (whole.episode_start_time_step, whole.episode_end_time_step) == (0, 9)


# --- #246: dynamic topology ------------------------------------------------------------------------


@pytest.mark.parametrize("window", [(0, 1100), (950, 1045), (500, 1499)])
def test_short_member_file_is_aligned_with_a_late_simulation_start(window):
    start, end = window
    env = CityLearnEnv(
        _schema(TOPOLOGY_DEMO), simulation_start_time_step=start, simulation_end_time_step=end, random_seed=0,
    )
    building = env._topology_service.member_pool["Building_18"]
    first_row = pd.read_csv(TOPOLOGY_DEMO / "Building_18.csv", nrows=1)["non_shiftable_load"].iloc[0]

    # Building_18 joins at global step 1000, and its file starts there (row 0 = 2.123).
    assert building.energy_simulation.non_shiftable_load[1000 - start] == pytest.approx(first_row)


def test_business_as_usual_sidecar_aligns_short_member_files():
    env = CityLearnEnv(
        _schema(TOPOLOGY_DEMO), simulation_start_time_step=900, simulation_end_time_step=1100,
        episode_time_steps=[(950, 1045)], random_seed=0,
    )
    env.reset()
    sidecar = env._business_as_usual_baseline_service._new_sidecar_env()
    building = sidecar._topology_service.member_pool["Building_18"]
    offset = 1000 - sidecar.episode_tracker.simulation_start_time_step

    assert building.energy_simulation.non_shiftable_load[offset] == pytest.approx(2.123)


def _copy_dataset(tmp_path: Path, root: Path) -> Path:
    target = tmp_path / root.name
    shutil.copytree(root, target)
    return target


def test_short_member_file_that_ends_early_warns(tmp_path):
    root = _copy_dataset(tmp_path, TOPOLOGY_DEMO)
    frame = pd.read_csv(root / "Building_18.csv")
    frame.iloc[:100].to_csv(root / "Building_18.csv", index=False)

    with pytest.warns(UserWarning, match="ends before the member's active steps"):
        env = CityLearnEnv(_schema(root), simulation_start_time_step=950, simulation_end_time_step=1200, random_seed=0)

    building = env._topology_service.member_pool["Building_18"]
    np.testing.assert_allclose(
        building.energy_simulation.non_shiftable_load[50:150], frame["non_shiftable_load"].iloc[:100], rtol=1.0e-6,
    )
    # The 3.6.0 carry-forward of the last row is kept for the steps the file does not cover.
    assert building.energy_simulation.non_shiftable_load[200] == pytest.approx(frame["non_shiftable_load"].iloc[99])


def test_rejoining_member_with_an_unrelated_short_file_raises():
    events = [
        {"id": "join", "time_step": 1000, "operation": "add_member", "target_member_id": "Building_18"},
        {"id": "leave", "time_step": 1010, "operation": "remove_member", "target_member_id": "Building_18"},
        {"id": "rejoin", "time_step": 1020, "operation": "add_member", "target_member_id": "Building_18"},
    ]

    with pytest.raises(ValueError, match=r"has 2200 rows, but the member is active in 2 windows"):
        CityLearnEnv(
            _schema(TOPOLOGY_DEMO, topology_events=events),
            simulation_start_time_step=990, simulation_end_time_step=1030, random_seed=0,
        )


def test_joining_member_starts_thermal_storage_at_initial_soc():
    schema = _schema(CHALLENGE_2023, interface="entity", topology_mode="dynamic", simulation_end_time_step=24)
    schema["buildings"]["Building_3"]["include"] = False
    schema["buildings"]["Building_3"]["dhw_storage"]["attributes"]["initial_soc"] = 0.8
    schema["topology_events"] = [
        {"id": "join", "time_step": 20, "operation": "add_member", "target_member_id": "Building_3"},
    ]
    env = CityLearnEnv(schema, random_seed=0)
    env.reset()
    building = env._topology_service.member_pool["Building_3"]

    while env.time_step < 20:
        env.step(_zero_entity_actions(env))

    tank = building.dhw_storage
    assert tank.capacity > 0.0 and tank.initial_soc == pytest.approx(0.8)
    assert tank.energy_init == pytest.approx(tank.initial_soc*tank.capacity, rel=1.0e-6)
    assert building.electrical_storage.soc[20] == pytest.approx(building.electrical_storage.initial_soc)


def test_district_series_keep_departed_members_and_skip_future_members():
    schema = _schema(TOPOLOGY_DEMO, simulation_end_time_step=120)
    schema["topology_events"] = [
        {"id": "leave", "time_step": 50, "operation": "remove_member", "target_member_id": "Building_1"},
        {"id": "join", "time_step": 60, "operation": "add_member", "target_member_id": "Building_18"},
    ]
    env = CityLearnEnv(schema, random_seed=0)
    env.reset()

    while env.time_step < 49:
        env.step(_zero_entity_actions(env))

    before = np.array(env.non_shiftable_load[:49], dtype=float)
    _run_episode(env, _zero_entity_actions)
    after = np.array(env.non_shiftable_load, dtype=float)
    joined = np.array(env._topology_service.member_pool["Building_18"].non_shiftable_load, dtype=float)

    np.testing.assert_allclose(after[:49], before)
    # The joining member counts from its first step only.
    members = [b for b in env._kpi_service._get_kpi_buildings() if b.name != "Building_18"]
    others = sum(np.array(b.non_shiftable_load, dtype=float)[:60] for b in members if b.name != "Building_1")
    leaver = np.array(env._topology_service.member_pool["Building_1"].non_shiftable_load, dtype=float)
    np.testing.assert_allclose(after[50:60], others[50:60], rtol=1.0e-6)
    np.testing.assert_allclose(after[:50], others[:50] + leaver[:50], rtol=1.0e-6)
    assert joined[:60].sum() > 0.0


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([], id="no-event"),
        pytest.param([{
            "id": "pv", "time_step": 10, "operation": "remove_asset", "target_member_id": "Building_11",
            "target_asset_type": "pv", "target_asset_id": "pv",
        }], id="remove-pv"),
        pytest.param([{
            "id": "leave", "time_step": 10, "operation": "remove_member", "target_member_id": "Building_2",
        }], id="remove-member"),
    ],
)
def test_terminal_step_event_does_not_add_an_unexecuted_interval(events):
    env = CityLearnEnv(_schema(TOPOLOGY_DEMO, simulation_end_time_step=10, topology_events=events), random_seed=0)
    env.reset()
    executed = _run_episode(env, _zero_entity_actions)

    assert len(executed) == env.time_steps - 1 == 10
    for name in ("Building_1", "Building_11"):
        building = env._topology_service.member_pool[name]
        assert building.time_step == 9
        assert len(building.net_electricity_consumption) == 10
    assert len(env.net_electricity_consumption) == 10


# --- #234: control step and dataset step -----------------------------------------------------------


def test_control_step_different_from_dataset_step_is_rejected():
    with pytest.raises(ValueError, match="does not resample"):
        CityLearnEnv(
            str(CHALLENGE_2022_EVS / "schema.json"), central_agent=True, simulation_end_time_step=23,
            seconds_per_time_step=900,
        )


def test_compress_mode_keeps_the_3_6_0_behaviour():
    env = CityLearnEnv(
        str(CHALLENGE_2022_EVS / "schema.json"), central_agent=True, simulation_end_time_step=23,
        seconds_per_time_step=900, time_step_ratio_mode="compress",
    )
    assert env.time_step_ratio == pytest.approx(0.25)

    with pytest.raises(ValueError, match="time_step_ratio_mode"):
        CityLearnEnv(str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, time_step_ratio_mode="resample")


def test_time_step_ratio_argument_must_match_the_data():
    env = CityLearnEnv(str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, time_step_ratio=1)
    assert env.time_step_ratio == pytest.approx(1.0)

    with pytest.raises(ValueError, match="does not match"):
        CityLearnEnv(str(MINUTE_EV_DEMO / "schema.json"), central_agent=True, time_step_ratio=4)


def test_irregular_first_row_no_longer_rescales_energy(tmp_path):
    root = _copy_dataset(tmp_path, MINUTE_EV_DEMO)
    frame = pd.read_csv(root / "Building_1.csv")
    frame.loc[1, "minutes"] = 30
    frame.to_csv(root / "Building_1.csv", index=False)

    # 3.6.0 inferred 1800 s from rows 0 and 1 and halved every energy; an 8-row file with a typo
    # mixes cadences and is now refused.
    with pytest.raises(ValueError, match="Cannot infer the dataset time step"):
        CityLearnEnv(str(root / "schema.json"), central_agent=True)


def test_dataset_step_is_inferred_from_every_row():
    hour = np.repeat(np.arange(24), 4)
    minutes = np.tile([0, 15, 30, 45], 24)

    assert EnergySimulation.infer_time_step_seconds(hour, minutes) == 900.0

    typo = minutes.copy()
    typo[1] = 30
    assert EnergySimulation.infer_time_step_seconds(hour, typo) == 900.0

    # A daylight-saving jump in local-time data is an isolated irregular interval.
    dst = hour.copy()
    dst[40:] = (dst[40:] + 1) % 24
    assert EnergySimulation.infer_time_step_seconds(dst, minutes) == 900.0

    # Padded rows of a member that joins mid-window repeat their stamp and are left out.
    padded = np.concatenate([np.repeat(hour[:1], 30), hour])
    padded_minutes = np.concatenate([np.repeat(minutes[:1], 30), minutes])
    assert EnergySimulation.infer_time_step_seconds(padded, padded_minutes) == 900.0

    # 15-second rows with minute stamps cannot be timed from the stamps.
    assert EnergySimulation.infer_time_step_seconds(
        np.zeros(8, dtype=int), np.repeat([0, 1], 4), seconds_per_time_step=15,
    ) == 15.0

    mixed_hour = np.concatenate([np.repeat(np.arange(4), 4), np.arange(4, 16)])
    mixed_minutes = np.concatenate([np.tile([0, 15, 30, 45], 4), np.zeros(12, dtype=int)])
    with pytest.raises(ValueError, match="mix cadences"):
        EnergySimulation.infer_time_step_seconds(mixed_hour, mixed_minutes)


# --- #286: members that leave and rejoin with member-window files ------------------------------------

REJOIN_EVENTS = [
    {"id": "join", "time_step": 1000, "operation": "add_member", "target_member_id": "Building_18"},
    {"id": "leave", "time_step": 1010, "operation": "remove_member", "target_member_id": "Building_18"},
    {"id": "rejoin", "time_step": 1020, "operation": "add_member", "target_member_id": "Building_18"},
    {"id": "leave_again", "time_step": 1040, "operation": "remove_member", "target_member_id": "Building_18"},
]
REJOIN_SPANS = [(1000, 1010), (1020, 1040)]


def _concatenated_rows(frame: pd.DataFrame, spans) -> pd.DataFrame:
    """Rows of the member-window file `frame` (row 0 = global step 1000) for each span, concatenated.

    Each span reads from a different offset of the original file, so every active step has a distinct row.
    """

    return pd.concat(
        [frame.iloc[start - 1000 + 100 * index: end - 1000 + 100 * index] for index, (start, end) in enumerate(spans)],
        ignore_index=True,
    )


def _rejoin_dataset(tmp_path: Path, spans=REJOIN_SPANS, drop_rows: int = 0):
    """Copy of the dynamic demo whose Building_18 files hold the concatenated activation windows."""

    root = _copy_dataset(tmp_path, TOPOLOGY_DEMO)
    frame = _concatenated_rows(pd.read_csv(root / "Building_18.csv"), spans)
    frame.iloc[:len(frame) - drop_rows].to_csv(root / "Building_18.csv", index=False)
    charger = _concatenated_rows(pd.read_csv(root / "charger_18_1.csv"), spans)
    charger.iloc[:len(charger) - drop_rows].to_csv(root / "charger_18_1.csv", index=False)
    return root, frame


def _expected_rejoin_load(frame, spans=REJOIN_SPANS):
    expected, offset = {}, 0

    for start, end in spans:
        for step in range(start, end):
            expected[step] = frame["non_shiftable_load"].iloc[offset + step - start]
        offset += end - start

    return expected


@pytest.mark.parametrize(
    "window", [(990, 1050), (1005, 1030), (1012, 1025), (1025, 1045), (0, 1100), (1100, 1200), (0, 500)],
)
def test_rejoining_member_reads_the_concatenated_window_file(tmp_path, window):
    root, frame = _rejoin_dataset(tmp_path)
    start, end = window
    env = CityLearnEnv(
        _schema(root, topology_events=REJOIN_EVENTS), simulation_start_time_step=start, simulation_end_time_step=end,
        random_seed=0,
    )
    load = env._topology_service.member_pool["Building_18"].energy_simulation.non_shiftable_load

    assert len(load) == end - start + 1
    for step, value in _expected_rejoin_load(frame).items():
        if start <= step <= end:
            assert load[step - start] == pytest.approx(value)


def test_rejoining_member_is_observed_at_each_activation(tmp_path):
    root, frame = _rejoin_dataset(tmp_path)
    expected = _expected_rejoin_load(frame)
    env = CityLearnEnv(
        _schema(root, topology_events=REJOIN_EVENTS),
        simulation_start_time_step=990, simulation_end_time_step=1050, random_seed=0,
    )
    env.reset()
    building = env._topology_service.member_pool["Building_18"]
    active = []

    while not env.terminated:
        step = 990 + env.time_step
        if "Building_18" in [b.name for b in env.buildings]:
            active.append(step)
        env.step(_zero_entity_actions(env))

    assert active == sorted(expected)
    recorded = np.array(building.non_shiftable_load, dtype=float)
    for step in (1000, 1009, 1020, 1039):
        assert recorded[step - 990] == pytest.approx(expected[step])


@pytest.mark.parametrize("episode", [(995, 1025), (1025, 1045)])
def test_rejoining_member_with_episode_splits(tmp_path, episode):
    root, frame = _rejoin_dataset(tmp_path)
    env = CityLearnEnv(
        _schema(root, topology_events=REJOIN_EVENTS), simulation_start_time_step=990, simulation_end_time_step=1050,
        episode_time_steps=[episode], random_seed=0,
    )
    load = env._topology_service.member_pool["Building_18"].energy_simulation.non_shiftable_load
    start, end = episode

    assert len(load) == end - start + 1
    for step, value in _expected_rejoin_load(frame).items():
        if start <= step <= end:
            assert load[step - start] == pytest.approx(value)


def test_rejoining_member_file_with_a_wrong_length_raises(tmp_path):
    root, _ = _rejoin_dataset(tmp_path, drop_rows=1)

    with pytest.raises(ValueError, match=r"has 29 rows.*2 windows.*need 30 rows"):
        CityLearnEnv(
            _schema(root, topology_events=REJOIN_EVENTS), simulation_start_time_step=990,
            simulation_end_time_step=1050, random_seed=0,
        )


def test_rejoining_member_with_a_full_horizon_file_is_unchanged(tmp_path):
    root = _copy_dataset(tmp_path, TOPOLOGY_DEMO)
    original = pd.read_csv(root / "Building_18.csv")
    full = pd.concat([original.iloc[[0] * 1000], original, original.iloc[[-1] * (8760 - 3200)]], ignore_index=True)
    assert len(full) == 8760
    full.to_csv(root / "Building_18.csv", index=False)
    schema = _schema(root, topology_events=REJOIN_EVENTS)
    schema["buildings"]["Building_18"]["chargers"] = {}
    env = CityLearnEnv(schema, simulation_start_time_step=990, simulation_end_time_step=1050, random_seed=0)
    load = env._topology_service.member_pool["Building_18"].energy_simulation.non_shiftable_load

    np.testing.assert_allclose(load, full["non_shiftable_load"].iloc[990:1051])
