from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.agents.baseline import RBCSmartPolicy
from citylearn.citylearn import CityLearnEnv
from citylearn.multi_community import MultiCommunityEnv


ROOT = Path(__file__).resolve().parents[1]
MINUTE_SCHEMA = ROOT / "tests" / "data" / "minute_ev_demo" / "schema.json"
ZERO_EV_PUBLIC_SCHEMA = ROOT / "data" / "datasets" / "citylearn_challenge_2022_phase_all" / "schema.json"

PEAK_FLEXIBILITY_KPIS = {
    "building_peak_window_mean_load_reduction_to_business_as_usual_kw",
    "building_peak_window_p90_load_reduction_to_business_as_usual_kw",
    "building_ev_peak_window_net_export_kwh",
    "building_ev_peak_window_mean_net_discharge_kw",
    "building_ev_peak_window_p90_net_discharge_kw",
    "district_peak_window_mean_load_reduction_to_business_as_usual_kw",
    "district_peak_window_p90_load_reduction_to_business_as_usual_kw",
    "district_ev_peak_window_net_export_kwh",
    "district_ev_peak_window_mean_net_discharge_kw",
    "district_ev_peak_window_p90_net_discharge_kw",
}


def _build_peak_flex_dataset(
    dataset_dir: Path,
    *,
    rows: int = 25,
    with_evs: bool = True,
) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hours = [t % 24 for t in range(rows)]
    weather = pd.DataFrame(
        {
            "outdoor_dry_bulb_temperature": [20.0] * rows,
            "outdoor_relative_humidity": [50.0] * rows,
            "diffuse_solar_irradiance": [0.0] * rows,
            "direct_solar_irradiance": [0.0] * rows,
            "outdoor_dry_bulb_temperature_predicted_1": [20.0] * rows,
            "outdoor_dry_bulb_temperature_predicted_2": [20.0] * rows,
            "outdoor_dry_bulb_temperature_predicted_3": [20.0] * rows,
            "outdoor_relative_humidity_predicted_1": [50.0] * rows,
            "outdoor_relative_humidity_predicted_2": [50.0] * rows,
            "outdoor_relative_humidity_predicted_3": [50.0] * rows,
            "diffuse_solar_irradiance_predicted_1": [0.0] * rows,
            "diffuse_solar_irradiance_predicted_2": [0.0] * rows,
            "diffuse_solar_irradiance_predicted_3": [0.0] * rows,
            "direct_solar_irradiance_predicted_1": [0.0] * rows,
            "direct_solar_irradiance_predicted_2": [0.0] * rows,
            "direct_solar_irradiance_predicted_3": [0.0] * rows,
        }
    )
    weather.to_csv(dataset_dir / "weather.csv", index=False)
    pd.DataFrame({"carbon_intensity": [0.2] * rows}).to_csv(dataset_dir / "carbon_intensity.csv", index=False)
    pd.DataFrame(
        {
            "electricity_pricing": [0.35 if 16 <= h < 19 else 0.10 for h in hours],
            "electricity_pricing_predicted_1": [0.10] * rows,
            "electricity_pricing_predicted_2": [0.10] * rows,
            "electricity_pricing_predicted_3": [0.10] * rows,
        }
    ).to_csv(dataset_dir / "pricing.csv", index=False)

    for b_name, base_load in (("Building_1", 5.0), ("Building_2", 4.0)):
        pd.DataFrame(
            {
                "month": [1] * rows,
                "hour": hours,
                "minutes": [0] * rows,
                "day_type": [1] * rows,
                "daylight_savings_status": [0] * rows,
                "indoor_dry_bulb_temperature": [21.0] * rows,
                "average_unmet_cooling_setpoint_difference": [0.0] * rows,
                "indoor_relative_humidity": [45.0] * rows,
                "non_shiftable_load": [base_load] * rows,
                "dhw_demand": [0.0] * rows,
                "cooling_demand": [0.0] * rows,
                "heating_demand": [0.0] * rows,
                "solar_generation": [0.0] * rows,
            }
        ).to_csv(dataset_dir / f"{b_name}.csv", index=False)

    observations = {
        "hour": {"active": True, "shared_in_central_agent": True},
        "non_shiftable_load": {"active": True, "shared_in_central_agent": False},
        "electrical_storage_soc": {"active": True, "shared_in_central_agent": False},
        "net_electricity_consumption": {"active": True, "shared_in_central_agent": False},
        "electricity_pricing": {"active": True, "shared_in_central_agent": True},
    }
    actions = {"electrical_storage": {"active": True}}

    if with_evs:
        observations.update(
            {
                "electric_vehicle_charger_connected_state": {"active": True, "shared_in_central_agent": False},
                "connected_electric_vehicle_at_charger_battery_capacity": {"active": True, "shared_in_central_agent": False},
                "connected_electric_vehicle_at_charger_departure_time": {"active": True, "shared_in_central_agent": False},
                "connected_electric_vehicle_at_charger_required_soc_departure": {"active": True, "shared_in_central_agent": False},
                "connected_electric_vehicle_at_charger_soc": {"active": True, "shared_in_central_agent": False},
            }
        )
        actions["electric_vehicle_storage"] = {"active": True}
        for idx, ev_name in enumerate(("EV_1", "EV_2"), start=1):
            pd.DataFrame(
                {
                    "electric_vehicle_charger_state": [1] * rows,
                    "electric_vehicle_id": [ev_name] * rows,
                    "electric_vehicle_departure_time": [float(max(rows - 1 - t, 1)) for t in range(rows)],
                    "electric_vehicle_required_soc_departure": [0.60] * rows,
                    "electric_vehicle_estimated_arrival_time": [np.nan] * rows,
                    "electric_vehicle_estimated_soc_arrival": [0.65 if t == 0 else np.nan for t in range(rows)],
                }
            ).to_csv(dataset_dir / f"charger_{idx}.csv", index=False)

    buildings_cfg = {}
    for idx, (b_name, max_ev_kw) in enumerate((("Building_1", 10.0), ("Building_2", 8.0)), start=1):
        b_cfg = {
            "include": True,
            "energy_simulation": f"{b_name}.csv",
            "weather": "weather.csv",
            "carbon_intensity": "carbon_intensity.csv",
            "pricing": "pricing.csv",
            "inactive_observations": [],
            "inactive_actions": [],
            "electrical_storage": {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": 30.0,
                    "nominal_power": 6.0,
                    "initial_soc": 0.80,
                    "efficiency": 1.0,
                    "loss_coefficient": 0.0,
                    "capacity_loss_coefficient": 0.0,
                    "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                },
            },
        }
        if with_evs:
            b_cfg["chargers"] = {
                f"charger_{idx}": {
                    "type": "citylearn.electric_vehicle_charger.Charger",
                    "charger_simulation": f"charger_{idx}.csv",
                    "autosize": False,
                    "attributes": {
                        "nominal_power": max_ev_kw,
                        "efficiency": 1.0,
                        "charger_type": 0,
                        "max_charging_power": max_ev_kw,
                        "min_charging_power": 0.0,
                        "max_discharging_power": max_ev_kw,
                        "min_discharging_power": 0.0,
                    },
                }
            }
        buildings_cfg[b_name] = b_cfg

    schema = {
        "random_seed": 0,
        "root_directory": str(dataset_dir),
        "central_agent": True,
        "interface": "flat",
        "simulation_start_time_step": 0,
        "simulation_end_time_step": rows - 1,
        "episode_time_steps": rows,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": 3600,
        "observations": observations,
        "actions": actions,
        "reward_function": {
            "type": "citylearn.reward_function.RewardFunction",
            "attributes": {},
        },
        "buildings": buildings_cfg,
    }
    if with_evs:
        schema["electric_vehicles_def"] = {
            ev_name: {
                "include": True,
                "battery": {
                    "type": "citylearn.energy_model.Battery",
                    "autosize": False,
                    "attributes": {
                        "capacity": 60.0,
                        "nominal_power": 12.0,
                        "initial_soc": 0.65,
                        "efficiency": 1.0,
                        "loss_coefficient": 0.0,
                        "capacity_loss_coefficient": 0.0,
                        "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    },
                },
            }
            for ev_name in ("EV_1", "EV_2")
        }
    return schema


def test_zero_ev_run_keeps_ev_contribution_zero_with_and_without_bau(tmp_path: Path):
    """Regression test for maintainer review:
    In a run with zero EVs and active stationary batteries shaving peak load, EV peak-window
    KPIs must report 0.0 kW / 0.0 kWh both with BAU enabled and with BAU disabled, while
    district/building load-reduction-to-BAU KPIs report the stationary battery peak reduction
    when BAU is enabled and NaN when BAU is disabled.
    """
    schema = _build_peak_flex_dataset(tmp_path / "zero_evs", rows=25, with_evs=False)
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        # Stationary batteries discharge during peak hours 16, 17, 18 (e.g. -3, -4, -6 kW total => mean ~4.33 kW!)
        peak_bess = {16: (-0.25, -0.25), 17: (-0.5, -1.0 / 6.0), 18: (-0.5, -0.5)}
        for t in range(24):
            if t in peak_bess:
                act = np.array(peak_bess[t], dtype="float32")
            else:
                act = np.zeros(2, dtype="float32")
            env.step([act])

        kpis_bau = env.evaluate_v2(
            include_business_as_usual=True,
            include_peak_flexibility_audit=True,
            flexibility_peak_hours=(16, 19),
        )
        kpis_no_bau = env.evaluate_v2(
            include_business_as_usual=False,
            include_peak_flexibility_audit=True,
            flexibility_peak_hours=(16, 19),
        )
        assert kpis_bau[["unit", "description"]].notna().all().all()
        assert kpis_no_bau[["unit", "description"]].notna().all().all()

        with_bau = {(r.level, r.name, r.cost_function): r.value for r in kpis_bau.itertuples()}
        no_bau = {(r.level, r.name, r.cost_function): r.value for r in kpis_no_bau.itertuples()}

        # EV contribution is strictly 0.0 both with BAU enabled and with BAU disabled!
        for lookup in (with_bau, no_bau):
            for scope in (("district", "District"), ("building", "Building_1"), ("building", "Building_2")):
                prefix = scope[0]
                assert lookup[(*scope, f"{prefix}_ev_peak_window_net_export_kwh")] == pytest.approx(0.0, abs=1e-12)
                assert lookup[(*scope, f"{prefix}_ev_peak_window_mean_net_discharge_kw")] == pytest.approx(0.0, abs=1e-12)
                assert lookup[(*scope, f"{prefix}_ev_peak_window_p90_net_discharge_kw")] == pytest.approx(0.0, abs=1e-12)

        # Whole-district load reduction relative to BAU reflects stationary battery peak shaving when BAU=True...
        bau_env = env.run_business_as_usual_baseline().env
        expected_reduction_kw = (
            np.asarray(bau_env.net_electricity_consumption[16:19], dtype="float64")
            - np.asarray(env.net_electricity_consumption[16:19], dtype="float64")
        )
        assert with_bau[("district", "District", "district_peak_window_mean_load_reduction_to_business_as_usual_kw")] == pytest.approx(
            float(np.mean(expected_reduction_kw)), abs=1e-5
        )
        assert with_bau[("district", "District", "district_peak_window_p90_load_reduction_to_business_as_usual_kw")] == pytest.approx(
            float(np.percentile(expected_reduction_kw, 10)), abs=1e-5
        )

        # ...and is NaN (undefined) when BAU=False!
        assert np.isnan(no_bau[("district", "District", "district_peak_window_mean_load_reduction_to_business_as_usual_kw")])
        assert np.isnan(no_bau[("district", "District", "district_peak_window_p90_load_reduction_to_business_as_usual_kw")])
        assert np.isnan(no_bau[("building", "Building_1", "building_peak_window_mean_load_reduction_to_business_as_usual_kw")])
    finally:
        env.close()

    # Also verify on bundled public zero-EV dataset
    pub_env = CityLearnEnv(str(ZERO_EV_PUBLIC_SCHEMA), episode_time_steps=24, central_agent=True, offline=True, render_mode="none")
    try:
        agent = RBCSmartPolicy(pub_env)
        obs, _ = pub_env.reset(seed=0)
        while not pub_env.terminated:
            obs, *_ = pub_env.step(agent.predict(obs))
        pub_bau = pub_env.evaluate_v2(include_business_as_usual=True, include_peak_flexibility_audit=True)
        pub_no_bau = pub_env.evaluate_v2(include_business_as_usual=False, include_peak_flexibility_audit=True)
        for frame in (pub_bau, pub_no_bau):
            ev_rows = frame[frame["cost_function"].str.contains("_ev_peak_window_")]
            assert np.allclose(ev_rows["value"].to_numpy(dtype="float64"), 0.0, atol=1e-12)
    finally:
        pub_env.close()


def test_ev_v2g_peak_window_contribution_is_invariant_to_bau_and_validates_hours(tmp_path: Path):
    schema = _build_peak_flex_dataset(tmp_path / "with_evs", rows=25, with_evs=True)
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        v2g_b1 = {16: -0.2, 17: -0.6, 18: -1.0}      # 10 kW charger -> 2, 6, 10 kW discharge
        v2g_b2 = {16: -0.25, 17: -0.5, 18: -0.75}    # 8 kW charger  -> 2, 4, 6 kW discharge
        for t in range(24):
            if t in v2g_b1:
                act = np.array([0.0, v2g_b1[t], 0.0, v2g_b2[t]], dtype="float32")
            else:
                act = np.zeros(4, dtype="float32")
            env.step([act])

        kpis_no_bau = env.evaluate_v2(
            include_business_as_usual=False,
            include_peak_flexibility_audit=True,
            flexibility_peak_hours=(16, 19),
        )
        kpis_with_bau = env.evaluate_v2(
            include_business_as_usual=True,
            include_peak_flexibility_audit=True,
            flexibility_peak_hours=(16, 19),
        )
        no_bau = {(r.level, r.name, r.cost_function): r.value for r in kpis_no_bau.itertuples()}
        with_bau = {(r.level, r.name, r.cost_function): r.value for r in kpis_with_bau.itertuples()}

        b1_dis = np.array([2.0, 6.0, 10.0], dtype="float64")
        b2_dis = np.array([2.0, 4.0, 6.0], dtype="float64")
        dist_dis = b1_dis + b2_dis

        # EV metrics are identical whether BAU is enabled or disabled!
        for lookup in (no_bau, with_bau):
            assert lookup[("building", "Building_1", "building_ev_peak_window_net_export_kwh")] == pytest.approx(18.0, abs=1e-5)
            assert lookup[("building", "Building_2", "building_ev_peak_window_net_export_kwh")] == pytest.approx(12.0, abs=1e-5)
            assert lookup[("district", "District", "district_ev_peak_window_net_export_kwh")] == pytest.approx(30.0, abs=1e-5)
            assert lookup[("building", "Building_1", "building_ev_peak_window_mean_net_discharge_kw")] == pytest.approx(
                float(np.mean(b1_dis)), abs=1e-5
            )
            assert lookup[("building", "Building_1", "building_ev_peak_window_p90_net_discharge_kw")] == pytest.approx(
                float(np.percentile(b1_dis, 10)), abs=1e-5
            )
            assert lookup[("district", "District", "district_ev_peak_window_mean_net_discharge_kw")] == pytest.approx(
                float(np.mean(dist_dis)), abs=1e-5
            )
            assert lookup[("district", "District", "district_ev_peak_window_p90_net_discharge_kw")] == pytest.approx(
                float(np.percentile(dist_dis, 10)), abs=1e-5
            )

        assert np.isnan(no_bau[("district", "District", "district_peak_window_mean_load_reduction_to_business_as_usual_kw")])
        bau_env = env.run_business_as_usual_baseline().env
        expected_bau_diff = (
            np.asarray(bau_env.net_electricity_consumption[16:19], dtype="float64")
            - np.asarray(env.net_electricity_consumption[16:19], dtype="float64")
        )
        assert with_bau[("district", "District", "district_peak_window_mean_load_reduction_to_business_as_usual_kw")] == pytest.approx(
            float(np.mean(expected_bau_diff)), abs=1e-5
        )
        assert with_bau[("district", "District", "district_peak_window_p90_load_reduction_to_business_as_usual_kw")] == pytest.approx(
            float(np.percentile(expected_bau_diff, 10)), abs=1e-5
        )

        # Overnight wrap-around window (22, 2) has zero V2G activity in this schedule -> 0.0 kWh / 0.0 kW
        wrap_kpis = env.evaluate_v2(
            include_business_as_usual=False,
            include_peak_flexibility_audit=True,
            flexibility_peak_hours=(22, 2),
        )
        wrap_lookup = {(r.level, r.name, r.cost_function): r.value for r in wrap_kpis.itertuples()}
        assert wrap_lookup[("district", "District", "district_ev_peak_window_mean_net_discharge_kw")] == pytest.approx(0.0, abs=1e-9)

        with pytest.raises(ValueError, match="flexibility_peak_hours"):
            env.evaluate_v2(include_business_as_usual=False, include_peak_flexibility_audit=True, flexibility_peak_hours=(16, 16))
    finally:
        env.close()


def test_multi_community_portfolio_and_default_bit_identity(tmp_path: Path):
    schema_a = _build_peak_flex_dataset(tmp_path / "mc_a", rows=25, with_evs=True)
    schema_b = _build_peak_flex_dataset(tmp_path / "mc_b", rows=25, with_evs=False)
    portfolio = MultiCommunityEnv(
        communities=[
            {"community_id": "a", "schema": schema_a, "weight": 1.0, "env_kwargs": {"offline": True, "render_mode": "none"}},
            {"community_id": "b", "schema": schema_b, "weight": 1.0, "env_kwargs": {"offline": True, "render_mode": "none"}},
        ]
    )
    try:
        portfolio.reset(seed=0)
        for t in range(24):
            act_a = np.array([0.0, -0.5, 0.0, -0.5], dtype="float32") if 16 <= t < 19 else np.zeros(4, dtype="float32")
            act_b = np.array([-0.5, -0.5], dtype="float32") if 16 <= t < 19 else np.zeros(2, dtype="float32")
            portfolio.step({"a": [act_a], "b": [act_b]})

        mc_kpis = portfolio.evaluate_v2(include_business_as_usual=False, include_peak_flexibility_audit=True)
        assert mc_kpis[["unit", "description"]].notna().all().all()
        port_export = float(
            mc_kpis[
                (mc_kpis["level"] == "portfolio")
                & (mc_kpis["cost_function"] == "portfolio_ev_peak_window_net_export_kwh")
            ]["value"].iloc[0]
        )
        # Community a exports (5 + 4) kW * 3 h = 27 kWh; Community b has zero EVs (0 kWh) -> sum = 27 kWh
        assert port_export == pytest.approx(27.0, abs=1e-5)
    finally:
        portfolio.close()

    env = CityLearnEnv(str(MINUTE_SCHEMA), offline=True, render_mode="none", random_seed=7)
    try:
        env.reset(seed=7)
        while not env.terminated:
            env.step([np.array([0.3, 0.4], dtype="float32")])
        kpis_default = env.evaluate_v2(include_business_as_usual=True)
        kpis_explicit_false = env.evaluate_v2(include_business_as_usual=True, include_peak_flexibility_audit=False)
        assert set(kpis_default["cost_function"]).isdisjoint(PEAK_FLEXIBILITY_KPIS)
        pd.testing.assert_frame_equal(kpis_default, kpis_explicit_false, check_exact=True)
    finally:
        env.close()
