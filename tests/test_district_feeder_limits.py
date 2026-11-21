from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv
from citylearn.internal.balance_audit import IndependentBalanceAudit


MINUTE_DATASET_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "minute_ev_demo"
MINUTE_SCHEMA = MINUTE_DATASET_DIR / "schema.json"

DISTRICT_FEEDER_KPIS = {
    "district_feeder_curtailed_import_kwh",
    "district_feeder_curtailed_export_kwh",
    "district_feeder_max_violation_kw",
}


def _build_two_building_feeder_dataset(
    tmp_path: Path,
    *,
    name: str = "feeder_district",
    rows: int = 4,
    seconds_per_time_step: int = 3600,
    base_loads_b1=None,
    base_loads_b2=None,
    solar_b1=None,
    solar_b2=None,
    b1_battery_power: float = 6.0,
    b2_battery_power: float = 4.0,
    b1_battery_efficiency: float = 0.90,
    b2_battery_efficiency: float = 0.90,
    b1_charger_max_charge: float = 10.0,
    b1_charger_min_charge: float = 0.0,
    b1_charger_max_discharge: float = 10.0,
    b1_charger_min_discharge: float = 0.0,
    b2_charger_max_charge: float = 8.0,
    b2_charger_min_charge: float = 0.0,
    b2_charger_max_discharge: float = 8.0,
    b2_charger_min_discharge: float = 0.0,
    ev_connected: bool = True,
    district_electrical_service=None,
    interface: str = "flat",
) -> dict:
    dataset_dir = tmp_path / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    hours = [((t * seconds_per_time_step) // 3600) % 24 for t in range(rows)]
    minutes = [((t * seconds_per_time_step) % 3600) // 60 for t in range(rows)]
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
            "electricity_pricing": [0.30 if 16 <= h < 19 else 0.12 for h in hours],
            "electricity_pricing_predicted_1": [0.12] * rows,
            "electricity_pricing_predicted_2": [0.12] * rows,
            "electricity_pricing_predicted_3": [0.12] * rows,
        }
    ).to_csv(dataset_dir / "pricing.csv", index=False)

    b1_load = list(base_loads_b1) if base_loads_b1 is not None else [2.0] * rows
    b2_load = list(base_loads_b2) if base_loads_b2 is not None else [1.5] * rows
    b1_pv = list(solar_b1) if solar_b1 is not None else [0.0] * rows
    b2_pv = list(solar_b2) if solar_b2 is not None else [0.0] * rows

    for b_name, loads, pv_series in (
        ("Building_1", b1_load, b1_pv),
        ("Building_2", b2_load, b2_pv),
    ):
        pd.DataFrame(
            {
                "month": [1] * rows,
                "hour": hours,
                "minutes": minutes,
                "day_type": [1] * rows,
                "daylight_savings_status": [0] * rows,
                "indoor_dry_bulb_temperature": [21.0] * rows,
                "average_unmet_cooling_setpoint_difference": [0.0] * rows,
                "indoor_relative_humidity": [45.0] * rows,
                "non_shiftable_load": loads,
                "dhw_demand": [0.0] * rows,
                "cooling_demand": [0.0] * rows,
                "heating_demand": [0.0] * rows,
                "solar_generation": pv_series,
            }
        ).to_csv(dataset_dir / f"{b_name}.csv", index=False)

    for idx, ev_name in enumerate(("EV_1", "EV_2"), start=1):
        if ev_connected:
            charger_df = pd.DataFrame(
                {
                    "electric_vehicle_charger_state": [1] * rows,
                    "electric_vehicle_id": [ev_name] * rows,
                    "electric_vehicle_departure_time": [float(max(rows - 1 - t, 1)) for t in range(rows)],
                    "electric_vehicle_required_soc_departure": [0.60] * rows,
                    "electric_vehicle_estimated_arrival_time": [np.nan] * rows,
                    "electric_vehicle_estimated_soc_arrival": [0.50 if t == 0 else np.nan for t in range(rows)],
                }
            )
        else:
            charger_df = pd.DataFrame(
                {
                    "electric_vehicle_charger_state": [0] * rows,
                    "electric_vehicle_id": [""] * rows,
                    "electric_vehicle_departure_time": [np.nan] * rows,
                    "electric_vehicle_required_soc_departure": [np.nan] * rows,
                    "electric_vehicle_estimated_arrival_time": [np.nan] * rows,
                    "electric_vehicle_estimated_soc_arrival": [np.nan] * rows,
                }
            )
        charger_df.to_csv(dataset_dir / f"charger_{idx}.csv", index=False)

    schema = {
        "random_seed": 0,
        "root_directory": str(dataset_dir),
        "central_agent": True,
        "interface": interface,
        "simulation_start_time_step": 0,
        "simulation_end_time_step": rows - 1,
        "episode_time_steps": rows,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": seconds_per_time_step,
        "observations": {
            "month": {"active": True, "shared_in_central_agent": True},
            "day_type": {"active": True, "shared_in_central_agent": True},
            "hour": {"active": True, "shared_in_central_agent": True},
            "minutes": {"active": True, "shared_in_central_agent": True},
            "outdoor_dry_bulb_temperature": {"active": True, "shared_in_central_agent": True},
            "carbon_intensity": {"active": True, "shared_in_central_agent": True},
            "non_shiftable_load": {"active": True, "shared_in_central_agent": False},
            "solar_generation": {"active": True, "shared_in_central_agent": False},
            "electrical_storage_soc": {"active": True, "shared_in_central_agent": False},
            "net_electricity_consumption": {"active": True, "shared_in_central_agent": False},
            "electricity_pricing": {"active": True, "shared_in_central_agent": True},
            "electric_vehicle_charger_connected_state": {"active": True, "shared_in_central_agent": False},
            "connected_electric_vehicle_at_charger_battery_capacity": {"active": True, "shared_in_central_agent": False},
            "connected_electric_vehicle_at_charger_departure_time": {"active": True, "shared_in_central_agent": False},
            "connected_electric_vehicle_at_charger_required_soc_departure": {"active": True, "shared_in_central_agent": False},
            "connected_electric_vehicle_at_charger_soc": {"active": True, "shared_in_central_agent": False},
        },
        "actions": {
            "electrical_storage": {"active": True},
            "electric_vehicle_storage": {"active": True},
        },
        "reward_function": {
            "type": "citylearn.reward_function.RewardFunction",
            "attributes": {},
        },
        "electric_vehicles_def": {
            "EV_1": {
                "include": True,
                "battery": {
                    "type": "citylearn.energy_model.Battery",
                    "autosize": False,
                    "attributes": {
                        "capacity": 60.0,
                        "nominal_power": 15.0,
                        "initial_soc": 0.50,
                        "efficiency": 1.0,
                        "loss_coefficient": 0.0,
                        "capacity_loss_coefficient": 1.0e-3,
                        "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    },
                },
            },
            "EV_2": {
                "include": True,
                "battery": {
                    "type": "citylearn.energy_model.Battery",
                    "autosize": False,
                    "attributes": {
                        "capacity": 40.0,
                        "nominal_power": 15.0,
                        "initial_soc": 0.50,
                        "efficiency": 1.0,
                        "loss_coefficient": 0.0,
                        "capacity_loss_coefficient": 1.0e-3,
                        "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    },
                },
            },
        },
        "buildings": {
            "Building_1": {
                "include": True,
                "energy_simulation": "Building_1.csv",
                "weather": "weather.csv",
                "carbon_intensity": "carbon_intensity.csv",
                "pricing": "pricing.csv",
                "inactive_observations": [],
                "inactive_actions": ["electrical_storage"] if b1_battery_power <= 0 else [],
                "pv": {
                    "type": "citylearn.energy_model.PV",
                    "autosize": False,
                    "attributes": {"nominal_power": 1.0, "generation_mode": "absolute"},
                },
                "electrical_storage": {
                    "type": "citylearn.energy_model.Battery",
                    "autosize": False,
                    "attributes": {
                        "capacity": 20.0 if b1_battery_power > 0 else 0.0,
                        "nominal_power": b1_battery_power,
                        "initial_soc": 0.50,
                        "efficiency": b1_battery_efficiency,
                        "loss_coefficient": 0.0,
                        "capacity_loss_coefficient": 0.0,
                        "power_efficiency_curve": [[0.0, b1_battery_efficiency], [1.0, b1_battery_efficiency]],
                        "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    },
                },
                "chargers": {
                    "charger_1": {
                        "type": "citylearn.electric_vehicle_charger.Charger",
                        "charger_simulation": "charger_1.csv",
                        "autosize": False,
                        "attributes": {
                            "nominal_power": max(b1_charger_max_charge, b1_charger_max_discharge, 1.0),
                            "efficiency": 1.0,
                            "charger_type": 0,
                            "max_charging_power": b1_charger_max_charge,
                            "min_charging_power": b1_charger_min_charge,
                            "max_discharging_power": b1_charger_max_discharge,
                            "min_discharging_power": b1_charger_min_discharge,
                        },
                    }
                },
            },
            "Building_2": {
                "include": True,
                "energy_simulation": "Building_2.csv",
                "weather": "weather.csv",
                "carbon_intensity": "carbon_intensity.csv",
                "pricing": "pricing.csv",
                "inactive_observations": [],
                "inactive_actions": ["electrical_storage"] if b2_battery_power <= 0 else [],
                "pv": {
                    "type": "citylearn.energy_model.PV",
                    "autosize": False,
                    "attributes": {"nominal_power": 1.0, "generation_mode": "absolute"},
                },
                "electrical_storage": {
                    "type": "citylearn.energy_model.Battery",
                    "autosize": False,
                    "attributes": {
                        "capacity": 20.0 if b2_battery_power > 0 else 0.0,
                        "nominal_power": b2_battery_power,
                        "initial_soc": 0.50,
                        "efficiency": b2_battery_efficiency,
                        "loss_coefficient": 0.0,
                        "capacity_loss_coefficient": 0.0,
                        "power_efficiency_curve": [[0.0, b2_battery_efficiency], [1.0, b2_battery_efficiency]],
                        "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                        "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    },
                },
                "chargers": {
                    "charger_2": {
                        "type": "citylearn.electric_vehicle_charger.Charger",
                        "charger_simulation": "charger_2.csv",
                        "autosize": False,
                        "attributes": {
                            "nominal_power": max(b2_charger_max_charge, b2_charger_max_discharge, 1.0),
                            "efficiency": 1.0,
                            "charger_type": 0,
                            "max_charging_power": b2_charger_max_charge,
                            "min_charging_power": b2_charger_min_charge,
                            "max_discharging_power": b2_charger_max_discharge,
                            "min_discharging_power": b2_charger_min_discharge,
                        },
                    }
                },
            },
        },
    }
    if district_electrical_service is not None:
        schema["district_electrical_service"] = deepcopy(district_electrical_service)
    return schema


def test_avoidable_violation_with_solar_ev_minimum_power_and_stationary_battery(tmp_path: Path):
    """Regression test for maintainer reproducer:
    5 kW solar surplus, 10 kW EV charging request with 7 kW minimum, 4 kW stationary battery
    charging request, and 1 kW limit in each direction -> EV paused (0 kW), battery held at 4 kW,
    district exports 1 kW (-1.0 kW net), meeting both limits with 0.0 kW violation.
    """
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="avoidable_deadband",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[5.0, 5.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=4.0,
        b2_battery_efficiency=1.0,
        b1_charger_max_charge=10.0,
        b1_charger_min_charge=7.0,
        b2_charger_max_charge=8.0,
        b2_charger_min_charge=0.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            # Active actions: [b1_ev (10 kW), b2_storage (4 kW), b2_ev (0 kW)]
            env.step([np.array([1.0, 1.0, 0.0], dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1, b2 = env.buildings
        assert float(b1.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(b2.electrical_storage.electricity_consumption[0]) == pytest.approx(4.0, abs=1e-6)
        assert float(env.net_electricity_consumption[0]) == pytest.approx(-1.0, abs=1e-6)
        assert float(env.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env.feeder_curtailed_import_kwh[0]) == pytest.approx(10.0, abs=1e-5)
        assert float(env.feeder_curtailed_export_kwh[0]) == pytest.approx(0.0, abs=1e-6)

        kpis = env.evaluate_v2(include_business_as_usual=False)
        lookup = {(r.level, r.name, r.cost_function): r.value for r in kpis.itertuples()}
        assert lookup[("district", "District", "district_feeder_max_violation_kw")] == pytest.approx(0.0, abs=1e-6)
        assert lookup[("district", "District", "district_feeder_curtailed_import_kwh")] == pytest.approx(10.0, abs=1e-5)
        assert lookup[("district", "District", "district_feeder_curtailed_export_kwh")] == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()


def test_avoidable_export_violation_with_ev_minimum_discharging_power(tmp_path: Path):
    """Symmetric deadband check on export curtailment:
    5 kW passive load, -10 kW EV V2G discharge request with 7 kW minimum, -4 kW battery discharge,
    and 1 kW limit in each direction -> EV V2G paused (0 kW), battery held at -4 kW,
    district imports 1 kW (+1.0 kW net), meeting both limits with 0.0 kW violation.
    """
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="avoidable_export_deadband",
        rows=2,
        base_loads_b1=[5.0, 5.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[0.0, 0.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=4.0,
        b2_battery_efficiency=1.0,
        b1_charger_max_discharge=10.0,
        b1_charger_min_discharge=7.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            # Active actions: [b1_ev (-10 kW), b2_storage (-4 kW), b2_ev (0 kW)]
            env.step([np.array([-1.0, -1.0, 0.0], dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1, b2 = env.buildings
        assert float(b1.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(b2.electrical_storage.electricity_consumption[0]) == pytest.approx(-4.0, abs=1e-6)
        assert float(env.net_electricity_consumption[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(env.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env.feeder_curtailed_export_kwh[0]) == pytest.approx(10.0, abs=1e-5)
    finally:
        env.close()


def test_unavoidable_solar_surplus_continues_and_records_4kw_exceedance(tmp_path: Path):
    """Unavoidable violation case:
    Fixed 5 kW solar surplus, no available control to absorb it, and 1 kW export limit ->
    simulation continues, reports 5 kW export (-5.0 kW net), and records 4 kW exceedance.
    """
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="unavoidable_solar",
        rows=3,
        base_loads_b1=[0.0, 0.0, 0.0],
        base_loads_b2=[0.0, 0.0, 0.0],
        solar_b1=[3.0, 3.0, 3.0],
        solar_b2=[2.0, 2.0, 2.0],
        b1_battery_power=0.0,
        b2_battery_power=0.0,
        ev_connected=False,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            for _ in range(2):
                # Active actions: [b1_ev, b2_ev] (both chargers have no connected EV)
                env.step([np.zeros(2, dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        net = np.asarray(env.net_electricity_consumption, dtype="float64")
        assert np.allclose(net, [-5.0, -5.0], atol=1e-6)
        assert np.allclose(env.feeder_curtailed_import_kwh, [0.0, 0.0], atol=1e-9)
        assert np.allclose(env.feeder_curtailed_export_kwh, [0.0, 0.0], atol=1e-9)
        assert np.allclose(env.feeder_limit_violation_kw, [4.0, 4.0], atol=1e-6)

        kpis = env.evaluate_v2(include_business_as_usual=False)
        lookup = {(r.level, r.name, r.cost_function): r.value for r in kpis.itertuples()}
        assert lookup[("district", "District", "district_energy_grid_total_export_control_kwh")] == pytest.approx(10.0, abs=1e-5)
        assert lookup[("district", "District", "district_feeder_max_violation_kw")] == pytest.approx(4.0, abs=1e-6)
        assert lookup[("district", "District", "district_feeder_curtailed_export_kwh")] == pytest.approx(0.0, abs=1e-9)
    finally:
        env.close()


def test_independent_one_sided_import_and_export_limits(tmp_path: Path):
    # 1) Import limit only (export_kw unset / None): curtails import, leaves 26 kW export unconstrained.
    schema_imp = _build_two_building_feeder_dataset(
        tmp_path,
        name="import_only",
        rows=3,
        base_loads_b1=[2.0, 1.0, 1.0],
        base_loads_b2=[1.5, 1.0, 1.0],
        district_electrical_service={"enabled": True, "limits": {"import_kw": 15.0}},
    )
    env_imp = CityLearnEnv(schema_imp, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env_imp.reset(seed=0)
        env_imp.step([np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")])   # 31.5 kW -> capped to 15.0 kW
        env_imp.step([np.array([-1.0, -1.0, -1.0, -1.0], dtype="float32")])  # -26.0 kW -> unconstrained export
        net = np.asarray(env_imp.net_electricity_consumption, dtype="float64")
        assert net[0] == pytest.approx(15.0, abs=1e-4)
        assert net[1] == pytest.approx(-26.0, abs=1e-4)
        assert np.allclose(env_imp.feeder_limit_violation_kw, [0.0, 0.0], atol=1e-6)
    finally:
        env_imp.close()

    # 2) Export limit only (import_kw unset / None) on entity interface: leaves 31.5 kW import unconstrained, caps export to 12.0 kW.
    schema_exp = _build_two_building_feeder_dataset(
        tmp_path,
        name="export_only",
        rows=3,
        base_loads_b1=[2.0, 1.0, 1.0],
        base_loads_b2=[1.5, 1.0, 1.0],
        interface="entity",
        district_electrical_service={"enabled": True, "limits": {"export_kw": 12.0}},
    )
    env_exp = CityLearnEnv(schema_exp, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env_exp.reset(seed=0)
        with IndependentBalanceAudit(env_exp) as balance_audit:
            pos_payload = {
                "tables": {name: np.full(space.shape, 1.0, dtype="float32") for name, space in env_exp.action_space["tables"].items()}
            }
            neg_payload = {
                "tables": {name: np.full(space.shape, -1.0, dtype="float32") for name, space in env_exp.action_space["tables"].items()}
            }
            env_exp.step(pos_payload)
            _, _, _, _, info = env_exp.step(neg_payload)
            charger_exec = {
                row["owner_module_id"]: row["applied_power_kw"]
                for row in info["entity_action_execution"]["entries"]
                if row["action_name"] == "electric_vehicle_storage"
            }
            assert charger_exec["Building_1/charger_1"] == pytest.approx(-5.0, abs=1e-4)
            assert charger_exec["Building_2/charger_2"] == pytest.approx(-4.0, abs=1e-4)
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        net = np.asarray(env_exp.net_electricity_consumption, dtype="float64")
        assert net[0] == pytest.approx(31.5, abs=1e-4)
        assert net[1] == pytest.approx(-12.0, abs=1e-4)
        assert np.allclose(env_exp.feeder_curtailed_export_kwh, [0.0, 14.0], atol=1e-3)
        assert np.allclose(env_exp.feeder_limit_violation_kw, [0.0, 0.0], atol=1e-6)
    finally:
        env_exp.close()


def test_proportional_import_curtailment_and_passive_load_violation(tmp_path: Path):
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="prop_import",
        rows=4,
        base_loads_b1=[2.0, 2.0, 9.0, 2.0],
        base_loads_b2=[1.5, 1.5, 8.0, 1.5],
        district_electrical_service={"enabled": True, "limits": {"import_kw": 15.0}},
    )
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            for _ in range(3):
                env.step([np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1, b2 = env.buildings
        b1_ctrl_0 = float(b1.electrical_storage.electricity_consumption[0]) + float(b1.chargers_electricity_consumption[0])
        b2_ctrl_0 = float(b2.electrical_storage.electricity_consumption[0]) + float(b2.chargers_electricity_consumption[0])
        assert b1_ctrl_0 / b2_ctrl_0 == pytest.approx(16.0 / 12.0, rel=1e-4)

        net = np.asarray(env.net_electricity_consumption, dtype="float64")
        assert net == pytest.approx([15.0, 15.0, 17.0], abs=1e-4)
        assert env.feeder_limit_violation_kw == pytest.approx([0.0, 0.0, 2.0], abs=1e-4)
    finally:
        env.close()


def test_subhourly_time_varying_envelopes_and_config_validation(tmp_path: Path):
    import_envelope = [14.0, 8.0, 20.0, 20.0, 20.0]
    export_envelope = [20.0, 20.0, 6.0, 10.0, 20.0]
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="subhourly_envelope",
        rows=5,
        seconds_per_time_step=900,
        base_loads_b1=[0.5] * 5,
        base_loads_b2=[0.5] * 5,
        district_electrical_service={
            "enabled": True,
            "limits": {"import_kw": import_envelope, "export_kw": export_envelope},
        },
    )
    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        env.step([np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")])
        env.step([np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")])
        env.step([np.array([-1.0, -1.0, -1.0, -1.0], dtype="float32")])
        env.step([np.array([-1.0, -1.0, -1.0, -1.0], dtype="float32")])

        district_kw = np.asarray(env.net_electricity_consumption, dtype="float64") / 0.25
        assert district_kw == pytest.approx([14.0, 8.0, -6.0, -10.0], abs=1e-3)
        assert np.allclose(env.feeder_limit_violation_kw, 0.0, atol=1e-6)
    finally:
        env.close()

    with pytest.raises(ValueError, match="must be >= 0.0"):
        CityLearnEnv(
            str(MINUTE_SCHEMA),
            offline=True,
            district_electrical_service={"enabled": True, "limits": {"import_kw": -5.0}},
        )

    with pytest.raises(ValueError, match="unknown keys"):
        CityLearnEnv(
            str(MINUTE_SCHEMA),
            offline=True,
            district_electrical_service={"enabled": True, "limits": {"imports_kw": 10.0}},
        )


def test_disabled_district_feeder_preserves_strict_bit_identity():
    env_default = CityLearnEnv(str(MINUTE_SCHEMA), offline=True, render_mode="none", random_seed=7)
    env_explicit_off = CityLearnEnv(
        str(MINUTE_SCHEMA),
        offline=True,
        render_mode="none",
        random_seed=7,
        district_electrical_service={"enabled": False, "limits": {"import_kw": 0.5, "export_kw": 0.5}},
    )
    try:
        obs_a, _ = env_default.reset(seed=7)
        obs_b, _ = env_explicit_off.reset(seed=7)
        assert env_default.feeder_curtailed_import_kwh is None
        assert env_explicit_off.feeder_curtailed_import_kwh is None
        assert env_explicit_off.feeder_curtailed_export_kwh is None
        assert env_explicit_off.feeder_limit_violation_kw is None
        assert np.array_equal(obs_a[0], obs_b[0])

        step_idx = 0
        while not env_default.terminated:
            action = [np.array([0.4 if step_idx % 2 == 0 else -0.3, 0.5], dtype="float32")]
            out_a = env_default.step(action)
            out_b = env_explicit_off.step(action)
            assert np.array_equal(out_a[0][0], out_b[0][0])
            assert out_a[1] == out_b[1]
            step_idx += 1

        assert env_default.net_electricity_consumption == env_explicit_off.net_electricity_consumption
        assert env_default.net_electricity_consumption_cost == env_explicit_off.net_electricity_consumption_cost
        assert env_default.net_electricity_consumption_emission == env_explicit_off.net_electricity_consumption_emission

        kpis_a = env_default.evaluate_v2(include_business_as_usual=True)
        kpis_b = env_explicit_off.evaluate_v2(include_business_as_usual=True)
        assert set(kpis_a["cost_function"]).isdisjoint(DISTRICT_FEEDER_KPIS)
        pd.testing.assert_frame_equal(kpis_a, kpis_b, check_exact=True)
    finally:
        env_default.close()
        env_explicit_off.close()


def test_joint_charging_and_discharging_deadband_coordination_and_mirror(tmp_path: Path):
    """Point 1 regression:
    Zero passive load/PV, EV_1 at fixed +7 kW charging, EV_2 at fixed -5 kW discharging,
    and 1 kW limits both ways -> pausing both yields 0 kW net and 0 kW violation.
    Also tests the exact import/export mirror (+5 kW charging and -7 kW discharging).
    """
    # Case A: +7 kW charging, -5 kW discharging
    schema_a = _build_two_building_feeder_dataset(
        tmp_path,
        name="joint_7c_5d",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[0.0, 0.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=0.0,
        b1_charger_max_charge=7.0,
        b1_charger_min_charge=7.0,
        b2_charger_max_discharge=5.0,
        b2_charger_min_discharge=5.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    env_a = CityLearnEnv(schema_a, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env_a.reset(seed=0)
        with IndependentBalanceAudit(env_a) as audit_a:
            env_a.step([np.array([1.0, -1.0], dtype="float32")])
            proof_a = audit_a.finish(check_kpis=True)
        assert proof_a["status"] == "pass"
        b1, b2 = env_a.buildings
        assert float(b1.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(b2.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_a.net_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_a.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_a.feeder_curtailed_import_kwh[0]) == pytest.approx(7.0, abs=1e-5)
        assert float(env_a.feeder_curtailed_export_kwh[0]) == pytest.approx(5.0, abs=1e-5)
    finally:
        env_a.close()

    # Case B (Mirror): +5 kW charging, -7 kW discharging
    schema_b = _build_two_building_feeder_dataset(
        tmp_path,
        name="joint_5c_7d",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[0.0, 0.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=0.0,
        b1_charger_max_charge=5.0,
        b1_charger_min_charge=5.0,
        b2_charger_max_discharge=7.0,
        b2_charger_min_discharge=7.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    env_b = CityLearnEnv(schema_b, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env_b.reset(seed=0)
        with IndependentBalanceAudit(env_b) as audit_b:
            env_b.step([np.array([1.0, -1.0], dtype="float32")])
            proof_b = audit_b.finish(check_kpis=True)
        assert proof_b["status"] == "pass"
        b1, b2 = env_b.buildings
        assert float(b1.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(b2.chargers_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_b.net_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_b.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_b.feeder_curtailed_import_kwh[0]) == pytest.approx(5.0, abs=1e-5)
        assert float(env_b.feeder_curtailed_export_kwh[0]) == pytest.approx(7.0, abs=1e-5)
    finally:
        env_b.close()


def test_twelve_6kw_plus_one_10kw_chargers_and_cutoff_runtime(tmp_path: Path):
    """Point 2 regression:
    10 kW fixed PV, twelve chargers at fixed 6 kW and one at fixed 10 kW, all requesting charge,
    and 1 kW limits both ways -> keeps only the 10 kW charger on (0 kW net, 0 kW violation).
    Also checks correctness and runtime across the 12..24 device cutoffs.
    """
    import time
    from citylearn.internal.electrical_limits import (
        EXACT_DEADBAND_SUBSET_CUTOFF,
        allocate_district_feeder_envelope,
        district_feeder_violation_kw,
    )

    assert EXACT_DEADBAND_SUBSET_CUTOFF == 24

    # Check allocator correctness & runtime across counts around 12, 13, 16, 17, 24, 26, 30 (identical & heterogeneous)
    for n_six in (11, 12, 15, 16, 23, 25, 29):
        # 1) Identical 6 kW chargers + one 10 kW charger (exact for all sizes via multiplicity grouping)
        controls = {f"c_{i}": 6.0 for i in range(n_six)}
        controls["c_ten"] = 10.0
        mins = dict(controls)
        total_before = -10.0 + sum(controls.values())

        t0 = time.perf_counter()
        alloc = allocate_district_feeder_envelope(
            total_before, controls, 1.0, 1.0, controls_min_power_kw=mins
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.15, f"n_six={n_six} identical took {elapsed:.3f}s"
        assert alloc.exact is True
        assert alloc.scales["c_ten"] == pytest.approx(1.0, abs=1e-9)
        assert all(alloc.scales[f"c_{i}"] == pytest.approx(0.0, abs=1e-9) for i in range(n_six))
        assert alloc.projected_power_kw == pytest.approx(0.0, abs=1e-6)
        assert district_feeder_violation_kw(alloc.projected_power_kw, 1.0, 1.0) == 0.0

        # 2) Distinct/heterogeneous ~6.0x kW chargers + one 10 kW charger (exact <= 24, Tier 3 DP > 24)
        het_controls = {f"c_{i}": 6.0 + 0.01 * (i + 1) for i in range(n_six)}
        het_controls["c_ten"] = 10.0
        het_mins = dict(het_controls)
        het_before = -10.0 + sum(het_controls.values())

        t0 = time.perf_counter()
        het_alloc = allocate_district_feeder_envelope(
            het_before, het_controls, 1.0, 1.0, controls_min_power_kw=het_mins
        )
        het_elapsed = time.perf_counter() - t0
        assert het_elapsed < 0.15, f"n_six={n_six} heterogeneous took {het_elapsed:.3f}s"
        assert het_alloc.exact is (n_six + 1 <= 24)
        assert district_feeder_violation_kw(het_alloc.projected_power_kw, 1.0, 1.0) == 0.0
        assert het_alloc.scales["c_ten"] == pytest.approx(1.0, abs=1e-9)

    # 3) 28 distinct fixed-power chargers (> 24 cutoff, all unique -> Tier 3 DP + polish) where {pair_a, pair_b}
    # (6.70 + 8.30 = 15.00 kW) is the unique feasible subset out of 2^28 = 268,435,456 subsets for 15 kW PV
    # and [-1, +1] kW limits, and is not a prefix of any greedy ordering.
    pair_controls = {"c_small": 4.00, "pair_a": 6.70, "pair_b": 8.30}
    for i in range(25):
        pair_controls[f"c_{i}"] = round(12.10 + 0.07 * i, 2)  # 12.10 .. 13.78 kW
    pair_alloc = allocate_district_feeder_envelope(
        -15.0 + sum(pair_controls.values()),
        pair_controls,
        1.0,
        1.0,
        controls_min_power_kw=dict(pair_controls),
    )
    assert pair_alloc.exact is False
    assert pair_alloc.relaxed_violation_kw == 0.0
    assert district_feeder_violation_kw(pair_alloc.projected_power_kw, 1.0, 1.0) == 0.0
    assert pair_alloc.scales["pair_a"] == pytest.approx(1.0, abs=1e-9)
    assert pair_alloc.scales["pair_b"] == pytest.approx(1.0, abs=1e-9)
    assert pair_alloc.projected_power_kw == pytest.approx(0.0, abs=1e-6)

    # End-to-end simulator check with 12 x 6 kW + 1 x 10 kW chargers and 10 kW PV
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="thirteen_chargers",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[10.0, 10.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=0.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0, "export_kw": 1.0}},
    )
    ds_dir = Path(schema["root_directory"])
    schema["electric_vehicles_def"] = {}
    b1_chargers = {}
    for idx in range(13):
        ev_name = f"EV_G_{idx}"
        ch_name = f"ch_g_{idx}"
        power_kw = 10.0 if idx == 12 else 6.0
        pd.DataFrame(
            {
                "electric_vehicle_charger_state": [1, 1],
                "electric_vehicle_id": [ev_name, ev_name],
                "electric_vehicle_departure_time": [1.0, 1.0],
                "electric_vehicle_required_soc_departure": [0.80, 0.80],
                "electric_vehicle_estimated_arrival_time": [np.nan, np.nan],
                "electric_vehicle_estimated_soc_arrival": [0.40, np.nan],
            }
        ).to_csv(ds_dir / f"{ch_name}.csv", index=False)
        schema["electric_vehicles_def"][ev_name] = {
            "include": True,
            "battery": {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": 60.0,
                    "nominal_power": 20.0,
                    "initial_soc": 0.40,
                    "efficiency": 1.0,
                    "loss_coefficient": 0.0,
                    "capacity_loss_coefficient": 0.0,
                    "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                },
            },
        }
        b1_chargers[ch_name] = {
            "type": "citylearn.electric_vehicle_charger.Charger",
            "charger_simulation": f"{ch_name}.csv",
            "autosize": False,
            "attributes": {
                "nominal_power": power_kw,
                "efficiency": 1.0,
                "charger_type": 0,
                "max_charging_power": power_kw,
                "min_charging_power": power_kw,
                "max_discharging_power": 0.0,
                "min_discharging_power": 0.0,
            },
        }
    schema["buildings"]["Building_1"]["chargers"] = b1_chargers
    schema["buildings"]["Building_2"].pop("chargers", None)
    schema["buildings"]["Building_2"]["inactive_observations"] = [
        "electric_vehicle_charger_connected_state",
        "connected_electric_vehicle_at_charger_battery_capacity",
        "connected_electric_vehicle_at_charger_departure_time",
        "connected_electric_vehicle_at_charger_required_soc_departure",
        "connected_electric_vehicle_at_charger_soc",
    ]
    schema["buildings"]["Building_2"]["inactive_actions"] = ["electrical_storage", "electric_vehicle_storage"]

    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            env.step([np.ones(13, dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1 = env.buildings[0]
        by_id = {c.charger_id: float(c.electricity_consumption[0]) for c in b1.electric_vehicle_chargers}
        assert by_id["ch_g_12"] == pytest.approx(10.0, abs=1e-5)
        for idx in range(12):
            assert by_id[f"ch_g_{idx}"] == pytest.approx(0.0, abs=1e-6)
        assert float(env.net_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(env.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()


def test_independent_enumerative_reference_matches_zero_avoidable_violations():
    """Compare allocate_district_feeder_envelope against an independent brute-force 2^N reference
    across 3,000 generated cases (joint charging & discharging, deadbands, floors, 1..12 devices).
    """
    from citylearn.internal.electrical_limits import (
        allocate_district_feeder_envelope,
        district_feeder_violation_kw,
    )

    rng = np.random.default_rng(20261121)

    def _reference_min_violation(base_kw, controls, mins, floors, imp, exp):
        cids = list(controls.keys())
        n = len(cids)
        options = []
        for cid in cids:
            p = controls[cid]
            u = abs(p)
            pm = min(max(mins.get(cid, 0.0), 0.0), u)
            fm = min(max(floors.get(cid, 0.0), 0.0), u)
            lo = max(pm, fm)
            if p > 0:
                iv_on = (lo, u)
            else:
                iv_on = (-u, -lo)
            if fm <= 1e-9 and lo > 1e-9:
                options.append(((0.0, 0.0), iv_on))
            else:
                options.append((iv_on,))

        best_v = float("inf")
        # Recursive exact enumeration of Cartesian product of device intervals
        def _dfs(idx, sum_lo, sum_hi):
            nonlocal best_v
            if idx == n:
                p_lo = base_kw + sum_lo
                p_hi = base_kw + sum_hi
                v = 0.0
                if imp is not None:
                    v = max(v, p_lo - imp)
                if exp is not None:
                    v = max(v, -exp - p_hi)
                if v < best_v:
                    best_v = v
                return
            for lo_i, hi_i in options[idx]:
                _dfs(idx + 1, sum_lo + lo_i, sum_hi + hi_i)

        _dfs(0, 0.0, 0.0)
        return 0.0 if best_v <= 1e-5 else best_v

    avoidable = 0
    for case_idx in range(3000):
        n_dev = 15 if case_idx < 40 else int(rng.integers(1, 11))
        base_kw = float(rng.uniform(-15.0, 15.0))
        controls = {}
        mins = {}
        floors = {}
        for i in range(n_dev):
            cid = f"d{i}"
            sign = 1.0 if rng.random() < 0.6 else -1.0
            u = float(np.round(rng.uniform(1.0, 12.0), 2))
            controls[cid] = sign * u
            mode = rng.random()
            if mode < 0.35:
                mins[cid] = u  # fixed minimum == max
            elif mode < 0.75:
                mins[cid] = float(np.round(rng.uniform(0.5, u), 2))
            else:
                mins[cid] = 0.0
            if rng.random() < 0.15:
                floors[cid] = float(np.round(rng.uniform(0.2, u), 2))

        imp = None if rng.random() < 0.15 else float(np.round(rng.uniform(0.0, 12.0), 2))
        exp = None if rng.random() < 0.15 else float(np.round(rng.uniform(0.0, 12.0), 2))
        total_kw = base_kw + sum(controls.values())

        alloc = allocate_district_feeder_envelope(
            total_kw,
            controls,
            imp,
            exp,
            controls_min_power_kw=mins,
            controls_floor_power_kw=floors,
        )
        # Verify every device scale respects its deadband and floor
        for cid, s in alloc.scales.items():
            assert -1e-9 <= s <= 1.0 + 1e-9
            u = abs(controls[cid])
            applied_mag = s * u
            fl = floors.get(cid, 0.0)
            mn = mins.get(cid, 0.0)
            if fl > 1e-9:
                assert applied_mag + 1e-5 >= max(fl, mn)
            elif applied_mag > 1e-6:
                assert applied_mag + 1e-5 >= mn

        actual_v = district_feeder_violation_kw(alloc.projected_power_kw, imp, exp)
        ref_v = _reference_min_violation(base_kw, controls, mins, floors, imp, exp)
        if actual_v > ref_v + 1e-4:
            avoidable += 1

    assert avoidable == 0


def test_preserves_building_and_per_phase_limits_during_district_curtailment(tmp_path: Path):
    """Point 3 regression:
    Building A has 5 kW PV, 5 kW battery charge request, 1 kW total export cap and 0.34 kW
    per-phase export caps. Building B has 4 kW passive load and requests 4 kW battery charging.
    With a 4 kW district import cap, district curtailment meets the 4 kW district cap while
    preserving Building A's 1 kW total and 0.34 kW per-phase export caps (0 violations).
    Also tests the unavoidable combined-scope case where preserving Building A's export cap
    leaves a district import residual that is recorded while simulation continues.
    """
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="combined_building_phase_district",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[4.0, 4.0],
        solar_b1=[5.0, 5.0],
        solar_b2=[0.0, 0.0],
        b1_battery_power=5.0,
        b2_battery_power=4.0,
        b1_battery_efficiency=1.0,
        b2_battery_efficiency=1.0,
        ev_connected=False,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 4.0}},
    )
    schema["buildings"]["Building_1"]["charging_constraints"] = {"enabled": True}
    schema["buildings"]["Building_1"]["electrical_service"] = {
        "mode": "three_phase",
        "default_split": "balanced",
        "limits": {
            "total": {"import_kw": 20.0, "export_kw": 1.0},
            "per_phase": {
                "L1": {"import_kw": 10.0, "export_kw": 0.34},
                "L2": {"import_kw": 10.0, "export_kw": 0.34},
                "L3": {"import_kw": 10.0, "export_kw": 0.34},
            },
        },
    }
    schema["buildings"]["Building_1"]["electrical_storage"]["attributes"]["phase_connection"] = "all_phases"

    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            # Action order: [b1_storage (+5 kW), b1_ev (0), b2_storage (+4 kW), b2_ev (0)]
            env.step([np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1, b2 = env.buildings
        b1_net = float(b1.net_electricity_consumption[0])
        assert -b1_net <= 1.0 + 1e-5
        for ph in ("L1", "L2", "L3"):
            ph_kw = float(b1._charging_phase_power_history_kw[ph][0])
            assert -ph_kw <= 0.34 + 1e-5
        assert float(env.net_electricity_consumption[0]) == pytest.approx(4.0, abs=1e-4)
        assert float(env.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)

        kpis = env.evaluate_v2(include_business_as_usual=False)
        lookup = {(r.level, r.name, r.cost_function): r.value for r in kpis.itertuples()}
        assert lookup[("building", "Building_1", "building_electrical_service_phase_violations_energy_total_kwh")] == pytest.approx(
            0.0, abs=1e-6
        )
        assert lookup[("district", "District", "district_feeder_max_violation_kw")] == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()

    # Unavoidable combined-scope case: Building B has 6 kW passive load and 2 kW district import cap.
    # Building A cannot export more than 1 kW (battery >= 4 kW -> net >= -1 kW), so district net is +5 kW
    # and records a 3 kW district violation while preserving Building A's limits.
    schema_unavoid = deepcopy(schema)
    schema_unavoid["district_electrical_service"] = {"enabled": True, "limits": {"import_kw": 2.0}}
    pd.DataFrame(
        {
            **pd.read_csv(Path(schema["root_directory"]) / "Building_2.csv").to_dict("list"),
            "non_shiftable_load": [6.0, 6.0],
        }
    ).to_csv(Path(schema["root_directory"]) / "Building_2.csv", index=False)

    env_u = CityLearnEnv(schema_unavoid, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env_u.reset(seed=0)
        env_u.step([np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")])
        b1_u, b2_u = env_u.buildings
        assert -float(b1_u.net_electricity_consumption[0]) <= 1.0 + 1e-5
        assert float(b2_u.electrical_storage.electricity_consumption[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(env_u.net_electricity_consumption[0]) == pytest.approx(5.0, abs=1e-3)
        assert float(env_u.feeder_limit_violation_kw[0]) == pytest.approx(3.0, abs=1e-3)
        kpis_u = env_u.evaluate_v2(include_business_as_usual=False)
        lookup_u = {(r.level, r.name, r.cost_function): r.value for r in kpis_u.itertuples()}
        assert lookup_u[("building", "Building_1", "building_electrical_service_phase_violations_energy_total_kwh")] == pytest.approx(
            0.0, abs=1e-6
        )
    finally:
        env_u.close()


def test_dataset_anchored_schedules_nonzero_episode_start_and_empty_reset_histories(tmp_path: Path):
    """Point 4 regression:
    1) Episode starting at dataset step 3 with 4 rows (steps 3..6) rejects short schedule [5, 4, 3, 2, 1].
    2) Valid dataset-anchored schedule [9, 8, 7, 5, 4, 3, 2] applies [5, 4, 3] at steps 3, 4, 5.
    3) After reset(), public feeder histories are empty (len == 0) even when initial passive load exceeds the cap.
    """
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="nonzero_start_schedule",
        rows=7,
        base_loads_b1=[4.0] * 7,
        base_loads_b2=[4.0] * 7,  # 8 kW passive load > 1 kW cap before any action!
    )
    schema["simulation_start_time_step"] = 3
    schema["simulation_end_time_step"] = 6
    schema["episode_time_steps"] = 4

    # Short schedule [5, 4, 3, 2, 1] (length 5 < required 6 for executed steps 3, 4, 5) must fail on construction
    with pytest.raises(ValueError, match="sequence has length 5"):
        CityLearnEnv(
            deepcopy(schema),
            offline=True,
            render_mode="none",
            district_electrical_service={"enabled": True, "limits": {"import_kw": [5.0, 4.0, 3.0, 2.0, 1.0]}},
        )

    # Check empty feeder histories after reset() even when passive load (8 kW) exceeds import_kw (1 kW)
    env_passive = CityLearnEnv(
        deepcopy(schema),
        offline=True,
        render_mode="none",
        district_electrical_service={"enabled": True, "limits": {"import_kw": 1.0}},
    )
    try:
        env_passive.reset(seed=0)
        assert len(env_passive.feeder_curtailed_import_kwh) == 0
        assert len(env_passive.feeder_curtailed_export_kwh) == 0
        assert len(env_passive.feeder_limit_violation_kw) == 0

        env_passive.step([np.zeros(4, dtype="float32")])
        assert env_passive.feeder_limit_violation_kw == pytest.approx([7.0], abs=1e-5)
    finally:
        env_passive.close()

    # Valid dataset-indexed schedule [9, 8, 7, 5, 4, 3, 2] with low passive load (1 kW total) and full charging request
    schema_sched = _build_two_building_feeder_dataset(
        tmp_path,
        name="valid_dataset_anchored",
        rows=7,
        base_loads_b1=[0.5] * 7,
        base_loads_b2=[0.5] * 7,
    )
    schema_sched["simulation_start_time_step"] = 3
    schema_sched["simulation_end_time_step"] = 6
    schema_sched["episode_time_steps"] = 4

    env = CityLearnEnv(
        schema_sched,
        offline=True,
        render_mode="none",
        physics_invariant_checks=True,
        district_electrical_service={"enabled": True, "limits": {"import_kw": [9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0]}},
    )
    try:
        env.reset(seed=0)
        assert len(env.feeder_limit_violation_kw) == 0
        for _ in range(3):
            env.step([np.ones(4, dtype="float32")])
        assert np.asarray(env.net_electricity_consumption, dtype="float64") == pytest.approx([5.0, 4.0, 3.0], abs=1e-4)
        assert np.allclose(env.feeder_limit_violation_kw, 0.0, atol=1e-6)
    finally:
        env.close()


def test_effective_schema_physical_hash_and_compare_manifests_distinguish_feeder_overrides(tmp_path: Path):
    """Point 5 regression:
    Two exported runs of the same schema file with constructor import caps of 2 kW and 5 kW
    must record the effective normalized configuration in effective_schema, produce distinct
    physical_configuration_sha256 digests, and return 'incompatible' from compare_manifests()
    and compare_exports(), while runs with equivalent normalized limits pair cleanly.
    """
    import json
    from citylearn.internal.comparison import compare_exports, compare_manifests

    schema_dict = _build_two_building_feeder_dataset(
        tmp_path,
        name="manifest_schema",
        rows=3,
        base_loads_b1=[0.5, 0.5, 0.5],
        base_loads_b2=[0.5, 0.5, 0.5],
    )
    schema_path = tmp_path / "manifest_schema" / "schema.json"
    schema_path.write_text(json.dumps(schema_dict), encoding="utf-8")

    def _run_and_export(out_dir: Path, feeder_cfg: dict, controller_name: str) -> Path:
        env = CityLearnEnv(
            str(schema_path),
            offline=True,
            render_mode="none",
            render_directory=out_dir,
            export_kpis_on_episode_end=False,
            random_seed=0,
            district_electrical_service=feeder_cfg,
        )
        try:
            env.reset(seed=0)
            env.set_experiment_identity(controller_name, "v1", {"charge": 1.0})
            while not env.terminated:
                env.step([np.ones(4, dtype="float32")])
            env.export_final_kpis(filepath="kpis.csv")
            return Path(env.new_folder_path) / "kpis.csv", np.asarray(env.net_electricity_consumption, dtype="float64")
        finally:
            env.close()

    kpi_2kw, net_2kw = _run_and_export(
        tmp_path / "exp_2kw",
        {"enabled": True, "limits": {"import_kw": 2.0}},
        "ctrl-2kw",
    )
    kpi_5kw, net_5kw = _run_and_export(
        tmp_path / "exp_5kw",
        {"enabled": True, "limits": {"import_kw": 5.0}},
        "ctrl-5kw",
    )
    kpi_2kw_alt, _ = _run_and_export(
        tmp_path / "exp_2kw_alt",
        {"enabled": "true", "limits": {"total": {"import_kw": 2, "export_kw": float("inf")}}},
        "ctrl-2kw-alt",
    )

    assert net_2kw == pytest.approx([2.0, 2.0], abs=1e-4)
    assert net_5kw == pytest.approx([5.0, 5.0], abs=1e-4)

    m_2kw = json.loads(kpi_2kw.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    m_5kw = json.loads(kpi_5kw.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    m_2kw_alt = json.loads(kpi_2kw_alt.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert m_2kw_alt["comparison_context"]["physical_configuration_sha256"] == (
        m_2kw["comparison_context"]["physical_configuration_sha256"]
    )

    assert m_2kw["effective_schema"]["district_electrical_service"] == {
        "enabled": True,
        "limits": {"import_kw": 2.0, "export_kw": None},
    }
    assert m_5kw["effective_schema"]["district_electrical_service"] == {
        "enabled": True,
        "limits": {"import_kw": 5.0, "export_kw": None},
    }
    assert m_2kw["comparison_context"]["physical_configuration_sha256"] != (
        m_5kw["comparison_context"]["physical_configuration_sha256"]
    )

    diff_report = compare_manifests(m_2kw, m_5kw)
    assert diff_report["status"] == "incompatible"
    assert "physical_configuration_sha256" in diff_report["mismatches"]
    assert "district_electrical_service" in diff_report["mismatches"]
    assert compare_exports(kpi_2kw, kpi_5kw)["status"] == "incompatible"

    # Equivalent normalized configurations pair cleanly
    paired_report = compare_exports(kpi_2kw, kpi_2kw_alt)
    assert paired_report["status"] == "paired", paired_report


def test_seventeen_heterogeneous_chargers_zero_limit_pv_absorption(tmp_path: Path):
    """Maintainer 17-charger large-fleet simulator reproducer:
    17 chargers with fixed min==max charging power below, EV batteries with sufficient headroom
    and power, zero passive load, 91.215 kW fixed PV, and 0 kW import and export limits.
    Running 1-based chargers [2, 5, 6, 9, 12, 13, 14, 17] absorbs 91.215 kW exactly with 0 kW
    feeder violation. Also verifies exact vs relaxation certificates on DistrictFeederAllocation.
    """
    from citylearn.internal.electrical_limits import (
        allocate_district_feeder_envelope,
        district_feeder_violation_kw,
    )

    powers = [
        17.923, 11.115, 16.484, 13.677, 7.607, 20.507, 20.605, 17.084, 9.610,
        11.552, 11.484, 7.976, 11.590, 11.761, 9.540, 12.925, 11.049,
    ]
    expected_one_based = {2, 5, 6, 9, 12, 13, 14, 17}
    assert sum(powers[i - 1] for i in expected_one_based) == pytest.approx(91.215, abs=1e-9)

    # 1) Direct allocator verification + exactness certificate
    controls = {f"ch_{i + 1}": p for i, p in enumerate(powers)}
    alloc = allocate_district_feeder_envelope(
        -91.215 + sum(powers),
        controls,
        0.0,
        0.0,
        controls_min_power_kw=dict(controls),
    )
    assert alloc.exact is True
    assert alloc.relaxed_violation_kw == 0.0
    assert alloc.projected_power_kw == pytest.approx(0.0, abs=1e-9)
    assert district_feeder_violation_kw(alloc.projected_power_kw, 0.0, 0.0) == 0.0
    active_one_based = {i + 1 for i in range(17) if alloc.scales[f"ch_{i + 1}"] > 0.5}
    assert active_one_based == expected_one_based

    # Continuous-relaxation certificate when PV exceeds total controllable demand
    unavoid_cont = allocate_district_feeder_envelope(
        -250.0 + sum(powers),
        controls,
        0.0,
        0.0,
        controls_min_power_kw=dict(controls),
    )
    assert unavoid_cont.exact is True
    assert unavoid_cont.relaxed_violation_kw == pytest.approx(250.0 - sum(powers), abs=1e-6)

    # Discrete-deadband unavoidable violation (continuous hull [-3, +4] covers 0, discrete {-3, +4} misses [-1, +1])
    unavoid_disc = allocate_district_feeder_envelope(
        4.0,
        {"ev": 7.0},
        1.0,
        1.0,
        controls_min_power_kw={"ev": 7.0},
    )
    assert unavoid_disc.exact is True
    assert unavoid_disc.relaxed_violation_kw == 0.0
    assert district_feeder_violation_kw(unavoid_disc.projected_power_kw, 1.0, 1.0) == pytest.approx(2.0, abs=1e-6)

    # 2) Full end-to-end CityLearnEnv simulator verification
    schema = _build_two_building_feeder_dataset(
        tmp_path,
        name="seventeen_chargers_pv",
        rows=2,
        base_loads_b1=[0.0, 0.0],
        base_loads_b2=[0.0, 0.0],
        solar_b1=[91.215, 91.215],
        solar_b2=[0.0, 0.0],
        b1_battery_power=0.0,
        b2_battery_power=0.0,
        district_electrical_service={"enabled": True, "limits": {"import_kw": 0.0, "export_kw": 0.0}},
    )
    ds_dir = Path(schema["root_directory"])
    schema["electric_vehicles_def"] = {}
    b1_chargers = {}
    for idx, power_kw in enumerate(powers, start=1):
        ev_name = f"EV_17_{idx}"
        ch_name = f"ch_17_{idx}"
        pd.DataFrame(
            {
                "electric_vehicle_charger_state": [1, 1],
                "electric_vehicle_id": [ev_name, ev_name],
                "electric_vehicle_departure_time": [1.0, 1.0],
                "electric_vehicle_required_soc_departure": [0.90, 0.90],
                "electric_vehicle_estimated_arrival_time": [np.nan, np.nan],
                "electric_vehicle_estimated_soc_arrival": [0.20, np.nan],
            }
        ).to_csv(ds_dir / f"{ch_name}.csv", index=False)
        schema["electric_vehicles_def"][ev_name] = {
            "include": True,
            "battery": {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": 80.0,
                    "nominal_power": 30.0,
                    "initial_soc": 0.20,
                    "efficiency": 1.0,
                    "loss_coefficient": 0.0,
                    "capacity_loss_coefficient": 0.0,
                    "power_efficiency_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "capacity_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                    "discharge_power_curve": [[0.0, 1.0], [1.0, 1.0]],
                },
            },
        }
        b1_chargers[ch_name] = {
            "type": "citylearn.electric_vehicle_charger.Charger",
            "charger_simulation": f"{ch_name}.csv",
            "autosize": False,
            "attributes": {
                "nominal_power": power_kw,
                "efficiency": 1.0,
                "charger_type": 0,
                "max_charging_power": power_kw,
                "min_charging_power": power_kw,
                "max_discharging_power": 0.0,
                "min_discharging_power": 0.0,
            },
        }
    schema["buildings"]["Building_1"]["chargers"] = b1_chargers
    schema["buildings"]["Building_2"].pop("chargers", None)
    schema["buildings"]["Building_2"]["inactive_observations"] = [
        "electric_vehicle_charger_connected_state",
        "connected_electric_vehicle_at_charger_battery_capacity",
        "connected_electric_vehicle_at_charger_departure_time",
        "connected_electric_vehicle_at_charger_required_soc_departure",
        "connected_electric_vehicle_at_charger_soc",
    ]
    schema["buildings"]["Building_2"]["inactive_actions"] = ["electrical_storage", "electric_vehicle_storage"]

    env = CityLearnEnv(schema, offline=True, render_mode="none", physics_invariant_checks=True)
    try:
        env.reset(seed=0)
        with IndependentBalanceAudit(env) as balance_audit:
            env.step([np.ones(17, dtype="float32")])
            proof = balance_audit.finish(check_kpis=True)
        assert proof["status"] == "pass"

        b1 = env.buildings[0]
        by_id = {c.charger_id: float(c.electricity_consumption[0]) for c in b1.electric_vehicle_chargers}
        for idx, power_kw in enumerate(powers, start=1):
            expected_kw = power_kw if idx in expected_one_based else 0.0
            assert by_id[f"ch_17_{idx}"] == pytest.approx(expected_kw, abs=1e-5)
        assert float(env.net_electricity_consumption[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(env.feeder_limit_violation_kw[0]) == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()


