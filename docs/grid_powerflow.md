# Distribution-Grid Power Flow

CityLearn can run an AC power flow on a SimBench low-voltage network at every control step. The building (and EV charger) power of the step becomes the load of the network. The result is bus voltages, line and transformer loading and losses. The feature is optional: a schema without a `grid` block never imports `pandapower`, `simbench` or `lightsim2grid` and every simulated quantity is bit-identical to a run without the feature.

The power flow reads the settled building state. It does not change building dispatch, rewards, costs or the existing KPIs. Use the observations to give an agent the grid state of the previous step.

## Installation

```console
pip install "citylearn[grid]"
```

The extra installs `pandapower`, `simbench` and `lightsim2grid`. `lightsim2grid` 1.3 or newer is required for the `lightsim2grid` backend.

## Schema

Add a top-level `grid` block to `schema.json`.

```json
"grid": {
  "simbench_code": "1-LV-rural1--0-sw",
  "backend": "lightsim2grid",
  "mode": "ac",
  "observations": ["voltage_min_pu", "line_loading_max_percent"]
}
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` keeps the block in the schema without activating it. |
| `simbench_code` | none | SimBench grid code. Use a low-voltage code such as `1-LV-rural1--0-sw`. One of `simbench_code` or `network_file` is required. |
| `network_file` | none | Path to a pandapower JSON network. Loads the file instead of a SimBench grid. |
| `backend` | `lightsim2grid` | `lightsim2grid` solves with a lightsim2grid `LSGrid` that is updated in place and warm started. `pandapower_lightsim2grid` calls `pandapower.runpp(lightsim2grid=True)`. `pandapower` calls `pandapower.runpp`. |
| `mode` | `ac` | `ac` solves every step. `linear` updates a sensitivity model. |
| `ac_every_n_steps` | `0` | In `linear` mode, run a full AC solve every N steps and re-anchor the linear model on it. `0` never re-anchors. The first step after `reset()` is an AC step when the value is above 0. |
| `building_loads` | none | Mapping from building name to the index of a network load. Unlisted buildings take the in-service loads in order and wrap around. Buildings that share a load add their power. |
| `charger_loads` | none | Mapping from charger id to a network load. A charger without an entry sits on the load of its building. |
| `load_scale` | `1.0` | Factor applied to all building and charger power before it enters the network. |
| `power_factor` | `0.98` | Constant power factor that sets the reactive power of every load, `q = p * tan(acos(pf))`, for import and export. |
| `unmapped_elements` | `zero` | `zero` sets the base power of all network loads and static generators to 0 before CityLearn writes its loads. `keep` leaves the SimBench base values. |
| `voltage_limits_pu` | `[0.9, 1.1]` | Limits of the voltage KPIs (EN 50160 style). |
| `overload_limit_percent` | `100.0` | Loading above this value counts as overloading. |
| `voltage_buses` | `lv` | `lv` monitors buses with `vn_kv <= 1.0`. `all` monitors every bus. |
| `linearization_load_kw` | `2.0` | Per-load active power of the point where the sensitivities are computed. |
| `observations` | `[]` | Grid values appended to the observation vector. See below. |
| `validate_backend` | `true` | Compare the `lightsim2grid` backend with `pandapower` on the reference point and raise on a difference above 1e-4 pu. |
| `max_iteration`, `tolerance` | `30`, `1e-8` | Newton-Raphson settings of the `lightsim2grid` backend. |

Dynamic topology is not supported with a `grid` block.

## Mapping

CityLearn computes the net power of building `b` at step `t` as `net_electricity_consumption[t] / dt_hours` in kW, converts it to MW and adds it to the mapped network load. For a building with chargers, the charger power moves from the building load to the charger load. Positive values are import. The network is a balanced single-phase equivalent, so phase unbalance is not modelled.

## Observations

`observations` appends values to the flat observation vector, after the building observations. The values are those of the most recent solved step. Before the first step they are the no-load solution.

| Name | Appended values |
|---|---|
| `voltage_min_pu`, `voltage_max_pu` | Lowest and highest monitored bus voltage. |
| `line_loading_max_percent` | Highest line loading. |
| `transformer_loading_max_percent` | Highest transformer loading. |
| `losses_kw` | Line and transformer active losses. |
| `building_voltage_pu` | One value per building, the voltage at its load bus. |

Grid observations need `central_agent=True` and `interface='flat'`.

## KPIs

`env.grid_kpis()` returns a data frame with the columns of `evaluate_v2()` (`cost_function`, `value`, `name`, `level`, `unit`, `description`) for the steps simulated so far. The rows are not part of `evaluate_v2()`, so the KPI catalogue and the existing KPI tables stay unchanged.

| KPI | Definition |
|---|---|
| `grid_voltage_violation_time_total_hours` | Time with any monitored bus outside the voltage limits. |
| `grid_voltage_undervoltage_time_total_hours`, `grid_voltage_overvoltage_time_total_hours` | The same, split by direction. |
| `grid_voltage_violation_bus_time_total_hours` | Bus-hours outside the limits. |
| `grid_voltage_minimum_pu`, `grid_voltage_maximum_pu` | Extreme monitored bus voltages. |
| `grid_overload_time_total_hours` | Time with any line or transformer above the loading limit. |
| `grid_overload_energy_total_kvah` | Apparent energy above the loading limit, summed over lines and transformers. |
| `grid_loading_peak_line_percent`, `grid_loading_peak_transformer_percent` | Peak loadings. |
| `grid_losses_total_kwh` | Line and transformer active losses. |
| `grid_losses_ratio_to_import_ratio` | Losses divided by the imported energy of the mapped loads. |

The per-step series are in `env.grid.series` (arrays indexed by time step) and the latest state is in `env.grid.last`.

## Linear mode

At construction the service solves the AC power flow with every load at `linearization_load_kw` and computes the sensitivities of bus voltages, line loading, transformer loading and losses to each load by finite differences. Each step then evaluates `y = y_anchor + S (p - p_anchor)` with one matrix-vector product. The anchor is the no-load solution until the first AC step. With `ac_every_n_steps > 0` the anchor becomes the latest AC solution. The sensitivities stay fixed.

The model is accurate near the anchor. Loading is the magnitude of a current, so it is not linear near zero load or across a sign change of the flow. Check the error on your dataset with the benchmark script.

## Benchmark and error

```console
python scripts/manual/bench_grid_powerflow.py --steps 720 --load-scale 2.0
```

The script runs the same seeded random actions for each configuration and prints the milliseconds per step and the maximum absolute error of every configuration against the `pandapower` AC run.

Reference run: `citylearn_challenge_2022_phase_all_plus_evs` (17 buildings), SimBench `1-LV-rural1--0-sw` (13 loads), 719 steps, `--load-scale 1.0`, one shared Windows machine under load. Treat the times as orders of magnitude.

| Configuration | Grid time per step | Max abs error vs `pandapower` AC: voltage (pu) / line loading (%) / transformer loading (%) / losses (kW) |
|---|---|---|
| off | 0 ms | not applicable |
| `pandapower` AC | about 70 to 95 ms | reference |
| `pandapower_lightsim2grid` AC | about 87 ms | 0 |
| `lightsim2grid` AC | 1.2 ms | 2e-7 / 1e-4 / 4e-4 / 5e-6 |
| `linear` | 0.6 ms | 5e-4 / 0.06 / 0.68 / 1.2 |
| `linear`, `ac_every_n_steps: 24` | 2.4 ms | 5e-4 / 0.07 / 0.75 / 1.2 |

The `pandapower_lightsim2grid` backend still pays pandapower's conversion and result extraction on every call, which is why only the direct `lightsim2grid` backend is fast enough for RL. The `pandapower` row of this run let pandapower select lightsim2grid itself (`lightsim2grid="auto"`); the code now passes `lightsim2grid=False` for that backend, so rerun the script for a clean figure.

## Limits

- The network is a balanced single-phase equivalent.
- The `lightsim2grid` backend ignores pandapower switches. `validate_backend` compares it with `pandapower` at the reference point.
- Building power is mapped to loads by assignment, not by the position of the building in a real feeder. Set `building_loads` when the position matters.
- The KPIs are not part of the KPI catalogue yet.
