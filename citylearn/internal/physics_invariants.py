from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from citylearn.energy_model import Battery, HeatPump
from citylearn.internal.physics_version import physics_version_of
from citylearn.internal.units import power_kw_to_energy_kwh

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class PhysicsInvariantError(AssertionError):
    """A hard physics invariant was violated.

    Subclasses ``AssertionError`` so existing handlers keep working, but is raised
    explicitly, so ``python -O`` does not disable the checks.
    """


class CityLearnPhysicsInvariantService:
    """Runtime physics/assertion checks evaluated per control step."""

    SOC_EPS = 1.0e-4
    ENERGY_EPS = 5.0e-4
    # Share of the compared magnitudes added to ENERGY_EPS: series are float32 (relative precision 1.2e-7),
    # so the absolute tolerance alone is too tight for buildings that move thousands of kWh per step.
    RELATIVE_EPS = 1.0e-6

    def __init__(self, env: "CityLearnEnv"):
        self.env = env
        self._battery_energy_before_actions = None
        # Largest violation of each identity so far, in kWh (SOC units for the battery reserve); at most
        # the tolerance when the checks pass. Read by the bundled-data audit.
        self.max_residuals = {}

    def capture_battery_energy_before_actions(self, time_step: int):
        """Snapshot actual SOC after arrivals/losses and before control actions.

        Do not read ``energy_init``: the independent check must detect a model
        that silently replaces measured energy with its operational reserve.
        """
        batteries = [
            (f'{building.name}.electrical_storage', building.electrical_storage)
            for building in self.env.buildings
        ]
        batteries.extend((f'{ev.name}.battery', ev.battery)
                         for ev in self.env.electric_vehicles)
        seen = set()
        snapshot = []
        for name, battery in batteries:
            if battery is None or id(battery) in seen:
                continue
            seen.add(id(battery))
            energy = float(battery.soc[time_step]) * float(battery.capacity)
            snapshot.append((name, battery, energy, battery.stored_energy_transfer))
        self._battery_energy_before_actions = (int(time_step), snapshot)

    def _assert_battery_energy_changes(self, time_step: int):
        snapshot = self._battery_energy_before_actions
        self._battery_energy_before_actions = None
        if snapshot is None or snapshot[0] != time_step:
            return
        for name, battery, energy_before, transfer_before in snapshot[1]:
            energy_after = float(battery.soc[time_step]) * float(battery.capacity)
            expected_change = battery.stored_energy_transfer - transfer_before
            self._require(abs(energy_after - energy_before - expected_change) <= self.ENERGY_EPS, (
                f"Stored battery energy mismatch for '{name}' at t={time_step}: "
                f"before={energy_before:.6f} kWh, after={energy_after:.6f} kWh, "
                f"expected_change={expected_change:.6f} kWh."
            ))

    @staticmethod
    def _require(condition: bool, message: str):
        if not condition:
            raise PhysicsInvariantError(message)

    def _require_at_most(self, identity: str, excess: float, tolerance: float, message: str):
        """Record ``excess`` (left side minus right side) of ``identity`` and raise when it exceeds ``tolerance``."""

        if excess > self.max_residuals.get(identity, -np.inf):
            self.max_residuals[identity] = excess

        if excess > tolerance:
            raise PhysicsInvariantError(f'{message} Identity: {identity}.')

    @classmethod
    def _tolerance(cls, *magnitudes: float) -> float:
        """Absolute tolerance in kWh for a comparison of values of the given magnitudes."""

        return cls.ENERGY_EPS + cls.RELATIVE_EPS*max((abs(value) for value in magnitudes), default=0.0)

    @staticmethod
    def _state(values, idx: int, name: str, default: float = 0.0) -> float:
        """Read a simulated state value; a NaN or inf value is a violation, not zero."""

        try:
            value = values[idx]
        except (IndexError, KeyError, TypeError):
            return float(default)

        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            raise PhysicsInvariantError(f'Non-finite {name}: {scalar!r}.')

        return scalar

    @staticmethod
    def _to_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    @classmethod
    def _safe_index(cls, values, idx: int, default: float = 0.0) -> float:
        """Read a state value at ``idx``; missing values default, non-finite values are violations."""

        return cls._state(values, idx, 'value', default)

    def assert_step_invariants(self, time_step: int):
        """Raise ``PhysicsInvariantError`` (an ``AssertionError``) if hard physics invariants are violated."""

        t = int(max(time_step, 0))
        buildings = list(self.env.buildings)
        topology = getattr(self.env, "_topology_service", None)
        if (
            getattr(self.env, "topology_mode", "static") == "dynamic"
            and topology is not None
            and hasattr(topology, "active_member_ids_at")
        ):
            active_member_ids = set(topology.active_member_ids_at(t))
            buildings = [
                building
                for building in buildings
                if str(getattr(building, "name", "")) in active_member_ids
            ]

        for building in buildings:
            for check in (
                self._assert_building_energy_balance,
                self._assert_outage_local_balance,
                self._assert_outage_no_storage_spill,
                self._assert_thermal_energy_identity,
                self._assert_served_within_demand,
                self._assert_battery_reserve,
                self._assert_soc_bounds,
                self._assert_storage_power_bounds,
                self._assert_charger_power_bounds,
            ):
                try:
                    check(building, t)
                except PhysicsInvariantError as error:
                    if str(error).startswith('Non-finite'):
                        raise PhysicsInvariantError(
                            f"{error} Found by {check.__name__.lstrip('_')} for '{building.name}' at t={t}."
                        ) from None
                    raise
        self._assert_battery_energy_changes(t)

    def _assert_building_energy_balance(self, building, t: int):
        if bool(getattr(building, "power_outage", False)):
            return

        net = self._safe_index(building.net_electricity_consumption, t, 0.0)
        terms = (
            self._safe_index(building.cooling_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.heating_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.dhw_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.non_shiftable_load_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.electrical_storage.electricity_consumption, t, 0.0)
            + self._safe_index(building.solar_generation, t, 0.0)
            + self._safe_index(building.chargers_electricity_consumption, t, 0.0)
            + self._safe_index(building.deferrable_appliances_electricity_consumption, t, 0.0)
            + self._safe_index(getattr(building, 'escalators_electricity_consumption', []), t, 0.0)
        )

        self._require(abs(net - terms) <= self.ENERGY_EPS, (
            f"Net energy balance mismatch for '{building.name}' at t={t}: "
            f"net={net:.6f} kWh, terms={terms:.6f} kWh."
        ))

    def _assert_outage_local_balance(self, building, t: int):
        outage_signal = self._safe_index(getattr(building, "power_outage_signal", []), t, 0.0)
        if not bool(getattr(building, "simulate_power_outage", False)) or outage_signal < 1.0:
            return

        solar_generation = self._safe_index(building.solar_generation, t, 0.0)
        storage_energy = self._safe_index(building.electrical_storage.electricity_consumption, t, 0.0)
        charger_energy = self._safe_index(building.chargers_electricity_consumption, t, 0.0)

        local_sources = (
            max(-solar_generation, 0.0)
            + max(-storage_energy, 0.0)
            + max(-charger_energy, 0.0)
        )
        local_loads = (
            self._safe_index(building.cooling_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.heating_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.dhw_device.electricity_consumption, t, 0.0)
            + self._safe_index(building.non_shiftable_load_device.electricity_consumption, t, 0.0)
            + max(storage_energy, 0.0)
            + max(charger_energy, 0.0)
            + self._safe_index(building.deferrable_appliances_electricity_consumption, t, 0.0)
            + self._safe_index(getattr(building, 'escalators_electricity_consumption', []), t, 0.0)
        )

        self._require(local_loads <= local_sources + self.ENERGY_EPS, (
            f"Outage local energy balance violated for '{building.name}' at t={t}: "
            f"loads={local_loads:.6f} kWh, local_sources={local_sources:.6f} kWh."
        ))

        if physics_version_of(building) >= 2:
            # Storage discharges only into the islanded load that PV leaves (#229); physics version 1
            # can discharge more and books the surplus as PV curtailment.
            storage_discharge = max(-storage_energy, 0.0) + max(-charger_energy, 0.0)
            uncovered_load = max(local_loads - max(-solar_generation, 0.0), 0.0)
            self._require(storage_discharge <= uncovered_load + self.ENERGY_EPS, (
                f"Outage storage discharge exceeds the load not covered by PV for '{building.name}' at t={t}: "
                f"discharge={storage_discharge:.6f} kWh, uncovered_load={uncovered_load:.6f} kWh."
            ))

    @staticmethod
    def _thermal_end_uses(building):
        """Yield ``(name, device, storage, demand, dataset_output, heating)`` per thermal end use."""

        yield ('cooling', building.cooling_device, building.cooling_storage, building.cooling_demand,
               building.energy_from_cooling_device, False)
        yield ('heating', building.heating_device, building.heating_storage, building.heating_demand,
               building.energy_from_heating_device, True)
        yield ('dhw', building.dhw_device, building.dhw_storage, building.dhw_demand,
               building.energy_from_dhw_device, True)

    @staticmethod
    def _cop(device, building, t: int, heating: bool) -> float:
        """Coefficient of performance of a heat pump, or the efficiency of an electric heater."""

        if isinstance(device, HeatPump):
            return float(device.get_cop(float(building.weather.outdoor_dry_bulb_temperature[t]), heating))

        return float(device.efficiency)

    def _assert_thermal_energy_identity(self, building, t: int):
        """Electricity x COP (or efficiency) equals the device output to the building plus the storage charge.

        The two sides come from different records: the electricity consumption of the device, and the thermal
        energy it delivered (`energy_from_*_device`) and stored (`energy_balance`). A wrong value in a device
        model breaks the identity even when the net balance still closes. An outage can only clip the
        electricity of a device, so the identity is an upper bound then.
        """

        outage = bool(getattr(building, 'power_outage', False))

        for name, device, storage, _, dataset_output, heating in self._thermal_end_uses(building):
            electricity = self._state(device.electricity_consumption, t, f'{name} electricity consumption')
            output = building._dataset_energy_to_control_step(self._state(dataset_output, t, f'{name} device output'))
            charge = max(self._state(storage.energy_balance, t, f'{name} storage energy balance'), 0.0)
            heat = electricity*self._cop(device, building, t, heating)
            expected = max(output, 0.0) + charge
            tolerance = self._tolerance(heat, expected)
            message = (
                f"Thermal energy identity violated for '{building.name}' at t={t} ({name}): "
                f"electricity*COP={heat:.6f} kWh, device_output+storage_charge={expected:.6f} kWh "
                f"(electricity={electricity:.6f} kWh, device_output={output:.6f} kWh, storage_charge={charge:.6f} kWh)."
            )
            self._require_at_most(f'{name}_heat_le_output_plus_charge', heat - expected, tolerance, message)

            if not outage:
                self._require_at_most(f'{name}_heat_ge_output_plus_charge', expected - heat, tolerance, message)

    def _assert_served_within_demand(self, building, t: int):
        """Heat delivered to the building (device output plus storage discharge) and the served
        non-shiftable load never exceed the demand."""

        for name, device, storage, demand_series, _, heating in self._thermal_end_uses(building):
            demand = building._dataset_energy_to_control_step(self._state(demand_series, t, f'{name} demand'))
            balance = self._state(storage.energy_balance, t, f'{name} storage energy balance')
            electricity = self._state(device.electricity_consumption, t, f'{name} electricity consumption')
            served = electricity*self._cop(device, building, t, heating) - max(balance, 0.0) + max(-balance, 0.0)
            self._require_at_most(f'{name}_served_le_demand', served - demand, self._tolerance(served, demand), (
                f"Served {name} energy exceeds the demand for '{building.name}' at t={t}: "
                f"served={served:.6f} kWh, demand={demand:.6f} kWh."
            ))

        load = building._dataset_energy_to_control_step(self._state(building.non_shiftable_load, t, 'non-shiftable load'))
        served = self._state(building.non_shiftable_load_device.electricity_consumption, t,
                             'non-shiftable load electricity consumption')
        self._require_at_most('non_shiftable_served_le_demand', served - load, self._tolerance(served, load), (
            f"Served non-shiftable load exceeds the demand for '{building.name}' at t={t}: "
            f"served={served:.6f} kWh, demand={load:.6f} kWh."
        ))

    def _assert_battery_reserve(self, building, t: int):
        """The stationary battery never discharges below its depth-of-discharge reserve.

        The standby loss acts on the stored energy before the control action, so an idle battery at the
        reserve drifts below it. The floor is therefore the lower of the reserve and the previous SOC after
        one step of standby loss (the initial SOC at the first step).
        """

        battery = getattr(building, 'electrical_storage', None)
        if not isinstance(battery, Battery) or float(battery.capacity) <= 0.0:
            return

        previous = (float(battery.initial_soc) if t == 0
                    else self._state(battery.soc, t - 1, 'electrical storage SOC')*(1.0 - float(battery.loss_coefficient)))
        floor = min(1.0 - float(battery.depth_of_discharge), previous)
        soc = self._state(battery.soc, t, 'electrical storage SOC')
        self._require_at_most('battery_soc_ge_reserve', floor - soc, self.SOC_EPS, (
            f"Battery SOC below the depth-of-discharge reserve for '{building.name}' at t={t}: "
            f"soc={soc:.6f}, floor={floor:.6f} (depth_of_discharge={float(battery.depth_of_discharge):.3f})."
        ))

    def _assert_outage_no_storage_spill(self, building, t: int):
        """During an outage, battery and vehicle discharge never exceed the local load that PV leaves.

        Uses the PV curtailment the building recorded, not the load arrays: discharge that no local load
        needs leaves PV curtailed, which is a spill to the failed grid. Physics version 1 books that
        surplus as curtailment by design (#229).
        """

        if (not bool(getattr(building, 'simulate_power_outage', False))
                or self._safe_index(getattr(building, 'power_outage_signal', []), t, 0.0) < 1.0
                or physics_version_of(building) < 2):
            return

        discharge = (max(-self._state(building.electrical_storage.electricity_consumption, t, 'electrical storage'), 0.0)
                     + max(-self._state(building.chargers_electricity_consumption, t, 'chargers'), 0.0))
        curtailment = self._state(building.pv_curtailment, t, 'PV curtailment')
        excess = min(discharge, curtailment) if discharge > self.ENERGY_EPS else 0.0
        self._require_at_most('outage_discharge_le_load_after_pv', excess, self._tolerance(discharge, curtailment), (
            f"Outage storage spill for '{building.name}' at t={t}: discharge={discharge:.6f} kWh "
            f"while pv_curtailment={curtailment:.6f} kWh."
        ))

    def _assert_soc_bounds(self, building, t: int):
        storages = [
            ("cooling_storage", getattr(building, "cooling_storage", None)),
            ("heating_storage", getattr(building, "heating_storage", None)),
            ("dhw_storage", getattr(building, "dhw_storage", None)),
            ("electrical_storage", getattr(building, "electrical_storage", None)),
        ]

        for name, storage in storages:
            if storage is None:
                continue

            soc = self._safe_index(getattr(storage, "soc", []), t, 0.0)
            self._require(-self.SOC_EPS <= soc <= 1.0 + self.SOC_EPS, (
                f"{name} SOC out of bounds for '{building.name}' at t={t}: soc={soc:.6f}."
            ))

        for charger in building.electric_vehicle_chargers or []:
            for ev in self._iter_charger_evs(charger):
                soc = self._safe_index(ev.battery.soc, t, 0.0)
                self._require(-self.SOC_EPS <= soc <= 1.0 + self.SOC_EPS, (
                    f"EV SOC out of bounds for '{ev.name}' at t={t}: soc={soc:.6f}."
                ))

    def _assert_storage_power_bounds(self, building, t: int):
        step_seconds = max(self._to_scalar(building.seconds_per_time_step, 1.0), 1.0)
        battery = getattr(building, "electrical_storage", None)
        if battery is None:
            return

        nominal_power_kw = self._to_scalar(getattr(battery, "nominal_power", 0.0), 0.0)
        if nominal_power_kw <= 0.0:
            return

        energy_kwh = self._safe_index(building.electrical_storage_electricity_consumption, t, 0.0)
        max_abs_energy_kwh = power_kw_to_energy_kwh(nominal_power_kw, step_seconds)
        self._require(abs(energy_kwh) <= max_abs_energy_kwh + self.ENERGY_EPS, (
            f"Battery power bound exceeded for '{building.name}' at t={t}: "
            f"energy={energy_kwh:.6f} kWh, limit={max_abs_energy_kwh:.6f} kWh/step."
        ))

    def _assert_charger_power_bounds(self, building, t: int):
        step_seconds = max(self._to_scalar(building.seconds_per_time_step, 1.0), 1.0)

        for charger in building.electric_vehicle_chargers or []:
            max_ch_kw = self._to_scalar(getattr(charger, "max_charging_power", 0.0), 0.0)
            max_dis_kw = self._to_scalar(getattr(charger, "max_discharging_power", 0.0), 0.0)
            max_power_kw = max(max_ch_kw, max_dis_kw)
            if max_power_kw <= 0.0:
                continue

            applied_energy_kwh = self._safe_index(charger.electricity_consumption, t, 0.0)
            commanded_energy_kwh = self._safe_index(charger.past_charging_action_values_kwh, t, 0.0)
            max_abs_energy_kwh = power_kw_to_energy_kwh(max_power_kw, step_seconds)

            self._require(abs(applied_energy_kwh) <= max_abs_energy_kwh + self.ENERGY_EPS, (
                f"Charger power bound exceeded at '{building.name}:{charger.charger_id}' t={t}: "
                f"applied={applied_energy_kwh:.6f} kWh, limit={max_abs_energy_kwh:.6f} kWh/step."
            ))
            self._require(abs(commanded_energy_kwh) <= max_abs_energy_kwh + self.ENERGY_EPS, (
                f"Charger command bound exceeded at '{building.name}:{charger.charger_id}' t={t}: "
                f"commanded={commanded_energy_kwh:.6f} kWh, limit={max_abs_energy_kwh:.6f} kWh/step."
            ))

    @staticmethod
    def _iter_charger_evs(charger) -> Iterable:
        yielded = set()
        for ev in (
            getattr(charger, "connected_electric_vehicle", None),
            getattr(charger, "incoming_electric_vehicle", None),
        ):
            if ev is None:
                continue
            ev_id = getattr(ev, "name", id(ev))
            if ev_id in yielded:
                continue
            yielded.add(ev_id)
            yield ev
