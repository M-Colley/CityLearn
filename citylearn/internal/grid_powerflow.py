"""Optional low-voltage distribution-grid power flow (pandapower, SimBench, lightsim2grid).

Enabled by a top-level ``"grid"`` block in the schema. Without the block the environment never imports
this module's optional dependencies and no simulated quantity changes.

Each control step the service reads the settled net building power (and charger power), writes it into the
loads of a SimBench low-voltage network in place, solves the balanced AC power flow (or a linearised
update) and stores bus voltages, line and transformer loading and losses.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Mapping

import numpy as np
import pandas as pd

from citylearn.internal.optional_dependencies import missing_dependency

BACKENDS = ('pandapower', 'pandapower_lightsim2grid', 'lightsim2grid')
MODES = ('ac', 'linear')
OBSERVATIONS = (
    'voltage_min_pu', 'voltage_max_pu', 'line_loading_max_percent', 'transformer_loading_max_percent',
    'losses_kw', 'building_voltage_pu',
)
_OBSERVATION_LIMITS = {
    'voltage_min_pu': (0.5, 1.5), 'voltage_max_pu': (0.5, 1.5), 'line_loading_max_percent': (0.0, 1000.0),
    'transformer_loading_max_percent': (0.0, 1000.0), 'losses_kw': (0.0, 1.0e4), 'building_voltage_pu': (0.5, 1.5),
}
_DEFAULTS = {
    'enabled': True, 'simbench_code': None, 'network_file': None, 'backend': 'lightsim2grid', 'mode': 'ac',
    'ac_every_n_steps': 0, 'power_factor': 0.98, 'load_scale': 1.0, 'unmapped_elements': 'zero',
    'building_loads': None, 'charger_loads': None, 'voltage_limits_pu': (0.9, 1.1),
    'overload_limit_percent': 100.0, 'voltage_buses': 'lv', 'linearization_load_kw': 2.0,
    'observations': (), 'validate_backend': True, 'max_iteration': 30, 'tolerance': 1.0e-8,
}


def parse_grid_config(block: Mapping[str, Any]) -> dict:
    """Validate the schema ``grid`` block and return it with defaults applied."""

    if not isinstance(block, Mapping):
        raise ValueError('schema "grid" must be an object.')

    unknown = sorted(set(block) - set(_DEFAULTS))

    if unknown:
        raise ValueError(f'Unknown schema "grid" keys {unknown}; valid keys are {sorted(_DEFAULTS)}.')

    cfg = {**_DEFAULTS, **block}
    cfg['observations'] = tuple(cfg['observations'] or ())

    if cfg['simbench_code'] is None and cfg['network_file'] is None:
        raise ValueError('schema "grid" needs "simbench_code" (e.g. "1-LV-rural1--0-sw") or "network_file".')

    if cfg['backend'] not in BACKENDS:
        raise ValueError(f'grid.backend must be one of {BACKENDS}, got {cfg["backend"]!r}.')

    if cfg['mode'] not in MODES:
        raise ValueError(f'grid.mode must be one of {MODES}, got {cfg["mode"]!r}.')

    if cfg['unmapped_elements'] not in ('zero', 'keep'):
        raise ValueError('grid.unmapped_elements must be "zero" or "keep".')

    if cfg['voltage_buses'] not in ('lv', 'all'):
        raise ValueError('grid.voltage_buses must be "lv" or "all".')

    unknown_obs = sorted(set(cfg['observations']) - set(OBSERVATIONS))

    if unknown_obs:
        raise ValueError(f'Unknown grid.observations {unknown_obs}; valid names are {list(OBSERVATIONS)}.')

    low, high = (float(v) for v in cfg['voltage_limits_pu'])

    if not 0.0 < low < high:
        raise ValueError('grid.voltage_limits_pu must be [low, high] with 0 < low < high.')

    cfg['voltage_limits_pu'] = (low, high)

    if not 0.0 < float(cfg['power_factor']) <= 1.0:
        raise ValueError('grid.power_factor must be in (0, 1].')

    if int(cfg['ac_every_n_steps']) < 0:
        raise ValueError('grid.ac_every_n_steps must be >= 0 (0: never re-anchor the linear model).')

    return cfg


class _PandapowerBackend:
    """Reference backend: ``pp.runpp`` on the shared network object (optionally with lightsim2grid inside)."""

    def __init__(self, net, cfg, use_lightsim: bool):
        self.net, self.cfg, self.use_lightsim = net, cfg, use_lightsim
        self.first = True

    def solve(self, p_mw: np.ndarray, q_mvar: np.ndarray, load_index) -> None:
        import pandapower as pp

        self.net.load.loc[load_index, 'p_mw'] = p_mw
        self.net.load.loc[load_index, 'q_mvar'] = q_mvar
        kwargs = dict(
            init='dc' if self.first else 'results', numba=_numba_available(), max_iteration=self.cfg['max_iteration'],
            lightsim2grid=self.use_lightsim,
        )

        pp.runpp(self.net, **kwargs)
        self.first = False

    def results(self) -> dict:
        n = self.net
        line = n.res_line[['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar']].to_numpy(dtype=float)
        trafo = n.res_trafo[['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar']].to_numpy(dtype=float)
        return {
            'vm': n.res_bus.vm_pu.to_numpy(dtype=float), 'line': line, 'trafo': trafo,
            'line_loading': n.res_line.loading_percent.to_numpy(dtype=float),
            'trafo_loading': n.res_trafo.loading_percent.to_numpy(dtype=float),
            'losses_mw': float(line[:, 0].sum() + line[:, 2].sum() + trafo[:, 0].sum() + trafo[:, 2].sum()),
        }


class _LightSimBackend:
    """Direct ``LSGrid`` backend: in-place load updates, warm start, no pandas in the hot loop."""

    def __init__(self, net, cfg):
        try:
            from lightsim2grid.network import init_from_pandapower
        except ImportError as exc:
            raise missing_dependency('The lightsim2grid grid backend', 'lightsim2grid', 'grid') from exc

        self.net, self.cfg = net, cfg
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.grid = init_from_pandapower(net)
        # A flat start (zero angles) diverges on grids whose transformers shift the phase angle (SimBench
        # uses 150 degrees); start from the pandapower solution, which carries the shifts.
        self.v = (
            net.res_bus.vm_pu.to_numpy(dtype=float)
            * np.exp(1j * np.deg2rad(net.res_bus.va_degree.to_numpy(dtype=float)))
        )
        sn = net.trafo.sn_mva.to_numpy(dtype=float) * net.trafo.parallel.to_numpy(dtype=float)
        self.trafo_rated_ka_hv = sn / (np.sqrt(3.0) * net.trafo.vn_hv_kv.to_numpy(dtype=float))
        self.trafo_rated_ka_lv = sn / (np.sqrt(3.0) * net.trafo.vn_lv_kv.to_numpy(dtype=float))
        self.line_rated_ka = (
            net.line.max_i_ka.to_numpy(dtype=float) * net.line.df.to_numpy(dtype=float)
            * net.line.parallel.to_numpy(dtype=float)
        )
        self.load_pos = {int(i): k for k, i in enumerate(net.load.index)}
        self.changed = np.zeros(len(net.load), dtype=bool)
        self.p_all = np.asarray(self.grid.get_load_target_p(), dtype=float).copy()
        self.q_all = np.zeros(len(net.load))

    def solve(self, p_mw: np.ndarray, q_mvar: np.ndarray, load_index) -> None:
        pos = [self.load_pos[int(i)] for i in load_index]
        self.p_all[pos] = p_mw
        self.q_all[pos] = q_mvar
        self.changed[pos] = True
        self.grid.update_loads_p(self.changed, self.p_all)
        self.grid.update_loads_q(self.changed, self.q_all)
        v = self.grid.ac_pf(self.v, int(self.cfg['max_iteration']), float(self.cfg['tolerance']))

        if v.shape[0] == 0:
            raise RuntimeError(f'lightsim2grid AC power flow diverged: {self.grid.get_solver().get_error()}')

        self.v[:] = v

    def results(self) -> dict:
        g = self.grid
        l1, l2 = g.get_line_res1_full(), g.get_line_res2_full()
        t1, t2 = g.get_trafo_res1_full(), g.get_trafo_res2_full()
        line = np.column_stack([l1[0], l1[1], l2[0], l2[1]])
        trafo = np.column_stack([t1[0], t1[1], t2[0], t2[1]])
        return {
            'vm': np.abs(self.v), 'line': line, 'trafo': trafo,
            # Currents are in kA.
            'line_loading': np.maximum(l1[3], l2[3]) / self.line_rated_ka * 100.0,
            'trafo_loading': np.maximum(t1[3] / self.trafo_rated_ka_hv, t2[3] / self.trafo_rated_ka_lv) * 100.0,
            'losses_mw': float(line[:, 0].sum() + line[:, 2].sum() + trafo[:, 0].sum() + trafo[:, 2].sum()),
        }


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def load_network(cfg: Mapping[str, Any]):
    """Load the network and coerce the object-dtype limit columns that SimBench writes (``None``)."""

    try:
        import pandapower as pp
    except ImportError as exc:
        raise missing_dependency('The grid power flow', 'pandapower', 'grid') from exc

    if cfg['network_file'] is not None:
        net = pp.from_json(str(cfg['network_file']))
    else:
        try:
            import simbench as sb
        except ImportError as exc:
            raise missing_dependency('SimBench grids', 'simbench', 'grid') from exc

        net = sb.get_simbench_net(str(cfg['simbench_code']))

    for table in ('load', 'sgen'):
        for column in ('min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar'):
            if column in net[table]:
                net[table][column] = pd.to_numeric(net[table][column], errors='coerce').astype(float)

    return net


class GridPowerFlowService:
    """Per-step power flow of a pandapower network driven by CityLearn building and charger power."""

    def __init__(self, env, block: Mapping[str, Any]):
        self.env = env
        self.cfg = parse_grid_config(block)
        cfg = self.cfg
        self.dt_h = float(env.seconds_per_time_step) / 3600.0
        self.net = load_network(cfg)
        net = self.net

        if cfg['unmapped_elements'] == 'zero':
            net.load['p_mw'] = 0.0
            net.load['q_mvar'] = 0.0
            net.sgen['p_mw'] = 0.0
            net.sgen['q_mvar'] = 0.0

        self.load_index = np.asarray(net.load.index[net.load.in_service], dtype=int)

        if len(self.load_index) == 0:
            raise ValueError('The grid network has no in-service loads to map buildings to.')

        self._map_elements()
        bus_vn = net.bus.vn_kv.to_numpy(dtype=float)
        self.voltage_bus = np.flatnonzero(bus_vn <= 1.0) if cfg['voltage_buses'] == 'lv' else np.arange(len(bus_vn))

        if len(self.voltage_bus) == 0:
            raise ValueError('The grid network has no low-voltage buses (vn_kv <= 1.0).')

        self.load_bus_position = np.array([int(net.bus.index.get_loc(b)) for b in net.load.loc[self.load_index, 'bus']])
        self._tan_phi = float(np.tan(np.arccos(float(cfg['power_factor']))))
        self._backend = None
        self._sens = None
        self._anchor = None
        self._since_anchor = 0
        self._n_ac = 0
        self.series = {}
        self.last = None
        self.initialize()

    # mapping -------------------------------------------------------------------------------------------------------
    def _map_elements(self):
        cfg, env = self.cfg, self.env
        names = [b.name for b in env.buildings]
        explicit = dict(cfg['building_loads'] or {})
        unknown = sorted(set(explicit) - set(names))

        if unknown:
            raise ValueError(f'grid.building_loads names unknown buildings {unknown}.')

        valid = set(int(i) for i in self.load_index)
        bad = sorted(set(explicit.values()) - valid)

        if bad:
            raise ValueError(f'grid.building_loads references loads {bad} that are not in-service loads of the network.')

        self.building_load = {}
        cycle = 0

        for name in names:
            if name in explicit:
                self.building_load[name] = int(explicit[name])
            else:
                self.building_load[name] = int(self.load_index[cycle % len(self.load_index)])
                cycle += 1

        charger_cfg = dict(cfg['charger_loads'] or {})
        self.charger_load = {}

        for b in env.buildings:
            for c in b.electric_vehicle_chargers or []:
                load = int(charger_cfg.get(c.charger_id, self.building_load[b.name]))

                if load not in valid:
                    raise ValueError(f'grid.charger_loads maps {c.charger_id} to load {load}, which is not in service.')

                self.charger_load[c.charger_id] = load

        self._building_pos = np.array([self._pos(self.building_load[b.name]) for b in env.buildings], dtype=int)
        self._charger_refs = [
            (b, c, self._pos(self.charger_load[c.charger_id]))
            for b in env.buildings for c in (b.electric_vehicle_chargers or [])
        ]

    def _pos(self, load_id: int) -> int:
        return int(np.flatnonzero(self.load_index == load_id)[0])

    # solving -------------------------------------------------------------------------------------------------------
    def initialize(self):
        """Build the backend, validate it against pandapower and precompute linear sensitivities."""

        cfg, net = self.cfg, self.net
        pf = _PandapowerBackend(net, cfg, use_lightsim=cfg['backend'] == 'pandapower_lightsim2grid')
        ref_p = np.full(len(self.load_index), float(cfg['linearization_load_kw']) / 1000.0)
        pf.solve(ref_p, ref_p * self._tan_phi, self.load_index)
        pp_vm = pf.results()['vm']

        if cfg['backend'] == 'lightsim2grid':
            ls = _LightSimBackend(net, cfg)
            ls.solve(ref_p, ref_p * self._tan_phi, self.load_index)

            if cfg['validate_backend']:
                err = float(np.max(np.abs(ls.results()['vm'] - pp_vm)))

                if err > 1.0e-4:
                    raise ValueError(
                        f'The lightsim2grid backend differs from pandapower by {err:.2e} pu on the reference case '
                        '(lightsim2grid ignores switches and supports a subset of pandapower elements). '
                        'Use grid.backend "pandapower_lightsim2grid" or "pandapower".'
                    )

            self._backend = ls
        else:
            self._backend = pf

        index = net.bus.index
        self._line_from = index.get_indexer(net.line.from_bus)
        self._line_to = index.get_indexer(net.line.to_bus)
        self._trafo_hv = index.get_indexer(net.trafo.hv_bus)
        self._trafo_lv = index.get_indexer(net.trafo.lv_bus)
        vn = net.bus.vn_kv.to_numpy(dtype=float)
        self._line_vn = vn[self._line_from]
        self._line_vn_to = vn[self._line_to]
        self._line_rated_ka = (
            net.line.max_i_ka.to_numpy(dtype=float) * net.line.df.to_numpy(dtype=float)
            * net.line.parallel.to_numpy(dtype=float)
        )
        self._trafo_rated_mva = net.trafo.sn_mva.to_numpy(dtype=float) * net.trafo.parallel.to_numpy(dtype=float)
        self._line_rated_mva = np.sqrt(3.0) * self._line_vn * self._line_rated_ka
        self._ref_p_mw = ref_p
        self._ref_y = self._pack(self._backend.results())
        self._sens = self._sensitivities()

    def _pack(self, res: dict) -> np.ndarray:
        return np.concatenate([res['vm'], res['line'].ravel(), res['trafo'].ravel(), [res['losses_mw']]])

    def _unpack(self, y: np.ndarray):
        nb, nl = len(self.net.bus), len(self.net.line)
        return (
            y[:nb], y[nb:nb + 4 * nl].reshape(nl, 4), y[nb + 4 * nl:-1].reshape(len(self.net.trafo), 4), float(y[-1]),
        )

    def _loading(self, y: np.ndarray):
        """Line and transformer loading (percent) from the flows and voltages in the packed vector."""

        vm, line, trafo, _ = self._unpack(y)
        root3 = np.sqrt(3.0)
        i_from = np.hypot(line[:, 0], line[:, 1]) / (root3 * self._line_vn * vm[self._line_from])
        i_to = np.hypot(line[:, 2], line[:, 3]) / (root3 * self._line_vn_to * vm[self._line_to])
        s_hv = np.hypot(trafo[:, 0], trafo[:, 1]) / (vm[self._trafo_hv] * self._trafo_rated_mva)
        s_lv = np.hypot(trafo[:, 2], trafo[:, 3]) / (vm[self._trafo_lv] * self._trafo_rated_mva)
        return np.maximum(i_from, i_to) / self._line_rated_ka * 100.0, np.maximum(s_hv, s_lv) * 100.0

    def _sensitivities(self) -> np.ndarray:
        n = len(self.load_index)
        y0 = self._ref_y
        step = 1.0e-3  # MW
        s = np.zeros((len(y0), n))

        for k in range(n):
            p = self._ref_p_mw.copy()
            p[k] += step
            self._backend.solve(p, p * self._tan_phi, self.load_index)
            s[:, k] = (self._pack(self._backend.results()) - y0) / step

        self._backend.solve(self._ref_p_mw, self._ref_p_mw * self._tan_phi, self.load_index)
        return s

    def load_power_mw(self) -> np.ndarray:
        """Net active power of every network load (MW) at the current CityLearn time step, load convention."""

        env, scale = self.env, float(self.cfg['load_scale']) / 1000.0 / self.dt_h
        p = np.zeros(len(self.load_index))
        t = int(env.time_step)

        for b, pos in zip(env.buildings, self._building_pos):
            p[pos] += float(b.net_electricity_consumption[t]) * scale

        for b, c, pos in self._charger_refs:
            charger = float(c.electricity_consumption[t]) * scale
            p[pos] += charger
            p[self._pos(self.building_load[b.name])] -= charger

        return p

    def _anchor_at(self, p: np.ndarray, res: dict):
        """Store an AC solution as the point the linear model extends from."""

        y = self._pack(res)
        line_d, trafo_d = self._loading(y)
        self._anchor = (p.copy(), y, res['line_loading'] - line_d, res['trafo_loading'] - trafo_d)

    def step(self):
        """Solve the power flow for the current time step and record the result."""

        cfg = self.cfg
        t = int(self.env.time_step)
        p = self.load_power_mw()
        every = int(cfg['ac_every_n_steps'])
        use_ac = cfg['mode'] == 'ac' or (every > 0 and self._since_anchor >= every)

        if use_ac:
            self._backend.solve(p, p * self._tan_phi, self.load_index)
            res = self._backend.results()
            self._anchor_at(p, res)
            self._since_anchor = 1
            self._n_ac += 1
            vm, line, trafo, losses_mw = res['vm'], res['line_loading'], res['trafo_loading'], res['losses_mw']
        else:
            p0, y0, line_bias, trafo_bias = self._anchor
            y = y0 + self._sens @ (p - p0)
            vm, _, _, losses_mw = self._unpack(y)
            line, trafo = self._loading(y)
            line, trafo = line + line_bias, trafo + trafo_bias
            self._since_anchor += 1

        self._record(t, vm, line, trafo, losses_mw, p)

    def _record(self, t: int, vm_all: np.ndarray, line: np.ndarray, trafo: np.ndarray, losses_mw: float, p_mw: np.ndarray):
        vm = vm_all[self.voltage_bus]
        losses_kw = float(losses_mw) * 1000.0
        low, high = self.cfg['voltage_limits_pu']
        lim = float(self.cfg['overload_limit_percent'])
        over_line = np.maximum(line - lim, 0.0) / 100.0 * self._line_rated_mva
        over_trafo = np.maximum(trafo - lim, 0.0) / 100.0 * self._trafo_rated_mva
        self.last = {'vm': vm, 'line': line, 'trafo': trafo, 'losses_kw': losses_kw, 'p_mw': p_mw}

        if t >= len(self.series['vm_min']):
            return

        s = self.series
        s['vm_min'][t] = vm.min()
        s['vm_max'][t] = vm.max()
        s['under_buses'][t] = int(np.sum(vm < low))
        s['over_buses'][t] = int(np.sum(vm > high))
        s['line_max'][t] = line.max() if len(line) else 0.0
        s['trafo_max'][t] = trafo.max() if len(trafo) else 0.0
        s['overload_kva'][t] = (float(over_line.sum()) + float(over_trafo.sum())) * 1000.0
        s['overloaded'][t] = float(np.any(line > lim) or np.any(trafo > lim))
        s['losses_kw'][t] = losses_kw
        s['load_kw'][t] = float(p_mw.sum()) * 1000.0

    def reset(self):
        n = int(self.env.episode_tracker.episode_time_steps)
        keys = ('vm_min', 'vm_max', 'under_buses', 'over_buses', 'line_max', 'trafo_max', 'overload_kva',
                'overloaded', 'losses_kw', 'load_kw')
        self.series = {k: np.full(n, np.nan) for k in keys}
        zero = np.zeros(len(self.load_index))
        self._backend.solve(zero, zero, self.load_index)
        res = self._backend.results()
        self.last = {
            'vm': res['vm'][self.voltage_bus], 'line': res['line_loading'], 'trafo': res['trafo_loading'],
            'losses_kw': res['losses_mw'] * 1000.0, 'p_mw': zero,
        }
        # The linear model is anchored at the no-load solution until the first AC step.
        self._anchor_at(zero, res)
        # With periodic re-anchoring the first step is an AC step.
        self._since_anchor = int(self.cfg['ac_every_n_steps'])
        self._n_ac = 0

    # observations --------------------------------------------------------------------------------------------------
    def observation_names(self) -> List[str]:
        names = []

        for o in self.cfg['observations']:
            if o == 'building_voltage_pu':
                names += [f'grid_voltage_pu_{b.name}' for b in self.env.buildings]
            else:
                names.append(f'grid_{o}')

        return names

    def observation_limits(self):
        low, high = [], []

        for o in self.cfg['observations']:
            n = len(self.env.buildings) if o == 'building_voltage_pu' else 1
            low += [_OBSERVATION_LIMITS[o][0]] * n
            high += [_OBSERVATION_LIMITS[o][1]] * n

        return low, high

    def observation_values(self) -> List[float]:
        last, out = self.last, []
        lv = {int(b): i for i, b in enumerate(self.voltage_bus)}

        for o in self.cfg['observations']:
            if o == 'voltage_min_pu':
                out.append(float(np.min(last['vm'])))
            elif o == 'voltage_max_pu':
                out.append(float(np.max(last['vm'])))
            elif o == 'line_loading_max_percent':
                out.append(float(np.max(last['line'])) if len(last['line']) else 0.0)
            elif o == 'transformer_loading_max_percent':
                out.append(float(np.max(last['trafo'])) if len(last['trafo']) else 0.0)
            elif o == 'losses_kw':
                out.append(float(last['losses_kw']))
            else:
                for bus in self.load_bus_position[self._building_pos]:
                    out.append(float(last['vm'][lv[int(bus)]]) if int(bus) in lv else float('nan'))

        return out

    # KPIs ----------------------------------------------------------------------------------------------------------
    def kpis(self) -> pd.DataFrame:
        """District KPIs of the steps simulated so far, in the ``evaluate_v2`` row format."""

        s = self.series
        done = ~np.isnan(s['vm_min'])
        dt = self.dt_h
        low, high = self.cfg['voltage_limits_pu']
        nan = float('nan')

        def total(key):
            return float(np.nansum(s[key][done]))

        def extreme(key, fn):
            return float(fn(s[key][done])) if done.any() else nan

        steps_under = int(np.sum(s['under_buses'][done] > 0))
        steps_over = int(np.sum(s['over_buses'][done] > 0))
        steps_any = int(np.sum((s['under_buses'][done] + s['over_buses'][done]) > 0))
        imported = float(np.sum(np.maximum(s['load_kw'][done], 0.0))) * dt
        rows = [
            ('grid_voltage_violation_time_total_hours', steps_any * dt, 'h',
             f'Time with any monitored bus outside [{low}, {high}] pu.'),
            ('grid_voltage_undervoltage_time_total_hours', steps_under * dt, 'h', f'Time with any bus below {low} pu.'),
            ('grid_voltage_overvoltage_time_total_hours', steps_over * dt, 'h', f'Time with any bus above {high} pu.'),
            ('grid_voltage_violation_bus_time_total_hours', (total('under_buses') + total('over_buses')) * dt, 'h',
             'Bus-hours outside the voltage limits.'),
            ('grid_voltage_minimum_pu', extreme('vm_min', np.nanmin), 'pu', 'Lowest monitored bus voltage.'),
            ('grid_voltage_maximum_pu', extreme('vm_max', np.nanmax), 'pu', 'Highest monitored bus voltage.'),
            ('grid_overload_time_total_hours', total('overloaded') * dt, 'h',
             'Time with any line or transformer above the loading limit.'),
            ('grid_overload_energy_total_kvah', total('overload_kva') * dt, 'kVAh',
             'Apparent energy above the loading limit summed over lines and transformers.'),
            ('grid_loading_peak_line_percent', extreme('line_max', np.nanmax), '%', 'Highest line loading.'),
            ('grid_loading_peak_transformer_percent', extreme('trafo_max', np.nanmax), '%',
             'Highest transformer loading.'),
            ('grid_losses_total_kwh', total('losses_kw') * dt, 'kWh', 'Line and transformer active losses.'),
            ('grid_losses_ratio_to_import_ratio', total('losses_kw') * dt / imported if imported > 0 else nan, 'ratio',
             'Losses divided by the imported energy of the mapped loads.'),
        ]
        return pd.DataFrame(
            [{'cost_function': n, 'value': v, 'name': 'District', 'level': 'district', 'unit': u, 'description': d}
             for n, v, u, d in rows]
        )

    def get_metadata(self) -> Mapping[str, Any]:
        return dict(self.cfg)

    def __getstate__(self):
        state = self.__dict__.copy()
        lightsim = isinstance(self._backend, _LightSimBackend)
        state['_backend'] = None if lightsim else self._backend
        state['_rebuild_lightsim'] = lightsim
        return state

    def __setstate__(self, state):
        rebuild = state.pop('_rebuild_lightsim', False)
        self.__dict__.update(state)

        if rebuild:
            self._backend = _LightSimBackend(self.net, self.cfg)
