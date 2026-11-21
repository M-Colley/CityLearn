"""Side-effect-free allocation of signed electrical-service requests, in kW.

Positive power imports/consumes; negative power exports/supplies. Callers provide
their own base-load estimate and phase split. This calculation reduces requests
proportionally; it neither dispatches devices nor guarantees that fixed loads
alone satisfy the service limits. Device physics and sequential outage supply
checks remain the runtime's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Mapping, Optional, Sequence, Tuple, TypedDict


class PowerRequest(TypedDict):
    request_total_kw: float
    request_phase_kw: Mapping[str, float]


@dataclass(frozen=True)
class ElectricalServiceAllocation:
    scales: Dict[str, float]
    total_kw: float
    phase_kw: Dict[str, float]
    requested_violation_kw: float
    residual_violation_kw: float


def _finite_scalar(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if isfinite(value) else default


def _compute_totals(
    base_total_kw: float,
    base_phase_kw: Mapping[str, float],
    controls: Mapping[str, PowerRequest],
    scales: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    total_kw = float(base_total_kw)
    phase_kw = {phase: float(value) for phase, value in base_phase_kw.items()}
    for control_id, control in controls.items():
        scale = scales[control_id]
        total_kw += control['request_total_kw'] * scale
        for phase_name, value in control['request_phase_kw'].items():
            phase_kw[phase_name] = phase_kw.get(phase_name, 0.0) + value * scale
    return total_kw, phase_kw


def electrical_service_violation_kw(
    total_kw: float,
    phase_kw: Mapping[str, float],
    total_limits: Mapping[str, Optional[float]],
    per_phase_limits: Mapping[str, Mapping[str, Optional[float]]],
    phase_names: Sequence[str],
) -> float:
    """Sum requested/residual excess over total and phase limits, in kW.

    This pressure metric counts each scope separately; it is not unmet energy.
    Missing/non-finite limits are unbounded, as in the service configuration.
    """
    violation_kw = 0.0
    for phase_name in (None, *phase_names):
        value = total_kw if phase_name is None else phase_kw.get(phase_name, 0.0)
        value = _finite_scalar(value, 0.0)
        limits = total_limits if phase_name is None else per_phase_limits.get(phase_name, {})
        import_limit = _finite_scalar(limits.get('import_kw'), float('nan'))
        export_limit = _finite_scalar(limits.get('export_kw'), float('nan'))
        if isfinite(import_limit):
            violation_kw += max(value - import_limit, 0.0)
        if isfinite(export_limit):
            violation_kw += max(-value - export_limit, 0.0)
    return float(violation_kw)


def _scale_scope(
    current_value_kw: float,
    limit_kw: Optional[float],
    controls: Mapping[str, PowerRequest],
    scales: Dict[str, float],
    phase_name: Optional[str],
    *,
    export: bool,
) -> bool:
    limit_kw = _finite_scalar(limit_kw, float('nan'))
    current_value_kw = _finite_scalar(current_value_kw, 0.0)
    if not isfinite(limit_kw):
        return False
    magnitude_kw = max(-current_value_kw, 0.0) if export else current_value_kw
    if magnitude_kw <= limit_kw + 1.0e-9:
        return False

    relevant = []
    for control_id, control in controls.items():
        component_kw = (control['request_total_kw'] if phase_name is None
                        else control['request_phase_kw'].get(phase_name, 0.0))
        if (component_kw < 0.0 if export else component_kw > 0.0) and scales[control_id] > 0.0:
            relevant.append((control_id, abs(component_kw)))
    if not relevant:
        return False

    current_relevant_kw = sum(scales[key] * component for key, component in relevant)
    if current_relevant_kw <= 1.0e-9:
        return False
    if export:
        fixed_kw = current_value_kw + current_relevant_kw
        allowed_kw = limit_kw + fixed_kw
    else:
        fixed_kw = current_value_kw - current_relevant_kw
        allowed_kw = limit_kw - fixed_kw
    factor = 0.0 if allowed_kw <= 0.0 else min(1.0, allowed_kw / current_relevant_kw)
    if factor >= 1.0 - 1.0e-9:
        return False
    for control_id, _ in relevant:
        scales[control_id] *= factor
    return True


def allocate_electrical_service(
    base_total_kw: float,
    base_phase_kw: Mapping[str, float],
    controls: Mapping[str, PowerRequest],
    total_limits: Mapping[str, Optional[float]],
    per_phase_limits: Mapping[str, Mapping[str, Optional[float]]],
    phase_names: Sequence[str],
) -> ElectricalServiceAllocation:
    """Return reductions and pressure without mutating any input or device.

    Preserve the established eight-pass maximum, scope order (total, then the
    supplied phases) and import-before-export order. Every scope reads totals
    after earlier reductions; opposite-sign controls can couple scopes, requiring
    a later pass. Scales only decrease, preserving proportional sharing within
    each constrained group. There is no redistribution or service prioritisation.
    """
    scales = {control_id: 1.0 for control_id in controls}
    requested_total_kw, requested_phase_kw = _compute_totals(base_total_kw, base_phase_kw, controls, scales)
    requested_violation_kw = electrical_service_violation_kw(
        requested_total_kw, requested_phase_kw, total_limits, per_phase_limits, phase_names)

    for _ in range(8):
        changed = False
        for phase_name in (None, *phase_names):
            limits = total_limits if phase_name is None else per_phase_limits.get(phase_name, {})
            for export in (False, True):
                total_kw, phase_kw = _compute_totals(base_total_kw, base_phase_kw, controls, scales)
                changed |= _scale_scope(
                    total_kw if phase_name is None else phase_kw.get(phase_name, 0.0),
                    limits.get('export_kw' if export else 'import_kw'),
                    controls, scales, phase_name, export=export,
                )
        if not changed:
            break

    total_kw, phase_kw = _compute_totals(base_total_kw, base_phase_kw, controls, scales)
    total_kw = _finite_scalar(total_kw, 0.0)
    phase_kw = {phase: _finite_scalar(value, 0.0) for phase, value in phase_kw.items()}
    residual_violation_kw = electrical_service_violation_kw(
        total_kw, phase_kw, total_limits, per_phase_limits, phase_names)
    return ElectricalServiceAllocation(
        scales, total_kw, phase_kw, requested_violation_kw, residual_violation_kw)


@dataclass(frozen=True)
class DistrictFeederAllocation:
    scales: Dict[str, float]
    projected_power_kw: float
    curtailed: bool


# Exact enumeration cutoffs and bounded district-scale search parameters.
EXACT_DEADBAND_SUBSET_CUTOFF = 16
EXACT_GROUP_COMBINATIONS_CUTOFF = 65536
BOUNDED_DP_BUCKETS = 2048


def _parse_district_limit_spec(value, path: str, *, enabled: bool):
    """Parse a static non-negative limit [kW] or a dataset-indexed per-step sequence of limits."""
    import math as _math
    import numpy as np

    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f'{path} must be a non-negative number, sequence, or null.')

    if isinstance(value, (int, float, np.number)):
        scalar = float(value)
        if _math.isnan(scalar):
            raise ValueError(f'{path} cannot be NaN.')
        if _math.isinf(scalar):
            if scalar < 0.0:
                raise ValueError(f'{path} must be >= 0.0 when provided.')
            return None
        if scalar < 0.0:
            raise ValueError(f'{path} must be >= 0.0 when provided.')
        return scalar

    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f'{path} must be a non-negative number, sequence, or null.')

    try:
        items = list(np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
    except TypeError as exc:
        raise ValueError(f'{path} must be a non-negative number, sequence, or null.') from exc

    if len(items) == 0:
        if enabled:
            raise ValueError(f'{path} sequence cannot be empty when district_electrical_service is enabled.')
        return None

    parsed_items = []
    for idx, item in enumerate(items):
        if item is None:
            parsed_items.append(None)
            continue
        if isinstance(item, bool):
            raise ValueError(f'{path}[{idx}] must be a non-negative number or null.')
        try:
            scalar = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{path}[{idx}] must be a non-negative number or null.') from exc
        if _math.isnan(scalar):
            parsed_items.append(None)
            continue
        if _math.isinf(scalar):
            if scalar < 0.0:
                raise ValueError(f'{path}[{idx}] must be >= 0.0 when provided.')
            parsed_items.append(None)
            continue
        if scalar < 0.0:
            raise ValueError(f'{path}[{idx}] must be >= 0.0 when provided.')
        parsed_items.append(scalar)

    return tuple(parsed_items)


def parse_district_electrical_service_config(config) -> Dict[str, object]:
    """Validate and normalize ``district_electrical_service`` configuration."""
    from citylearn.utilities import parse_bool

    if config is None:
        return {'enabled': False, 'import_kw': None, 'export_kw': None}

    if not isinstance(config, Mapping):
        raise ValueError('district_electrical_service must be a mapping or None.')

    unknown = set(config.keys()) - {'enabled', 'limits'}
    if unknown:
        raise ValueError(f'district_electrical_service has unknown keys {sorted(unknown)}; use enabled and limits.')

    enabled = bool(parse_bool(config.get('enabled', False), default=False, path='district_electrical_service.enabled'))
    limits = config.get('limits')
    if limits is None:
        limits = {}
    elif not isinstance(limits, Mapping):
        raise ValueError('district_electrical_service.limits must be a mapping when provided.')

    unknown_limits = set(limits.keys()) - {'import_kw', 'export_kw', 'total'}
    if unknown_limits:
        raise ValueError(
            f'district_electrical_service.limits has unknown keys {sorted(unknown_limits)}; '
            'use import_kw and export_kw.'
        )

    total_block = limits.get('total')
    if total_block is not None and not isinstance(total_block, Mapping):
        raise ValueError('district_electrical_service.limits.total must be a mapping when provided.')
    if isinstance(total_block, Mapping):
        unknown_total = set(total_block.keys()) - {'import_kw', 'export_kw'}
        if unknown_total:
            raise ValueError(
                f'district_electrical_service.limits.total has unknown keys {sorted(unknown_total)}; '
                'use import_kw and export_kw.'
            )

    raw_import = limits.get('import_kw', total_block.get('import_kw') if isinstance(total_block, Mapping) else None)
    raw_export = limits.get('export_kw', total_block.get('export_kw') if isinstance(total_block, Mapping) else None)

    return {
        'enabled': enabled,
        'import_kw': _parse_district_limit_spec(
            raw_import, 'district_electrical_service.limits.import_kw', enabled=enabled
        ),
        'export_kw': _parse_district_limit_spec(
            raw_export, 'district_electrical_service.limits.export_kw', enabled=enabled
        ),
    }


def _serialize_limit_spec(spec):
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec) if isfinite(float(spec)) else None
    if isinstance(spec, tuple):
        return [None if item is None else float(item) for item in spec]
    return None


def serialize_district_electrical_service_config(parsed_config: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """Return a canonical JSON-serializable representation of ``parsed_config`` for schemas and manifests."""
    if not parsed_config or not bool(parsed_config.get('enabled', False)):
        return {'enabled': False, 'limits': {'import_kw': None, 'export_kw': None}}

    return {
        'enabled': True,
        'limits': {
            'import_kw': _serialize_limit_spec(parsed_config.get('import_kw')),
            'export_kw': _serialize_limit_spec(parsed_config.get('export_kw')),
        },
    }


def validate_district_feeder_schedule_bounds(
    parsed_config: Optional[Mapping[str, object]],
    *,
    simulation_start_time_step: int = 0,
    simulation_end_time_step: int = 0,
    episode_start_time_step: Optional[int] = None,
    episode_end_time_step: Optional[int] = None,
) -> None:
    """Validate that dataset-indexed sequence limits cover all executed dataset time steps."""
    if not parsed_config or not bool(parsed_config.get('enabled', False)):
        return

    sim_start = max(int(simulation_start_time_step or 0), 0)
    sim_end = max(int(simulation_end_time_step or 0), sim_start)
    # A window [start, end] with end > start executes transitions at dataset steps start .. end - 1.
    max_executed_step = sim_end - 1 if sim_end > sim_start else sim_start

    if episode_start_time_step is not None and episode_end_time_step is not None:
        ep_start = max(int(episode_start_time_step), 0)
        ep_end = max(int(episode_end_time_step), ep_start)
        ep_max_executed = ep_end - 1 if ep_end > ep_start else ep_start
        max_executed_step = max(max_executed_step, ep_max_executed)

    required_len = max_executed_step + 1
    for key in ('import_kw', 'export_kw'):
        spec = parsed_config.get(key)
        if isinstance(spec, tuple) and len(spec) < required_len:
            raise ValueError(
                f'district_electrical_service.limits.{key} sequence has length {len(spec)}, '
                f'which does not cover executed dataset time step {max_executed_step} '
                f'(schedules are anchored at dataset step 0 and need at least {required_len} entries).'
            )


def resolve_district_feeder_limits_at(
    parsed_config: Mapping[str, object],
    time_step: int,
    *,
    episode_start_time_step: int = 0,
    episode_time_steps: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve scalar ``(import_kw, export_kw)`` feeder limits at dataset step ``episode_start_time_step + time_step``."""
    del episode_time_steps  # Sequences are always anchored to dataset step 0.
    resolved = []
    step = max(int(time_step), 0)
    offset = max(int(episode_start_time_step), 0)
    global_step = offset + step

    for key in ('import_kw', 'export_kw'):
        spec = parsed_config.get(key)
        if spec is None or isinstance(spec, float):
            resolved.append(spec)
            continue
        if isinstance(spec, tuple):
            if global_step >= len(spec):
                raise ValueError(
                    f'district_electrical_service.limits.{key} sequence has length {len(spec)}, '
                    f'which does not cover dataset time step {global_step} '
                    f'(episode_start_time_step={offset}, time_step={step}).'
                )
            resolved.append(spec[global_step])
            continue
        resolved.append(None)

    return resolved[0], resolved[1]


def district_feeder_violation_kw(
    total_power_kw: float,
    import_limit_kw: Optional[float],
    export_limit_kw: Optional[float],
    *,
    residual_eps_kw: float = 1.0e-5,
) -> float:
    """Residual excess over ``import_limit_kw`` or ``export_limit_kw`` in [kW]."""
    power = _finite_scalar(total_power_kw, 0.0)
    violation = 0.0
    imp = _finite_scalar(import_limit_kw, float('nan'))
    exp = _finite_scalar(export_limit_kw, float('nan'))
    if isfinite(imp):
        excess = max(power - imp, 0.0)
        if excess > residual_eps_kw:
            violation += excess
    if isfinite(exp):
        excess = max(-power - exp, 0.0)
        if excess > residual_eps_kw:
            violation += excess
    return float(violation)


def _waterfill_deadband_subset(
    group_ids: Sequence[str],
    active_ids: Sequence[str],
    u_kw: Mapping[str, float],
    m_kw: Mapping[str, float],
    target_sum_kw: float,
) -> Tuple[Dict[str, float], bool]:
    """Allocate ``target_sum_kw`` across ``active_ids`` respecting ``m_kw[i] <= x[i] <= u_kw[i]``."""
    alloc = {cid: 0.0 for cid in group_ids}
    if not active_ids or target_sum_kw <= 1.0e-12:
        return alloc, True

    u_sum = sum(u_kw[cid] for cid in active_ids)
    if target_sum_kw >= u_sum - 1.0e-12:
        for cid in active_ids:
            alloc[cid] = u_kw[cid]
        return alloc, True

    m_sum = sum(m_kw[cid] for cid in active_ids)
    if target_sum_kw <= m_sum + 1.0e-12:
        for cid in active_ids:
            alloc[cid] = m_kw[cid]
        return alloc, (m_sum <= 1.0e-12)

    unpinned = list(active_ids)
    rem = float(target_sum_kw)
    pinned_any = False

    while unpinned:
        rem_u = sum(u_kw[cid] for cid in unpinned)
        if rem_u <= 1.0e-12:
            break
        alpha = rem / rem_u
        below = [cid for cid in unpinned if alpha * u_kw[cid] + 1.0e-12 < m_kw[cid]]
        if not below:
            for cid in unpinned:
                alloc[cid] = min(u_kw[cid], max(m_kw[cid], alpha * u_kw[cid]))
            break
        pinned_any = True
        for cid in below:
            alloc[cid] = m_kw[cid]
            rem = max(0.0, rem - m_kw[cid])
        below_set = set(below)
        unpinned = [cid for cid in unpinned if cid not in below_set]

    return alloc, (not pinned_any)


def _evaluate_joint_subset_arrays(
    base_kw: float,
    lp_arr,
    up_arr,
    ln_arr,
    un_arr,
    u_all_pos: float,
    u_all_neg: float,
    imp: float,
    exp: float,
):
    """Vectorized evaluation of feeder violation and minimum curtailed power for candidate subsets."""
    import numpy as np

    p_lo = base_kw + lp_arr - un_arr
    p_hi = base_kw + up_arr - ln_arr
    p_raw = base_kw + up_arr - un_arr

    viol = np.zeros_like(p_lo, dtype='float64')
    if isfinite(imp):
        viol = np.maximum(viol, p_lo - imp)
    if isfinite(exp):
        viol = np.maximum(viol, -exp - p_hi)

    x_pos = up_arr.copy()
    x_neg = un_arr.copy()
    if isfinite(imp):
        over = p_raw > imp
        if np.any(over):
            x_pos[over] = np.minimum(up_arr[over], np.maximum(lp_arr[over], imp - base_kw + un_arr[over]))
    if isfinite(exp):
        under = p_raw < -exp
        if np.any(under):
            x_neg[under] = np.minimum(un_arr[under], np.maximum(ln_arr[under], base_kw + up_arr[under] + exp))

    curtailed = np.maximum(0.0, (u_all_pos - x_pos) + (u_all_neg - x_neg))
    return viol, curtailed, x_pos, x_neg


def _bounded_dp_candidate_masks(
    branch_ids: Sequence[str],
    pos_set: set,
    l_kw: Mapping[str, float],
    u_kw: Mapping[str, float],
    l0_pos: float,
    u0_pos: float,
    l0_neg: float,
    u0_neg: float,
) -> Sequence[int]:
    """Bounded quantized-interval dynamic program + greedy trajectories for heterogeneous district scale."""
    masks = {0, (1 << len(branch_ids)) - 1}
    for idx in range(len(branch_ids)):
        masks.add(1 << idx)

    # Greedy prefix orders (by minimum power, maximum power, and deadband ratio).
    indexed = list(enumerate(branch_ids))
    for order in (
        sorted(indexed, key=lambda it: (l_kw[it[1]], -u_kw[it[1]])),
        sorted(indexed, key=lambda it: (-u_kw[it[1]], l_kw[it[1]])),
        sorted(indexed, key=lambda it: (l_kw[it[1]] / max(u_kw[it[1]], 1.0e-12), -u_kw[it[1]])),
    ):
        running = 0
        for idx, _ in order:
            running |= 1 << idx
            masks.add(running)

    # Quantized 1D Pareto DP over signed lower reach L_signed = L_pos - U_neg in [-max_neg, +max_pos].
    min_reach = l0_pos - (u0_neg + sum(u_kw[cid] for cid in branch_ids if cid not in pos_set))
    max_reach = (l0_pos + sum(l_kw[cid] for cid in branch_ids if cid in pos_set)) - u0_neg
    span = max(max_reach - min_reach, 1.0e-9)
    buckets = BOUNDED_DP_BUCKETS

    def bucket_of(l_signed: float) -> int:
        b = int(round((l_signed - min_reach) / span * (buckets - 1)))
        return 0 if b < 0 else (buckets - 1 if b >= buckets else b)

    # dp[bucket] = (l_signed, r_signed, active_u_total, lp, up, ln, un, mask)
    l0_sig = l0_pos - u0_neg
    r0_sig = u0_pos - l0_neg
    b0 = bucket_of(l0_sig)
    init_state = (l0_sig, r0_sig, u0_pos + u0_neg, l0_pos, u0_pos, l0_neg, u0_neg, 0)
    dp_wide: Dict[int, tuple] = {b0: init_state}
    dp_low: Dict[int, tuple] = {b0: init_state}
    dp_full: Dict[int, tuple] = {b0: init_state}

    for idx, cid in enumerate(branch_ids):
        bit = 1 << idx
        li = l_kw[cid]
        ui = u_kw[cid]
        is_pos = cid in pos_set

        next_wide = dict(dp_wide)
        next_low = dict(dp_low)
        next_full = dict(dp_full)
        seen_masks = set()
        combined_states = []
        for table in (dp_wide, dp_low, dp_full):
            for st in table.values():
                if st[-1] not in seen_masks:
                    seen_masks.add(st[-1])
                    combined_states.append(st)

        for _, _, u_tot, lp, up, ln, un, mask in combined_states:
            if is_pos:
                nlp, nup, nln, nun = lp + li, up + ui, ln, un
            else:
                nlp, nup, nln, nun = lp, up, ln + li, un + ui
            nl_sig = nlp - nun
            nr_sig = nup - nln
            nu_tot = u_tot + ui
            nb = bucket_of(nl_sig)
            cand = (nl_sig, nr_sig, nu_tot, nlp, nup, nln, nun, mask | bit)

            prev_w = next_wide.get(nb)
            if prev_w is None or (cand[1], -cand[0], cand[2]) > (prev_w[1], -prev_w[0], prev_w[2]):
                next_wide[nb] = cand

            prev_l = next_low.get(nb)
            if prev_l is None or (-cand[0], cand[1], cand[2]) > (-prev_l[0], prev_l[1], prev_l[2]):
                next_low[nb] = cand

            prev_f = next_full.get(nb)
            if prev_f is None or (cand[2], cand[1], -cand[0]) > (prev_f[2], prev_f[1], -prev_f[0]):
                next_full[nb] = cand

        dp_wide = next_wide
        dp_low = next_low
        dp_full = next_full

    for table in (dp_wide, dp_low, dp_full):
        for st in table.values():
            masks.add(st[-1])

    return sorted(masks)


def allocate_district_feeder_envelope(
    total_power_kw: float,
    controls_power_kw: Mapping[str, float],
    import_limit_kw: Optional[float],
    export_limit_kw: Optional[float],
    *,
    controls_min_power_kw: Optional[Mapping[str, float]] = None,
    controls_floor_power_kw: Optional[Mapping[str, float]] = None,
    tolerance_kw: float = 1.0e-6,
) -> DistrictFeederAllocation:
    """Curtail controllable charging and discharging jointly to satisfy shared feeder limits.

    Every controllable device ``i`` has requested magnitude ``u_i = |P_i| > 0``, physical deadband
    ``p_min,i >= 0`` (e.g. EV charger minimum charging/discharging power), and optional building
    service curtailment floor ``f_i >= 0`` (the minimum magnitude required to preserve its own
    building and per-phase limits). When ``f_i == 0`` and ``p_min,i > 0``, the device may either
    pause (``0 kW``) or operate in ``[min(p_min,i, u_i), u_i]``; when ``f_i > 0`` or ``p_min,i == 0``,
    its feasible range is the single connected interval ``[max(p_min,i, f_i), u_i]``.

    Search strategy over pauseable minimum-power devices ``D``:
    1. Exact subset enumeration across all ``2**|D|`` joint charging/discharging on/off subsets
       when ``|D| <= 16`` (``EXACT_DEADBAND_SUBSET_CUTOFF``).
    2. Exact multiplicity-grouped enumeration when ``|D| > 16`` and grouping identical
       ``(sign, min_kw, max_kw)`` devices yields at most ``65,536`` combinations
       (``EXACT_GROUP_COMBINATIONS_CUTOFF``).
    3. Bounded quantized-interval dynamic programming (``2,048`` buckets across reachable net power)
       combined with greedy prefix trajectories for larger heterogeneous districts.
    """
    import numpy as np

    scales: Dict[str, float] = {cid: 1.0 for cid in controls_power_kw}
    imp = _finite_scalar(import_limit_kw, float('nan'))
    exp = _finite_scalar(export_limit_kw, float('nan'))
    total_kw = _finite_scalar(total_power_kw, 0.0)

    import_ok = (not isfinite(imp)) or (total_kw <= imp + tolerance_kw)
    export_ok = (not isfinite(exp)) or (-total_kw <= exp + tolerance_kw)
    if import_ok and export_ok:
        return DistrictFeederAllocation(scales=scales, projected_power_kw=total_kw, curtailed=False)

    min_map = controls_min_power_kw or {}
    floor_map = controls_floor_power_kw or {}
    base_kw = float(total_power_kw) - sum(float(p) for p in controls_power_kw.values())

    pos_ids = [cid for cid, p in controls_power_kw.items() if float(p) > 1.0e-9]
    neg_ids = [cid for cid, p in controls_power_kw.items() if float(p) < -1.0e-9]
    if not pos_ids and not neg_ids:
        return DistrictFeederAllocation(scales=scales, projected_power_kw=total_kw, curtailed=False)

    pos_set = set(pos_ids)
    u_kw: Dict[str, float] = {}
    l_kw: Dict[str, float] = {}
    branch_ids: list = []
    fixed_pos_ids: list = []
    fixed_neg_ids: list = []

    for cid in pos_ids + neg_ids:
        ui = abs(float(controls_power_kw[cid]))
        p_min = min(max(_finite_scalar(min_map.get(cid, 0.0), 0.0), 0.0), ui)
        f_min = min(max(_finite_scalar(floor_map.get(cid, 0.0), 0.0), 0.0), ui)
        li = max(p_min, f_min)
        u_kw[cid] = ui
        l_kw[cid] = li
        if f_min <= 1.0e-9 and li > 1.0e-9:
            branch_ids.append(cid)
        elif cid in pos_set:
            fixed_pos_ids.append(cid)
        else:
            fixed_neg_ids.append(cid)

    l0_pos = sum(l_kw[cid] for cid in fixed_pos_ids)
    u0_pos = sum(u_kw[cid] for cid in fixed_pos_ids)
    l0_neg = sum(l_kw[cid] for cid in fixed_neg_ids)
    u0_neg = sum(u_kw[cid] for cid in fixed_neg_ids)
    u_all_pos = sum(u_kw[cid] for cid in pos_ids)
    u_all_neg = sum(u_kw[cid] for cid in neg_ids)

    def _finalize_subset(active_branch_ids: Sequence[str]):
        act_pos = fixed_pos_ids + [cid for cid in active_branch_ids if cid in pos_set]
        act_neg = fixed_neg_ids + [cid for cid in active_branch_ids if cid not in pos_set]
        lp = sum(l_kw[cid] for cid in act_pos)
        up = sum(u_kw[cid] for cid in act_pos)
        ln = sum(l_kw[cid] for cid in act_neg)
        un = sum(u_kw[cid] for cid in act_neg)

        p_lo = base_kw + lp - un
        p_hi = base_kw + up - ln
        p_raw = base_kw + up - un

        viol = 0.0
        if isfinite(imp):
            viol = max(viol, p_lo - imp)
        if isfinite(exp):
            viol = max(viol, -exp - p_hi)

        target_pos = up
        target_neg = un
        if isfinite(imp) and p_raw > imp:
            target_pos = min(up, max(lp, imp - base_kw + un))
        elif isfinite(exp) and p_raw < -exp:
            target_neg = min(un, max(ln, base_kw + up + exp))

        alloc_pos, unpinned_pos = _waterfill_deadband_subset(pos_ids, act_pos, u_kw, l_kw, target_pos)
        alloc_neg, unpinned_neg = _waterfill_deadband_subset(neg_ids, act_neg, u_kw, l_kw, target_neg)
        curtailed_kw = max(0.0, (u_all_pos - target_pos) + (u_all_neg - target_neg))
        viol_key = 0.0 if viol <= 1.0e-7 else round(viol, 9)
        score = (
            viol_key,
            round(curtailed_kw, 7),
            0 if (unpinned_pos and unpinned_neg) else 1,
            -len(active_branch_ids),
        )
        return score, alloc_pos, alloc_neg

    m = len(branch_ids)
    best_subset: Sequence[str]

    if m == 0:
        best_subset = ()
    elif m <= EXACT_DEADBAND_SUBSET_CUTOFF:
        # Tier 1: Exact vectorized enumeration of all 2**m joint on/off subsets.
        lp_arr = np.array([l0_pos], dtype='float64')
        up_arr = np.array([u0_pos], dtype='float64')
        ln_arr = np.array([l0_neg], dtype='float64')
        un_arr = np.array([u0_neg], dtype='float64')
        cnt_arr = np.array([0], dtype='int32')

        for cid in branch_ids:
            li = l_kw[cid]
            ui = u_kw[cid]
            if cid in pos_set:
                lp_arr = np.concatenate([lp_arr, lp_arr + li])
                up_arr = np.concatenate([up_arr, up_arr + ui])
                ln_arr = np.concatenate([ln_arr, ln_arr])
                un_arr = np.concatenate([un_arr, un_arr])
            else:
                lp_arr = np.concatenate([lp_arr, lp_arr])
                up_arr = np.concatenate([up_arr, up_arr])
                ln_arr = np.concatenate([ln_arr, ln_arr + li])
                un_arr = np.concatenate([un_arr, un_arr + ui])
            cnt_arr = np.concatenate([cnt_arr, cnt_arr + 1])

        viol, curtailed, _, _ = _evaluate_joint_subset_arrays(
            base_kw, lp_arr, up_arr, ln_arr, un_arr, u_all_pos, u_all_neg, imp, exp
        )
        eff_viol = np.where(viol <= 1.0e-7, 0.0, viol)
        min_v = float(np.min(eff_viol))
        v_mask = eff_viol <= min_v + 1.0e-8
        min_k = float(np.min(curtailed[v_mask]))
        cand_indices = np.flatnonzero(v_mask & (curtailed <= min_k + 1.0e-5))
        if cand_indices.size > 64:
            order = np.lexsort((-cnt_arr[cand_indices], curtailed[cand_indices], eff_viol[cand_indices]))
            cand_indices = cand_indices[order[:64]]

        best_score = None
        best_subset = ()
        for mask_int in cand_indices.tolist():
            sub = [branch_ids[bit] for bit in range(m) if (mask_int >> bit) & 1]
            sc, _, _ = _finalize_subset(sub)
            if best_score is None or sc < best_score:
                best_score = sc
                best_subset = sub
    else:
        # Check Tier 2: Exact multiplicity-grouped enumeration when identical devices compress combinations.
        groups: Dict[tuple, list] = {}
        for cid in branch_ids:
            gkey = (1 if cid in pos_set else -1, round(l_kw[cid], 6), round(u_kw[cid], 6))
            groups.setdefault(gkey, []).append(cid)

        group_list = list(groups.items())
        total_combos = 1
        for _, members in group_list:
            total_combos *= (len(members) + 1)
            if total_combos > EXACT_GROUP_COMBINATIONS_CUTOFF:
                break

        if total_combos <= EXACT_GROUP_COMBINATIONS_CUTOFF:
            lp_arr = np.array([l0_pos], dtype='float64')
            up_arr = np.array([u0_pos], dtype='float64')
            ln_arr = np.array([l0_neg], dtype='float64')
            un_arr = np.array([u0_neg], dtype='float64')
            cnt_arr = np.array([0], dtype='int32')
            radices = []

            for (sgn, _, _), members in group_list:
                ng = len(members)
                radices.append(ng + 1)
                ks = np.arange(ng + 1, dtype='float64')
                # Exact cumulative sums of actual member values preserve any sub-1e-6 differences.
                l_cum = np.concatenate([[0.0], np.cumsum([l_kw[c] for c in members], dtype='float64')])
                u_cum = np.concatenate([[0.0], np.cumsum([u_kw[c] for c in members], dtype='float64')])
                if sgn > 0:
                    lp_arr = (lp_arr[:, None] + l_cum[None, :]).ravel()
                    up_arr = (up_arr[:, None] + u_cum[None, :]).ravel()
                    ln_arr = np.repeat(ln_arr, ng + 1)
                    un_arr = np.repeat(un_arr, ng + 1)
                else:
                    lp_arr = np.repeat(lp_arr, ng + 1)
                    up_arr = np.repeat(up_arr, ng + 1)
                    ln_arr = (ln_arr[:, None] + l_cum[None, :]).ravel()
                    un_arr = (un_arr[:, None] + u_cum[None, :]).ravel()
                cnt_arr = (cnt_arr[:, None] + ks.astype('int32')[None, :]).ravel()

            viol, curtailed, _, _ = _evaluate_joint_subset_arrays(
                base_kw, lp_arr, up_arr, ln_arr, un_arr, u_all_pos, u_all_neg, imp, exp
            )
            eff_viol = np.where(viol <= 1.0e-7, 0.0, viol)
            min_v = float(np.min(eff_viol))
            v_mask = eff_viol <= min_v + 1.0e-8
            min_k = float(np.min(curtailed[v_mask]))
            cand_indices = np.flatnonzero(v_mask & (curtailed <= min_k + 1.0e-5))
            if cand_indices.size > 64:
                order = np.lexsort((-cnt_arr[cand_indices], curtailed[cand_indices], eff_viol[cand_indices]))
                cand_indices = cand_indices[order[:64]]

            best_score = None
            best_subset = ()
            for flat_idx in cand_indices.tolist():
                rem = int(flat_idx)
                counts = []
                for rad in reversed(radices):
                    counts.append(rem % rad)
                    rem //= rad
                counts.reverse()
                sub = []
                for (_, members), k_take in zip(group_list, counts):
                    if k_take > 0:
                        sub.extend(members[:k_take])
                sc, _, _ = _finalize_subset(sub)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_subset = sub
        else:
            # Tier 3: Bounded quantized-interval DP + greedy candidate masks.
            cand_masks = _bounded_dp_candidate_masks(
                branch_ids, pos_set, l_kw, u_kw, l0_pos, u0_pos, l0_neg, u0_neg
            )
            best_score = None
            best_subset = ()
            for mask_int in cand_masks:
                sub = [branch_ids[bit] for bit in range(m) if (mask_int >> bit) & 1]
                sc, _, _ = _finalize_subset(sub)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_subset = sub

    _, alloc_pos, alloc_neg = _finalize_subset(best_subset)
    for cid in pos_ids:
        scales[cid] = min(1.0, max(0.0, alloc_pos[cid] / u_kw[cid]))
    for cid in neg_ids:
        scales[cid] = min(1.0, max(0.0, alloc_neg[cid] / u_kw[cid]))

    projected = _finite_scalar(
        base_kw
        + sum(u_kw[cid] * scales[cid] for cid in pos_ids)
        - sum(u_kw[cid] * scales[cid] for cid in neg_ids),
        0.0,
    )
    curtailed = any(scale < 1.0 - 1.0e-9 for scale in scales.values())
    return DistrictFeederAllocation(scales=scales, projected_power_kw=projected, curtailed=curtailed)
