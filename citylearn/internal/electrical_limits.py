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
    exact: bool = True
    relaxed_violation_kw: float = 0.0


# Exact enumeration cutoffs and bounded district-scale search parameters.
EXACT_DIRECT_SUBSET_CUTOFF = 14
EXACT_DEADBAND_SUBSET_CUTOFF = 24
EXACT_MITM_SIDE_CUTOFF = 4096
EXACT_GROUP_COMBINATIONS_CUTOFF = 65536
BOUNDED_DP_BUCKETS = 512
FEEDER_ZERO_VIOLATION_EPS_KW = 1.0e-5


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


def _expand_group_half(
    half_groups: Sequence[tuple],
    l_kw: Mapping[str, float],
    u_kw: Mapping[str, float],
    l0: float,
    r0: float,
    up0: float,
    un0: float,
):
    import numpy as np

    la = np.array([l0], dtype='float64')
    ra = np.array([r0], dtype='float64')
    upa = np.array([up0], dtype='float64')
    una = np.array([un0], dtype='float64')
    cnta = np.array([0], dtype='int32')
    radices = []

    for (sgn, _, _), members in half_groups:
        ng = len(members)
        radices.append(ng + 1)
        ks = np.arange(ng + 1, dtype='int32')
        l_cum = np.concatenate([[0.0], np.cumsum([l_kw[c] for c in members], dtype='float64')])
        u_cum = np.concatenate([[0.0], np.cumsum([u_kw[c] for c in members], dtype='float64')])
        if sgn > 0:
            dl_cum = l_cum
            dr_cum = u_cum
            dup_cum = u_cum
            dun_cum = np.zeros(ng + 1, dtype='float64')
        else:
            dl_cum = -u_cum
            dr_cum = -l_cum
            dup_cum = np.zeros(ng + 1, dtype='float64')
            dun_cum = u_cum
        la = (la[:, None] + dl_cum[None, :]).ravel()
        ra = (ra[:, None] + dr_cum[None, :]).ravel()
        upa = (upa[:, None] + dup_cum[None, :]).ravel()
        una = (una[:, None] + dun_cum[None, :]).ravel()
        cnta = (cnta[:, None] + ks[None, :]).ravel()

    return la, ra, upa, una, cnta, radices


def _decode_group_flat_index(flat_idx: int, half_groups: Sequence[tuple], radices: Sequence[int]) -> list:
    rem = int(flat_idx)
    counts = []
    for rad in reversed(radices):
        counts.append(rem % rad)
        rem //= rad
    counts.reverse()
    chosen = []
    for (_, members), k_take in zip(half_groups, counts):
        if k_take > 0:
            chosen.extend(members[:k_take])
    return chosen


def _solve_grouped_mitm(
    left_groups: Sequence[tuple],
    right_groups: Sequence[tuple],
    l_kw: Mapping[str, float],
    u_kw: Mapping[str, float],
    base_kw: float,
    l0_pos: float,
    u0_pos: float,
    l0_neg: float,
    u0_neg: float,
    imp: float,
    exp: float,
    finalize_fn,
) -> Sequence[str]:
    """Exact Horowitz-Sahni meet-in-the-middle solver over partitioned multiplicity groups."""
    import numpy as np

    la, ra, upa, una, cnta, rad_l = _expand_group_half(
        left_groups, l_kw, u_kw, base_kw + l0_pos - u0_neg, base_kw + u0_pos - l0_neg, u0_pos, u0_neg
    )
    lb, rb, upb, unb, cntb, rad_r = _expand_group_half(
        right_groups, l_kw, u_kw, 0.0, 0.0, 0.0, 0.0
    )

    order = np.argsort(lb, kind='mergesort')
    lbs = lb[order]
    rbs = rb[order]
    upbs = upb[order]
    unbs = unb[order]
    wbs = upbs + unbs
    cntbs = cntb[order]
    pref_r = np.maximum.accumulate(rbs)
    nr = len(lbs)

    has_imp = isfinite(imp)
    has_exp = isfinite(exp)

    def _feas_mask(v_slack: float, tol: float):
        if has_imp:
            hi = np.searchsorted(lbs, imp + v_slack + tol - la, side='right') - 1
        else:
            hi = np.full(len(la), nr - 1, dtype='int64')
        ok = hi >= 0
        if has_exp:
            ok &= (ra + pref_r[np.maximum(hi, 0)] >= -exp - v_slack - tol)
        return ok, hi

    ok0, hi = _feas_mask(0.0, FEEDER_ZERO_VIOLATION_EPS_KW)
    min_v = 0.0
    if not np.any(ok0):
        v_cand = np.full(len(la), np.inf, dtype='float64')
        has_hi = hi >= 0
        if np.any(has_hi) and has_exp:
            v_cand[has_hi] = np.minimum(
                v_cand[has_hi], np.maximum(0.0, -exp - ra[has_hi] - pref_r[hi[has_hi]])
            )
        has_nx = (hi + 1) < nr
        if np.any(has_nx):
            nx = hi[has_nx] + 1
            v_imp = (la[has_nx] + lbs[nx] - imp) if has_imp else 0.0
            v_exp = np.maximum(0.0, -exp - ra[has_nx] - rbs[nx]) if has_exp else 0.0
            v_cand[has_nx] = np.minimum(v_cand[has_nx], np.maximum(v_imp, v_exp))
        lo_v = 0.0
        hi_v = float(np.min(v_cand))
        for _ in range(40):
            if hi_v - lo_v <= 1.0e-11:
                break
            mid = 0.5 * (lo_v + hi_v)
            okm, _ = _feas_mask(mid, 0.0)
            if np.any(okm):
                hi_v = mid
            else:
                lo_v = mid
        min_v = hi_v
        ok0, hi = _feas_mask(min_v, 5.0e-9)

    tol_v = FEEDER_ZERO_VIOLATION_EPS_KW if min_v == 0.0 else 5.0e-9
    p_size = 1 << max((nr - 1).bit_length(), 0)
    max_r = np.full(2 * p_size, -np.inf, dtype='float64')
    max_w = np.full(2 * p_size, -np.inf, dtype='float64')
    max_up = np.full(2 * p_size, -np.inf, dtype='float64')
    max_un = np.full(2 * p_size, -np.inf, dtype='float64')
    max_cnt = np.full(2 * p_size, -1, dtype='int32')
    max_r[p_size:p_size + nr] = rbs
    max_w[p_size:p_size + nr] = wbs
    max_up[p_size:p_size + nr] = upbs
    max_un[p_size:p_size + nr] = unbs
    max_cnt[p_size:p_size + nr] = cntbs
    step = p_size >> 1
    while step >= 1:
        s0, s1 = step, 2 * step
        c0, c1 = 2 * step, 4 * step
        max_r[s0:s1] = np.maximum(max_r[c0:c1:2], max_r[c0 + 1:c1:2])
        max_w[s0:s1] = np.maximum(max_w[c0:c1:2], max_w[c0 + 1:c1:2])
        max_up[s0:s1] = np.maximum(max_up[c0:c1:2], max_up[c0 + 1:c1:2])
        max_un[s0:s1] = np.maximum(max_un[c0:c1:2], max_un[c0 + 1:c1:2])
        max_cnt[s0:s1] = np.maximum(max_cnt[c0:c1:2], max_cnt[c0 + 1:c1:2])
        step >>= 1

    pref_w = np.maximum.accumulate(wbs)
    pref_up = np.maximum.accumulate(upbs)
    pref_un = np.maximum.accumulate(unbs)
    pref_cnt = np.maximum.accumulate(cntbs)

    f_idx = np.flatnonzero(ok0)
    imp_c = (imp + min_v - base_kw) if has_imp else np.inf
    exp_c = (exp + min_v + base_kw) if has_exp else np.inf

    f_hi = hi[f_idx]
    ub_a = (upa[f_idx] + una[f_idx]) + pref_w[f_hi]
    if has_imp:
        ub_a = np.minimum(ub_a, 2.0 * (una[f_idx] + pref_un[f_hi]) + imp_c)
    if has_exp:
        ub_a = np.minimum(ub_a, 2.0 * (upa[f_idx] + pref_up[f_hi]) + exp_c)
    ub_cnt = cnta[f_idx] + pref_cnt[f_hi]

    sort_ord = np.lexsort((-ub_cnt, -ub_a))
    f_idx = f_idx[sort_ord]
    ub_a = ub_a[sort_ord]
    ub_cnt = ub_cnt[sort_ord]

    best_served = -1.0
    best_cnt = -1
    best_unpinned = False
    best_score = None
    best_subset: Sequence[str] = ()
    tie_checks = 0

    mr_l = max_r.tolist()
    mw_l = max_w.tolist()
    mup_l = max_up.tolist()
    mun_l = max_un.tolist()
    mcnt_l = max_cnt.tolist()
    wbs_l = wbs.tolist()
    upbs_l = upbs.tolist()
    unbs_l = unbs.tolist()
    cntbs_l = cntbs.tolist()

    for k_pos, ai in enumerate(f_idx.tolist()):
        uba = float(ub_a[k_pos])
        ubc = int(ub_cnt[k_pos])
        if uba < best_served - 1.0e-5:
            break
        prune_ties = best_unpinned or (tie_checks >= 16)
        if prune_ties and uba <= best_served + 1.0e-5 and ubc <= best_cnt:
            continue

        lim_hi = int(hi[ai])
        req_r = (-exp - min_v - tol_v - float(ra[ai])) if has_exp else -np.inf
        ua = float(upa[ai])
        na = float(una[ai])
        wa_i = ua + na
        ca = int(cnta[ai])

        stack = [(1, 0, p_size - 1)]
        while stack:
            node, slo, shi = stack.pop()
            if slo > lim_hi or mr_l[node] < req_r:
                continue
            ub = wa_i + mw_l[node]
            if has_imp:
                ub = min(ub, 2.0 * (na + mun_l[node]) + imp_c)
            if has_exp:
                ub = min(ub, 2.0 * (ua + mup_l[node]) + exp_c)
            if ub < best_served - 1.0e-5:
                continue
            prune_ties = best_unpinned or (tie_checks >= 16)
            if prune_ties and ub <= best_served + 1.0e-5 and ca + mcnt_l[node] <= best_cnt:
                continue
            if slo == shi:
                if slo >= nr:
                    continue
                srv = wa_i + wbs_l[slo]
                if has_imp:
                    srv = min(srv, 2.0 * (na + unbs_l[slo]) + imp_c)
                if has_exp:
                    srv = min(srv, 2.0 * (ua + upbs_l[slo]) + exp_c)
                if srv < best_served - 1.0e-5:
                    continue
                tot_cnt = ca + cntbs_l[slo]
                if prune_ties and srv <= best_served + 1.0e-5 and tot_cnt <= best_cnt:
                    continue

                cand_sub = (
                    _decode_group_flat_index(ai, left_groups, rad_l)
                    + _decode_group_flat_index(int(order[slo]), right_groups, rad_r)
                )
                sc, _, _ = finalize_fn(cand_sub)
                if srv > best_served + 1.0e-5:
                    best_served = srv
                    best_cnt = tot_cnt
                    best_unpinned = (sc[2] == 0)
                    best_score = sc
                    best_subset = cand_sub
                    tie_checks = 0
                else:
                    tie_checks += 1
                    if best_score is None or sc < best_score:
                        if srv > best_served:
                            best_served = srv
                        best_score = sc
                        best_subset = cand_sub
                        best_cnt = tot_cnt
                        if sc[2] == 0:
                            best_unpinned = True
                continue
            mid = (slo + shi) >> 1
            stack.append((node << 1, slo, mid))
            if mid + 1 <= lim_hi:
                stack.append(((node << 1) | 1, mid + 1, shi))

    return best_subset


def _polish_subset_mask(
    init_mask: int,
    branch_ids: Sequence[str],
    pos_set: set,
    l_kw: Mapping[str, float],
    u_kw: Mapping[str, float],
    base_kw: float,
    l0_pos: float,
    u0_pos: float,
    l0_neg: float,
    u0_neg: float,
    u_all_pos: float,
    u_all_neg: float,
    imp: float,
    exp: float,
) -> int:
    """Vectorized 1-flip and 1-swap neighborhood hill-climb for large heterogeneous districts."""
    import numpy as np

    m = len(branch_ids)
    if m == 0:
        return init_mask

    dl = np.array([l_kw[c] if c in pos_set else -u_kw[c] for c in branch_ids], dtype='float64')
    dr = np.array([u_kw[c] if c in pos_set else -l_kw[c] for c in branch_ids], dtype='float64')
    dup = np.array([u_kw[c] if c in pos_set else 0.0 for c in branch_ids], dtype='float64')
    dun = np.array([0.0 if c in pos_set else u_kw[c] for c in branch_ids], dtype='float64')

    active = np.array([(init_mask >> idx) & 1 for idx in range(m)], dtype=bool)

    def _state_metrics(act_vec):
        lp = l0_pos + float(np.sum(np.where(act_vec & (dup > 0), dl, 0.0)))
        up = u0_pos + float(np.sum(dup[act_vec]))
        ln = l0_neg + float(np.sum(np.where(act_vec & (dun > 0), -dr, 0.0)))
        un = u0_neg + float(np.sum(dun[act_vec]))
        v_arr, k_arr, _, _ = _evaluate_joint_subset_arrays(
            base_kw,
            np.array([lp], dtype='float64'),
            np.array([up], dtype='float64'),
            np.array([ln], dtype='float64'),
            np.array([un], dtype='float64'),
            u_all_pos,
            u_all_neg,
            imp,
            exp,
        )
        vv = 0.0 if float(v_arr[0]) <= FEEDER_ZERO_VIOLATION_EPS_KW else float(v_arr[0])
        return vv, float(k_arr[0]), lp, up, ln, un

    cur_v, cur_k, lp, up, ln, un = _state_metrics(active)
    if cur_v <= FEEDER_ZERO_VIOLATION_EPS_KW:
        return init_mask

    is_pos = dup > 0.0
    l_vec = np.array([l_kw[c] for c in branch_ids], dtype='float64')
    u_vec = np.array([u_kw[c] for c in branch_ids], dtype='float64')

    for _ in range(32):
        if cur_v <= FEEDER_ZERO_VIOLATION_EPS_KW:
            break
        # 1) All m single-bit flips
        sgn_flip = np.where(active, -1.0, 1.0)
        lp_1 = lp + np.where(is_pos, sgn_flip * l_vec, 0.0)
        up_1 = up + np.where(is_pos, sgn_flip * u_vec, 0.0)
        ln_1 = ln + np.where(~is_pos, sgn_flip * l_vec, 0.0)
        un_1 = un + np.where(~is_pos, sgn_flip * u_vec, 0.0)
        v1, k1, _, _ = _evaluate_joint_subset_arrays(
            base_kw, lp_1, up_1, ln_1, un_1, u_all_pos, u_all_neg, imp, exp
        )
        v1 = np.where(v1 <= FEEDER_ZERO_VIOLATION_EPS_KW, 0.0, v1)
        b1 = int(np.lexsort((k1, v1))[0])
        best_step_v = float(v1[b1])
        best_step_k = float(k1[b1])
        best_move = ('flip', b1, -1)

        # 2) All on-off pair swaps (cap at m <= 120 so m^2/4 <= 3,600)
        on_idx = np.flatnonzero(active)
        off_idx = np.flatnonzero(~active)
        if on_idx.size > 0 and off_idx.size > 0 and on_idx.size * off_idx.size <= 4096:
            lp_on = np.where(is_pos[on_idx], -l_vec[on_idx], 0.0)[:, None]
            up_on = np.where(is_pos[on_idx], -u_vec[on_idx], 0.0)[:, None]
            ln_on = np.where(~is_pos[on_idx], -l_vec[on_idx], 0.0)[:, None]
            un_on = np.where(~is_pos[on_idx], -u_vec[on_idx], 0.0)[:, None]

            lp_off = np.where(is_pos[off_idx], l_vec[off_idx], 0.0)[None, :]
            up_off = np.where(is_pos[off_idx], u_vec[off_idx], 0.0)[None, :]
            ln_off = np.where(~is_pos[off_idx], l_vec[off_idx], 0.0)[None, :]
            un_off = np.where(~is_pos[off_idx], u_vec[off_idx], 0.0)[None, :]

            lp_2 = (lp + lp_on + lp_off).ravel()
            up_2 = (up + up_on + up_off).ravel()
            ln_2 = (ln + ln_on + ln_off).ravel()
            un_2 = (un + un_on + un_off).ravel()
            v2, k2, _, _ = _evaluate_joint_subset_arrays(
                base_kw, lp_2, up_2, ln_2, un_2, u_all_pos, u_all_neg, imp, exp
            )
            v2 = np.where(v2 <= FEEDER_ZERO_VIOLATION_EPS_KW, 0.0, v2)
            b2 = int(np.lexsort((k2, v2))[0])
            if (float(v2[b2]), float(k2[b2])) < (best_step_v - 1.0e-9, best_step_k):
                best_step_v = float(v2[b2])
                best_step_k = float(k2[b2])
                best_move = ('swap', int(on_idx[b2 // off_idx.size]), int(off_idx[b2 % off_idx.size]))

        if (best_step_v < cur_v - 1.0e-9) or (abs(best_step_v - cur_v) <= 1.0e-9 and best_step_k < cur_k - 1.0e-7):
            if best_move[0] == 'flip':
                active[best_move[1]] = ~active[best_move[1]]
            else:
                active[best_move[1]] = False
                active[best_move[2]] = True
            cur_v, cur_k, lp, up, ln, un = _state_metrics(active)
        else:
            break

    out_mask = 0
    for idx in np.flatnonzero(active).tolist():
        out_mask |= 1 << int(idx)
    return out_mask


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
       when ``|D| <= 24`` (``EXACT_DEADBAND_SUBSET_CUTOFF``, using direct vectorized evaluation up
       to ``14`` devices and Horowitz-Sahni meet-in-the-middle with segment-tree pruning for ``15..24``).
    2. Exact multiplicity-grouped meet-in-the-middle when ``|D| > 24`` and partitioning identical
       ``(sign, min_kw, max_kw)`` device groups yields at most ``4,096`` combinations per half
       (up to ``16,777,216`` grouped combinations).
    3. Bounded ``512``-bucket quantized-interval dynamic programming with greedy trajectories and
       ``1-flip / 1-swap`` local neighborhood polish for larger heterogeneous districts, setting
       ``exact=False`` alongside ``relaxed_violation_kw`` (the continuous convex-hull lower bound).
    """
    import numpy as np

    scales: Dict[str, float] = {cid: 1.0 for cid in controls_power_kw}
    imp = _finite_scalar(import_limit_kw, float('nan'))
    exp = _finite_scalar(export_limit_kw, float('nan'))
    total_kw = _finite_scalar(total_power_kw, 0.0)

    import_ok = (not isfinite(imp)) or (total_kw <= imp + tolerance_kw)
    export_ok = (not isfinite(exp)) or (-total_kw <= exp + tolerance_kw)
    if import_ok and export_ok:
        return DistrictFeederAllocation(
            scales=scales,
            projected_power_kw=total_kw,
            curtailed=False,
            exact=True,
            relaxed_violation_kw=0.0,
        )

    min_map = controls_min_power_kw or {}
    floor_map = controls_floor_power_kw or {}
    base_kw = float(total_power_kw) - sum(float(p) for p in controls_power_kw.values())

    pos_ids = [cid for cid, p in controls_power_kw.items() if float(p) > 1.0e-9]
    neg_ids = [cid for cid, p in controls_power_kw.items() if float(p) < -1.0e-9]
    if not pos_ids and not neg_ids:
        rel_v = 0.0
        if isfinite(imp):
            rel_v = max(rel_v, total_kw - imp)
        if isfinite(exp):
            rel_v = max(rel_v, -exp - total_kw)
        return DistrictFeederAllocation(
            scales=scales,
            projected_power_kw=total_kw,
            curtailed=False,
            exact=True,
            relaxed_violation_kw=(0.0 if rel_v <= 1.0e-7 else float(rel_v)),
        )

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

    # Continuous convex-hull relaxation lower bound on feeder violation (relaxing each deadband to [0, u_i]).
    relaxed_lo_kw = base_kw + l0_pos - u_all_neg
    relaxed_hi_kw = base_kw + u_all_pos - l0_neg
    relaxed_v = 0.0
    if isfinite(imp):
        relaxed_v = max(relaxed_v, relaxed_lo_kw - imp)
    if isfinite(exp):
        relaxed_v = max(relaxed_v, -exp - relaxed_hi_kw)
    relaxed_violation_kw = 0.0 if relaxed_v <= 1.0e-7 else float(relaxed_v)

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
        viol_key = 0.0 if viol <= FEEDER_ZERO_VIOLATION_EPS_KW else round(viol, 9)
        score = (
            viol_key,
            round(curtailed_kw, 5),
            0 if (unpinned_pos and unpinned_neg) else 1,
            -len(active_branch_ids),
        )
        return score, alloc_pos, alloc_neg

    m = len(branch_ids)
    best_subset: Sequence[str]
    search_exact = True

    if m == 0:
        best_subset = ()
    elif m <= EXACT_DIRECT_SUBSET_CUTOFF:
        # Tier 1a: Direct vectorized enumeration of all 2**m joint on/off subsets (m <= 14).
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
        eff_viol = np.where(viol <= FEEDER_ZERO_VIOLATION_EPS_KW, 0.0, viol)
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
        # Tier 1b / Tier 2: Group identical (sign, min_kw, max_kw) devices and partition into two halves
        # for Horowitz-Sahni meet-in-the-middle (always exact for m <= 26 since 2^13 == 8,192, and exact
        # for arbitrarily large fleets when partitioned group combinations stay <= 8,192 per half).
        groups: Dict[tuple, list] = {}
        for cid in branch_ids:
            gkey = (1 if cid in pos_set else -1, round(l_kw[cid], 6), round(u_kw[cid], 6))
            groups.setdefault(gkey, []).append(cid)

        sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        left_groups: list = []
        right_groups: list = []
        prod_l = 1
        prod_r = 1
        for item in sorted_groups:
            rad = len(item[1]) + 1
            if prod_l <= prod_r:
                left_groups.append(item)
                prod_l *= rad
            else:
                right_groups.append(item)
                prod_r *= rad

        if m <= EXACT_DEADBAND_SUBSET_CUTOFF or max(prod_l, prod_r) <= EXACT_MITM_SIDE_CUTOFF:
            if max(prod_l, prod_r) > EXACT_MITM_SIDE_CUTOFF:
                # Guaranteed <= 8,192 per half when splitting individual devices for m <= 26.
                half = m // 2
                left_groups = [((1 if c in pos_set else -1, l_kw[c], u_kw[c]), [c]) for c in branch_ids[:half]]
                right_groups = [((1 if c in pos_set else -1, l_kw[c], u_kw[c]), [c]) for c in branch_ids[half:]]
            best_subset = _solve_grouped_mitm(
                left_groups,
                right_groups,
                l_kw,
                u_kw,
                base_kw,
                l0_pos,
                u0_pos,
                l0_neg,
                u0_neg,
                imp,
                exp,
                _finalize_subset,
            )
        else:
            # Tier 3: Bounded quantized-interval DP + greedy candidate masks + 1-flip/1-swap local polish.
            search_exact = False
            cand_masks = _bounded_dp_candidate_masks(
                branch_ids, pos_set, l_kw, u_kw, l0_pos, u0_pos, l0_neg, u0_neg
            )
            n_cand = len(cand_masks)
            lp_c = np.full(n_cand, l0_pos, dtype='float64')
            up_c = np.full(n_cand, u0_pos, dtype='float64')
            ln_c = np.full(n_cand, l0_neg, dtype='float64')
            un_c = np.full(n_cand, u0_neg, dtype='float64')
            cnt_c = np.zeros(n_cand, dtype='int32')
            for bit, cid in enumerate(branch_ids):
                on = np.array([(mk >> bit) & 1 for mk in cand_masks], dtype=bool)
                if cid in pos_set:
                    lp_c[on] += l_kw[cid]
                    up_c[on] += u_kw[cid]
                else:
                    ln_c[on] += l_kw[cid]
                    un_c[on] += u_kw[cid]
                cnt_c[on] += 1

            viol_c, curt_c, _, _ = _evaluate_joint_subset_arrays(
                base_kw, lp_c, up_c, ln_c, un_c, u_all_pos, u_all_neg, imp, exp
            )
            eff_vc = np.where(viol_c <= FEEDER_ZERO_VIOLATION_EPS_KW, 0.0, viol_c)
            top_order = np.lexsort((-cnt_c, curt_c, eff_vc))
            seed_mask = int(cand_masks[int(top_order[0])])
            if float(eff_vc[int(top_order[0])]) > FEEDER_ZERO_VIOLATION_EPS_KW:
                polished_mask = _polish_subset_mask(
                    seed_mask,
                    branch_ids,
                    pos_set,
                    l_kw,
                    u_kw,
                    base_kw,
                    l0_pos,
                    u0_pos,
                    l0_neg,
                    u0_neg,
                    u_all_pos,
                    u_all_neg,
                    imp,
                    exp,
                )
                eval_masks = [polished_mask] + [int(cand_masks[int(i)]) for i in top_order[:16]]
            else:
                eval_masks = [int(cand_masks[int(i)]) for i in top_order[:32]]

            best_score = None
            best_subset = ()
            seen_eval = set()
            for mask_int in eval_masks:
                if mask_int in seen_eval:
                    continue
                seen_eval.add(mask_int)
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
    return DistrictFeederAllocation(
        scales=scales,
        projected_power_kw=projected,
        curtailed=curtailed,
        exact=search_exact,
        relaxed_violation_kw=relaxed_violation_kw,
    )
