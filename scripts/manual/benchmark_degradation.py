"""Run one episode with and without a battery `degradation` setting and report KPIs, capacity and ms per step.

    python scripts/manual/benchmark_degradation.py [dataset] [--steps N] [--cell CELL] [--interval N]
"""
import argparse
import copy
import hashlib
import json
import time

import numpy as np

from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet


def build_schema(dataset, steps, degradation):
    schema = DataSet().get_schema(dataset)
    schema['simulation_end_time_step'] = steps - 1
    schema['root_directory'] = DataSet().get_dataset(dataset) if 'root_directory' not in schema or not schema['root_directory'] else schema['root_directory']
    if degradation:
        for building in schema['buildings'].values():
            storage = building.get('electrical_storage')
            if storage and storage.get('type'):
                storage.setdefault('attributes', {})['degradation'] = copy.deepcopy(degradation)
        for ev in (schema.get('electric_vehicles_def') or {}).values():
            ev['battery']['attributes']['degradation'] = {**degradation, 'temperature': 25.0}
    return schema


def run(dataset, steps, degradation):
    env = CityLearnEnv(build_schema(dataset, steps, degradation), central_agent=True)
    env.reset()
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    while not env.terminated:
        env.step([rng.uniform(-1, 1, size=len(env.action_space[0].low)).astype('float32').tolist()])
    elapsed = time.perf_counter() - start
    batteries = [b.electrical_storage for b in env.buildings] + [e.battery for e in env.electric_vehicles]
    caps = [float(b.degraded_capacity / b.capacity) if b.capacity else 1.0 for b in batteries]
    digest = hashlib.sha256(np.concatenate([np.asarray(b.energy_balance, float) for b in batteries]).tobytes()).hexdigest()[:16]
    kpis = env.evaluate_v2() if hasattr(env, 'evaluate_v2') else None
    kpi_hash = hashlib.sha256(json.dumps(kpis.to_dict() if hasattr(kpis, 'to_dict') else str(kpis), sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {'ms_per_step': 1000 * elapsed / steps, 'min_soh': min(caps), 'mean_soh': float(np.mean(caps)), 'balance_sha': digest, 'kpi_sha': kpi_hash}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', nargs='?', default='citylearn_challenge_2022_phase_all_plus_evs')
    p.add_argument('--steps', type=int, default=720)
    p.add_argument('--cell', default='Nmc111_Gr_Kokam75Ah_Battery')
    p.add_argument('--interval', type=int, default=24)
    p.add_argument('--default-only', action='store_true', help='Run only the unmodified configuration (for comparing code versions).')
    a = p.parse_args()
    cfg = {'model': 'blast_lite', 'cell': a.cell, 'update_interval_steps': a.interval, 'temperature': 'outdoor_dry_bulb'}
    for name, d in [('default', None)] + ([] if a.default_only else [('blast_lite', cfg)]):
        print(name, run(a.dataset, a.steps, d))
