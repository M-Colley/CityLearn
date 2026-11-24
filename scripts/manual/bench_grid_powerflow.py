"""Benchmark the optional grid power flow and report the linear-vs-AC error on a dataset episode.

Usage: python scripts/manual/bench_grid_powerflow.py [--dataset NAME] [--steps N] [--simbench-code CODE]

The same seeded random actions drive every configuration, so the error table compares the power-flow
modes on identical building trajectories.
"""

import argparse
import json
import time
import warnings

import numpy as np

from pathlib import Path

from citylearn.citylearn import CityLearnEnv

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    'off': None,
    'pandapower_ac': {'backend': 'pandapower', 'mode': 'ac'},
    'pandapower_lightsim2grid_ac': {'backend': 'pandapower_lightsim2grid', 'mode': 'ac'},
    'lightsim2grid_ac': {'backend': 'lightsim2grid', 'mode': 'ac'},
    'linear': {'backend': 'lightsim2grid', 'mode': 'linear'},
    'linear_ac_every_24': {'backend': 'lightsim2grid', 'mode': 'linear', 'ac_every_n_steps': 24},
}
SERIES = ('vm_min', 'vm_max', 'line_max', 'trafo_max', 'losses_kw')


def run(dataset, steps, grid, seed):
    folder = ROOT / 'data' / 'datasets' / dataset
    schema = json.loads((folder / 'schema.json').read_text(encoding='utf-8'))
    schema['root_directory'] = str(folder)

    if grid is not None:
        schema['grid'] = grid

    env = CityLearnEnv(schema, central_agent=True, interface='flat', episode_time_steps=steps, random_seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    low, high = env.action_space[0].low, env.action_space[0].high
    spent = [0.0]

    if env.grid is not None:
        original = env.grid.step

        def timed():
            tic = time.perf_counter()
            original()
            spent[0] += time.perf_counter() - tic

        env.grid.step = timed

    start = time.perf_counter()

    for _ in range(steps - 1):
        env.step([rng.uniform(low, high).tolist()])

    elapsed = time.perf_counter() - start
    return env, elapsed / (steps - 1) * 1000.0, spent[0] / (steps - 1) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='citylearn_challenge_2022_phase_all_plus_evs')
    parser.add_argument('--steps', type=int, default=720)
    parser.add_argument('--simbench-code', default='1-LV-rural1--0-sw')
    parser.add_argument('--load-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    results, series = {}, {}

    for name, cfg in CONFIGS.items():
        grid = None if cfg is None else {'simbench_code': args.simbench_code, 'load_scale': args.load_scale, **cfg}
        env, ms, grid_ms = run(args.dataset, args.steps, grid, args.seed)
        results[name] = {'env_ms_per_step': round(ms, 3), 'grid_ms_per_step': round(grid_ms, 3)}

        if env.grid is not None:
            series[name] = {k: env.grid.series[k][:args.steps - 1].copy() for k in SERIES}

    reference = series['pandapower_ac']

    for name, values in series.items():
        results[name]['max_abs_error_vs_pandapower_ac'] = {
            k: float(np.nanmax(np.abs(values[k] - reference[k]))) for k in SERIES
        }

    print(json.dumps({'dataset': args.dataset, 'steps': args.steps, 'grid': args.simbench_code, 'results': results},
                     indent=2))


if __name__ == '__main__':
    main()
