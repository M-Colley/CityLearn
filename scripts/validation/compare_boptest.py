"""Compare the CityLearn LSTM thermal model with BOPTEST ``bestest_hydronic_heat_pump``.

Offline (default): replays recorded BOPTEST results shipped in ``data/validation/boptest``.
Live: ``--base-url http://127.0.0.1:8000`` records a new open-loop run from a BOPTEST service first
(the recorded schedule is a constant modulation plus an on/off setback; see ``--schedule``).

    python scripts/validation/compare_boptest.py --output-json out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from citylearn.validation import boptest as bt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'validation' / 'boptest' / 'bestest_hydronic_heat_pump'
SCHEMA = ROOT / 'data' / 'datasets' / 'quebec_neighborhood_without_demand_response_set_points' / 'schema.json'
BUILDING = 'resstock-amy2018-2021-release-1-247942'
PERIODS = {'peak_heat_day': 'results_peak_heat_day.csv', 'typical_heat_day': 'results_typical_heat_day.csv'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--periods', nargs='+', default=list(PERIODS), choices=list(PERIODS))
    p.add_argument('--schema', type=Path, default=SCHEMA)
    p.add_argument('--building', default=BUILDING)
    p.add_argument('--weather', type=Path, default=DATA / 'weather_subset.csv')
    p.add_argument('--base-url', help='BOPTEST service URL for a live recording instead of the recorded results.')
    p.add_argument('--start-day', type=float, default=9.0, help='Live run start (day of year).')
    p.add_argument('--days', type=float, default=7.0, help='Live run length in days.')
    p.add_argument('--modulation', type=float, default=0.5, help='Live run heat pump modulation (daytime).')
    p.add_argument('--output-json', type=Path)
    args = p.parse_args(argv)

    emulator = bt.CityLearnThermalEmulator(bt.load_schema_spec(args.schema, args.building))
    sources = {k: DATA / v for k, v in PERIODS.items() if k in args.periods}

    if args.base_url:
        step, start = bt.HOUR_S, args.start_day * 86400.0
        n = int(args.days * 24)
        hours = (start / step + np.arange(n)) % 24
        schedule = np.where((hours >= 6) & (hours < 22), args.modulation, 0.0)
        raw = bt.record_open_loop(bt.BOPTESTClient(args.base_url), 'bestest_hydronic_heat_pump', start, n * step, step, schedule)
        live = Path('boptest_live_results.csv')
        raw.to_csv(live, index=False)
        sources = {'live_open_loop': live}

    results = {}

    for name, path in sources.items():
        trajectory = bt.load_reference_trajectory(path, args.weather)
        results[name] = bt.compare(trajectory, emulator)
        print(bt.format_report(name, results[name]))

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
