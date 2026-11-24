"""Command line: `python -m citylearn.price_carbon {entsoe,gb,scenario} ...`."""

from __future__ import annotations

import argparse
import logging

from citylearn.price_carbon.common import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m citylearn.price_carbon', description='Write CityLearn pricing.csv and carbon_intensity.csv.')
    sub = parser.add_subparsers(dest='source', required=True)

    def common(p, time_range=True):
        p.add_argument('--out', required=True, help='Output directory.')
        p.add_argument('--horizons', type=float, nargs=3, default=(1, 2, 3), metavar='H', help='Prediction horizons in hours for the three predicted columns.')
        if time_range:
            p.add_argument('--start', required=True, help='First step, ISO time (interpreted in --tz unless it has an offset).')
            p.add_argument('--end', required=True, help='End, exclusive.')
            p.add_argument('--resolution-minutes', type=int, default=60)
            p.add_argument('--max-gap-steps', type=int, default=3, help='Longest gap that is interpolated; longer gaps are an error.')

    e = sub.add_parser('entsoe', help='ENTSO-E day-ahead prices and generation-mix carbon (key in ENTSOE_API_KEY).')
    common(e)
    e.add_argument('--zone', required=True, help='Bidding zone, e.g. DE_LU, FR, NL.')
    e.add_argument('--tz', default='Europe/Brussels', help='Time zone for --start and --end.')

    g = sub.add_parser('gb', help='Great Britain carbon intensity (no key). Writes carbon_intensity.csv only.')
    common(g)
    g.add_argument('--tz', default='UTC')
    g.add_argument('--field', choices=['actual', 'forecast'], default='actual')

    s = sub.add_parser('scenario', help='Future scenario from a PyPSA dispatch model.')
    common(s, time_range=False)
    s.add_argument('--config', required=True, help='Scenario YAML.')
    s.add_argument('--solver', default='highs')
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    pricing = None
    if args.source == 'entsoe':
        from citylearn.price_carbon.entsoe import entsoe_datasets
        pricing, carbon, grid, meta = entsoe_datasets(args.zone, args.start, args.end, args.resolution_minutes, args.tz, args.horizons, args.max_gap_steps)
    elif args.source == 'gb':
        from citylearn.price_carbon.gb import gb_carbon
        carbon, grid, meta = gb_carbon(args.start, args.end, args.resolution_minutes, args.tz, args.field, args.max_gap_steps)
    else:
        from citylearn.price_carbon.scenario import load_config, scenario_datasets
        pricing, carbon, grid, meta = scenario_datasets(load_config(args.config), horizons_hours=args.horizons, solver_name=args.solver)
    for path in write_outputs(args.out, pricing, carbon, meta, grid):
        print(path)


if __name__ == '__main__':
    main()
