# Price and Carbon Datasets

`citylearn.price_carbon` writes `pricing.csv` and `carbon_intensity.csv` for a time range at the dataset's resolution. Three sources are available: ENTSO-E day-ahead prices and generation mix for a bidding zone, the public Carbon Intensity API for Great Britain, and a PyPSA dispatch model for a future scenario. The module is not imported by `CityLearnEnv`, so it adds no simulation cost. It writes files; point a schema's `pricing` and `carbon_intensity` entries at them.

```console
pip install "citylearn[price_carbon]"   # entsoe-py, PyPSA, HiGHS
```

## Output

| File | Columns | Unit |
|---|---|---|
| `pricing.csv` | `electricity_pricing`, `electricity_pricing_predicted_1/2/3` | currency/kWh |
| `carbon_intensity.csv` | `carbon_intensity` | kgCO2/kWh |
| `price_carbon_metadata.json` | source, units, factors, first and last step (UTC), counts | |

`carbon_intensity.csv` has no predicted columns because CityLearn reads only `carbon_intensity`.

The three predicted price columns hold the actual price `--horizons` hours ahead (default 1, 2 and 3 hours): a perfect-foresight forecast. The last steps repeat the final price. Replace them if a controller must be evaluated with imperfect forecasts.

The grid is contiguous in absolute time from `--start` (interpreted in `--tz` unless it carries an offset) to `--end` (exclusive). A local day at a daylight-saving change therefore has 23 or 25 hourly steps. Native data at 15, 30 or 60 minutes is averaged or held onto the grid. Up to `--max-gap-steps` (default 3) missing steps are interpolated; a longer gap is an error. Prices can be negative and are kept as they are.

## ENTSO-E (EU bidding zones)

Request a free key at transparency.entsoe.eu and export it. The tool reads it only from `ENTSOE_API_KEY` and never logs or writes it.

```console
export ENTSOE_API_KEY=...
python -m citylearn.price_carbon entsoe --zone DE_LU --start 2024-01-01 --end 2025-01-01 \
    --tz Europe/Berlin --resolution-minutes 60 --out data/my_dataset
```

Prices are EUR/MWh divided by 1000. Carbon intensity is `sum(generation_i * factor_i) / sum(generation_i)` over the generation types with a factor, using the lifecycle medians of IPCC AR5 WG3 Annex III (Table A.III.2, gCO2eq/kWh): coal 820, gas 490, biomass 230, geothermal 38, hydropower 24, nuclear 12, solar PV (utility) 48, wind onshore 11, wind offshore 12. Pumped-storage output counts as hydropower and net consumption is clipped to zero. Oil, waste, peat, marine and "other" types have no AR5 median and are left out; the metadata lists them and their share, and the tool stops when it exceeds 10 % of the generation. The result is a production-based average that ignores imports and is not a marginal intensity.

## Great Britain (no key)

```console
python -m citylearn.price_carbon gb --start 2024-01-01 --end 2025-01-01 --resolution-minutes 30 --out data/gb
```

The tool calls `api.carbonintensity.org.uk/intensity` in chunks of 13 days. `--field actual` (default) falls back to the forecast where the actual value is null. The values are national averages in gCO2/kWh, converted to kgCO2/kWh. The command writes `carbon_intensity.csv` only.

## Future scenarios (PyPSA)

A YAML file defines snapshots, buses, a load, generators with marginal cost (currency/MWh) and `emission_factor` (gCO2/kWh), optional storage units and links. HiGHS solves the least-cost dispatch. The price is the marginal price of the bus balance. The carbon intensity is the dispatched emissions divided by the dispatched energy (`intensity_basis: system`, or `bus` for the generators at the price bus). A load-shedding generator keeps the model feasible; the tool warns when it is used.

```console
python -m citylearn.price_carbon scenario --config examples/price_carbon/scenario.yaml --out data/scenario_2035
```

See `examples/price_carbon/scenario.yaml` and the module docstring of `citylearn/price_carbon/scenario.py` for the keys. Profiles come from a CSV (`file`, `column`), a repeated list (`values`) or a seeded synthetic series (`seed`, `mean`, `sd`). The marginal price of a tight single-node system is the cost of the marginal generator, so scarcity prices appear only when the shedding cost or a binding capacity sets them.

## Python

```python
from citylearn.price_carbon.entsoe import entsoe_datasets
from citylearn.price_carbon.common import write_outputs

pricing, carbon, grid, meta = entsoe_datasets('DE_LU', '2024-01-01', '2025-01-01', 60, 'Europe/Berlin')
write_outputs('data/my_dataset', pricing, carbon, meta, grid)
```
