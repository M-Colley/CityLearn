"""Dataset tooling for real and scenario electricity prices and carbon intensity.

Writes CityLearn `pricing.csv` and `carbon_intensity.csv`. Not imported by `CityLearnEnv`, so it adds no
runtime cost. ENTSO-E and PyPSA need the `price_carbon` extra: `pip install "citylearn[price_carbon]"`.
"""
