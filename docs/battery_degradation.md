# Battery ageing with BLAST-Lite

Optional semi-empirical capacity fade for building batteries and EV batteries. It replaces the linear
`capacity_loss_coefficient` model of one battery with a [BLAST-Lite](https://github.com/NatLabRockies/BLAST-Lite)
cell model that depends on SOC, temperature, depth of discharge, C-rate and time. Without the setting nothing changes:
KPIs and simulation outputs are bit-identical to a run without this feature.

## Install

```console
python -m pip install "citylearn[degradation]"
```

BLAST-Lite pins `numpy<2`, so the extra restricts NumPy for the whole environment.

## Configure

Add `degradation` to the attributes of `electrical_storage` or of an EV `battery` in the schema:

```json
"electrical_storage": {
  "type": "citylearn.energy_model.Battery",
  "attributes": {
    "capacity": 6.4,
    "nominal_power": 5.0,
    "degradation": {
      "model": "blast_lite",
      "cell": "Lfp_Gr_250AhPrismatic",
      "update_interval_steps": 24,
      "temperature": "outdoor_dry_bulb"
    }
  }
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `blast_lite` | The only model. |
| `cell` | `Nmc111_Gr_Kokam75Ah_Battery` | Class name in `blast.models`. `citylearn.degradation.available_cells()` lists them. LFP: `Lfp_Gr_250AhPrismatic`, `Lfp_Gr_SonyMurata3Ah_Battery`. NMC: `Nmc111_Gr_Kokam75Ah_Battery`, `NMC_Gr_75Ah_A`, `Nmc622_Gr_DENSO50Ah_Battery`, `Nmc811_GrSi_LGM50_5Ah_Battery`. |
| `update_interval_steps` | 24 | Steps between BLAST-Lite state updates. |
| `temperature` | 25.0 | Constant in [C], or `outdoor_dry_bulb` for the building weather. An EV battery accepts only a constant. |
| `degradation_scalar` | 1.0 | Multiplier on the model degradation rate. |

Unknown keys, unknown cells and `update_interval_steps < 1` raise `ValueError`.

## What the battery does

- The battery records the SOC at the end of each step, as a fraction of its current capacity, with the temperature and the time.
- Every `update_interval_steps` steps it passes the batch to `update_battery_state`. The last sample starts the next batch, so no interval is lost.
- The capacity is `capacity * q`, with `q` from BLAST-Lite. It stays constant between updates, and `degraded_capacity`, `capacity_history` and `capacity_before_step` read it as before.
- CityLearn has no resistance model, so BLAST-Lite resistance growth is not applied.
- Steps after the last completed interval of an episode do not enter the capacity. `reset()` starts a new BLAST-Lite model.
- Between samples BLAST-Lite interpolates SOC linearly. EV batteries that are not charged in a step contribute no sample for it.

Use `update_interval_steps` of 2 or more. BLAST-Lite 1.1.3 computes the C-rate of a batch from `np.trapz` over
the interval rates without the first interval, so a one-step batch has C-rate 0 and fade depends on the interval.
A fix is proposed upstream.

## Cost

`scripts/manual/benchmark_battery_charge.py` times `Battery.charge` over 8760 steps. On a loaded Windows
machine, best of three, Python 3.12:

| Model | us per step |
| --- | --- |
| linear (default) | 115 |
| BLAST-Lite, interval 168 | 95 |
| BLAST-Lite, interval 24 | 130 |
| BLAST-Lite, interval 1 | 1144 |

A BLAST-Lite update costs about 0.7 to 0.8 ms per call, so the cost per step is that divided by the interval.

## Check against PyBaMM

`scripts/manual/validate_degradation_pybamm.py` compares BLAST-Lite with a PyBaMM SPMe simulation with
solvent-diffusion-limited SEI growth for one duty cycle. It is offline and not part of the runtime path:

```console
python -m pip install pybamm
python scripts/manual/validate_degradation_pybamm.py --cycles 100 --c-rate 0.5 --temperature 25
```

For the LG M50 cell (`Nmc811_GrSi_LGM50_5Ah_Battery` and PyBaMM `Chen2020`), 100 full cycles at C/2 and 25 C with a one-hour hold at full charge:

| Model | Capacity fade |
| --- | --- |
| PyBaMM SPMe + SEI | 0.24 % |
| BLAST-Lite | 4.60 % |

The difference is 4.4 percentage points. The PyBaMM SEI parameters are not fitted to ageing data, while BLAST-Lite is
fitted to measured cell ageing, so the script indicates the order of magnitude and does not calibrate either model.
