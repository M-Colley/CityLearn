"""Compare BLAST-Lite capacity fade with a PyBaMM SPMe + SEI simulation for one cycling duty. Offline only.

    pip install citylearn[degradation] pybamm
    python scripts/manual/validate_degradation_pybamm.py --cycles 100 --c-rate 0.5 --temperature 25

The default pair is the LG M50 cell: BLAST-Lite `Nmc811_GrSi_LGM50_5Ah_Battery` and PyBaMM `Chen2020`.
Both run full-window equivalent cycles. PyBaMM capacity is the discharge capacity of the last cycle relative to
the first; BLAST-Lite is driven through `citylearn.degradation.BlastLiteDegradation`, as CityLearn uses it.
The difference is a sanity check of the order of magnitude, not a calibration of either model.
"""
import argparse
import json

import numpy as np

from citylearn.degradation import BlastLiteDegradation


def pybamm_fade(cycles, c_rate, temperature_c, parameter_set, v_min, v_max):
    import pybamm

    pybamm.set_logging_level('ERROR')
    model = pybamm.lithium_ion.SPMe({'SEI': 'solvent-diffusion limited', 'SEI porosity change': 'true'})
    parameters = pybamm.ParameterValues(parameter_set)
    parameters['Ambient temperature [K]'] = temperature_c + 273.15
    parameters['Initial temperature [K]'] = temperature_c + 273.15
    cycle = (
        f'Discharge at {c_rate}C until {v_min} V',
        f'Charge at {c_rate}C until {v_max} V',
        f'Hold at {v_max} V until C/50',
    )
    solution = pybamm.Simulation(model, parameter_values=parameters, experiment=pybamm.Experiment([cycle]*cycles)).solve(calc_esoh=False)
    capacity = np.array([c.steps[0]['Discharge capacity [A.h]'].entries[-1] for c in solution.cycles])
    return capacity[-1]/capacity[0], capacity[0]


def blast_fade(cell, cycles, c_rate, temperature_c, step_seconds=60.0, interval_steps=60):
    model = BlastLiteDegradation(cell=cell, update_interval_steps=interval_steps, temperature=temperature_c)
    steps_half = int(round(3600.0/c_rate/step_seconds))
    # Discharge 1 -> 0, charge 0 -> 1, then a 1 h hold at full charge standing for the constant-voltage step.
    trajectory = np.concatenate([np.linspace(1, 0, steps_half + 1)[1:], np.linspace(0, 1, steps_half + 1)[1:], np.ones(int(3600/step_seconds))])
    model.reset(1.0)
    step = 0
    for _ in range(cycles):
        for soc in trajectory:
            model.record(step, step_seconds, float(soc))
            step += 1
    return model.q


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cell', default='Nmc811_GrSi_LGM50_5Ah_Battery')
    p.add_argument('--parameter-set', default='Chen2020')
    p.add_argument('--v-min', type=float, default=2.5)
    p.add_argument('--v-max', type=float, default=4.2)
    p.add_argument('--cycles', type=int, default=100)
    p.add_argument('--c-rate', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=25.0)
    a = p.parse_args()
    q_pybamm, q0 = pybamm_fade(a.cycles, a.c_rate, a.temperature, a.parameter_set, a.v_min, a.v_max)
    q_blast = blast_fade(a.cell, a.cycles, a.c_rate, a.temperature)
    result = {
        'cell': a.cell, 'parameter_set': a.parameter_set, 'cycles': a.cycles, 'c_rate': a.c_rate, 'temperature_c': a.temperature,
        'pybamm_capacity_ah_first': q0, 'pybamm_retention': q_pybamm, 'blast_retention': q_blast,
        'fade_pybamm_pct': 100*(1 - q_pybamm), 'fade_blast_pct': 100*(1 - q_blast),
        'fade_difference_pp': 100*(q_pybamm - q_blast),
    }
    print(json.dumps(result, indent=2))
