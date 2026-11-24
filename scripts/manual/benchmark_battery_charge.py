"""Time `Battery.charge` per step with the linear model and with BLAST-Lite (best of several repeats)."""
import time

from citylearn.base import EpisodeTracker
from citylearn.energy_model import Battery

STEPS = 8760


def make(degradation):
    tracker = EpisodeTracker(0, STEPS - 1)
    tracker.next_episode(STEPS, rolling_episode_split=False, random_episode_split=False, random_seed=0)
    battery = Battery(
        capacity=100.0, nominal_power=50.0, initial_soc=0.5, efficiency=0.9, loss_coefficient=0.0,
        capacity_loss_coefficient=1e-4, power_efficiency_curve=[[0.0, 0.9], [1.0, 0.9]],
        capacity_power_curve=[[0.0, 1.0], [1.0, 1.0]], depth_of_discharge=1.0, seconds_per_time_step=3600,
        episode_tracker=tracker, random_seed=0, degradation=degradation,
    )
    battery.reset()
    return battery


def run(degradation):
    best = float('inf')
    for _ in range(3):
        battery = make(degradation)
        start = time.perf_counter()
        for step in range(STEPS):
            battery.charge(30.0 if step % 24 < 6 else -10.0 if step % 24 < 18 else 0.0)
            if step < STEPS - 1:
                battery.next_time_step()
        best = min(best, time.perf_counter() - start)
    return 1e6*best/STEPS, battery.degraded_capacity/battery.capacity


if __name__ == '__main__':
    print('linear                    us/step=%.1f soh=%.4f' % run(None))
    for interval in (1, 24, 168):
        print('blast_lite interval=%-4d  us/step=%.1f soh=%.4f' % ((interval,) + run({'model': 'blast_lite', 'update_interval_steps': interval})))
