"""Tests of the BOPTEST validation harness. No Docker or network: the REST API is mocked and the
trajectory is a recorded BOPTEST result file (``data/validation/boptest``)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citylearn.validation import boptest as bt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'validation' / 'boptest' / 'bestest_hydronic_heat_pump'
SCHEMA = ROOT / 'data' / 'datasets' / 'quebec_neighborhood_without_demand_response_set_points' / 'schema.json'
BUILDING = 'resstock-amy2018-2021-release-1-247942'


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body, self.status_code, self.text = body, status_code, str(body)

    def json(self):
        return self._body


class FakeSession:
    """Replays BOPTEST v2 API responses; the results payload comes from a recorded results file."""

    def __init__(self, results_csv, fail_on=None):
        self.calls = []
        self.recorded = pd.read_csv(results_csv)
        self.fail_on = fail_on

    def request(self, method, url, timeout=None, json=None, **_):
        path = url.split('://', 1)[1].split('/', 1)[1]
        self.calls.append((method, '/' + path.rsplit('/', 1)[0], json))

        if self.fail_on and self.fail_on in path:
            return FakeResponse({'status': 500, 'message': 'boom', 'payload': None})
        if path.startswith('testcases/') and path.endswith('/select'):
            return FakeResponse({'testid': 'abc'})
        if path.startswith('results/'):
            columns = ['time', *bt.HEAT_PUMP_POINTS]
            return FakeResponse({'status': 200, 'message': 'ok', 'payload': {c: self.recorded[c].tolist() for c in columns}})
        if path.startswith('missing'):
            return FakeResponse({}, status_code=404)

        return FakeResponse({'status': 200, 'message': 'ok', 'payload': {}})


@pytest.fixture
def results_csv():
    return DATA / 'results_peak_heat_day.csv'


def test_record_open_loop_request_sequence(results_csv):
    session = FakeSession(results_csv)
    client = bt.BOPTESTClient('http://boptest.test', session=session)
    frame = bt.record_open_loop(client, 'bestest_hydronic_heat_pump', 777600.0, 3 * 3600.0, 3600.0, [0.0, 0.5, 1.0])

    sequence = [(m, p) for m, p, _ in session.calls]
    assert sequence == [
        ('POST', '/testcases/bestest_hydronic_heat_pump'), ('PUT', '/initialize'), ('PUT', '/step'),
        ('POST', '/advance'), ('POST', '/advance'), ('POST', '/advance'), ('PUT', '/results'), ('PUT', '/stop'),
    ]
    assert [c[2]['oveHeaPumY_u'] for c in session.calls if c[1] == '/advance'] == [0.0, 0.5, 1.0]
    assert all(c[2]['oveHeaPumY_activate'] == 1 for c in session.calls if c[1] == '/advance')
    assert client.testid is None
    assert list(frame.columns) == ['time', *bt.HEAT_PUMP_POINTS]


def test_record_open_loop_rejects_wrong_schedule_length_before_any_request(results_csv):
    session = FakeSession(results_csv)
    with pytest.raises(ValueError, match='expected 2'):
        bt.record_open_loop(bt.BOPTESTClient('http://boptest.test', session=session), 'x', 0.0, 7200.0, 3600.0, [1.0])
    assert session.calls == []


def test_error_status_raises_and_test_is_stopped(results_csv):
    session = FakeSession(results_csv, fail_on='advance')
    client = bt.BOPTESTClient('http://boptest.test', session=session)
    with pytest.raises(bt.BOPTESTError, match='boom'):
        bt.record_open_loop(client, 'x', 0.0, 3600.0, 3600.0, [1.0])
    assert session.calls[-1][1] == '/stop'


def test_http_error_raises(results_csv):
    client = bt.BOPTESTClient('http://boptest.test', session=FakeSession(results_csv))
    with pytest.raises(bt.BOPTESTError, match='404'):
        client._request('GET', '/missing/1')


def test_calendar_matches_boptest_testcase():
    # BOPTEST zero time is Tuesday 1 January 2019; occupied before 07:00, from 20:00 and on weekends.
    hours = np.array([0, 6, 7, 12, 19, 20, 23]) * 3600.0
    assert bt.occupants(hours).tolist() == [5, 5, 0, 0, 0, 5, 5]
    saturday = 4 * 86400.0 + 12 * 3600.0
    monday = 6 * 86400.0 + 12 * 3600.0
    assert bt.occupants(np.array([saturday, monday])).tolist() == [5, 0]
    low, high = bt.comfort_band(np.array([12 * 3600.0, 3 * 3600.0]))
    assert (low.tolist(), high.tolist()) == ([15, 21], [30, 24])


def test_discomfort_kh_counts_distance_outside_band():
    # 12:00 on a weekday is unoccupied (15 to 30 C); 03:00 is occupied (21 to 24 C).
    time_s = np.array([4 * 3600.0, 13 * 3600.0])  # step ends, so the steps start at 03:00 and 12:00
    assert bt.discomfort_kh(np.array([20.0, 14.0]), time_s) == pytest.approx(1.0 + 1.0)
    assert bt.discomfort_kh(np.array([22.0, 20.0]), time_s) == 0.0


def test_reference_trajectory_conserves_recorded_energy(results_csv):
    raw = pd.read_csv(results_csv)
    trajectory = bt.load_reference_trajectory(results_csv, DATA / 'weather_subset.csv')
    assert trajectory['time_s'].diff().dropna().eq(3600.0).all()
    assert trajectory[['t_zone_c', 't_out_c', 'heat_kwh', 'electric_kwh']].notna().all().all()
    # Energy inside the resampled window equals the trapezoid integral of the recorded power over that window.
    t0, t1 = trajectory['time_s'].iloc[0] - 3600.0, trajectory['time_s'].iloc[-1]
    inside = raw[(raw['time'] >= t0) & (raw['time'] <= t1)]
    reference = np.trapezoid(inside['reaQFloHea_y'], inside['time']) / 3.6e6
    assert trajectory['heat_kwh'].sum() == pytest.approx(reference, rel=0.02)
    assert trajectory['t_zone_c'].between(15, 30).all()


@pytest.fixture(scope='module')
def emulator():
    pytest.importorskip('torch')
    return bt.CityLearnThermalEmulator(bt.load_schema_spec(SCHEMA, BUILDING))


def test_emulator_warm_up_copies_measurements_and_rolls_out(emulator, results_csv):
    trajectory = bt.load_reference_trajectory(results_csv, DATA / 'weather_subset.csv').iloc[:40]
    lookback = emulator.dynamics.lookback
    free = emulator.simulate(trajectory, free_run=True)
    step = emulator.simulate(trajectory, free_run=False)
    measured = trajectory['t_zone_c'].to_numpy()
    assert np.array_equal(free[:lookback], measured[:lookback])
    assert np.isfinite(free).all() and np.isfinite(step).all()
    assert np.array_equal(free, emulator.simulate(trajectory, free_run=True))  # deterministic (eval mode)


def test_compare_reports_all_metrics(emulator, results_csv):
    trajectory = bt.load_reference_trajectory(results_csv, DATA / 'weather_subset.csv').iloc[:60]
    result = bt.compare(trajectory, emulator)
    assert result['steps'] == 60 - emulator.dynamics.lookback
    for key in ('free_run', 'one_step', 'constant_mean'):
        assert set(result[key]) == {'rmse_k', 'bias_k', 'mae_k', 'discomfort_kh'}
    assert result['constant_mean']['bias_k'] == pytest.approx(0.0, abs=1e-9)
    assert result['electric_difference_pct'] == pytest.approx(
        100 * (result['citylearn_electric_kwh'] / result['boptest']['electric_kwh'] - 1))
    assert 'RMSE' in bt.format_report('x', result)
