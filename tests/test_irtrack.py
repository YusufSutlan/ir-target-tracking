import numpy as np

from irtrack.detect import Detector
from irtrack.metrics import clear_mot
from irtrack.pipeline import run_scenario
from irtrack.sim import Scenario, simulate
from irtrack.track import Kalman, Tracker


def test_simulation_is_deterministic():
    a, ta = simulate(Scenario(n_frames=20, seed=3))
    b, tb = simulate(Scenario(n_frames=20, seed=3))
    assert np.array_equal(a, b) and ta == tb


def test_detector_finds_a_single_blob_and_ignores_static_clutter():
    sc = Scenario(n_frames=30, n_targets=1, target_amplitude=0.3, seed=5)
    frames, truth = simulate(sc)
    det = Detector()
    det.warmup(frames[:10])
    d = det(frames[20])
    x, y = truth[20][0]
    assert len(d) >= 1
    assert min(np.hypot(*(p - np.array([x, y]))) for p in d) < 1.5


def test_kalman_estimates_velocity():
    kf = Kalman(np.array([0.0, 0.0]))
    for k in range(1, 25):
        kf.predict()
        kf.update(np.array([2.0 * k, -1.0 * k]))
    assert abs(kf.x[2] - 2.0) < 0.2 and abs(kf.x[3] + 1.0) < 0.2


def test_tracker_confirms_after_three_hits_and_drops_noise():
    tr = Tracker()
    out = [tr.step(np.array([[10.0 + k, 20.0]])) for k in range(4)]
    assert out[0] == [] and out[1] == [] and len(out[3]) == 1
    tr2 = Tracker()
    tr2.step(np.array([[5.0, 5.0]]))
    tr2.step(np.empty((0, 2)))
    tr2.step(np.empty((0, 2)))
    assert tr2.tracks == []


def test_mota_perfect_and_with_miss():
    truth = [{0: (1.0, 1.0)}, {0: (2.0, 2.0)}]
    assert clear_mot(truth, [{5: np.array([1.0, 1.0])}, {5: np.array([2.0, 2.0])}])["mota"] == 1.0
    assert clear_mot(truth, [{5: np.array([1.0, 1.0])}, {}])["mota"] == 0.5


def test_high_contrast_scenario_is_tracked_well_and_low_contrast_is_harder():
    easy = run_scenario(Scenario(target_amplitude=0.3, seed=0))
    hard = run_scenario(Scenario(target_amplitude=0.04, seed=0))
    assert easy["tracking"]["mota"] > 0.75
    assert easy["tracking"]["mota"] > hard["tracking"]["mota"]
