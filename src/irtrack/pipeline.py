"""Run detector + tracker over a scenario and score it."""

from __future__ import annotations

import numpy as np

from .detect import Detector
from .metrics import clear_mot, detection_pr
from .sim import Scenario, simulate
from .track import Tracker

WARMUP = 10


def run_scenario(sc: Scenario, k: float = 5.0):
    frames, truth = simulate(sc)
    detector = Detector(k=k)
    detector.warmup(frames[:WARMUP])
    tracker = Tracker()

    dets, tracks_out = [], []
    for frame in frames[WARMUP:]:
        d = detector(frame)
        dets.append(d)
        tracks_out.append({t.id: t.kf.x[:2].copy() for t in tracker.step(d)})
    truth = truth[WARMUP:]
    return {
        "frames": frames[WARMUP:],
        "truth": truth,
        "detections": dets,
        "tracks": tracks_out,
        "detection": detection_pr(truth, dets),
        "tracking": clear_mot(truth, tracks_out),
    }
