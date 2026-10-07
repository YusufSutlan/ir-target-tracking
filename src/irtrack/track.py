"""Multi-target tracker: constant-velocity Kalman filters + Hungarian data association."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_GATE_2DOF_99 = 9.21


class Kalman:
    """State [x, y, vx, vy], position-only measurements."""

    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def __init__(self, z: np.ndarray, q: float = 0.15, r: float = 0.5):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([r, r, 25.0, 25.0])
        self.Q = q * np.diag([0.25, 0.25, 1.0, 1.0])
        self.R = r * np.eye(2)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = self.H @ self.P @ self.H.T + self.R
        return z - self.H @ self.x, S

    def update(self, z: np.ndarray) -> None:
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class Track:
    id: int
    kf: Kalman
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    history: list = field(default_factory=list)


class Tracker:
    """M-of-N style track management: confirm after `confirm_hits` detections, delete after
    `max_misses` consecutive misses (tentative tracks are dropped on their first miss)."""

    def __init__(self, confirm_hits: int = 3, max_misses: int = 6):
        self.confirm_hits, self.max_misses = confirm_hits, max_misses
        self.tracks: list[Track] = []
        self._ids = count()

    def step(self, detections: np.ndarray) -> list[Track]:
        for t in self.tracks:
            t.kf.predict()
            t.age += 1

        matched_t, matched_d = set(), set()
        if self.tracks and len(detections):
            cost = np.full((len(self.tracks), len(detections)), 1e6)
            for i, t in enumerate(self.tracks):
                for j, z in enumerate(detections):
                    y, S = t.kf.innovation(z)
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 < CHI2_GATE_2DOF_99:
                        cost[i, j] = d2
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    self.tracks[i].kf.update(detections[j])
                    self.tracks[i].hits += 1
                    self.tracks[i].misses = 0
                    matched_t.add(i)
                    matched_d.add(j)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
            if t.hits >= self.confirm_hits:
                t.confirmed = True

        self.tracks = [
            t for t in self.tracks
            if t.misses <= (self.max_misses if t.confirmed else 1)
        ]  # fmt: skip
        for j, z in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(next(self._ids), Kalman(z)))

        for t in self.tracks:
            t.history.append(t.kf.x[:2].copy())
        return [t for t in self.tracks if t.confirmed and t.misses == 0]
