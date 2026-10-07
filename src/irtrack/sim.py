"""Synthetic infrared (IR) surveillance sequences.

Small, dim, moving hot spots (vehicles / aircraft seen from far away) over cloud-like
clutter, with static hot clutter (sun glints, buildings), sensor noise and
maneuvering targets that can cross each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Scenario:
    n_frames: int = 120
    height: int = 240
    width: int = 320
    n_targets: int = 4
    target_amplitude: float = 0.20  # peak contrast over local background (0..1 intensity scale)
    noise_sigma: float = 0.02
    n_static_clutter: int = 6
    seed: int = 0


def _blob(h: int, w: int, x: float, y: float, amp: float, sigma: float = 1.2) -> np.ndarray:
    """Gaussian point-spread function rendered only near (x, y) for speed."""
    img = np.zeros((h, w), dtype=np.float32)
    r = int(4 * sigma) + 1
    x0, y0 = int(round(x)), int(round(y))
    ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1))
    xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1))
    if len(ys) == 0 or len(xs) == 0:
        return img
    gx, gy = np.meshgrid(xs, ys)
    img[np.ix_(ys, xs)] = amp * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma**2))
    return img


def simulate(sc: Scenario) -> tuple[np.ndarray, list[dict[int, tuple[float, float]]]]:
    """Return (frames [T, H, W] float32, truth[t] = {target_id: (x, y)})."""
    rng = np.random.default_rng(sc.seed)
    h, w = sc.height, sc.width

    # Cloud clutter: low-frequency noise plus a vertical gradient (hot ground, cold sky).
    clutter = gaussian_filter(rng.normal(size=(h, w)), sigma=18)
    clutter = 0.12 * clutter / clutter.std()
    clutter += np.linspace(0.10, 0.35, h)[:, None]
    static = np.zeros((h, w), dtype=np.float32)
    for _ in range(sc.n_static_clutter):
        static += _blob(h, w, rng.uniform(10, w - 10), rng.uniform(10, h - 10), rng.uniform(0.15, 0.3), 1.5)
    background = (clutter + static).astype(np.float32)

    # Targets: constant velocity with one random heading change; they may leave the field of view.
    states = []
    for tid in range(sc.n_targets):
        x, y = rng.uniform(20, w - 20), rng.uniform(20, h - 20)
        speed, heading = rng.uniform(1.0, 2.5), rng.uniform(0, 2 * np.pi)
        turn_at = int(rng.integers(sc.n_frames // 4, max(sc.n_frames // 4 + 1, 3 * sc.n_frames // 4)))
        states.append(
            dict(id=tid, x=x, y=y, speed=speed, heading=heading, turn_at=turn_at,
                 turn=rng.uniform(-1.0, 1.0), amp=sc.target_amplitude * rng.uniform(0.8, 1.2))
        )  # fmt: skip

    frames = np.empty((sc.n_frames, h, w), dtype=np.float32)
    truth: list[dict[int, tuple[float, float]]] = []
    for t in range(sc.n_frames):
        frame = background + rng.normal(0, sc.noise_sigma, (h, w)).astype(np.float32)
        visible = {}
        for s in states:
            if t == s["turn_at"]:
                s["heading"] += s["turn"]
            s["x"] += s["speed"] * np.cos(s["heading"])
            s["y"] += s["speed"] * np.sin(s["heading"])
            if 3 <= s["x"] < w - 3 and 3 <= s["y"] < h - 3:
                frame += _blob(h, w, s["x"], s["y"], s["amp"])
                visible[s["id"]] = (s["x"], s["y"])
        frames[t] = frame
        truth.append(visible)
    return frames, truth
