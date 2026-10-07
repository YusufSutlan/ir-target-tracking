"""Moving point-target detection: running background model, robust CFAR-style threshold."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass


class Detector:
    """Subtracts a slowly adapting background (removes clouds and static clutter) and
    thresholds the smoothed residual at `k` robust sigmas (median / MAD estimate)."""

    def __init__(self, k: float = 5.0, alpha: float = 0.04, min_area: int = 2, max_area: int = 60):
        self.k, self.alpha, self.min_area, self.max_area = k, alpha, min_area, max_area
        self.bg: np.ndarray | None = None

    def warmup(self, frames: np.ndarray) -> None:
        """Initialise the background with the temporal median of the first frames."""
        self.bg = np.median(frames, axis=0).astype(np.float32)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return detections as an (N, 2) array of sub-pixel (x, y) centroids."""
        if self.bg is None:
            self.bg = frame.copy()
        # Matched filter for a small Gaussian blob = Gaussian smoothing of the residual.
        residual = gaussian_filter(frame - self.bg, sigma=1.2)
        med = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - med))
        mask = residual > med + self.k * sigma

        # Update the background only where nothing was detected, so targets are not absorbed.
        update = ~(gaussian_filter(mask.astype(np.float32), 3) > 0)
        self.bg = np.where(update, (1 - self.alpha) * self.bg + self.alpha * frame, self.bg)

        labels, n = label(mask)
        if n == 0:
            return np.empty((0, 2))
        idx = np.arange(1, n + 1)
        areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        keep = idx[(areas >= self.min_area) & (areas <= self.max_area)]
        if len(keep) == 0:
            return np.empty((0, 2))
        com = center_of_mass(np.clip(residual - med, 0, None), labels, keep)
        return np.array([(x, y) for y, x in com])
