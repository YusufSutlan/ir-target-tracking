"""CLEAR-MOT style metrics (MOTA / MOTP) plus detection precision and recall."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match(truth: dict[int, tuple[float, float]], est: dict[int, np.ndarray], max_dist: float):
    """Optimal one-to-one matching of ground truth ids to estimate ids within max_dist."""
    if not truth or not est:
        return []
    tk, ek = list(truth), list(est)
    d = np.array([[np.hypot(*(np.array(truth[a]) - est[b])) for b in ek] for a in tk])
    cost = np.where(d <= max_dist, d, 1e6)
    return [(tk[i], ek[j], d[i, j]) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6]


def clear_mot(truth_seq, est_seq, max_dist: float = 4.0) -> dict[str, float]:
    gt = fn = fp = idsw = 0
    dists: list[float] = []
    last: dict[int, int] = {}
    for truth, est in zip(truth_seq, est_seq):
        gt += len(truth)
        pairs = match(truth, est, max_dist)
        for tid, eid, dist in pairs:
            if tid in last and last[tid] != eid:
                idsw += 1
            last[tid] = eid
            dists.append(dist)
        fn += len(truth) - len(pairs)
        fp += len(est) - len(pairs)
    return {
        "mota": 1 - (fn + fp + idsw) / max(gt, 1),
        "motp_px": float(np.mean(dists)) if dists else float("nan"),
        "false_negatives": fn,
        "false_positives": fp,
        "id_switches": idsw,
        "gt_objects": gt,
    }


def detection_pr(truth_seq, det_seq, max_dist: float = 3.0) -> dict[str, float]:
    tp = fp = fn = 0
    for truth, dets in zip(truth_seq, det_seq):
        est = {i: d for i, d in enumerate(dets)}
        pairs = match(truth, est, max_dist)
        tp += len(pairs)
        fn += len(truth) - len(pairs)
        fp += len(est) - len(pairs)
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
