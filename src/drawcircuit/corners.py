"""Round off corners that are too tight for a real road.

Wherever the centerline turns tighter than the limit, each point there moves a
little towards the midpoint of its neighbours, over and over (curve-shortening
flow, applied only where the line is too tight). That rounds the inside of the
corner: the line cuts across it and the turn spreads over a longer arc, while
the rest of the line stays as drawn. A hairpin whose legs are closer together
than the limit allows cannot be rounded that far; it stops once the line has
moved ``max_shift`` and is reported, to be redrawn wider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import geometry


@dataclass
class Corner:
    s: float          # arc length along the line where it turned tightest, px
    before: float     # its centerline radius as drawn, px
    after: float      # after rounding, px
    limit: float      # the radius asked for there, px
    shift: float      # how far the line moved, px

    @property
    def met(self) -> bool:
        return self.after >= 0.97 * self.limit


def curvature(pts: np.ndarray, step: float, scale: float = 3.0) -> np.ndarray:
    """Signed curvature of a closed polyline sampled every ``step`` px, measured over
    ~``scale`` px so that pencil jitter does not count."""
    p = gaussian_filter1d(pts, scale / step, axis=0, mode="wrap")
    d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2
    d2 = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1), 1e-9) ** 3


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the runs of True in a circular mask."""
    n = len(mask)
    if mask.all():
        return [np.arange(n)]
    k = int(np.argmin(mask))  # start from a False
    rolled = np.roll(mask, -k)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], rolled.astype(int), [0]])))
    return [(np.arange(a, b) + k) % n for a, b in zip(edges[::2], edges[1::2])]


def round_corners(line: np.ndarray, radius, *, step: float | None = None, max_shift: float = np.inf,
                  iterations: int = 4000, headroom: float = 1.2) -> tuple[np.ndarray, list[Corner]]:
    """The closed polyline ``line`` with each corner tighter than ``radius`` rounded off.

    ``radius`` is the smallest centerline radius allowed, px: a number, or a function
    of the arc-length fraction along the line (for a road whose width changes).
    Corners are rounded to ``headroom`` times the radius, so that fitting a spline
    to the result afterwards does not take them back under it. The line is worked on
    at points ``step`` px apart, by default a tenth of the radius (at least 1 px), so a
    wide corner rounds off in as many steps as a tight one.
    """
    limit_at = (lambda f: radius(f)) if callable(radius) else (lambda f: np.full(len(f), float(radius)))
    if step is None:
        step = max(1.0, float(limit_at(np.linspace(0.0, 1.0, 400, endpoint=False)).min()) / 10.0)
    pts = geometry.resample(line, spacing=step, closed=True)
    original = pts.copy()
    kappa0 = curvature(pts, step)
    tight0 = np.abs(kappa0) * limit_at(np.arange(len(pts)) / len(pts)) > 1.0
    if not tight0.any():
        return pts, []

    from scipy.spatial import cKDTree
    tree = cKDTree(original)
    for it in range(iterations):
        if it % 50 == 0:  # keep the points evenly spaced, and stop where the line has moved enough
            pts = geometry.resample(pts, spacing=step, closed=True)
            n = len(pts)
            limit = limit_at(np.arange(n) / n)
            free = tree.query(pts)[0] < max_shift
        tight = np.abs(curvature(pts, step, 2.0)) * limit * headroom > 1.0
        tight &= free
        if not tight.any():
            break
        # Move the tight points most and their neighbours less and less, so the line
        # stays smooth where the rounding fades out.
        weight = np.clip(4.0 * gaussian_filter1d(tight.astype(float), 6.0 / step, mode="wrap"), 0.0, 1.0)
        mid = (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0)) / 2
        pts = pts + 0.5 * weight[:, None] * (mid - pts)
    pts = geometry.resample(pts, spacing=step, closed=True)

    n = len(pts)
    kappa, limit = curvature(pts, step), limit_at(np.arange(n) / n)
    shift = tree.query(pts)[0]
    corners = []
    for run in _runs(tight0):
        worst0 = run[np.argmax(np.abs(kappa0[run]))]
        # the same corner on the new line: around the same fraction of the lap
        at = int(round(worst0 / len(original) * n))
        near = (np.arange(n) - at + n // 2) % n - n // 2
        near = np.abs(near) <= max(len(run), int(2 * limit.max() / step))
        corners.append(Corner(float(worst0) * step, 1.0 / abs(kappa0[worst0]),
                              1.0 / max(float(np.abs(kappa[near]).max()), 1e-12),
                              float(limit[near].max()), float(shift[near].max())))
    return pts, corners


def turns(pts: np.ndarray, step: float, min_curvature: float) -> list[tuple[float, float]]:
    """(arc length, total turning in degrees) of each corner: a stretch that keeps turning
    the same way at a curvature above ``min_curvature``."""
    kappa = curvature(pts, step)
    out = []
    for sign in (1, -1):
        for run in _runs(sign * kappa > min_curvature):
            out.append((float(run[len(run) // 2]) * step, float(np.degrees(abs(kappa[run].sum()) * step))))
    return sorted(out)
