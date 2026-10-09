"""Penalised least-squares cubic B-spline fitting (P-splines) for open and closed curves.

The curve parameter u in [0, 1] is chord length, so it is roughly proportional to
arc length. Knots are placed by a density that is higher where the curve bends.
Open curves use a clamped knot vector, so the curve starts and ends near the
drawn endpoints. Closed curves use a periodic (unclamped) knot vector with the
first ``degree`` control points repeated at the end, which keeps the curve
C2-continuous across the seam.

The number of control points grows until the fit is within ``tol`` pixels of
the drawn line. A small second-difference
penalty on the control points (``smooth``) keeps the control polygon well
behaved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d

from . import geometry


@dataclass
class Spline:
    """A planar B-spline curve.

    ``knots`` and ``control_points`` follow the standard convention
    ``len(knots) == len(control_points) + degree + 1``, so any de Boor evaluator
    reproduces the curve over ``domain``.
    """

    degree: int
    knots: np.ndarray
    control_points: np.ndarray
    periodic: bool
    rms_error: float = 0.0
    max_error: float = 0.0

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.knots[self.degree]), float(self.knots[len(self.control_points)])

    @property
    def unique_control_points(self) -> np.ndarray:
        """Control points without the periodic wrap-around repeats."""
        if self.periodic:
            return self.control_points[: len(self.control_points) - self.degree]
        return self.control_points

    def _bspline(self) -> BSpline:
        return BSpline(self.knots, self.control_points, self.degree,
                       extrapolate="periodic" if self.periodic else False)

    def __call__(self, u, nu: int = 0) -> np.ndarray:
        b = self._bspline()
        return (b.derivative(nu) if nu else b)(np.asarray(u, dtype=float))

    def params(self, count: int) -> np.ndarray:
        a, b = self.domain
        return np.linspace(a, b, count, endpoint=not self.periodic)

    def sample(self, count: int) -> np.ndarray:
        return self(self.params(count))

    def length(self, count: int = 4000) -> float:
        return geometry.length(self.sample(count), closed=self.periodic)


def _curvature(pts: np.ndarray, closed: bool, spacing: float, scale: float = 8.0) -> np.ndarray:
    """|curvature| (1/px) at samples ``spacing`` px apart, measured at a ``scale`` of
    several pixels so that hand jitter does not read as tight turns. Smoothing
    spreads a sharp corner out but keeps its total turning, which is what the
    knot density integrates."""
    mode = "wrap" if closed else "nearest"
    sigma = scale / spacing
    p = gaussian_filter1d(pts, sigma, axis=0, mode=mode)
    if closed:
        d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2
        d2 = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
    else:
        d1 = np.gradient(p, axis=0)
        d2 = np.gradient(d1, axis=0)
    speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    return gaussian_filter1d(k, sigma, mode=mode)


def _knots(n: int, k: int, closed: bool, u_at: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Knot vector whose spans each hold an equal share of the knot density.

    ``share`` rises from 0 to 1 along the curve and ``u_at`` gives the curve
    parameter at the same samples, so knots crowd where the density is high.
    """
    if closed:
        inner = np.interp(np.arange(n + 1) / n, share, u_at)
        inner[0], inner[-1] = 0.0, 1.0
        return np.concatenate([inner[n - k:n] - 1.0, inner, inner[1:k + 1] + 1.0])
    interior = np.interp(np.arange(1, n - k) / (n - k), share, u_at)
    return np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)])


def _design(u: np.ndarray, t: np.ndarray, n: int, k: int, closed: bool) -> np.ndarray:
    b = BSpline.design_matrix(u, t, k).toarray()
    if closed:  # fold the k wrap-around basis functions onto the first k control points
        b[:, :k] += b[:, n:]
        b = b[:, :n]
    return b


def _second_difference(n: int, closed: bool) -> np.ndarray:
    if not closed:
        return np.diff(np.eye(n), 2, axis=0)
    d = -2.0 * np.eye(n)
    d += np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), -1, axis=1)
    return d


def fit(points: np.ndarray, closed: bool, *, tol: float = 2.5, cp_spacing: float = 60.0,
        knots_per_radian: float = 4.0, smooth: float = 0.05, degree: int = 3,
        sample_spacing: float = 1.5, max_iter: int = 20) -> Spline:
    """Fit a cubic B-spline to a drawn polyline.

    ``points`` is an (N, 2) polyline; for closed curves do not repeat the first point.
    Knots are spaced ``cp_spacing`` apart on straights, with ``knots_per_radian``
    more for every radian the curve turns, so corners get the control points they
    need without crowding the straights.
    """
    pts = geometry.resample(points, spacing=sample_spacing, closed=closed)
    # Remove pixel staircase noise from the skeleton before measuring fit error.
    pts = gaussian_filter1d(pts, 1.0, axis=0, mode="wrap" if closed else "nearest")
    s = geometry.cumulative_length(pts, closed)
    total = s[-1]
    u = s[:-1] / total if closed else s / total
    if not closed:
        u[-1] = 1.0
    m = len(pts)

    weight = np.ones(m)
    if not closed:  # hold the curve to the drawn endpoints
        weight[[0, -1]] = 25.0
    wpts = pts * weight[:, None]

    density = 1.0 / cp_spacing + knots_per_radian * _curvature(pts, closed, sample_spacing)
    seg = np.diff(s)
    mid = (density + np.roll(density, -1)) / 2 if closed else (density[:-1] + density[1:]) / 2
    cum = np.concatenate([[0.0], np.cumsum(mid * seg)])
    u_at = s / total

    n_min = 2 * degree + 2 if closed else degree + 1
    n_max = max(n_min, m // 2)
    # Start sparse and add knots (keeping their distribution) until the fit is good,
    # which lands near the fewest control points that meet the tolerance.
    n = int(np.clip(round(0.15 * cum[-1]), n_min, n_max))
    for _ in range(max_iter):
        t = _knots(n, degree, closed, u_at, cum / cum[-1])
        b = _design(u, t, n, degree, closed)
        d = _second_difference(n, closed)
        a = b.T @ (b * weight[:, None]) + (smooth * m / n) * (d.T @ d)
        c = np.linalg.solve(a, b.T @ wpts)
        residual = np.linalg.norm(b @ c - pts, axis=1)
        # A high percentile rather than the max, so a few skeleton spikes do not
        # force the whole curve onto a dense control polygon.
        if np.percentile(residual, 98) <= tol or n >= n_max:
            break
        n = min(n_max, int(n * 1.2) + 1)

    control = np.vstack([c, c[:degree]]) if closed else c
    return Spline(degree, t, control, closed,
                  rms_error=float(np.sqrt(np.mean(residual ** 2))),
                  max_error=float(residual.max()))
