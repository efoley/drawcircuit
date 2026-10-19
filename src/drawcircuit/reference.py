"""A smooth reference line around a circuit, with the road width on either side.

This is the input a racing-line optimiser works from: a closed line plus the
distances ``w_left`` and ``w_right`` from it to the two edges, measured along its
normal. The line starts as the true midline (the points equally far from both
edges), which has kinks at sharp corners. It is smoothed with a Gaussian of
growing width until

    1 - w_inside * |curvature| >= min_factor

everywhere, where ``w_inside`` is the width on the side the line turns towards.
Where that factor reaches 0, the normals of neighbouring points cross inside the
road and the line can no longer serve as a coordinate frame for it. Smoothing
helps twice: tight turns get wider and the line moves towards their inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from . import bspline, geometry
from .bspline import Spline

MIN_FACTOR = 0.3


@dataclass
class Reference:
    line: Spline             # closed, starts beside the edges' start, runs in the direction of travel
    s: np.ndarray            # arc length along the line at each row, px (evenly spaced)
    u: np.ndarray            # line parameter at each row
    w_left: np.ndarray       # distance to the left edge along the left normal, px
    w_right: np.ndarray      # distance to the right edge along the right normal, px
    sigma: float             # Gaussian smoothing applied to the midline, px
    min_factor: float        # min over the lap of 1 - w_inside * |curvature|
    min_radius: float        # px
    max_deviation: float = 0.0  # farthest the line strays from the true midline, px

    @property
    def length(self) -> float:
        """Lap length along the line, px: the row spacing times the number of rows."""
        return float(self.s[1] - self.s[0]) * len(self.s)

    def u_at(self, s) -> np.ndarray:
        """Line parameter at arc length ``s`` px, wrapping around the lap."""
        total = self.length
        return np.interp(np.asarray(s, dtype=float) % total, np.append(self.s, total),
                         np.append(self.u, self.line.domain[1]))


def midline(a: np.ndarray, b: np.ndarray, grid: float = 0.5) -> np.ndarray | None:
    """Points equally far from two nested closed polylines, as a closed polyline.

    That is the zero level of d(outer) - d(inner), which is negative outside the
    road and positive in the infield, so it is evaluated only near the road.
    """
    outer, inner = (a, b) if abs(geometry.signed_area(a)) > abs(geometry.signed_area(b)) else (b, a)
    origin = outer.min(axis=0) - 4 * grid
    w, h = (np.ceil((outer.max(axis=0) - origin) / grid) + 5).astype(int)

    def fill(poly: np.ndarray) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round((poly - origin) / grid * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        return m.astype(bool)

    in_outer, in_inner = fill(outer), fill(inner)
    road = cv2.dilate((in_outer & ~in_inner).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    f = np.where(in_inner, 1.0, -1.0).astype(np.float32)
    ys, xs = np.nonzero(road)
    pts = origin + np.column_stack([xs, ys]) * grid
    f[ys, xs] = cKDTree(outer).query(pts)[0] - cKDTree(inner).query(pts)[0]

    contours = [c for c in find_contours(f, 0.0) if len(c) > 3 and np.allclose(c[0], c[-1])]
    if not contours:
        return None
    rc = max(contours, key=geometry.length)[:-1]
    line = origin + rc[:, ::-1] * grid
    if geometry.length(line, True) < 0.5 * geometry.length(inner, True):
        return None
    return line


def _cast(o: np.ndarray, d: np.ndarray, a: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Ray parameter t where o + t d crosses the segment a + s e (0 <= s <= 1); inf on a miss."""
    den = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    ao = a - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]) / den
        s = (ao[:, 0] * d[:, 1] - ao[:, 1] * d[:, 0]) / den
    return np.where((den != 0) & (s >= 0) & (s <= 1) & (t > 0), t, np.inf)


def ray_distances(origins: np.ndarray, dirs: np.ndarray, poly: np.ndarray,
                  window: tuple[np.ndarray, float] | None = None) -> np.ndarray:
    """Distance from each origin along its unit direction to the closed polyline (inf if never hit).

    With ``window`` = (fraction of the lap at each origin, w), only the part of
    ``poly`` within ``w`` of that fraction counts, so a ray cannot stop at the
    edge of another part of the lap that crosses this one.
    """
    e = np.roll(poly, -1, axis=0) - poly
    near = cKDTree(poly).query(origins)[0]
    radius = 3.0 * near + 2.0 * float(np.linalg.norm(e, axis=1).max()) + 1.0
    hits = cKDTree(poly + e / 2).query_ball_point(origins, radius)
    ray = np.repeat(np.arange(len(origins)), [len(h) for h in hits])
    seg = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])
    if window is not None:
        at, w = window
        frac = geometry.cumulative_length(poly, True)[:-1] / geometry.length(poly, True)
        near_lap = lambda f, k: np.abs((frac[k] - f + 0.5) % 1.0 - 0.5) <= w
        ok = near_lap(at[ray], seg)
        ray, seg = ray[ok], seg[ok]
    out = np.full(len(origins), np.inf)
    np.minimum.at(out, ray, _cast(origins[ray], dirs[ray], poly[seg], e[seg]))
    for i in np.nonzero(np.isinf(out))[0]:  # the crossing lies beyond the search radius
        k = np.arange(len(poly)) if window is None else np.nonzero(near_lap(window[0][i], np.arange(len(poly))))[0]
        out[i] = _cast(origins[i][None], dirs[i][None], poly[k], e[k]).min() if len(k) else np.inf
    return out


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    c = poly.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) > 0 for x, y in pts])


def _measure(line: Spline, left: np.ndarray, right: np.ndarray, spacing: float,
             sigma: float, local: float | None = None) -> Reference | None:
    """Rows at even arc length along ``line``; None if the line leaves the road.

    ``local`` (a fraction of the lap) is for laps that cross themselves: widths are
    measured only to the stretch of each edge beside the same part of the lap.
    """
    m = max(20000, 4 * int(line.length()))
    u_fine = np.linspace(0.0, 1.0, m + 1)
    speed = np.linalg.norm(line(u_fine, 1), axis=1)
    s_fine = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 / m)])
    n = max(16, int(round(s_fine[-1] / spacing)))
    s = np.arange(n) * s_fine[-1] / n
    u = np.interp(s, s_fine, u_fine)
    p, d1, d2 = line(u), line(u, 1), line(u, 2)
    if local is None and not np.all(_inside(left, p) ^ _inside(right, p)):
        return None  # nested loops: the road is inside exactly one
    normal = geometry.left_normals(d1)
    kappa = np.sum(d2 * normal, axis=1) / np.sum(d1 * d1, axis=1)  # > 0 where the line turns left
    window = None if local is None else (s / s_fine[-1], local)
    w_left, w_right = ray_distances(p, normal, left, window), ray_distances(p, -normal, right, window)
    if not (np.isfinite(w_left).all() and np.isfinite(w_right).all()):
        return None
    factor = np.minimum(1.0 - kappa * w_left, 1.0 + kappa * w_right)
    return Reference(line, s, u, w_left, w_right, sigma, float(factor.min()),
                     float(1.0 / max(np.abs(kappa).max(), 1e-12)))


def build(left: Spline, right: Spline, *, spacing: float = 2.0, min_factor: float = MIN_FACTOR,
          tol: float = 0.5, cp_spacing: float = 60.0, smooth: float = 0.05,
          sigma0: float = 2.0, growth: float = 1.4, center: Spline | None = None,
          local: float = 0.03) -> Reference | None:
    """Reference line between the two closed edges of a circuit, which both run in
    the direction of travel. Returns None if no midline can be traced.

    With ``center`` (the drawn centerline of a lap that crosses itself) the line
    starts from it instead of the midline, and widths are measured within
    ``local`` of the same fraction of the lap.
    """
    lp, rp = (geometry.resample(e.sample(max(1000, int(e.length()))), spacing=1.0, closed=True)
              for e in (left, right))
    mid = midline(lp, rp) if center is None else center.sample(max(2000, int(center.length())))
    if mid is None:
        return None
    mid = geometry.resample(mid, spacing=1.0, closed=True)
    if center is None and (geometry.signed_area(mid) < 0) != (geometry.signed_area(lp) < 0):
        mid = mid[::-1]  # the traced midline runs either way; a centerline already runs with the edges
    start = (left(left.domain[0]) + right(right.domain[0])) / 2
    mid = np.roll(mid, -int(np.argmin(np.linalg.norm(mid - start, axis=1))), axis=0)

    best, sigma, max_sigma = None, sigma0, geometry.length(mid, True) / 8
    while sigma < max_sigma:
        smoothed = gaussian_filter1d(mid, sigma, axis=0, mode="wrap")
        line = bspline.fit(smoothed, True, tol=tol, cp_spacing=cp_spacing, smooth=smooth)
        ref = _measure(line, lp, rp, spacing, sigma, None if center is None else local)
        if ref is None:
            break
        if best is None or ref.min_factor > best.min_factor:
            best = ref
        if ref.min_factor >= min_factor:
            break
        sigma *= growth
    if best is not None:
        best.max_deviation = float(geometry.nearest_distances(best.line(best.u), mid, dst_closed=True).max())
    return best
