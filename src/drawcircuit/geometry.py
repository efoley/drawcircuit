"""Small polyline helpers. Points are (x, y) rows in image pixels (y down)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def dedupe(poly: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate points."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    keep = np.concatenate([[True], np.linalg.norm(np.diff(poly, axis=0), axis=1) > eps])
    return poly[keep]


def cumulative_length(poly: np.ndarray, closed: bool = False) -> np.ndarray:
    """Arc length at each vertex; for closed curves one extra entry holds the total."""
    pts = np.vstack([poly, poly[:1]]) if closed else poly
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def length(poly: np.ndarray, closed: bool = False) -> float:
    if len(poly) < 2:
        return 0.0
    return float(cumulative_length(poly, closed)[-1])


def resample(poly: np.ndarray, spacing: float | None = None, count: int | None = None,
             closed: bool = False) -> np.ndarray:
    """Resample a polyline at uniform arc-length spacing (or to a fixed point count)."""
    poly = dedupe(poly)
    pts = np.vstack([poly, poly[:1]]) if closed else poly
    s = cumulative_length(poly, closed)
    total = s[-1]
    if count is None:
        count = max(int(round(total / spacing)), 3 if closed else 2)
    si = np.linspace(0.0, total, count, endpoint=not closed)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])])


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area. In image coordinates (y down) a negative value means the
    curve runs counter-clockwise as seen on the page."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def contains(outer: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for many points."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = outer[:, 0][None, :], outer[:, 1][None, :]
    x1, y1 = np.roll(outer[:, 0], -1)[None, :], np.roll(outer[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return np.count_nonzero(crosses & (x < xi), axis=1) % 2 == 1


def nearest_distances(src: np.ndarray, dst: np.ndarray, dst_spacing: float = 1.0,
                      dst_closed: bool = False) -> np.ndarray:
    """Distance from each point of ``src`` to the polyline ``dst`` (densely resampled)."""
    dense = resample(dst, spacing=dst_spacing, closed=dst_closed)
    return cKDTree(dense).query(src)[0]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v, dtype=float), where=n > 0)


def left_normals(tangents: np.ndarray) -> np.ndarray:
    """Unit normals pointing to the visual left of the travel direction (image coords, y down)."""
    t = unit(tangents)
    return np.column_stack([t[:, 1], -t[:, 0]])


def crosses_itself(poly: np.ndarray, closed: bool = False, spacing: float = 2.0) -> bool:
    """Whether the polyline crosses itself (as a figure-8 or an over/underpass does)."""
    pts = resample(poly, spacing=spacing, closed=closed)
    a = pts
    b = np.roll(pts, -1, axis=0) if closed else pts[1:]
    a = a[:len(b)]
    n = len(a)
    tree = cKDTree((a + b) / 2)
    reach = 2.0 * float(np.linalg.norm(b - a, axis=1).max())
    for i, j in tree.query_pairs(reach):
        gap = abs(i - j)
        if (min(gap, n - gap) if closed else gap) <= 1:  # neighbouring segments share an end
            continue
        d1, d2 = b[i] - a[i], b[j] - a[j]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if den == 0:
            continue
        r = a[j] - a[i]
        t = (r[0] * d2[1] - r[1] * d2[0]) / den
        u = (r[0] * d1[1] - r[1] * d1[0]) / den
        if 0 <= t < 1 and 0 <= u < 1:
            return True
    return False
