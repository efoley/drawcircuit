"""Turn a classified layout into oriented, aligned B-spline track edges."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from . import bspline, geometry, reference
from .bspline import Spline
from .layout import CENTERLINE, CIRCUIT, Layout
from .profile import Profile
from .reference import Reference


OFFSET_TOL = 0.5  # px: how closely the edges of a centerline drawing follow its offsets


@dataclass
class FitParams:
    tol: float
    cp_spacing: float
    smooth: float


@dataclass
class Track:
    kind: str                       # "circuit" or "segment"
    source: str                     # "edges" or "centerline"
    edges: dict[str, Spline]        # circuit: outer/inner, segment: left/right
    centerline: Spline | None = None
    direction: str | None = None    # circuits: "ccw" or "cw" as seen on the page
    width: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reference: Reference | None = None  # circuits: smooth line between the edges, with widths
    profile: Profile | None = None      # circuits drawn as a centerline with tick marks: grades
    m_per_px: float | None = None       # real-world scale, when the length was given

    @property
    def left_edge(self) -> str:
        """Name of the edge on the left in the direction of travel."""
        if self.kind != CIRCUIT:
            return "left"
        return "inner" if self.direction == "ccw" else "outer"

    @property
    def right_edge(self) -> str:
        return {"left": "right", "inner": "outer", "outer": "inner"}[self.left_edge]

    def length(self) -> float:
        """Length in px that a real-world length refers to: the reference line if
        there is one, else the centerline, else the mean of the two edges."""
        if self.reference is not None:
            return self.reference.length
        if self.centerline is not None:
            return self.centerline.length()
        return float(np.mean([e.length() for e in self.edges.values()]))


def _orient_loop(poly: np.ndarray, direction: str) -> np.ndarray:
    ccw = geometry.signed_area(poly) < 0  # y points down
    return poly if ccw == (direction == "ccw") else poly[::-1]


def _start_at(poly: np.ndarray, i: int) -> np.ndarray:
    return np.roll(poly, -int(i), axis=0)


def straight_middle(poly: np.ndarray, max_turn: float = np.radians(10), scale: float = 15.0,
                    spacing: float = 2.0) -> int:
    """Index of the point halfway along the longest straight of a closed polyline.

    A straight is a stretch over which the direction changes by at most
    ``max_turn``, measured on the line smoothed over ~``scale`` px so that hand
    jitter does not count. A track without straights gets its least curved stretch.
    """
    pts = geometry.resample(poly, spacing=spacing, closed=True)
    n = len(pts)
    smooth = gaussian_filter1d(pts, scale / spacing, axis=0, mode="wrap")
    step = np.roll(smooth, -1, axis=0) - smooth
    heading = np.unwrap(np.append(np.arctan2(step[:, 1], step[:, 0]), np.arctan2(step[0, 1], step[0, 0])))
    h = np.concatenate([heading[:-1], heading[:-1] + heading[-1] - heading[0]])  # two laps
    lo, hi = deque(), deque()  # indices of the window's minima / maxima, in order
    best_len, best_start, i = 0, 0, 0
    for j in range(len(h)):
        while lo and h[lo[-1]] >= h[j]:
            lo.pop()
        lo.append(j)
        while hi and h[hi[-1]] <= h[j]:
            hi.pop()
        hi.append(j)
        while h[hi[0]] - h[lo[0]] > max_turn or j - i >= n:
            i += 1
            if lo[0] < i:
                lo.popleft()
            if hi[0] < i:
                hi.popleft()
        if j - i + 1 > best_len:
            best_len, best_start = j - i + 1, i
    s = geometry.cumulative_length(poly, closed=True)
    target = ((best_start + best_len // 2) % n) * s[-1] / n
    return int(min(np.searchsorted(s, target), len(poly) - 1))


def _side(a: np.ndarray, b: np.ndarray) -> float:
    """> 0 when polyline ``b`` lies to the left of ``a`` (in a's direction of travel)."""
    pts = geometry.resample(a, count=100)[10:90]
    tangents = np.gradient(pts, axis=0)
    dense = geometry.resample(b, spacing=1.0)
    _, j = cKDTree(dense).query(pts)
    to_b = dense[j] - pts
    return float(np.mean(np.sign(np.sum(to_b * geometry.left_normals(tangents), axis=1))))


def _width_stats(a: Spline, b: Spline) -> dict[str, float]:
    pa, pb = a.sample(600), b.sample(600)
    if not a.periodic:  # ignore the ends, where drawn edges rarely line up
        pa = pa[60:-60]
    d = geometry.nearest_distances(pa, pb, dst_closed=b.periodic)
    return {"median": float(np.median(d)), "min": float(d.min()), "max": float(d.max())}


def _add_reference(track: Track, fp: FitParams) -> Track:
    if track.kind != CIRCUIT:
        return track
    ref = reference.build(track.edges[track.left_edge], track.edges[track.right_edge],
                          cp_spacing=fp.cp_spacing, smooth=fp.smooth)
    if ref is None:
        track.warnings.append("could not trace a reference line between the edges")
    elif ref.min_factor < reference.MIN_FACTOR:
        track.warnings.append(f"the reference line only reaches 1 - w*kappa = {ref.min_factor:.2f} "
                              f"(target {reference.MIN_FACTOR})")
    track.reference = ref
    return track


def build(layout: Layout, fp: FitParams, *, direction: str = "ccw",
          track_width: float | Callable[[Spline, np.ndarray], np.ndarray] = 60.0) -> Track:
    """Oriented, aligned track edges. For a centerline drawing ``track_width`` is the
    road width in px, or a function of (centerline, parameters) giving it along the line."""
    fit = lambda pts, closed: bspline.fit(pts, closed, tol=fp.tol, cp_spacing=fp.cp_spacing,
                                           smooth=fp.smooth)
    warnings = list(layout.warnings)
    closed = layout.kind == CIRCUIT

    if layout.source == CENTERLINE:
        line = layout.curves[0].points
        if closed:
            line = _orient_loop(line, direction)
            line = _start_at(line, straight_middle(line))
        elif line[0].sum() > line[-1].sum():  # segments start at the end nearer the top-left
            line = line[::-1]
        center = fit(line, closed)
        u = center.params(max(400, int(center.length() / 2)))
        p, dp, ddp = center(u), center(u, 1), center(u, 2)
        normal = geometry.left_normals(dp)
        speed = np.linalg.norm(dp, axis=1)
        curvature = np.abs(dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / np.maximum(speed, 1e-9) ** 3
        w = track_width(center, u) if callable(track_width) else np.full(len(u), float(track_width))
        if np.max(curvature * w / 2) > 1.0:
            worst = int(np.argmax(curvature * w))
            warnings.append(f"a corner is tighter than half the track width ({w[worst]:.0f} px); "
                            "the inside edge will fold over itself there")
        left, right = p + normal * w[:, None] / 2, p - normal * w[:, None] / 2
        # The edges are offsets of a line that is already smooth, so they are fitted
        # closely: the road is then as wide as asked, and width changes keep their length.
        fit = lambda pts, closed: bspline.fit(pts, closed, tol=min(fp.tol, OFFSET_TOL),
                                               cp_spacing=fp.cp_spacing, smooth=fp.smooth)
        if closed:
            inner_is_left = abs(geometry.signed_area(left)) < abs(geometry.signed_area(right))
            outer, inner = (right, left) if inner_is_left else (left, right)
            edges = {"outer": fit(outer, True), "inner": fit(inner, True)}
        else:
            edges = {"left": fit(left, False), "right": fit(right, False)}
        width = {"median": float(np.median(w)), "min": float(w.min()), "max": float(w.max())}
        return _add_reference(Track(layout.kind, CENTERLINE, edges, center,
                                    direction if closed else None, width, warnings), fp)

    a, b = (c.points for c in layout.curves)
    if closed:
        outer, inner = _orient_loop(a, direction), _orient_loop(b, direction)
        outer = _start_at(outer, straight_middle(outer))
        inner = _start_at(inner, np.argmin(np.linalg.norm(inner - outer[0], axis=1)))
        edges = {"outer": fit(outer, True), "inner": fit(inner, True)}
    else:
        same = np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
        swapped = np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0])
        if swapped < same:
            b = b[::-1]
        if a[0].sum() + b[0].sum() > a[-1].sum() + b[-1].sum():  # start nearer the top-left
            a, b = a[::-1], b[::-1]
        left, right = (b, a) if _side(a, b) > 0 else (a, b)
        edges = {"left": fit(left, False), "right": fit(right, False)}
    first, second = edges.values()
    return _add_reference(Track(layout.kind, layout.source, edges, None, direction if closed else None,
                                _width_stats(first, second), warnings), fp)
