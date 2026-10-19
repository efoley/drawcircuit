"""Turn a classified layout into oriented, aligned B-spline track edges."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from . import bspline, corners, geometry, reference
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
    rounded: list[corners.Corner] = field(default_factory=list)  # corners rounded to the limit
    profile: Profile | None = None      # circuits drawn as a centerline with tick marks: grades
    m_per_px: float | None = None       # real-world scale
    scale_from: str | None = None       # "length" (given) or "assumed road width"

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


def _rounded(layout: Layout, fp: FitParams, drawn: Spline, track_width, direction: str,
             warnings: list[str], inside_radius: float, max_turn: float) -> Track:
    """A circuit around the centerline ``drawn`` with its corners rounded so the inside
    edge turns no tighter than ``inside_radius``."""
    # Round the corners, then check the inside edge along the reference line, which is
    # what a racing line is planned on; round with more headroom if fitting tightened it.
    tight_fit = lambda pts, closed: bspline.fit(pts, closed, tol=min(fp.tol, OFFSET_TOL),
                                                 cp_spacing=fp.cp_spacing, smooth=fp.smooth)
    for headroom in (1.2, 1.4, 1.7):
        center, rounded = _round(drawn, track_width, inside_radius, headroom, tight_fit)
        track = _from_centerline(layout, fp, center, track_width, direction, list(warnings))
        if track.reference is None or inside_edge_radius(track.reference).min() >= 0.97 * inside_radius:
            break
    track.rounded = rounded
    # A corner keeps turning the same way, at a radius under ten times the limit (about 100 m).
    dense = geometry.resample(center.sample(max(2000, int(center.length()))), spacing=1.0, closed=True)
    at = lambda s: "({:.0f}, {:.0f}) px".format(*dense[int(s) % len(dense)])
    for s, turn in corners.turns(dense, 1.0, 1.0 / (10.0 * (inside_radius + track.width["median"] / 2))):
        if turn > max_turn:
            track.warnings.append(f"the corner at {at(s)} turns through {turn:.0f} degrees in one go "
                                  f"(more than {max_turn:.0f}); redraw it if it is not meant to be a spiral")
    for c in rounded:
        if not c.met:
            track.warnings.append(f"the corner near {at(c.s)} could only be rounded to a radius of "
                                  f"{c.after:.1f} px (wanted {c.limit:.1f} px): its legs are too close; redraw it wider")
    if track.reference is not None and inside_edge_radius(track.reference).min() < 0.97 * inside_radius:
        track.warnings.append(f"the inside edge still turns as tight as "
                              f"{inside_edge_radius(track.reference).min():.1f} px in places "
                              f"(wanted {inside_radius:.1f} px)")
    return track


def _round_drawn_edges(layout: Layout, fp: FitParams, edges: dict[str, Spline], direction: str,
                       warnings: list[str], inside_radius: float, max_turn: float) -> Track | None:
    """Round the corners of a circuit drawn with two edges: the road is rebuilt around
    the midline between them, as wide as drawn at each point along it, with the
    corners rounded. None if no midline can be traced."""
    lp, rp = (geometry.resample(e.sample(max(1000, int(e.length()))), spacing=1.0, closed=True)
              for e in edges.values())
    mid = reference.midline(lp, rp)
    if mid is None:
        return None
    mid = _orient_loop(mid, direction)
    mid = _start_at(mid, np.argmin(np.linalg.norm(mid - (lp[0] + rp[0]) / 2, axis=1)))
    center = bspline.fit(mid, True, tol=min(fp.tol, OFFSET_TOL), cp_spacing=fp.cp_spacing, smooth=fp.smooth)
    pts = center.sample(max(2000, int(center.length())))
    drawn = gaussian_filter1d(cKDTree(lp).query(pts)[0] + cKDTree(rp).query(pts)[0], 10.0, mode="wrap")
    frac = np.arange(len(pts)) / len(pts)
    width = lambda c, u: np.interp(np.asarray(u) % 1.0, frac, drawn, period=1.0)
    track = _rounded(layout, fp, center, width, direction, warnings, inside_radius, max_turn)
    track.source = layout.source
    return track


def _round(center: Spline, track_width, inside_radius: float, headroom: float,
           fit) -> tuple[Spline, list[corners.Corner]]:
    """Round the corners of a closed centerline so the inside edge turns no tighter
    than ``inside_radius``, given the road width along it."""
    u = center.params(max(2000, int(center.length())))
    w = track_width(center, u) if callable(track_width) else np.full(len(u), float(track_width))
    frac = np.arange(len(u)) / len(u)
    radius = lambda f: inside_radius + np.interp(f, frac, w, period=1.0) / 2
    line, rounded = corners.round_corners(center(u), radius, max_shift=2.0 * float(w.max()), headroom=headroom)
    return (fit(line, True) if rounded else center), rounded


def inside_edge_radius(ref: Reference) -> np.ndarray:
    """Radius of the inside edge at each reference row, px: the reference line's
    radius less the width on the side it turns towards (inf on a straight)."""
    d1, d2 = ref.line(ref.u, 1), ref.line(ref.u, 2)
    kappa = np.sum(d2 * geometry.left_normals(d1), axis=1) / np.sum(d1 * d1, axis=1)
    inside = np.where(kappa > 0, ref.w_left, ref.w_right)
    with np.errstate(divide="ignore"):
        return 1.0 / np.abs(kappa) - inside


def _from_centerline(layout: Layout, fp: FitParams, center: Spline, track_width, direction: str,
                     warnings: list[str]) -> Track:
    """Edges offset to either side of a fitted centerline, and the reference line."""
    closed = layout.kind == CIRCUIT
    u = center.params(max(400, int(center.length() / 2)))
    p, dp, ddp = center(u), center(u, 1), center(u, 2)
    normal = geometry.left_normals(dp)
    speed = np.linalg.norm(dp, axis=1)
    curvature = np.abs(dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / np.maximum(speed, 1e-9) ** 3
    w = track_width(center, u) if callable(track_width) else np.full(len(u), float(track_width))
    if np.max(curvature * w / 2) > 1.0:
        worst = int(np.argmax(curvature * w))
        warnings.append(f"a corner is tighter than half the track width ({w[worst]:.1f} px); "
                        "the inside edge will fold over itself there")
    left, right = p + normal * w[:, None] / 2, p - normal * w[:, None] / 2
    # The edges are offsets of a line that is already smooth, so they are fitted
    # closely: the road is then as wide as asked, and width changes keep their length.
    fit = lambda pts, closed: bspline.fit(pts, closed, tol=min(fp.tol, OFFSET_TOL),
                                           cp_spacing=fp.cp_spacing, smooth=fp.smooth)
    if closed:  # the line runs ``direction`` round, so the inner edge is on the left when ccw
        outer, inner = (right, left) if direction == "ccw" else (left, right)
        edges = {"outer": fit(outer, True), "inner": fit(inner, True)}
    else:
        edges = {"left": fit(left, False), "right": fit(right, False)}
    width = {"median": float(np.median(w)), "min": float(w.min()), "max": float(w.max())}
    return _add_reference(Track(layout.kind, CENTERLINE, edges, center, direction if closed else None,
                                width, warnings), fp)


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


def _start(loop: np.ndarray, start: tuple[float, float] | None) -> np.ndarray:
    """``loop`` starting where the circuit starts: at the point nearest ``start`` (a start
    line marked on the drawing, frame px), else halfway along its longest straight."""
    if start is None:
        return _start_at(loop, straight_middle(loop))
    loop = geometry.resample(loop, spacing=1.0, closed=True)  # a bridged pencil gap has no points
    return _start_at(loop, int(np.argmin(np.linalg.norm(loop - np.asarray(start, dtype=float), axis=1))))


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
    # A centerline drawing's edges are offsets of its centerline, so the reference line
    # starts from that line itself; tracing the midline between the edges would only
    # add noise. (A lap that crosses itself has no midline between nested edges anyway.)
    ref = reference.build(track.edges[track.left_edge], track.edges[track.right_edge],
                          cp_spacing=fp.cp_spacing, smooth=fp.smooth, center=track.centerline)
    if ref is None:
        track.warnings.append("could not trace a reference line between the edges")
    elif ref.min_factor < reference.MIN_FACTOR:
        track.warnings.append(f"the reference line only reaches 1 - w*kappa = {ref.min_factor:.2f} "
                              f"(target {reference.MIN_FACTOR})")
    track.reference = ref
    return track


def build(layout: Layout, fp: FitParams, *, direction: str = "ccw",
          track_width: float | Callable[[Spline, np.ndarray], np.ndarray] = 60.0,
          inside_radius: float | None = None, max_turn: float = 200.0,
          start: tuple[float, float] | None = None) -> Track:
    """Oriented, aligned track edges. For a centerline drawing ``track_width`` is the
    road width in px, or a function of (centerline, parameters) giving it along the line.

    With ``inside_radius`` (px), a centerline circuit's corners are rounded until the
    inside edge turns no tighter than that, and corners turning through more than
    ``max_turn`` degrees are reported.
    """
    fit = lambda pts, closed: bspline.fit(pts, closed, tol=fp.tol, cp_spacing=fp.cp_spacing,
                                           smooth=fp.smooth)
    warnings = list(layout.warnings)
    closed = layout.kind == CIRCUIT

    if layout.source == CENTERLINE:
        line = layout.curves[0].points
        if closed:
            line = _orient_loop(line, direction)
            line = _start(line, start)
        elif line[0].sum() > line[-1].sum():  # segments start at the end nearer the top-left
            line = line[::-1]
        drawn = fit(line, closed)
        if not (closed and inside_radius):
            return _from_centerline(layout, fp, drawn, track_width, direction, warnings)
        return _rounded(layout, fp, drawn, track_width, direction, warnings, inside_radius, max_turn)

    a, b = (c.points for c in layout.curves)
    if closed:
        outer, inner = _orient_loop(a, direction), _orient_loop(b, direction)
        outer = _start(outer, start)
        inner = _start_at(inner, np.argmin(np.linalg.norm(inner - outer[0], axis=1)))
        edges = {"outer": fit(outer, True), "inner": fit(inner, True)}
        if inside_radius:
            rounded = _round_drawn_edges(layout, fp, edges, direction, warnings, inside_radius, max_turn)
            if rounded is not None:
                return rounded
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
