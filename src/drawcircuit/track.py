"""Turn a classified layout into oriented, aligned B-spline track edges."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import bspline, geometry, reference
from .bspline import Spline
from .layout import CENTERLINE, CIRCUIT, Layout
from .reference import Reference


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

    @property
    def left_edge(self) -> str:
        """Name of the edge on the left in the direction of travel."""
        if self.kind != CIRCUIT:
            return "left"
        return "inner" if self.direction == "ccw" else "outer"

    @property
    def right_edge(self) -> str:
        return {"left": "right", "inner": "outer", "outer": "inner"}[self.left_edge]


def _orient_loop(poly: np.ndarray, direction: str) -> np.ndarray:
    ccw = geometry.signed_area(poly) < 0  # y points down
    return poly if ccw == (direction == "ccw") else poly[::-1]


def _start_at(poly: np.ndarray, i: int) -> np.ndarray:
    return np.roll(poly, -int(i), axis=0)


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
          track_width: float = 60.0) -> Track:
    fit = lambda pts, closed: bspline.fit(pts, closed, tol=fp.tol, cp_spacing=fp.cp_spacing,
                                           smooth=fp.smooth)
    warnings = list(layout.warnings)
    closed = layout.kind == CIRCUIT

    if layout.source == CENTERLINE:
        line = layout.curves[0].points
        if closed:
            line = _orient_loop(line, direction)
            line = _start_at(line, np.argmin(line[:, 1]))
        elif line[0].sum() > line[-1].sum():  # segments start at the end nearer the top-left
            line = line[::-1]
        center = fit(line, closed)
        u = center.params(max(400, int(center.length() / 2)))
        p, dp, ddp = center(u), center(u, 1), center(u, 2)
        normal = geometry.left_normals(dp)
        speed = np.linalg.norm(dp, axis=1)
        curvature = np.abs(dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / np.maximum(speed, 1e-9) ** 3
        if curvature.max() * track_width / 2 > 1.0:
            warnings.append(f"a corner is tighter than half the track width ({track_width:.0f} px); "
                            "the inside edge will fold over itself there")
        left, right = p + normal * track_width / 2, p - normal * track_width / 2
        if closed:
            inner_is_left = abs(geometry.signed_area(left)) < abs(geometry.signed_area(right))
            outer, inner = (right, left) if inner_is_left else (left, right)
            edges = {"outer": fit(outer, True), "inner": fit(inner, True)}
        else:
            edges = {"left": fit(left, False), "right": fit(right, False)}
        width = {"median": track_width, "min": track_width, "max": track_width}
        return _add_reference(Track(layout.kind, CENTERLINE, edges, center,
                                    direction if closed else None, width, warnings), fp)

    a, b = (c.points for c in layout.curves)
    if closed:
        outer, inner = _orient_loop(a, direction), _orient_loop(b, direction)
        outer = _start_at(outer, np.argmin(outer[:, 1]))
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
