"""JSON serialisation of a processed track, in world metres.

Everything in the file is in metres in a right-handed frame: x to the right of
the drawing, y towards its top, z up, with the origin at the lap's start/finish
point. No pixel or image information is written; the drawing is only an input.
"""

from __future__ import annotations

import re

import numpy as np

from .bspline import Spline
from .pipeline import Result
from .reference import Reference
from .track import Track

FORMAT_VERSION = 2


def _r(a, nd: int = 3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


class World:
    """The map from working-frame px (y down) to world metres (y up, origin at the start)."""

    def __init__(self, t: Track):
        self.m = float(t.m_per_px)
        self.origin = origin(t)

    def point(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p - self.origin) * np.array([self.m, -self.m])

    def length(self, v) -> float:
        return float(v) * self.m

    def text(self, message: str) -> str:
        """A warning with its "(x, y) px" points and "N px" lengths given in metres."""
        message = re.sub(r"\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\) px",
                         lambda g: "({:.0f}, {:.0f}) m".format(*self.point([float(g[1]), float(g[2])])), message)
        return re.sub(r"([-+]?)(\d+(?:\.\d+)?) px",
                      lambda g: f"{g[1]}{float(g[2]) * self.m:.1f} m", message)


def origin(t: Track) -> np.ndarray:
    """The start/finish point, frame px: where the reference line (or centerline) starts,
    or midway between the starts of the edges."""
    if t.reference is not None:
        return t.reference.line(t.reference.line.domain[0])
    if t.centerline is not None:
        return t.centerline(t.centerline.domain[0])
    first, second = t.edges.values()
    return (first(first.domain[0]) + second(second.domain[0])) / 2


def spline_to_dict(s: Spline, w: World, samples: int) -> dict:
    return {
        "degree": s.degree,
        "periodic": s.periodic,
        "domain": list(s.domain),
        "knots": _r(s.knots, 6),
        "control_points": _r(w.point(s.control_points)),  # a B-spline maps exactly under an affine map
        "length_m": round(w.length(s.length()), 3),
        "fit_error_m": {"rms": round(w.length(s.rms_error), 3), "max": round(w.length(s.max_error), 3)},
        "samples": _r(w.point(s.sample(samples))),
    }


def reference_to_dict(r: Reference, w: World, samples: int) -> dict:
    rows = np.column_stack([np.round(r.s * w.m, 3), np.round(r.u, 7),
                            np.round(r.w_left * w.m, 3), np.round(r.w_right * w.m, 3)])
    return {
        "line": spline_to_dict(r.line, w, samples),
        "smoothing_sigma_m": round(w.length(r.sigma), 3),
        "max_midline_deviation_m": round(w.length(r.max_deviation), 3),
        "min_radius_m": round(w.length(r.min_radius), 2),
        "min_1_minus_w_kappa": round(r.min_factor, 4),
        "spacing_m": round(float(r.s[1] - r.s[0]) * w.m, 6),
        "columns": ["s_m", "u", "w_left_m", "w_right_m"],
        "rows": rows.tolist(),
    }


def profile_to_dict(t: Track, w: World) -> dict:
    p = t.profile
    rows = []
    if p.graded:
        rows = np.column_stack([np.round(t.reference.s * w.m, 3), np.round(p.z * w.m, 3),
                                np.round(p.grade, 6)]).tolist()
    m = lambda v: round(w.length(v), 3)
    return {
        "vertical_radius_m": m(p.vertical_radius) if p.graded else None,
        "ticks": [{"s_m": m(s), "point": _r(w.point(pt)), "ease_m": m(p.eases[i]) if p.graded else None,
                   "moved_m": m(p.moved[i])}
                  for i, (s, pt) in enumerate(zip(p.ticks, p.points))],
        "sections": [{
            "index": sec.index,
            "label": sec.label,
            "grade": None if sec.grade is None else round(sec.grade, 6),
            "solved": sec.solved,
            "s_start_m": m(sec.s_start),
            "s_end_m": m(sec.s_end),
            "length_m": m(sec.length),
            "width_m": m(sec.width),
            "z_start_m": m(sec.z_start) if p.graded else None,
            "z_end_m": m(sec.z_end) if p.graded else None,
        } for sec in p.sections],
        "crossings": [{"s_over_m": m(c.s_over), "s_under_m": m(c.s_under), "point": _r(w.point(c.point)),
                       "clearance_m": m(c.clearance)} for c in p.crossings],
        "columns": ["s_m", "z_m", "grade"],
        "rows": rows,
    }


def result_to_dict(result: Result, samples: int = 256) -> dict:
    t = result.track
    w = World(t)
    return {
        "format": "drawcircuit-track",
        "version": FORMAT_VERSION,
        "units": "m",
        "axes": "x right and y up on the drawing, z up; origin at the start/finish point",
        "type": t.kind,
        "source": t.source,
        "direction": t.direction,
        "left_edge": t.left_edge,
        "right_edge": t.right_edge,
        "length_m": round(w.length(t.length()), 3),
        "scale_from": t.scale_from,
        "track_width_m": {k: round(w.length(v), 3) for k, v in t.width.items()},
        "edges": {name: spline_to_dict(s, w, samples) for name, s in t.edges.items()},
        "centerline": spline_to_dict(t.centerline, w, samples) if t.centerline is not None else None,
        "reference": reference_to_dict(t.reference, w, samples) if t.reference is not None else None,
        "profile": profile_to_dict(t, w) if t.profile is not None else None,
        "warnings": [w.text(message) for message in t.warnings],
    }
