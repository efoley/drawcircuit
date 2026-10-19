"""JSON serialisation of a processed track."""

from __future__ import annotations

import numpy as np

from .bspline import Spline
from .pipeline import Result
from .reference import Reference
from .track import Track

FORMAT_VERSION = 1


def _r(a, nd: int = 3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def spline_to_dict(s: Spline, samples: int) -> dict:
    return {
        "degree": s.degree,
        "periodic": s.periodic,
        "domain": list(s.domain),
        "knots": _r(s.knots, 6),
        "control_points": _r(s.control_points),
        "length_px": round(s.length(), 2),
        "fit_error_px": {"rms": round(s.rms_error, 3), "max": round(s.max_error, 3)},
        "samples": _r(s.sample(samples)),
    }


def reference_to_dict(r: Reference, samples: int) -> dict:
    rows = np.column_stack([np.round(r.s, 3), np.round(r.u, 7), np.round(r.w_left, 3), np.round(r.w_right, 3)])
    return {
        "line": spline_to_dict(r.line, samples),
        "smoothing_sigma_px": round(r.sigma, 2),
        "max_midline_deviation_px": round(r.max_deviation, 3),
        "min_radius_px": round(r.min_radius, 2),
        "min_1_minus_w_kappa": round(r.min_factor, 4),
        "spacing_px": round(float(r.s[1] - r.s[0]), 6),
        "columns": ["s_px", "u", "w_left", "w_right"],
        "rows": rows.tolist(),
    }


def profile_to_dict(t: Track) -> dict:
    p = t.profile
    rows = []
    if p.graded:
        rows = np.column_stack([np.round(t.reference.s, 3), np.round(p.z, 3), np.round(p.grade, 6)]).tolist()
    return {
        "vertical_radius_px": round(p.vertical_radius, 3) if p.graded else None,
        "ticks": [{"s_px": round(float(s), 3), "point": _r(pt),
                   "ease_px": round(float(p.eases[i]), 3) if p.graded else None,
                   "moved_px": round(float(p.moved[i]), 3)}
                  for i, (s, pt) in enumerate(zip(p.ticks, p.points))],
        "sections": [{
            "index": sec.index,
            "label": sec.label,
            "grade": None if sec.grade is None else round(sec.grade, 6),
            "solved": sec.solved,
            "s_start_px": round(sec.s_start, 3),
            "s_end_px": round(sec.s_end, 3),
            "length_px": round(sec.length, 3),
            "width_px": round(sec.width, 3),
            "z_start_px": round(sec.z_start, 3) if p.graded else None,
            "z_end_px": round(sec.z_end, 3) if p.graded else None,
        } for sec in p.sections],
        "crossings": [{"s_over_px": round(c.s_over, 3), "s_under_px": round(c.s_under, 3),
                       "point": _r(c.point), "clearance_px": round(c.clearance, 3)} for c in p.crossings],
        "columns": ["s_px", "z_px", "grade"],
        "rows": rows,
    }


def scale_to_dict(t: Track) -> dict:
    return {
        "m_per_px": round(t.m_per_px, 9),
        "length_m": round(t.length() * t.m_per_px, 3),
        "track_width_m": round(t.width["median"] * t.m_per_px, 3),
    }


def result_to_dict(result: Result, samples: int = 256) -> dict:
    t = result.track
    w, h = result.frame_size
    return {
        "format": "drawcircuit-track",
        "version": FORMAT_VERSION,
        "type": t.kind,
        "source": t.source,
        "direction": t.direction,
        "left_edge": t.left_edge,
        "right_edge": t.right_edge,
        "frame": {
            "width": w,
            "height": h,
            "units": "px",
            "y_axis": "down",
            "paper_detected": result.paper_detected,
            "input_to_frame": _r(result.homography, 9),
        },
        "track_width_px": {k: round(v, 2) for k, v in t.width.items()},
        "edges": {name: spline_to_dict(s, samples) for name, s in t.edges.items()},
        "centerline": spline_to_dict(t.centerline, samples) if t.centerline is not None else None,
        "reference": reference_to_dict(t.reference, samples) if t.reference is not None else None,
        "scale": scale_to_dict(t) if t.m_per_px is not None else None,
        "profile": profile_to_dict(t) if t.profile is not None else None,
        "warnings": t.warnings,
    }
