"""JSON serialisation of a processed track."""

from __future__ import annotations

import numpy as np

from .bspline import Spline
from .pipeline import Result

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


def result_to_dict(result: Result, samples: int = 256) -> dict:
    t = result.track
    w, h = result.frame_size
    return {
        "format": "drawcircuit-track",
        "version": FORMAT_VERSION,
        "type": t.kind,
        "source": t.source,
        "direction": t.direction,
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
        "warnings": t.warnings,
    }
