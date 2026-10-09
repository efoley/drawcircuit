"""Decide what was drawn: a full circuit or a segment, as two edges or as a centerline."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import geometry
from .strokes import Stroke


class TrackError(RuntimeError):
    """The drawing could not be interpreted as a track."""


CIRCUIT, SEGMENT = "circuit", "segment"
EDGES, CENTERLINE = "edges", "centerline"


@dataclass
class Layout:
    kind: str                 # CIRCUIT or SEGMENT
    source: str               # EDGES (two drawn edges) or CENTERLINE (one drawn line)
    curves: list[Stroke]      # [outer, inner] / [a, b] for EDGES, [centerline] otherwise
    candidates: list[Stroke] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def significant(strokes: list[Stroke], min_length: float, min_fraction: float = 0.2) -> list[Stroke]:
    """Strokes long enough to be part of the track, longest first."""
    strokes = sorted(strokes, key=lambda s: s.length, reverse=True)
    if not strokes:
        return []
    cutoff = max(min_length, min_fraction * strokes[0].length)
    return [s for s in strokes if s.length >= cutoff]


def _samples(s: Stroke, count: int = 200, trim: float = 0.0) -> np.ndarray:
    pts = geometry.resample(s.points, count=count, closed=s.closed)
    if trim and not s.closed:
        k = int(trim * count)
        pts = pts[k:count - k]
    return pts


def _separation(a: Stroke, b: Stroke) -> np.ndarray:
    """Distances between the two strokes, measured both ways (ends trimmed for open strokes)."""
    da = geometry.nearest_distances(_samples(a, trim=0.1), b.points, dst_closed=b.closed)
    db = geometry.nearest_distances(_samples(b, trim=0.1), a.points, dst_closed=a.closed)
    return np.concatenate([da, db])


def _edge_pair(a: Stroke, b: Stroke, stroke_width: float) -> tuple[str, Stroke, Stroke, float] | None:
    """If ``a`` and ``b`` look like the two edges of one road, return (kind, first, second, width)."""
    la, lb = a.length, b.length
    if max(la, lb) > 3.0 * min(la, lb):
        return None
    if a.closed and b.closed:
        if geometry.contains(a.points, _samples(b)).mean() > 0.98:
            outer, inner = a, b
        elif geometry.contains(b.points, _samples(a)).mean() > 0.98:
            outer, inner = b, a
        else:
            return None
        kind = CIRCUIT
    elif not a.closed and not b.closed:
        outer, inner, kind = a, b, SEGMENT
    else:
        return None
    d = _separation(outer, inner)
    width = float(np.median(d))
    if width < 2.5 * stroke_width:            # the same line traced twice
        return None
    if width > 0.35 * min(la, lb):            # too far apart to be one road
        return None
    if np.percentile(d, 90) > 3.0 * width:    # not running alongside each other
        return None
    return kind, outer, inner, width


def classify(strokes: list[Stroke], *, mode: str, stroke_width: float, min_length: float) -> Layout:
    """Pick the track out of the joined strokes.

    ``mode`` is "auto", "edges" or "centerline".
    """
    cands = significant(strokes, min_length)
    if not cands:
        raise TrackError("no strokes long enough to be a track were found")
    warnings: list[str] = []

    if mode in ("auto", "edges"):
        best = None
        for a, b in combinations(cands[:4], 2):
            pair = _edge_pair(a, b, stroke_width)
            if pair is None and a.closed != b.closed:
                # One edge of a loop may have a gap the joiner did not dare to close.
                o = a if not a.closed else b
                if o.end_gap <= 0.2 * o.length:
                    forced = Stroke(o.points, True)
                    pair = _edge_pair(forced, b if o is a else a, stroke_width)
                    if pair is not None:
                        warnings.append(f"closed a {o.end_gap:.0f} px gap in a track edge")
            # Prefer the widest valid pair, so a dashed centerline drawn between two
            # edges is not mistaken for one of them.
            if pair is not None and (best is None or pair[3] > best[3]):
                best = pair
        if best is not None:
            kind, first, second, _ = best
            used = sum(1 for s in cands if s.points is first.points or s.points is second.points)
            if len(cands) > used:
                warnings.append(f"ignored {len(cands) - used} other long stroke(s)")
            return Layout(kind, EDGES, [first, second], cands, warnings)
        if mode == "edges":
            raise TrackError("could not find two strokes that form the edges of a track")

    line = cands[0]
    if len(cands) > 1:
        warnings.append(f"treating the longest stroke as the centerline; ignored {len(cands) - 1} other(s)")
    return Layout(CIRCUIT if line.closed else SEGMENT, CENTERLINE, [line], cands, warnings)
