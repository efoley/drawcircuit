"""Trace a drawing along a route given as waypoints.

Some drawings are beyond the automatic stroke joiner: ticks at many corners,
faint pencil, a road that crosses itself, or a tunnel drawn as a hatched band.
The LLM driving the program (or a person) can then give the route as points in
travel order, read off the page image or the overlay (working-frame px).
Between consecutive points the line follows the cheapest path along the ink
skeleton, so the points only need to be roughly on the line, at each place
where the line could go more than one way. A leg marked "free" is drawn
straight instead, e.g. through a tunnel whose road is not drawn as a line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from . import geometry
from .layout import TrackError


@dataclass
class Route:
    points: np.ndarray   # (n, 2) working-frame px, in travel order
    free: np.ndarray     # bool (n,): the leg arriving at this point ignores the ink
    closed: bool = True


def load(path: str | Path) -> Route:
    """Read a route file: {"closed": true, "points": [[x, y], [x, y, "free"], ...]}."""
    doc = json.loads(Path(path).read_text())
    pts = doc["points"]
    return Route(np.array([p[:2] for p in pts], dtype=float),
                 np.array([len(p) > 2 and p[2] == "free" for p in pts]), bool(doc.get("closed", True)))


def _snap(p: np.ndarray, tree: cKDTree, coords: np.ndarray, radius: float, k: int) -> np.ndarray:
    d, i = tree.query(p)
    if d > radius:
        raise TrackError(f"route point {k + 1} ({p[0]:.0f}, {p[1]:.0f}) is {d:.0f} px from the drawn line")
    return coords[i]


def _leg(cost: np.ndarray, a: np.ndarray, b: np.ndarray, margin: float) -> np.ndarray:
    """Cheapest path from a to b (x, y px) through ``cost``, searched in a box around both."""
    h, w = cost.shape
    lo = np.maximum(np.floor(np.minimum(a, b) - margin), 0).astype(int)
    hi = np.minimum(np.ceil(np.maximum(a, b) + margin), [w - 1, h - 1]).astype(int)
    sub = cost[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    start = tuple(np.round(a - lo).astype(int)[::-1])
    end = tuple(np.round(b - lo).astype(int)[::-1])
    mcp = MCP_Geometric(sub)
    mcp.find_costs([start], [end])
    return np.array(mcp.traceback(end), dtype=float)[:, ::-1] + lo


def trace(route: Route, skeleton: np.ndarray, mask: np.ndarray, *, snap: float = 15.0,
          off_ink: float = 50.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The route as a dense polyline, which of its points lie on free legs, and warnings.

    Steps along the skeleton cost 1, across other ink 3 and across paper
    ``off_ink``, so a path bridges a pencil skip but does not cut across to a
    neighbouring line.
    """
    cost = np.where(skeleton, 1.0, np.where(mask, 3.0, off_ink))
    coords = np.column_stack(np.nonzero(skeleton)[::-1]).astype(float)
    if len(coords) == 0:
        raise TrackError("no drawn line to follow")
    tree = cKDTree(coords)
    pts = [p if free else _snap(p, tree, coords, snap, k)
           for k, (p, free) in enumerate(zip(route.points, route.free))]
    n = len(pts)
    legs = range(n) if route.closed else range(1, n)
    line, on_free, warnings = [], [], []
    for k in legs:
        a, b = pts[k - 1], pts[k]
        if route.free[k]:
            steps = max(2, int(np.ceil(np.linalg.norm(b - a))))
            seg = a + (b - a) * np.linspace(0.0, 1.0, steps + 1)[:, None]
        else:
            seg = _leg(cost, a, b, margin=max(30.0, 0.5 * float(np.linalg.norm(b - a))))
            ix = np.round(seg).astype(int)
            off = int(np.count_nonzero(~mask[ix[:, 1], ix[:, 0]]))
            if off > max(30, 0.25 * len(seg)):  # more than a pencil skip: is a point off the line?
                warnings.append(f"the route from point {k} to point {k + 1} leaves the ink for {off} px")
        line.append(seg[:-1])
        on_free.append(np.full(len(seg) - 1, bool(route.free[k])))
    if not route.closed:
        line.append(pts[-1][None])
        on_free.append(np.array([False]))
    poly, free = np.vstack(line), np.concatenate(on_free)

    # Round off the corners where straight free legs meet, and where they join the ink.
    if free.any():
        mode = "wrap" if route.closed else "nearest"
        near = gaussian_filter1d(free.astype(float), 12, mode=mode) > 0.01
        smooth = gaussian_filter1d(poly, 6, axis=0, mode=mode)
        poly = np.where(near[:, None], smooth, poly)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-9])
    return poly[keep], free[keep], warnings
