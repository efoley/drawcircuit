"""Tick marks drawn across a centerline, which split it into sections.

Ticks are found in the ink rather than the skeleton, because spur pruning
removes short ones. Ink farther than half a stroke width from the drawn line
is grouped into pieces. A piece that touches the line, reaches out from it by
more than a stroke width and a half, is straight and leaves the line steeply
is half of a tick (or all of a tick drawn on one side only). Halves on
opposite sides at the same place are one tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import geometry


@dataclass
class Tick:
    point: np.ndarray   # where the tick crosses the line, frame px
    reach: float        # how far it sticks out from the line, px


def find(mask: np.ndarray, line: np.ndarray, closed: bool, stroke_width: float, *,
         max_reach: float, min_angle: float = np.radians(45)) -> list[Tick]:
    """Ticks across the polyline ``line`` (the drawn centerline) in the ink ``mask``."""
    h, w = mask.shape
    raster = np.full((h, w), 255, np.uint8)
    cv2.polylines(raster, [np.round(line * 16).astype(np.int32)], closed, 0, 1, cv2.LINE_8, 4)
    dist = cv2.distanceTransform(raster, cv2.DIST_L2, 5)  # distance to the drawn line
    band = 0.5 * stroke_width + 1.5
    rest = (mask & (dist > band)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    touching = np.unique(labels[(rest > 0) & (dist <= band + 1.5)])

    tree = cKDTree(line)
    s = geometry.cumulative_length(line, closed)
    found = []  # (arc length of the crossing, crossing point, reach)
    for c in touching[touching > 0]:
        x0, y0, bw, bh = stats[c, :4]
        ys, xs = np.nonzero(labels[y0:y0 + bh, x0:x0 + bw] == c)
        pts = np.column_stack([xs + x0, ys + y0]).astype(float)
        d = dist[ys + y0, xs + x0]
        reach = float(d.max())
        if not max(1.5 * stroke_width, band + 4.0) <= reach <= max_reach or len(pts) < 5:
            continue
        # Measure direction away from the line, where the ink is the mark's alone.
        body = pts[d >= band + 0.5 * stroke_width] if np.sum(d >= band + 0.5 * stroke_width) >= 5 else pts
        center = body.mean(axis=0)
        axis = np.linalg.eigh((body - center).T @ (body - center))[1][:, 1]
        along, across = (body - center) @ axis, (body - center) @ np.array([-axis[1], axis[0]])
        if np.percentile(np.abs(across), 95) > max(0.6 * stroke_width, 0.15 * np.ptp(along)):
            continue  # not a straight mark: a letter or a scribble touching the line
        # The mark crosses the line where its axis does, near where it touches it.
        i = int(tree.query(pts[np.argmin(d)])[1])
        k = int(np.ceil(2.0 * stroke_width + 4.0))
        near = np.arange(i - k, i + k + 1)
        near = near % len(line) if closed else near[(near >= 0) & (near < len(line))]
        rel = line[near] - center
        i = int(near[np.argmin(np.abs(rel[:, 0] * axis[1] - rel[:, 1] * axis[0]))])
        k = max(2, int(round(stroke_width)))
        if closed:
            tangent = line[(i + k) % len(line)] - line[(i - k) % len(line)]
        else:
            tangent = line[min(i + k, len(line) - 1)] - line[max(i - k, 0)]
        tangent = geometry.unit(tangent[None])[0]
        if abs(float(axis @ tangent)) > np.cos(min_angle):
            continue
        found.append((float(s[i]), line[i], reach))

    # Halves of one tick cross the line within about a stroke width of each other.
    found.sort(key=lambda f: f[0])
    total, gap = s[-1], 2.0 * stroke_width + 4.0
    groups: list[list] = []
    for f in found:
        if groups and f[0] - groups[-1][-1][0] <= gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    if closed and len(groups) > 1 and groups[0][0][0] + total - groups[-1][-1][0] <= gap:
        groups[0] = groups.pop() + groups[0]
    ticks = []
    for g in groups:
        p = np.mean([f[1] for f in g], axis=0)
        ticks.append(Tick(line[int(tree.query(p)[1])], max(f[2] for f in g)))
    return ticks
