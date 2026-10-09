"""Join skeleton paths into whole strokes.

Pencil lines break where the pencil skipped, and pass through junctions where
lines touch or cross. Path ends are linked greedily, cheapest first, with a
cost that favours short gaps and straight continuation. A chain whose two ends
link up becomes a closed stroke.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import geometry


@dataclass
class Stroke:
    points: np.ndarray
    closed: bool

    @property
    def length(self) -> float:
        return geometry.length(self.points, self.closed)

    @property
    def end_gap(self) -> float:
        return float(np.linalg.norm(self.points[-1] - self.points[0]))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))


def _end_geometry(points: np.ndarray, at_end: bool, skip: float, back: float):
    """Position and outward unit tangent at one end of a polyline."""
    pts = points[::-1] if at_end else points
    s = geometry.cumulative_length(pts)
    total = s[-1]
    s0, s1 = min(skip, 0.2 * total), min(skip + back, 0.5 * total)
    if s1 - s0 < 1e-6:
        s0, s1 = 0.0, total
    p0 = np.array([np.interp(s0, s, pts[:, 0]), np.interp(s0, s, pts[:, 1])])
    p1 = np.array([np.interp(s1, s, pts[:, 0]), np.interp(s1, s, pts[:, 1])])
    return pts[0], geometry.unit(p0 - p1)


@dataclass
class JoinParams:
    max_gap: float            # longest pencil break bridged
    junction_radius: float    # ends closer than this are treated as meeting at a junction
    min_loop: float           # shortest stroke that may close into a loop
    lateral_tol: float        # how far a gap may stray from a smooth arc between its ends
    max_gap_angle: float = np.radians(50)   # gap direction vs each end's tangent (short gaps)
    max_bend: float = np.radians(75)        # turn between the two joined ends
    angle_weight: float = 10.0              # px of cost per radian of misalignment
    allow_close: bool = True


def _cost(pa, ta, pb, tb, jp: JoinParams) -> float | None:
    gap = pb - pa
    d = float(np.linalg.norm(gap))
    bend = _angle(ta, -tb)
    if bend > jp.max_bend:
        return None
    if d <= jp.junction_radius:
        return d + jp.angle_weight * bend
    if d > jp.max_gap:
        return None
    g = gap / d
    a1, a2 = _angle(ta, g), _angle(-tb, g)
    if max(a1, a2) > jp.max_gap_angle * (1.0 - 0.6 * d / jp.max_gap):  # long gaps must line up
        return None
    # A break in a smooth line is bridged by an arc, whose chord makes equal angles with
    # the tangents at both ends; reject ends that would need a kink to connect.
    skew = _signed_angle(ta, g) - _signed_angle(g, -tb)
    if d * abs(np.sin(skew / 2)) > max(jp.lateral_tol, 0.1 * d):
        return None
    return d + jp.angle_weight * (a1 + a2 + bend)


def _corner_indices(pts: np.ndarray, closed: bool, window: float, min_angle: float) -> list[int]:
    """Indices where the polyline turns by more than ``min_angle`` within +-``window`` px."""
    n = len(pts)
    if n < 5:
        return []
    work = np.vstack([pts, pts, pts]) if closed else pts
    s = geometry.cumulative_length(work)
    lo = np.searchsorted(s, s - window)
    hi = np.minimum(np.searchsorted(s, s + window), len(work) - 1)
    a, b = work - work[lo], work[hi] - work
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 0.7 * window) & (nb > 0.7 * window)
    cos = np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-9)
    turn = np.where(ok, np.arccos(np.clip(cos, -1, 1)), 0.0)
    if closed:
        turn = turn[n:2 * n]
    sharp = turn > min_angle
    out, i = [], 0
    while i < n:  # one corner per run of sharp points
        if sharp[i]:
            j = i
            while j + 1 < n and sharp[j + 1]:
                j += 1
            out.append(i + int(np.argmax(turn[i:j + 1])))
            i = j + 1
        else:
            i += 1
    return out


def split_corners(paths: list[np.ndarray], closed: list[bool], window: float,
                  min_angle: float = np.radians(55)):
    """Cut paths at sharp corners so the joiner can choose a straighter continuation
    (e.g. a road edge with a pencil gap right where a start line meets it).

    Returns (paths, closed flags, corner pairs); each pair names the two piece ends
    that met at a corner, which are re-linked if nothing better claims them.
    """
    out, out_closed, pairs = [], [], []
    for pts, c in zip(paths, closed):
        cuts = _corner_indices(pts, c, window, min_angle)
        if not cuts:
            out.append(pts)
            out_closed.append(c)
            continue
        if c:
            pts = np.roll(pts, -cuts[0], axis=0)
            pts = np.vstack([pts, pts[:1]])
            bounds = [k - cuts[0] for k in cuts] + [len(pts) - 1]
        else:
            bounds = [0] + cuts + [len(pts) - 1]
        first = len(out)
        for a, b in zip(bounds[:-1], bounds[1:]):
            out.append(pts[a:b + 1])
            out_closed.append(False)
        pairs += [(2 * k + 1, 2 * (k + 1)) for k in range(first, len(out) - 1)]
        if c:
            pairs.append((2 * (len(out) - 1) + 1, 2 * first))
    return out, out_closed, pairs


def join(paths: list[np.ndarray], closed: list[bool], jp: JoinParams,
         corner_pairs: list[tuple[int, int]] = ()) -> list[Stroke]:
    """Link open paths end-to-end. ``paths[i]`` is an (n, 2) polyline; its start is
    end id ``2 * i`` and its end is ``2 * i + 1``."""
    n = len(paths)
    lengths = [geometry.length(p, c) for p, c in zip(paths, closed)]
    back = max(8.0, 2.5 * jp.junction_radius)
    # End e = 2*i (start of path i) or 2*i + 1 (end of path i).
    pos = np.zeros((2 * n, 2))
    tan = np.zeros((2 * n, 2))
    for i, p in enumerate(paths):
        for side in (0, 1):
            pos[2 * i + side], tan[2 * i + side] = _end_geometry(p, bool(side), 2.0, back)

    open_ends = [e for e in range(2 * n) if not closed[e // 2]]
    candidates = []
    if open_ends:
        tree = cKDTree(pos[open_ends])
        for a, b in tree.query_pairs(max(jp.max_gap, jp.junction_radius)):
            ea, eb = open_ends[a], open_ends[b]
            c = _cost(pos[ea], tan[ea], pos[eb], tan[eb], jp)
            if c is not None:
                candidates.append((c, ea, eb))
    candidates.sort()

    parent = list(range(n))
    chain_len = list(lengths)
    chain_closed = list(closed)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    link: dict[int, int] = {}

    def connect(ea: int, eb: int) -> None:
        if ea in link or eb in link:
            return
        ra, rb = find(ea // 2), find(eb // 2)
        d = float(np.linalg.norm(pos[ea] - pos[eb]))
        if d > jp.junction_radius and min(chain_len[ra], chain_len[rb]) < 0.5 * d:
            return  # a pencil break sits between pieces of line, not next to a tiny mark
        if ra == rb:  # the two free ends of one chain: closing it into a loop
            if not jp.allow_close or chain_len[ra] < max(4.0 * d, jp.min_loop):
                return
            chain_closed[ra] = True
        else:
            parent[rb] = ra
            chain_len[ra] += chain_len[rb] + d
        link[ea], link[eb] = eb, ea

    for _, ea, eb in candidates:
        connect(ea, eb)
    for ea, eb in corner_pairs:  # genuine sharp corners nobody else continued through
        connect(ea, eb)

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)

    strokes = []
    for root, idx in members.items():
        if len(idx) == 1 and closed[idx[0]]:
            strokes.append(Stroke(paths[idx[0]], True))
            continue
        if chain_closed[root]:
            start = 2 * idx[0]
        else:
            start = next(e for i in idx for e in (2 * i, 2 * i + 1) if e not in link)
        parts, e = [], start
        while True:
            i, side = divmod(e, 2)
            parts.append(paths[i] if side == 0 else paths[i][::-1])
            nxt = link.get(e ^ 1)
            if nxt is None or nxt == start:
                break
            e = nxt
        pts = geometry.dedupe(np.vstack(parts))
        if chain_closed[root] and len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) < 1e-6:
            pts = pts[:-1]
        strokes.append(Stroke(pts, bool(chain_closed[root])))
    return strokes


def close_nearly_closed(strokes: list[Stroke], max_gap: float) -> list[Stroke]:
    """Close long open strokes whose ends nearly meet, whatever the end directions
    (e.g. a loop that was started and finished at a corner)."""
    for s in strokes:
        if not s.closed and s.end_gap <= max_gap and s.end_gap <= 0.05 * s.length:
            s.closed = True
    return strokes
