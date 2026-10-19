"""Thin the ink mask to a one-pixel skeleton and split it into paths.

A path runs between two graph nodes (stroke ends or junctions), or is a closed
loop with no nodes at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
END, JUNCTION = "end", "junction"


@dataclass
class Path:
    points: np.ndarray  # (n, 2) x, y pixel coordinates in order
    start: int          # node id, -1 for closed loops
    end: int
    closed: bool

    @property
    def length(self) -> float:
        pts = np.vstack([self.points, self.points[:1]]) if self.closed else self.points
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _adjacency(skel: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """8-connected pixel adjacency, minus diagonal links that duplicate a path
    through a shared 4-neighbour (those would create fake 3-cycles and junctions)."""
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    n = len(ys)
    coords = np.column_stack([xs, ys])
    if n == 0:
        return coords, []
    index = np.full((h + 2, w + 2), -1, np.int64)
    index[ys + 1, xs + 1] = np.arange(n)
    on = index >= 0
    src, dst = [], []
    for dy, dx in _NEIGHBOURS:
        nb = index[ys + 1 + dy, xs + 1 + dx]
        ok = nb >= 0
        if dy and dx:
            ok &= ~(on[ys + 1 + dy, xs + 1] | on[ys + 1, xs + 1 + dx])
        src.append(np.nonzero(ok)[0])
        dst.append(nb[ok])
    src_a, dst_a = np.concatenate(src), np.concatenate(dst)
    dst_a = dst_a[np.argsort(src_a, kind="stable")]
    counts = np.bincount(src_a, minlength=n)
    return coords, [a.tolist() for a in np.split(dst_a, np.cumsum(counts)[:-1])]


def trace(skel: np.ndarray) -> tuple[list[Path], list[str]]:
    """Split a skeleton into paths. Returns the paths and the kind of each node id."""
    coords, adj = _adjacency(skel)
    n = len(coords)
    deg = np.array([len(a) for a in adj], dtype=int)
    node = np.full(n, -1)
    kinds: list[str] = []

    for p in np.nonzero(deg >= 3)[0]:  # neighbouring junction pixels form one node
        if node[p] >= 0:
            continue
        nid = len(kinds)
        kinds.append(JUNCTION)
        node[p] = nid
        stack = [p]
        while stack:
            q = stack.pop()
            for r in adj[q]:
                if deg[r] >= 3 and node[r] < 0:
                    node[r] = nid
                    stack.append(r)
    for p in np.nonzero(deg == 1)[0]:
        node[p] = len(kinds)
        kinds.append(END)

    visited = np.zeros(n, dtype=bool)
    paths: list[Path] = []
    direct: set[tuple[int, int]] = set()
    for p in np.nonzero(node >= 0)[0]:
        for q in adj[p]:
            if node[q] >= 0:
                if node[q] != node[p] and (q, p) not in direct:
                    direct.add((p, q))
                    paths.append(Path(coords[[p, q]].astype(float), node[p], node[q], False))
                continue
            if visited[q]:
                continue
            seq, prev, cur = [p, q], p, q
            visited[q] = True
            while node[cur] < 0:
                a, b = adj[cur]
                prev, cur = cur, (b if a == prev else a)
                visited[cur] |= node[cur] < 0
                seq.append(cur)
            paths.append(Path(coords[seq].astype(float), node[p], node[cur], False))

    for p in np.nonzero((deg == 2) & ~visited)[0]:  # loops without any node
        if visited[p]:
            continue
        seq, prev, cur = [p], p, adj[p][0]
        visited[p] = True
        while cur != p:
            visited[cur] = True
            seq.append(cur)
            a, b = adj[cur]
            prev, cur = cur, (b if a == prev else a)
        paths.append(Path(coords[seq].astype(float), -1, -1, True))
    return paths, kinds


def skeleton_from_mask(mask: np.ndarray, spur_length: float, rounds: int = 5) -> np.ndarray:
    """Skeletonize and repeatedly remove short spurs (end -> junction branches)."""
    skel = skeletonize(mask)
    for _ in range(rounds):
        paths, kinds = trace(skel)
        changed = False
        for p in paths:
            if p.closed or {kinds[p.start], kinds[p.end]} != {END, JUNCTION}:
                continue
            if p.length >= spur_length:
                continue
            tip = p.points[:-1] if kinds[p.end] == JUNCTION else p.points[1:]
            skel[tip[:, 1].astype(int), tip[:, 0].astype(int)] = False
            changed = True
        if not changed:
            break
        skel = skeletonize(skel)
    return skel


def _leaving(pts: np.ndarray, back: float) -> np.ndarray:
    """Unit direction in which a polyline leaves its first point, measured over ~``back`` px."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    d = pts[min(int(np.searchsorted(s, min(back, 0.5 * s[-1]))), len(pts) - 1)] - pts[0]
    return d / max(float(np.linalg.norm(d)), 1e-9)


def drop_stubs(paths: list[Path], kinds: list[str], max_length: float,
               min_angle: float = np.radians(45)) -> list[Path]:
    """Remove short straight dead ends that stick out sideways from a line, such as
    tick marks drawn across it, and join the two halves of the line they leave.

    A stub runs from a free end to a junction where exactly two other path ends
    meet, on paths at least twice as long, and leaves the line at ``min_angle`` or
    more. Where the line also turns sharply at that junction, the joiner would
    otherwise follow the stub instead of the line.
    """
    def ends_at(node: int, among: list[Path]) -> list[tuple[Path, np.ndarray]]:
        """(path, its points running away from ``node``) for each path end at ``node``."""
        out = [(p, p.points) for p in among if not p.closed and p.start == node]
        return out + [(p, p.points[::-1]) for p in among if not p.closed and p.end == node]

    stubs: dict[int, list[Path]] = {}
    for p in paths:
        if p.closed or {kinds[p.start], kinds[p.end]} != {END, JUNCTION} or p.length > max_length:
            continue
        if np.linalg.norm(p.points[-1] - p.points[0]) < 0.85 * p.length:
            continue
        stubs.setdefault(p.start if kinds[p.start] == JUNCTION else p.end, []).append(p)

    drop, joins = set(), []
    for node, ss in stubs.items():
        line = [(p, pts) for p, pts in ends_at(node, paths) if all(p is not s for s in ss)]
        back = 2.0 * max(s.length for s in ss)
        if len(line) != 2 or min(p.length for p, _ in line) < back:
            continue
        tangent = _leaving(line[0][1], back) - _leaving(line[1][1], back)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        sides = [pts for p, pts in ends_at(node, ss)]
        if all(abs(float(_leaving(pts, back) @ tangent)) <= np.cos(min_angle) for pts in sides):
            drop.update(id(s) for s in ss)
            joins.append(node)

    out = [p for p in paths if id(p) not in drop]
    for node in joins:
        line = ends_at(node, out)
        if len(line) != 2:
            continue
        (a, pa), (b, pb) = line
        if a is b:
            merged = Path(a.points, -1, -1, True)
        else:
            far_a = a.end if a.start == node else a.start
            far_b = b.end if b.start == node else b.start
            merged = Path(np.vstack([pa[::-1], pb[1:]]), far_a, far_b, False)
        out = [p for p in out if p is not a and p is not b] + [merged]
    return out


def pieces(skel: np.ndarray, min_length: float, bridge_length: float,
           stub_length: float = 0.0) -> list[Path]:
    """Paths worth handing to the stroke joiner.

    Short junction-to-junction bridges (typical in the skeleton of an X crossing)
    are dropped; the joiner reconnects across them by direction instead. With
    ``stub_length``, dead ends up to that long are dropped as marks (``drop_stubs``).
    """
    paths, kinds = trace(skel)
    out = []
    for p in paths:
        length = p.length
        if not p.closed and kinds[p.start] == JUNCTION and kinds[p.end] == JUNCTION and length < bridge_length:
            continue
        if length >= min_length:
            out.append(p)
    if stub_length > 0:
        out = drop_stubs(out, kinds, stub_length)
    return out
