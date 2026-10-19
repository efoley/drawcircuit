"""Grades along a circuit: the sections between tick marks and the elevation they give.

Ticks drawn across the centerline split the lap into sections. Section 1 starts
at the first tick after the start line, and the last section runs through the
start line. Each section gets a grade (rise over horizontal run, so 0.05 for
5%), given in travel order. A section marked ``descent``, ``climb`` or ``?`` gets
the grade that brings the lap back to the height it started at. Alternatively
every section gets a grade and one tick moves along the lap until they close it.

The grade does not jump at a tick: it changes at a steady rate along a
parabolic vertical curve centred on the tick, as on a real road. The curve's
radius sets how gradual that is (a change of 6% over 60 m on a 1000 m radius).
Outside the curves the elevation is exactly that of the straight grades, so
the lap still closes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import geometry
from .bspline import Spline
from .layout import TrackError
from .reference import Reference

UNKNOWN = {"descent": -1, "down": -1, "downhill": -1, "climb": 1, "up": 1, "uphill": 1, "?": 0}
_RANGE = re.compile(r"([+-]?)(\d+(?:\.\d+)?)\s*%?\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*%?")
_SINGLE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%?")
VERTICAL_RADIUS_M = 1000.0  # default radius of the curves that ease grade changes


@dataclass
class Section:
    index: int             # 1-based, in travel order
    s_start: float         # px along the reference line; the last section wraps past the lap end
    s_end: float
    length: float          # px
    label: str | None = None    # the grade as given, e.g. "1-2%" or "descent"
    grade: float | None = None  # rise over run; None until grades are given
    solved: bool = False        # the grade was chosen to close the lap
    z_start: float = 0.0        # elevation at s_start, px
    z_end: float = 0.0
    width: float = 0.0          # typical road width (w_left + w_right), px


@dataclass
class Crossing:
    """Where the lap passes over itself, as on a bridge over a tunnel."""
    s_over: float        # s along the reference line of the upper road, px
    s_under: float       # s of the lower road, where the two are closest in height
    point: np.ndarray    # where they cross, frame px
    clearance: float     # how far the upper road is above the lower one there, px


@dataclass
class Profile:
    ticks: np.ndarray            # s along the reference line where each tick crosses it, increasing
    points: np.ndarray           # the tick positions on the reference line, frame px
    sections: list[Section]
    z: np.ndarray | None = None      # elevation at each reference row, px; 0 at the start line
    grade: np.ndarray | None = None  # grade at each reference row
    eases: np.ndarray | None = None  # length of the vertical curve centred on each tick, px
    moved: np.ndarray | None = None  # how far each tick was moved along the lap from where it was drawn, px
    vertical_radius: float = 0.0     # radius those curves aim for, px; 0 for abrupt changes
    crossings: list[Crossing] = field(default_factory=list)

    @property
    def graded(self) -> bool:
        return self.z is not None


def parse_grade(text: str) -> tuple[float | None, int]:
    """(grade, sign) for one grade as written: "5%", "-3", "1-2%" (the midpoint),
    "flat", or "descent" / "climb" / "?" (grade None, sign -1 / +1 / 0)."""
    t = text.strip().lower()
    if t in UNKNOWN:
        return None, UNKNOWN[t]
    if t == "flat":
        return 0.0, 0
    m = _RANGE.fullmatch(t)
    if m:
        return (-1 if m[1] == "-" else 1) * (float(m[2]) + float(m[3])) / 200.0, 0
    m = _SINGLE.fullmatch(t)
    if m:
        return float(m[1]) / 100.0, 0
    raise ValueError(f"cannot read the grade {text!r}: use e.g. 5%, -3%, 1-2%, flat, descent, climb or ?")


def _locate(ref: Reference, centerline: Spline | None, points: np.ndarray) -> np.ndarray:
    """Arc length along the reference line of each point, which lies on the centerline.

    The nearest point of the reference line is searched for only around the same
    fraction of the lap as on the centerline, so a tick on one leg of a hairpin
    cannot land on the other leg.
    """
    if len(points) == 0:
        return np.zeros(0)
    total = ref.length
    rows = ref.line(ref.u)
    if centerline is None:
        return ref.s[cKDTree(rows).query(points)[1]]
    dense = centerline.sample(max(2000, int(2 * centerline.length())))
    frac = geometry.cumulative_length(dense, True)[:-1] / geometry.length(dense, True)
    out = []
    for p, f in zip(points, frac[cKDTree(dense).query(points)[1]]):
        near = np.abs((ref.s / total - f + 0.5) % 1.0 - 0.5) < 0.05
        i = np.nonzero(near)[0][np.argmin(np.linalg.norm(rows[near] - p, axis=1))]
        out.append(ref.s[i])
    return np.asarray(out, dtype=float)


def section_widths(center: Spline, tick_points: list[np.ndarray], widths: dict[int, float],
                   default: float, taper: float) -> Callable[[np.ndarray], np.ndarray]:
    """Road width along a closed centerline, px, as a function of its parameter.

    Sections are numbered as in ``build``: section 1 starts at the first tick
    after the start of the line. Section k is ``widths[k]`` wide, else
    ``default``. Each change is spread over ``taper`` times its size, centred on
    its tick, along a smooth (sine) ramp, and never over more than half of either
    section beside it.
    """
    u = center.params(max(4000, int(4 * center.length())))
    pts = center(u)
    arc = geometry.cumulative_length(pts, True)
    total = arc[-1]
    u = np.append(u, center.domain[1])
    s = np.sort(arc[cKDTree(pts).query(np.asarray(tick_points, float).reshape(-1, 2))[1]])
    n = max(len(s), 1)
    bad = [k for k in widths if not 1 <= k <= n]
    if bad:
        raise TrackError(f"there is no section {bad[0]}: the drawing has {n} section(s)")
    w = np.array([widths.get(k + 1, default) for k in range(n)], dtype=float)
    if len(s) == 0:
        return lambda q: np.full(np.shape(q), w[0])
    lengths = np.diff(np.append(s, s[0] + total))
    change = w - np.roll(w, 1)  # at tick k, from section k - 1 (the last one for k = 0) to section k
    half = np.minimum(taper * np.abs(change) / 2, np.minimum(lengths, np.roll(lengths, 1)) / 2)

    def at(q: np.ndarray) -> np.ndarray:
        x = np.interp(np.asarray(q, dtype=float), u, arc)
        out = w[(np.searchsorted(s, x, side="right") - 1) % len(s)]
        for tick, dw, h in zip(s, change, half):
            if h > 0:
                d = (x - tick + total / 2) % total - total / 2
                ramp = (1 + np.sin(np.pi / 2 * np.clip(d / h, -1, 1))) / 2
                out = out + dw * (ramp - (d >= 0))
        return out

    return at


def _sections(s: np.ndarray, total: float) -> list[Section]:
    if len(s) == 0:
        return [Section(1, 0.0, total, total)]
    ends = np.append(s[1:], s[0] + total)
    return [Section(k + 1, a, b % total if k == len(s) - 1 else b, b - a)
            for k, (a, b) in enumerate(zip(s, ends))]


def _closing_shift(s: np.ndarray, sections: list[Section], grades: list[float | None], tick: int,
                   total: float) -> float:
    """How far tick ``tick`` (1-based) must move along the lap for the grades to close it."""
    if any(g is None for g in grades):
        raise TrackError("to move a tick so the lap closes, give every section a grade "
                         "(no descent, climb or ?)")
    if not 1 <= tick <= len(s):
        raise TrackError(f"there is no tick {tick}: the drawing has {len(s)} tick marks")
    k = tick - 1
    step = grades[k - 1] - grades[k]  # rise gained per px the tick moves forward
    if abs(step) < 1e-12:
        raise TrackError(f"tick {tick} has the same grade on both sides, so moving it cannot close the lap")
    shift = -sum(g * sec.length for g, sec in zip(grades, sections)) / step
    if not -sections[k - 1].length < shift < sections[k].length:
        raise TrackError(f"tick {tick} would have to move {shift:+.0f} px, past a neighbouring tick")
    if not 0 <= s[k] + shift < total:
        raise TrackError(f"tick {tick} would have to move {shift:+.0f} px, across the start line")
    return shift


def build(ref: Reference, centerline: Spline | None, tick_points: list[np.ndarray],
          grades: list[str] | None, warnings: list[str], vertical_radius: float = 0.0,
          move_tick: int | None = None, under: int | None = None, clearance: float = 0.0) -> Profile:
    """Sections between the ticks, and with ``grades`` the elevation along the lap.

    ``move_tick`` (1-based) moves that tick along the lap until the grades, all
    given, close it: e.g. the end of a descent with a set average grade. Where
    the lap crosses itself, section ``under`` passes at least ``clearance`` px
    below the other road: the open grades (descent, climb, ?) are changed as
    little as possible to make the room.
    """
    total = ref.length
    s = np.sort(_locate(ref, centerline, np.asarray(tick_points, float).reshape(-1, 2)))
    if len(s) > 1:  # one tick found twice
        s = s[np.append(True, np.diff(s) > 4.0)]
        if s[-1] - s[0] > total - 4.0:
            s = s[:-1]
    sections = _sections(s, total)
    moved = np.zeros(len(s))
    if grades is not None and len(grades) != len(sections):
        raise TrackError(f"{len(grades)} grades given, but the drawing has {len(sections)} sections "
                         f"({len(s)} tick marks); list one grade per section in travel order, "
                         "starting with the section after the first tick past the start line")
    if move_tick is not None:
        if grades is None:
            raise TrackError("moving a tick to close the lap needs the grades")
        moved[move_tick - 1] = _closing_shift(s, sections, [parse_grade(g)[0] for g in grades],
                                              move_tick, total)
        s = s + moved
        sections = _sections(s, total)
    points = ref.line(ref.u_at(s)) if len(s) else np.zeros((0, 2))
    for sec in sections:
        inside = (ref.s - sec.s_start) % total < sec.length
        sec.width = float(np.median((ref.w_left + ref.w_right)[inside])) if inside.any() else 0.0
    profile = Profile(s, points, sections, moved=moved)
    if grades is None:
        return profile

    signs = []
    for sec, text in zip(sections, grades):
        sec.label = text.strip()
        sec.grade, sign = parse_grade(text)
        signs.append(sign)
    rise = sum(sec.grade * sec.length for sec in sections if sec.grade is not None)
    open_ = [k for k, sec in enumerate(sections) if sec.grade is None]
    if open_:
        g = -rise / sum(sections[k].length for k in open_)
        for k in open_:
            sections[k].grade, sections[k].solved = g, True
    elif abs(rise) > 1e-4 * total:
        warnings.append(f"the grades do not close the lap: it finishes {rise:+.1f} px "
                        f"({100 * rise / total:+.2f}% of the lap length) above the start")

    pairs = overlap_pairs(ref)
    if under is not None:
        _make_room(ref, s, sections, open_, pairs, under, clearance, vertical_radius, warnings)
    for k in open_:
        if signs[k] * sections[k].grade < 0:
            warnings.append(f"section {k + 1} is marked {sections[k].label!r}, but the lap only "
                            f"closes with a grade of {100 * sections[k].grade:+.2f}% there")

    at, change, half, want, knots = _elevation(s, sections, total, vertical_radius)
    for k in np.nonzero(half < 0.9 * want)[0]:
        warnings.append(f"the grade change at tick {k + 1} ({100 * sections[k - 1].grade:+.1f}% to "
                        f"{100 * sections[k].grade:+.1f}%) is eased on only {100 * half[k] / want[k]:.0f}% "
                        "of the vertical radius: the sections beside it are too short")
    n = len(s)
    z0 = float(at(0.0)[0])  # nonzero only if a curve spans the start line
    profile.z, profile.grade = at(ref.s)
    profile.z = profile.z - z0
    for k, sec in enumerate(sections):
        sec.z_start = float(at(sec.s_start)[0]) - z0
        sec.z_end = float(at(sec.s_end)[0]) - z0 + (knots[-1] if n and k == n - 1 else 0.0)
    profile.eases, profile.vertical_radius = 2 * half, vertical_radius
    profile.crossings = crossings(ref, profile.z, pairs)
    for c in profile.crossings:
        if c.clearance < 0.99 * clearance:
            over, below = section_at(sections, c.s_over, total), section_at(sections, c.s_under, total)
            warnings.append(f"section {over.index} passes only {c.clearance:.1f} px above section "
                            f"{below.index} where they cross (want {clearance:.1f}); give the section "
                            "that goes underneath with --under")
    return profile


def section_at(sections: list[Section], s: float, total: float) -> Section:
    """The section that contains arc length ``s``."""
    return next((sec for sec in sections if (s - sec.s_start) % total < sec.length), sections[-1])


def _elevation(s: np.ndarray, sections: list[Section], total: float, vertical_radius: float):
    """Elevation along the lap for the sections' grades: (at, change, half, want, knots).

    The lap rises at each section's grade from tick to tick, with z = 0 at the start
    line (inside the last section), and each grade change is eased along a vertical
    curve around its tick, at most half as long as the sections on either side so
    that neighbouring curves do not overlap. ``at(x)`` gives (z, grade) at arc
    lengths ``x`` in [0, total], before z is shifted to be 0 at the start line.
    """
    n = len(s)
    breaks = np.concatenate([[0.0], s, [total]])
    slopes = np.array([sections[-1].grade] + [sec.grade for sec in sections[:n]])
    knots = np.concatenate([[0.0], np.cumsum(np.diff(breaks) * slopes)])
    change = np.array([sections[k].grade - sections[k - 1].grade for k in range(n)])
    want = vertical_radius * np.abs(change) / 2
    half = np.minimum(want, [min(sections[k - 1].length, sections[k].length) / 2 for k in range(n)])

    def at(x) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        z = np.interp(x, breaks, knots)
        g = slopes[np.clip(np.searchsorted(breaks, x, side="right") - 1, 0, n)]
        for tick, dg, h in zip(s, change, half):
            if h > 0:
                d = (x - tick + total / 2) % total - total / 2
                r = np.clip(h - np.abs(d), 0.0, None)
                z = z + dg / (4 * h) * r ** 2
                g = g - np.where(d >= 0, 1.0, -1.0) * dg / (2 * h) * r
        return z, g

    return at, change, half, want, knots


def _make_room(ref: Reference, s: np.ndarray, sections: list[Section], open_: list[int],
               pairs: tuple[np.ndarray, np.ndarray], under: int, clearance: float,
               vertical_radius: float, warnings: list[str]) -> None:
    """Change the open (solved) grades as little as possible so that section ``under``
    passes at least ``clearance`` below every part of the lap that crosses it, while
    the lap still closes."""
    total = ref.length
    if not 1 <= under <= len(sections):
        raise TrackError(f"there is no section {under}: the drawing has {len(sections)} sections")
    own = np.array([(ref.s - sec.s_start) % total < sec.length for sec in sections]).argmax(axis=0) == under - 1
    i, j = pairs
    cross = own[i] ^ own[j]
    if not cross.any():
        raise TrackError(f"section {under} does not cross another part of the lap")
    if len(open_) < 2:
        raise TrackError("to pass under another part of the lap, at least two sections need an open "
                         "grade (descent, climb or ?), so one can change while the lap still closes")
    lower, upper = np.where(own[i], i, j)[cross], np.where(own[i], j, i)[cross]
    lengths = np.array([sections[k].length for k in open_])

    def gaps(g: np.ndarray) -> np.ndarray:
        for k, v in zip(open_, g):
            sections[k].grade = v
        z = _elevation(s, sections, total, vertical_radius)[0](ref.s)[0]
        return z[upper] - z[lower]

    g = np.array([sections[k].grade for k in open_])
    for _ in range(10):
        gap = gaps(g)
        worst = int(np.argmin(gap))
        if gap[worst] >= 0.999 * clearance:
            break
        step = 1e-3
        slope = np.array([(gaps(g + step * np.eye(len(g))[a])[worst] - gap[worst]) / step for a in range(len(g))])
        # Closure (the open sections' total rise stays the same) and the clearance at the worst place.
        g = g + np.linalg.lstsq(np.vstack([lengths, slope]), [0.0, clearance - gap[worst]], rcond=None)[0]
    else:
        warnings.append(f"could not make {clearance:.1f} px of room under the crossing: "
                        f"section {under} passes {gap[worst]:.1f} px below")
    gaps(g)


def overlap_pairs(ref: Reference, apart: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of reference rows whose road surfaces overlap on the page although they
    are at least ``apart`` road widths apart along the lap: a crossing."""
    p = ref.line(ref.u)
    w = ref.w_left + ref.w_right
    total = ref.length
    pairs = np.array(sorted(cKDTree(p).query_pairs(float(w.max()))), dtype=int).reshape(-1, 2)
    i, j = pairs.T
    along = np.abs((ref.s[i] - ref.s[j] + total / 2) % total - total / 2)
    keep = (np.linalg.norm(p[i] - p[j], axis=1) < (w[i] + w[j]) / 2) & (along > apart * w.max())
    return i[keep], j[keep]


def crossings(ref: Reference, z: np.ndarray, pairs: tuple[np.ndarray, np.ndarray],
              apart: float = 10.0) -> list[Crossing]:
    """Where the lap passes over itself: one entry per place, at the pair of rows
    closest in height."""
    p = ref.line(ref.u)
    w = ref.w_left + ref.w_right
    total = ref.length
    i, j = pairs
    out: list[Crossing] = []
    seen = np.zeros(len(p), bool)
    for k in np.argsort(np.abs(z[i] - z[j])):
        a, b = i[k], j[k]
        if seen[a] or seen[b]:
            continue
        for c in (a, b):
            seen |= np.abs((ref.s - ref.s[c] + total / 2) % total - total / 2) < apart * w.max()
        over, under = (a, b) if z[a] >= z[b] else (b, a)
        out.append(Crossing(float(ref.s[over]), float(ref.s[under]), (p[a] + p[b]) / 2,
                            float(z[over] - z[under])))
    return out
