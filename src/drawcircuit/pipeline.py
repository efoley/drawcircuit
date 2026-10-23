"""End-to-end processing: image -> track edges as B-splines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

from . import geometry, preprocess, profile, skeleton, strokes, ticks, track
from . import layout as layout_mod
from . import route as route_mod
from .layout import CENTERLINE, CIRCUIT, SEGMENT, Layout, TrackError
from .track import FitParams, Track

REFERENCE_SIZE = 1600  # pixel defaults below are tuned for this working resolution


@dataclass
class Params:
    """Tunables. Pixel values are in working-frame pixels; ``None`` means a default
    scaled from the working resolution."""

    working_size: int = REFERENCE_SIZE  # long side of the working frame
    find_paper: bool = True             # rectify a sheet photographed on a darker surface
    mode: str = "auto"                  # auto | edges | centerline
    closed: bool | None = None          # force circuit (True) / segment (False); None = detect
    direction: str = "ccw"              # travel direction for circuits, as seen on the page
    track_width: float | None = None    # centerline mode only; default 3.5% of the frame diagonal
    ink_threshold: float | None = None  # override the automatic ink threshold (0..1)
    tol: float | None = None            # max distance of the spline from the drawn line
    cp_spacing: float | None = None     # control point spacing on straights
    smooth: float = 0.05                # control polygon smoothness penalty
    max_gap: float | None = None        # longest break in a pencil line that gets bridged
    margin: float | None = None         # ignore this close to the image border
    close_radius: float | None = None   # merge sketchy strokes closer than ~2x this
    length_m: float | None = None       # real length of the lap (or segment) in meters; sets the scale
    track_width_m: float | None = None  # centerline mode only: road width in meters (needs length_m)
    grades: list[str] | None = None     # grade of each section between tick marks, in travel order
    vertical_radius_m: float | None = None  # curves easing each grade change (needs length_m); 0: abrupt
    move_tick: int | None = None        # move this tick (1-based) along the lap until the grades close it
    section_widths_m: dict[int, float] | None = None  # road width of some sections (centerline circuits)
    width_taper: float = 5.0            # a width change is spread over this many times its size
    route: route_mod.Route | None = None  # follow these waypoints instead of joining strokes
    under: int | None = None            # section that passes under where the lap crosses itself
    clearance_m: float = 6.0            # how far above it the other road must be (needs length_m)
    min_inside_radius_m: float = 6.0    # round centerline corners so the inside edge is no tighter; 0: as drawn
    max_corner_turn_deg: float = 200.0  # report corners that turn through more (spirals)
    assumed_width_m: float = 10.0       # without length_m, the road is taken to be this wide
    start: tuple[float, float] | None = None  # start/finish line marked on the drawing (frame px)
    drop_ticks: list[tuple[float, float]] | None = None  # marks found as ticks that are not (frame px)
    add_ticks: list[tuple[float, float]] | None = None   # ticks where none is drawn (frame px)
    section_lengths_m: list[float] | None = None  # sections of these lengths from the start, not the drawn ticks

    def px(self, value: float | None, default: float) -> float:
        return float(value) if value is not None else default * self.working_size / REFERENCE_SIZE


@dataclass
class Result:
    track: Track
    frame_size: tuple[int, int]         # (width, height) of the working frame
    homography: np.ndarray              # input image px -> working frame px
    paper_detected: bool
    debug: dict = field(default_factory=dict)


def process(image: np.ndarray | str | Path, params: Params | None = None) -> Result:
    """Run the full pipeline on an image (BGR array or file path)."""
    p = params or Params()
    img = preprocess.load_image(image) if isinstance(image, (str, Path)) else image
    page = preprocess.prepare_page(img, p.working_size, p.find_paper)
    h, w = page.image.shape[:2]
    diag = float(np.hypot(w, h))
    debug: dict = {"page": page.image}

    ink = preprocess.segment_ink(
        page.image,
        margin=p.px(p.margin, 12),
        close_radius=p.px(p.close_radius, 3),
        background_kernel=p.px(None, 21),
        min_blob_area=p.px(None, 6) ** 2,
        max_hole_area=p.px(None, 20) ** 2,
        threshold=p.ink_threshold,
    )
    debug.update(ink=ink.strength, mask=ink.mask, valid=ink.valid)
    if not ink.mask.any():
        raise TrackError("no ink found on the page")

    # Typical stroke width: twice the distance from the skeleton to the stroke edge.
    dist = cv2.distanceTransform(ink.mask.astype(np.uint8), cv2.DIST_L2, 3)
    raw_skel = skeletonize(ink.mask)
    stroke_width = float(np.clip(2.0 * np.median(dist[raw_skel]), 2.0, None))

    junction_radius = max(p.px(None, 6), 2.5 * stroke_width)
    skel = skeleton.skeleton_from_mask(ink.mask, spur_length=max(p.px(None, 12), 3.0 * stroke_width))
    paths = skeleton.pieces(skel, min_length=p.px(None, 10), bridge_length=junction_radius,
                            stub_length=p.px(None, 60))
    debug.update(skeleton=skel, paths=paths)

    if p.route is not None:  # the line is given as waypoints: follow them along the ink
        line, free, warnings = route_mod.trace(p.route, skel, ink.mask,
                                               snap=max(p.px(None, 15), 2 * stroke_width))
        stroke = strokes.Stroke(line, p.route.closed)
        lay = Layout(CIRCUIT if p.route.closed else SEGMENT, CENTERLINE, [stroke], [stroke], warnings)
        debug.update(strokes=[stroke], layout=lay, route=p.route)
        return _finish(p, lay, ink, stroke_width, diag, (w, h), page, debug, free)

    pieces, piece_closed, corners = strokes.split_corners(
        [q.points for q in paths], [q.closed for q in paths], window=max(p.px(None, 10), 2.0 * stroke_width))
    joined = strokes.join(
        pieces, piece_closed,
        strokes.JoinParams(max_gap=p.px(p.max_gap, 120), junction_radius=junction_radius,
                           min_loop=p.px(None, 300), lateral_tol=1.25 * stroke_width,
                           allow_close=p.closed is not False),
        corners,
    )
    if p.closed is not False:
        joined = strokes.close_nearly_closed(joined, p.px(None, 90))
    if p.closed:
        for s in joined:
            s.closed = True
    debug["strokes"] = joined

    lay = layout_mod.classify(joined, mode=p.mode, stroke_width=stroke_width,
                              min_length=0.05 * diag)
    debug["layout"] = lay
    return _finish(p, lay, ink, stroke_width, diag, (w, h), page, debug)


def _finish(p: Params, lay: Layout, ink: preprocess.Ink, stroke_width: float, diag: float,
            frame_size: tuple[int, int], page: preprocess.Page, debug: dict,
            ignore: np.ndarray | None = None) -> Result:
    """Fit the track to the chosen line(s), with ticks, scale and grades."""
    if p.section_lengths_m is not None and (p.drop_ticks or p.add_ticks or p.section_widths_m):
        raise ValueError("sections given by length replace the drawn ticks: drop, add or widen none with them")
    marks = []
    if lay.source == CENTERLINE:
        line = lay.curves[0]
        marks = ticks.find(ink.mask, line.points, line.closed, stroke_width, max_reach=p.px(None, 100),
                           ignore=ignore)
        marks = _drop_ticks(marks, p.drop_ticks or [], near=p.px(None, 40))
        marks += _add_ticks(line.points, p.add_ticks or [], near=p.px(None, 40))
    debug.update(stroke_width=stroke_width, ticks=marks)

    fit = FitParams(tol=p.px(p.tol, 2.5), cp_spacing=p.px(p.cp_spacing, 60), smooth=p.smooth)
    t = _scaled_build(lay, fit, p, diag, marks)
    if p.section_lengths_m is not None:
        marks = _ticks_at(t, p.section_lengths_m)
        debug["ticks"] = marks
    if t.reference is not None and (marks or p.grades is not None):
        radius = (p.vertical_radius_m if p.vertical_radius_m is not None
                  else profile.VERTICAL_RADIUS_M) / t.m_per_px
        t.profile = profile.build(t.reference, t.centerline, [m.point for m in marks], p.grades,
                                  t.warnings, vertical_radius=radius, move_tick=p.move_tick, under=p.under,
                                  clearance=p.clearance_m / t.m_per_px)
    elif p.grades is not None:
        raise TrackError("grades need a circuit with a reference line")
    elif marks:
        t.warnings.append(f"ignored {len(marks)} tick mark(s): grades need a circuit with a reference line")
    return Result(t, frame_size, page.homography, page.paper_detected, debug)


def _drop_ticks(marks: list[ticks.Tick], points, near: float) -> list[ticks.Tick]:
    """``marks`` without the tick nearest each of ``points`` (an erased mark or a stray line
    that was taken for a tick)."""
    marks = list(marks)
    for x, y in points:
        d = [float(np.hypot(*(m.point - (x, y)))) for m in marks]
        if not d or min(d) > near:
            raise TrackError(f"no tick within {near:.0f} px of ({x:.0f}, {y:.0f}) to drop")
        marks.pop(int(np.argmin(d)))
    return marks


def _add_ticks(line: np.ndarray, points, near: float) -> list[ticks.Tick]:
    """Ticks where none is drawn, at the points of ``line`` nearest ``points`` (to split a
    section in two, e.g. for a steeper pitch inside it)."""
    out = []
    for x, y in points:
        d = np.hypot(*(line - (x, y)).T)
        if d.min() > near:
            raise TrackError(f"no line within {near:.0f} px of ({x:.0f}, {y:.0f}) to add a tick to")
        out.append(ticks.Tick(np.asarray(line[int(np.argmin(d))], dtype=float), 0.0))
    return out


def _ticks_at(t: Track, lengths_m: list[float]) -> list[ticks.Tick]:
    """Ticks that split the lap into sections of these lengths, meters in travel order from
    the start line (as a map's legend gives them), in place of the drawn ones."""
    if t.reference is None or t.scale_from != "length":
        raise TrackError("sections given by length need a circuit and its lap length")
    total, given = t.reference.length * t.m_per_px, float(sum(lengths_m))
    if abs(given - total) > 0.01 * total:
        raise TrackError(f"the section lengths add up to {given:.0f} m, not the lap's {total:.0f} m")
    s = np.concatenate([[0.0], np.cumsum(lengths_m[:-1])]) * t.reference.length / given
    return [ticks.Tick(point, 0.0) for point in t.reference.line(t.reference.u_at(s))]


def _scaled_build(lay: layout_mod.Layout, fit: FitParams, p: Params, diag: float,
                  marks: list[ticks.Tick]) -> Track:
    """Build the track. Given its real length, also set the scale and, for a centerline
    drawing, make the road ``track_width_m`` wide, or ``section_widths_m[k]`` in
    section k (the road width changes the reference line and so the length in px,
    so this takes a rebuild or two)."""
    if p.length_m is None and (p.track_width_m is not None or p.vertical_radius_m is not None
                               or p.section_widths_m):
        raise ValueError("lengths in meters need the real length of the track too, to set the scale")
    if p.section_widths_m:
        if p.track_width_m is None:
            raise ValueError("section widths need a road width in meters for the other sections")
        if lay.source != CENTERLINE or not lay.curves[0].closed:
            raise TrackError("section widths need a circuit drawn as a centerline")
    metric = p.track_width_m is not None and lay.source == CENTERLINE
    rounds = bool(p.min_inside_radius_m) and p.length_m is not None and lay.kind == CIRCUIT
    rounds = rounds and (metric or lay.source != CENTERLINE)  # a centerline needs its width in meters
    points = [m.point for m in marks]

    def width(m_per_px: float | None):
        if not metric:
            return p.track_width if p.track_width is not None else 0.035 * diag
        default = p.track_width_m / m_per_px
        if not p.section_widths_m:
            return default
        widths = {k: v / m_per_px for k, v in p.section_widths_m.items()}
        return lambda center, u: profile.section_widths(center, points, widths, default, p.width_taper)(u)

    m = None
    if metric or rounds:  # first guess at the scale, from the drawn line(s)
        lengths = []
        for line in lay.curves:
            pts = geometry.resample(line.points, spacing=2.0, closed=line.closed)
            pts = gaussian_filter1d(pts, 2.0, axis=0, mode="wrap" if line.closed else "nearest")
            lengths.append(geometry.length(pts, line.closed))
        m = p.length_m / float(np.mean(lengths))

    def build(m_per_px: float | None) -> Track:
        inside = p.min_inside_radius_m / m_per_px if rounds else None
        return track.build(lay, fit, direction=p.direction, track_width=width(m_per_px),
                           inside_radius=inside, max_turn=p.max_corner_turn_deg, start=p.start)

    t = build(m)
    if p.length_m is None:  # no real length: take the road to be the usual width
        t.m_per_px, t.scale_from = p.assumed_width_m / t.width["median"], "assumed road width"
        t.warnings.append(f"no real length given, so the road is taken to be {p.assumed_width_m:g} m wide "
                          "and corners are not rounded; give the lap length for a real scale")
        return t
    t.m_per_px, t.scale_from = p.length_m / t.length(), "length"
    for _ in range(3):
        if not (metric or rounds) or abs(t.m_per_px - m) <= 0.002 * m:
            break
        m = t.m_per_px
        t = build(m)
        t.m_per_px, t.scale_from = p.length_m / t.length(), "length"
    return t
