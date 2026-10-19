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
from .layout import CENTERLINE, TrackError
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
    marks = []
    if lay.source == CENTERLINE:
        line = lay.curves[0]
        marks = ticks.find(ink.mask, line.points, line.closed, stroke_width, max_reach=p.px(None, 100))
    debug.update(stroke_width=stroke_width, ticks=marks)

    fit = FitParams(tol=p.px(p.tol, 2.5), cp_spacing=p.px(p.cp_spacing, 60), smooth=p.smooth)
    t = _scaled_build(lay, fit, p, diag, marks)
    if t.reference is not None and (marks or p.grades is not None):
        if t.m_per_px is not None:
            radius = (p.vertical_radius_m if p.vertical_radius_m is not None
                      else profile.VERTICAL_RADIUS_M) / t.m_per_px
        else:  # no scale: a quarter of the lap, which is 1000 m on a 4 km lap
            radius = 0.25 * t.length()
        t.profile = profile.build(t.reference, t.centerline, [m.point for m in marks], p.grades,
                                  t.warnings, vertical_radius=radius, move_tick=p.move_tick)
    elif p.grades is not None:
        raise TrackError("grades need a circuit with a reference line")
    elif marks:
        t.warnings.append(f"ignored {len(marks)} tick mark(s): grades need a circuit with a reference line")
    return Result(t, (w, h), page.homography, page.paper_detected, debug)


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
    if metric:  # first guess at the scale, from the drawn line
        line = lay.curves[0]
        pts = geometry.resample(line.points, spacing=2.0, closed=line.closed)
        pts = gaussian_filter1d(pts, 2.0, axis=0, mode="wrap" if line.closed else "nearest")
        m = p.length_m / geometry.length(pts, line.closed)
    t = track.build(lay, fit, direction=p.direction, track_width=width(m))
    if p.length_m is None:
        return t
    t.m_per_px = p.length_m / t.length()
    for _ in range(3):
        if not metric or abs(t.m_per_px - m) <= 0.002 * m:
            break
        m = t.m_per_px
        t = track.build(lay, fit, direction=p.direction, track_width=width(m))
        t.m_per_px = p.length_m / t.length()
    return t
