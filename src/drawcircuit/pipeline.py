"""End-to-end processing: image -> track edges as B-splines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import layout as layout_mod
from . import preprocess, skeleton, strokes, track
from .layout import TrackError
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
    paths = skeleton.pieces(skel, min_length=p.px(None, 10), bridge_length=junction_radius)
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
    fit = FitParams(tol=p.px(p.tol, 2.5), cp_spacing=p.px(p.cp_spacing, 60), smooth=p.smooth)
    width = p.track_width if p.track_width is not None else 0.035 * diag
    result = track.build(lay, fit, direction=p.direction, track_width=width)
    debug["stroke_width"] = stroke_width
    return Result(result, (w, h), page.homography, page.paper_detected, debug)
