"""Overlay and debug images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pipeline import Result

EDGE_COLORS = {"outer": (40, 40, 220), "left": (40, 40, 220),
               "inner": (210, 110, 20), "right": (210, 110, 20)}
CENTER_COLOR = (60, 150, 60)
REFERENCE_COLOR = (150, 50, 150)
SHIFT = 4  # sub-pixel drawing precision for cv2


def _poly(img, pts, closed, color, thickness):
    cv2.polylines(img, [np.round(pts * (1 << SHIFT)).astype(np.int32)], closed, color,
                  thickness, cv2.LINE_AA, SHIFT)


def _pt(p):
    return tuple(int(v) for v in np.round(p))


def overlay(result: Result, show_control_points: bool = True) -> np.ndarray:
    page = result.debug["page"]
    img = cv2.addWeighted(page, 0.45, np.full_like(page, 255), 0.55, 0)
    t = result.track
    for name, s in t.edges.items():
        color = EDGE_COLORS.get(name, (0, 0, 0))
        if show_control_points:
            cp = s.unique_control_points
            soft = tuple(int(0.45 * c + 0.55 * 255) for c in color)
            _poly(img, cp, s.periodic, soft, 1)
            for q in cp:
                cv2.circle(img, _pt(q), 3, soft, -1, cv2.LINE_AA)
        _poly(img, s.sample(1500), s.periodic, color, 2)
    if t.centerline is not None:
        _poly(img, t.centerline.sample(1500), t.centerline.periodic, CENTER_COLOR, 1)
    if t.reference is not None:
        _poly(img, t.reference.line.sample(1500), True, REFERENCE_COLOR, 1)

    # Start marker and direction arrow halfway between the edges.
    first, second = t.edges.values()
    a0, b0 = first(first.domain[0]), second(second.domain[0])
    start = (a0 + b0) / 2
    heading = first(first.domain[0], 1) + second(second.domain[0], 1)
    heading = heading / max(np.linalg.norm(heading), 1e-9)
    cv2.line(img, _pt(a0), _pt(b0), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, _pt(start), _pt(start + 45 * heading), (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.35)

    label = f"{t.kind} ({'two edges drawn' if t.source == 'edges' else 'centerline drawn'})"
    legend = [(label, (0, 0, 0))] + [(n, EDGE_COLORS.get(n, (0, 0, 0))) for n in t.edges]
    if t.reference is not None:
        legend.append(("reference", REFERENCE_COLOR))
    for i, (text, color) in enumerate(legend):
        y = 32 + 30 * i
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img


def _palette(i: int) -> tuple[int, int, int]:
    hue = (i * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def debug_images(result: Result) -> dict[str, np.ndarray]:
    d = result.debug
    page = d["page"]
    faint = cv2.addWeighted(page, 0.3, np.full_like(page, 255), 0.7, 0)
    out = {"1_page": page}

    ink = (255 - np.clip(d["ink"] * 255 * 2, 0, 255)).astype(np.uint8)
    ink = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
    ink[~d["valid"]] = (ink[~d["valid"]] * 0.6 + np.array([0, 0, 120])).astype(np.uint8)
    out["2_ink_strength"] = ink
    out["3_ink_mask"] = np.where(d["mask"][..., None], 0, 255).astype(np.uint8).repeat(3, axis=2)

    skel = faint.copy()
    for i, p in enumerate(d["paths"]):
        _poly(skel, p.points, p.closed, _palette(i), 2)
    out["4_skeleton_paths"] = skel

    joined = faint.copy()
    used = {id(c.points) for c in d["layout"].curves}
    for i, s in enumerate(sorted(d["strokes"], key=lambda s: s.length, reverse=True)):
        color = _palette(i) if id(s.points) in used else (160, 160, 160)
        _poly(joined, s.points, s.closed, color, 3 if id(s.points) in used else 1)
        cv2.putText(joined, f"{i}{'*' if s.closed else ''}", _pt(s.points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    out["5_strokes"] = joined
    out["6_result"] = overlay(result)
    return out


def write_debug(result: Result, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, img in debug_images(result).items():
        cv2.imwrite(str(directory / f"{name}.png"), img)
