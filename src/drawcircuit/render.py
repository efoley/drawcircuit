"""Overlay and debug images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import geometry
from .pipeline import Result
from .track import Track

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
    if t.profile is not None:
        _draw_sections(img, t)

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
    if t.profile is not None and t.profile.graded:
        img = np.vstack([img, _profile_panel(t, img.shape[1])])
    return img


def grade_color(g: float | None) -> tuple[int, int, int]:
    """BGR: grey for flat, orange to dark red for climbs, light to dark blue for descents."""
    if g is None:
        return (120, 120, 120)
    a = min(abs(g) / 0.10, 1.0)
    if abs(g) < 0.0025:
        return (110, 110, 110)
    if g > 0:
        return tuple(int(v) for v in np.array([40, 170, 250]) * (1 - a) + np.array([30, 20, 150]) * a)
    return tuple(int(v) for v in np.array([235, 190, 90]) * (1 - a) + np.array([140, 40, 10]) * a)


def _text(img, text, org, scale, color, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _section_label(sec, graded: bool) -> str:
    if not graded:
        return str(sec.index)
    return f"{sec.index}: {100 * sec.grade:+.1f}%{'*' if sec.solved else ''}"


def _draw_sections(img: np.ndarray, t: Track) -> None:
    """Colour the reference line by grade, mark the ticks and number the sections."""
    ref, prof = t.reference, t.profile
    outward = 1.0 if t.left_edge == "outer" else -1.0
    half = t.width["median"] / 2
    for k, sec in enumerate(prof.sections):
        u = ref.u_at(np.linspace(sec.s_start, sec.s_start + sec.length, max(2, int(sec.length / 2))))
        color = grade_color(sec.grade) if prof.graded else ((60, 60, 60) if k % 2 else (40, 160, 240))
        _poly(img, ref.line(u), False, color, 4)
        mid = ref.u_at(sec.s_start + sec.length / 2)
        normal = geometry.left_normals(ref.line([mid], 1))[0] * outward
        text = _section_label(sec, prof.graded)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        c = ref.line(mid) + normal * (half + 14 + 0.5 * abs(normal[0]) * tw + 0.5 * abs(normal[1]) * th)
        _text(img, text, _pt(c - [tw / 2, -th / 2]), 0.7, color if prof.graded else (0, 0, 0))
    for s, p in zip(prof.ticks, prof.points):
        normal = geometry.left_normals(ref.line([ref.u_at(s)], 1))[0]
        bar = max(14.0, 1.5 * half + 6)
        cv2.line(img, _pt(p - normal * bar), _pt(p + normal * bar), (0, 0, 0), 3, cv2.LINE_AA)


def _nice_step(span: float, count: int) -> float:
    raw = span / max(count, 1)
    mag = 10 ** np.floor(np.log10(raw))
    return float(mag * min((m for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10))


def _profile_panel(t: Track, width: int, height: int = 300) -> np.ndarray:
    """Elevation against distance along the reference line, sections coloured by grade."""
    ref, prof = t.reference, t.profile
    scale, unit = (t.m_per_px, "m") if t.m_per_px else (1.0, "px")
    s = np.append(ref.s, ref.length) * scale
    z = np.append(prof.z, prof.z[-1] + prof.grade[-1] * (ref.length - ref.s[-1])) * scale
    panel = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 90, 30, 60, 50
    zlo, zhi = float(z.min()), float(z.max())
    if zhi - zlo < 1e-9:
        zlo, zhi = zlo - 1, zhi + 1
    pad = 0.08 * (zhi - zlo)
    zlo, zhi = zlo - pad, zhi + pad
    X = lambda v: left + (width - left - right) * v / s[-1]
    Y = lambda v: top + (height - top - bottom) * (zhi - v) / (zhi - zlo)

    step = _nice_step(zhi - zlo, 4)
    for v in np.arange(np.ceil(zlo / step) * step, zhi, step):
        cv2.line(panel, (left, int(Y(v))), (width - right, int(Y(v))), (230, 230, 230), 1)
        _text(panel, f"{v + 0.0:g}", (12, int(Y(v)) + 6), 0.5, (90, 90, 90), 1)
    for v in np.arange(0, s[-1] + 1e-9, _nice_step(s[-1], 8)):
        cv2.line(panel, (int(X(v)), height - bottom), (int(X(v)), height - bottom + 6), (90, 90, 90), 1)
        _text(panel, f"{v:g}", (int(X(v)) - 14, height - bottom + 24), 0.5, (90, 90, 90), 1)
    _text(panel, f"elevation ({unit}) against distance along the reference line ({unit}); "
          "* = solved to close the lap", (left, 24), 0.55, (60, 60, 60), 1)

    breaks = np.concatenate([[0.0], prof.ticks, [ref.length]]) * scale
    owner = [prof.sections[-1]] + prof.sections[:len(prof.ticks)]
    for (a, b), sec in zip(zip(breaks[:-1], breaks[1:]), owner):
        m = (s >= a) & (s <= b)
        if m.sum() < 2:
            continue
        poly = np.column_stack([X(s[m]), Y(z[m])])
        base = np.array([[poly[-1, 0], height - bottom], [poly[0, 0], height - bottom]])
        fill = tuple(int(0.35 * c + 0.65 * 255) for c in grade_color(sec.grade))
        cv2.fillPoly(panel, [np.round(np.vstack([poly, base]) * 16).astype(np.int32)], fill, cv2.LINE_AA, 4)
        _poly(panel, poly, False, grade_color(sec.grade), 2)
    for v in prof.ticks * scale:
        cv2.line(panel, (int(X(v)), top), (int(X(v)), height - bottom), (0, 0, 0), 1, cv2.LINE_AA)
    for sec in prof.sections:
        mid = ((sec.s_start + sec.length / 2) % ref.length) * scale
        _text(panel, str(sec.index), (int(X(mid)) - 6, top - 6), 0.55, (0, 0, 0), 1)
    cv2.line(panel, (left, height - bottom), (width - right, height - bottom), (90, 90, 90), 1)
    return panel


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
    for m in d.get("ticks", []):
        cv2.circle(joined, _pt(m.point), 9, (0, 0, 0), 2, cv2.LINE_AA)
    out["5_strokes"] = joined
    out["6_result"] = overlay(result)
    return out


def write_debug(result: Result, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, img in debug_images(result).items():
        cv2.imwrite(str(directory / f"{name}.png"), img)
