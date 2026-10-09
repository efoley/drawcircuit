"""Synthetic pencil sketches of tracks with known ground truth, for tests and demos.

    python -m drawcircuit.synth OUT_DIR    # writes a few example images
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

from . import geometry


@dataclass
class Sketch:
    image: np.ndarray            # BGR uint8
    kind: str                    # "circuit" | "segment"
    source: str                  # "edges" | "centerline"
    edges: list[np.ndarray]      # ground-truth edge polylines, image px
    centerline: np.ndarray       # ground-truth centerline, image px
    width: float


# --------------------------------------------------------------------------- shapes

def _min_radius(pts: np.ndarray, closed: bool) -> float:
    mode = "wrap" if closed else "nearest"
    p = gaussian_filter1d(geometry.resample(pts, spacing=2.0, closed=closed), 3, axis=0, mode=mode)
    if closed:
        d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2
        d2 = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
    else:
        d1, d2 = np.gradient(p, axis=0)[5:-5], np.gradient(np.gradient(p, axis=0), axis=0)[5:-5]
    k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    return float(1.0 / k.max())


def closed_centerline(rng: np.random.Generator, w: int, h: int, width: float) -> np.ndarray:
    """A random star-shaped loop that stays clear of the page border."""
    for _ in range(200):
        th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        r = np.ones_like(th)
        for k in range(2, 6):
            r += rng.uniform(0, 0.22 / k ** 0.7) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
        pts = np.column_stack([np.cos(th) * r, np.sin(th) * r])
        pts -= pts.min(axis=0)
        pts *= min((w - 2 * 0.12 * w) / pts[:, 0].max(), (h - 2 * 0.12 * h) / pts[:, 1].max())
        pts += [0.12 * w + (w - 2 * 0.12 * w - pts[:, 0].max()) / 2, 0.12 * h + (h - 2 * 0.12 * h - pts[:, 1].max()) / 2]
        if _min_radius(pts, True) > 0.9 * width:
            return pts
    raise RuntimeError("could not generate a smooth enough loop")


def open_centerline(rng: np.random.Generator, w: int, h: int, width: float) -> np.ndarray:
    for _ in range(200):
        xs = np.linspace(0.12 * w, 0.88 * w, 6)
        ys = h * (0.5 + rng.uniform(-0.28, 0.28, 6))
        pts = make_interp_spline(np.linspace(0, 1, 6), np.column_stack([xs, ys]), k=3)(np.linspace(0, 1, 2000))
        if _min_radius(pts, False) > 0.9 * width and pts[:, 1].min() > 0.1 * h and pts[:, 1].max() < 0.9 * h:
            return pts
    raise RuntimeError("could not generate a smooth enough segment")


def s_curve(w: int, h: int) -> np.ndarray:
    """Roughly the S-shaped segment from the example photo (portrait page)."""
    ctrl = np.array([[0.66, -0.02], [0.50, 0.17], [0.48, 0.36], [0.68, 0.55],
                     [0.82, 0.70], [0.74, 0.83], [0.48, 0.84], [0.20, 0.70]]) * [w, h]
    return make_interp_spline(np.linspace(0, 1, len(ctrl)), ctrl, k=3)(np.linspace(0, 1, 2000))


def figure_eight(w: int, h: int) -> np.ndarray:
    """A closed centerline that crosses itself once."""
    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    return np.column_stack([w / 2 + 0.36 * w * np.sin(t), h / 2 + 0.3 * h * np.sin(2 * t)])


def offset(center: np.ndarray, closed: bool, d: float) -> np.ndarray:
    p = geometry.resample(center, spacing=1.0, closed=closed)
    t = gaussian_filter1d(p, 2, axis=0, order=1, mode="wrap" if closed else "nearest")
    return p + geometry.left_normals(t) * d


# --------------------------------------------------------------------------- drawing

def pencil(shape, pts, closed, rng, *, thickness=2, darkness=0.6, wobble=1.0, gaps=2,
           retrace=False) -> np.ndarray:
    """Render one hand-drawn pencil line as an ink layer in [0, 1]."""
    h, w = shape
    mode = "wrap" if closed else "nearest"
    pts = geometry.resample(pts, spacing=1.0, closed=closed)
    n = len(pts)
    normals = geometry.left_normals(gaussian_filter1d(pts, 2, axis=0, order=1, mode=mode))
    wob = gaussian_filter1d(rng.normal(size=n), 25, mode=mode)
    pts = pts + normals * (wobble * wob / (wob.std() + 1e-9))[:, None]
    pressure = gaussian_filter1d(rng.normal(size=n), 40, mode=mode)
    pressure = np.clip(0.8 + 0.2 * pressure / (pressure.std() + 1e-9), 0.45, 1.0)
    keep = np.ones(n, bool)
    for _ in range(gaps):
        start, length = rng.integers(n), rng.integers(5, 14)
        keep[np.arange(start, start + length) % n] = False

    layer = np.zeros((h, w), np.uint8)
    passes = [(pts, 1.0)]
    if retrace:
        passes.append((pts + normals * rng.uniform(1.0, 2.0), 0.6))
    for p, strength in passes:
        last = n if closed else n - 1
        for i in range(0, last, 4):
            j = [(i + q) % n for q in range(5)] if closed else list(range(i, min(i + 5, n)))
            if len(j) < 2 or not keep[j].all():
                continue
            color = int(255 * pressure[i] * strength)
            cv2.polylines(layer, [np.round(p[j] * 16).astype(np.int32)], False, color,
                          thickness, cv2.LINE_AA, 4)
    grain = 0.55 + 0.45 * rng.random((h, w), dtype=np.float32)
    ink = cv2.GaussianBlur(layer.astype(np.float32) / 255.0 * grain, (0, 0), 0.6)
    return np.clip(ink * darkness, 0, 1)


def _smooth_field(rng, shape, scale) -> np.ndarray:
    h, w = shape
    small = rng.normal(size=(max(2, h // scale), max(2, w // scale))).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (field - field.mean()) / (field.std() + 1e-9)


def compose(ink: np.ndarray, rng, *, shadows: bool = False) -> np.ndarray:
    """Paper with ink, uneven light, optional hard-edged blue-ish shadows, noise."""
    h, w = ink.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    illum = 1.0 - 0.12 * ((gx * xx / w + gy * yy / h) + 1) / 2
    gray = rng.uniform(0.86, 0.94) * illum * (1.0 - ink)
    img = gray[..., None] * np.array([0.93, 0.97, 1.0], np.float32)  # warm paper (BGR)
    if shadows:
        s = (_smooth_field(rng, (h, w), 260) > 0.2).astype(np.float32)
        s = cv2.GaussianBlur(s, (0, 0), rng.uniform(6, 25))
        shade = np.array([0.74, 0.68, 0.62], np.float32)  # blue-ish shadow (BGR)
        img *= 1.0 - s[..., None] * (1.0 - shade)
    img += rng.normal(0, 0.012, img.shape).astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def on_table(page: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Photograph the page lying on a dark table, with perspective. Returns (photo, page->photo H)."""
    h, w = page.shape[:2]
    W, H = int(w * 1.35), int(h * 1.35)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    base = np.float32([[0.13 * W, 0.12 * H], [0.87 * W, 0.12 * H], [0.87 * W, 0.88 * H], [0.13 * W, 0.88 * H]])
    dst = (base + rng.uniform(-0.05, 0.05, (4, 2)) * [W, H]).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, m, (W, H), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.ones((h, w), np.float32), m, (W, H))[..., None]
    wood = 0.32 + 0.05 * _smooth_field(rng, (H, W), 40)[..., None]
    table = np.clip(wood * np.array([0.55, 0.75, 1.0]) * 255, 0, 255)
    photo = cover * warped + (1 - cover) * table
    return np.clip(photo, 0, 255).astype(np.uint8), m


def _scribbles(shape, rng, count: int) -> np.ndarray:
    """Short annotation marks (letters, arrows) away from the middle of the page."""
    h, w = shape
    ink = np.zeros(shape, np.float32)
    for _ in range(count):
        c = np.array([rng.uniform(0.05, 0.2) * w, rng.uniform(0.05, 0.95) * h])
        pts = c + np.cumsum(rng.normal(0, 12, (6, 2)), axis=0)
        ink = np.maximum(ink, pencil(shape, pts, False, rng, gaps=0))
    return ink


def make(kind: str = "circuit", source: str = "edges", *, seed: int = 0, size=(1600, 1236),
         width: float = 70.0, photo: bool = False, shadows: bool = False, gaps: int = 2,
         retrace: bool = False, scribbles: int = 0, shape: str | None = None,
         start_line: bool = False, darkness: float = 0.6) -> Sketch:
    rng = np.random.default_rng(seed)
    w, h = size
    closed = kind == "circuit"
    if shape == "s":
        center = s_curve(w, h)
    elif shape == "8":
        center = figure_eight(w, h)
    else:
        center = closed_centerline(rng, w, h, width) if closed else open_centerline(rng, w, h, width)
    left, right = offset(center, closed, width / 2), offset(center, closed, -width / 2)
    lines = [left, right] if source == "edges" else [geometry.resample(center, spacing=1.0, closed=closed)]

    ink = np.zeros((h, w), np.float32)
    for line in lines:
        ink = np.maximum(ink, pencil((h, w), line, closed, rng, gaps=gaps, retrace=retrace,
                                     darkness=darkness))
    if start_line:  # a start/finish line drawn across the road, overshooting both edges
        i = len(left) // 3
        a, b = left[i], right[np.argmin(np.linalg.norm(right - left[i], axis=1))]
        ext = (b - a) * 0.15
        ink = np.maximum(ink, pencil((h, w), np.array([a - ext, b + ext]), False, rng, gaps=0,
                                     darkness=darkness))
    if scribbles:
        ink = np.maximum(ink, _scribbles((h, w), rng, scribbles))
    img = compose(ink, rng, shadows=shadows)
    edges = [left, right]
    if photo:
        img, m = on_table(img, rng)
        warp = lambda p: cv2.perspectiveTransform(p[None].astype(np.float64), m)[0]
        edges, center = [warp(e) for e in edges], warp(center)
    return Sketch(img, kind, source, edges, center, width)


EXAMPLES = {
    "circuit_edges": dict(kind="circuit", source="edges", seed=1, scribbles=3),
    "circuit_centerline": dict(kind="circuit", source="centerline", seed=2),
    "segment_edges": dict(kind="segment", source="edges", seed=3),
    "segment_centerline": dict(kind="segment", source="centerline", seed=4),
    "circuit_photo": dict(kind="circuit", source="edges", seed=5, photo=True, retrace=True),
    "s_curve_shadows": dict(kind="segment", source="edges", seed=6, shape="s", size=(1200, 1600),
                            width=120.0, shadows=True),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "examples")
    out.mkdir(parents=True, exist_ok=True)
    for name, kw in EXAMPLES.items():
        cv2.imwrite(str(out / f"{name}.jpg"), make(**kw).image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(out / f"{name}.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
