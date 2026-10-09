"""Image loading, optional paper rectification and pencil/pen ink segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import sato


@dataclass
class Page:
    image: np.ndarray        # BGR uint8 in the working frame
    homography: np.ndarray   # 3x3, input image pixels -> working frame pixels
    paper_detected: bool


@dataclass
class Ink:
    strength: np.ndarray     # float32 in [0, 1]: how much darker than the local paper
    mask: np.ndarray         # bool, cleaned ink pixels
    valid: np.ndarray        # bool, pixels considered part of the paper


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # honours EXIF orientation
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


# --------------------------------------------------------------------------- paper

def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    c = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    return np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)


def find_paper(img: np.ndarray) -> np.ndarray | None:
    """Find a sheet of paper lying on a darker background.

    Returns the four corners (tl, tr, br, bl) in input pixels, or None when the
    photo/scan shows only paper or no clean quadrilateral is found. The checks
    are strict on purpose: a sunlit patch between shadows must not be mistaken
    for the sheet.
    """
    h, w = img.shape[:2]
    s = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    total = gray.shape[0] * gray.shape[1]
    if not 0.15 * total <= area <= 0.95 * total:
        return None
    hull = cv2.convexHull(contour)
    if area / max(cv2.contourArea(hull), 1.0) < 0.95:  # paper is convex; shadow patches are not
        return None
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            break
    else:
        return None
    if not 0.93 <= cv2.contourArea(approx) / area <= 1.07:
        return None
    inside = np.zeros_like(gray)
    cv2.fillConvexPoly(inside, approx, 255)
    if np.count_nonzero(inside == 0) < 0.03 * total:
        return None
    if gray[inside == 0].mean() > 0.7 * gray[inside > 0].mean():  # background must be clearly darker
        return None
    return _order_quad(approx.reshape(4, 2).astype(np.float64) / s)


def prepare_page(img: np.ndarray, working_size: int, find_sheet: bool) -> Page:
    """Rectify (if a sheet is found) and rescale so the long side is ``working_size``."""
    quad = find_paper(img) if find_sheet else None
    if quad is None:
        h, w = img.shape[:2]
        s = working_size / max(h, w)
        size = (round(w * s), round(h * s))
        page = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        m = np.diag([size[0] / w, size[1] / h, 1.0])
        return Page(page, m, False)

    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    s = working_size / max(w, h)
    size = (round(w * s), round(h * s))
    pre = min(1.0, 1.5 * s)  # shrink first so the warp does not alias thin lines
    src = cv2.resize(img, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA) if pre < 1 else img
    dst = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
    warp = cv2.getPerspectiveTransform(np.float32(quad * pre), dst)
    page = cv2.warpPerspective(src, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Page(page, warp @ np.diag([pre, pre, 1.0]), True)


# --------------------------------------------------------------------------- ink

def _odd(v: float) -> int:
    return max(3, int(round(v)) | 1)


def _fill_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed background regions smaller than ``max_area`` pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    small = stats[:, cv2.CC_STAT_AREA] <= max_area
    h, w = mask.shape
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    touches = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    fill = small & ~touches
    fill[0] = False
    return mask | fill[labels] if n > 1 else mask


def _drop_specks(mask: np.ndarray, min_area: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def paper_region(gray: np.ndarray, margin: float) -> np.ndarray:
    """Pixels that belong to the sheet: excludes dark table areas reaching the image
    border and a thin strip along the border itself."""
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    paper = (g > 0.45 * np.percentile(g, 90)).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    paper = _fill_holes(paper.astype(bool), np.inf)  # dark regions not reaching the border are on the paper
    m = int(round(margin))
    if m > 0:
        paper = cv2.erode(paper.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))).astype(bool)
        paper[:m], paper[-m:], paper[:, :m], paper[:, -m:] = False, False, False, False
    return paper


def _local_noise(strength: np.ndarray, center: float, block: int = 16) -> np.ndarray:
    """Per-pixel robust noise level (MAD) of the ink strength.

    Measured in blocks so it follows the lighting: shadows are noisier than sunlit
    paper because the same sensor noise is divided by a darker background.
    """
    h, w = strength.shape
    hb, wb = max(1, h // block), max(1, w // block)
    dev = np.abs(cv2.resize(strength, (wb * block, hb * block), interpolation=cv2.INTER_NEAREST) - center)
    blocks = dev.reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    local = median_filter(np.median(blocks, axis=2), size=5, mode="nearest")
    return np.maximum(1.4826 * cv2.resize(local, (w, h), interpolation=cv2.INTER_LINEAR), 0.004)


def _hysteresis(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Pixels of ``low`` that are 8-connected to some pixel of ``high``."""
    n, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(labels[high & low])] = True
    keep[0] = False
    return keep[labels]


def segment_ink(page: np.ndarray, *, margin: float, close_radius: float, background_kernel: float,
                min_blob_area: float, max_hole_area: float, threshold: float | None = None) -> Ink:
    """Find pencil/pen strokes, robust to uneven lighting and hard-edged shadows.

    The paper brightness is estimated with a morphological closing, which erases
    thin dark lines but keeps the (possibly sharp) edges of shadows, so strokes
    are measured relative to their local background. Thresholds are set in units
    of the local noise level.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    darkest = page.min(axis=2).astype(np.float32) / 255.0  # coloured pens are dark in some channel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(background_kernel),) * 2)
    background = cv2.morphologyEx(darkest, cv2.MORPH_CLOSE, k)
    background = cv2.GaussianBlur(background, (0, 0), 1.5)
    strength = np.clip(1.0 - darkest / np.maximum(background, 1e-3), 0.0, 1.0)
    strength = cv2.GaussianBlur(strength, (0, 0), 0.8)

    valid = paper_region(gray, margin)
    values = strength[valid] if valid.any() else strength.ravel()
    center = float(np.median(values))
    if threshold is not None:
        mask = _hysteresis(strength > 0.5 * threshold, strength > threshold)
    else:
        # Thresholds are lenient multiples of the local noise (stray marks become short
        # strokes that later stages ignore). Pixels must also reach a fraction of the
        # darkest ink within ~7 px: on a clean, crisp drawing the noise is ~0 and the
        # faint halo around each line would otherwise swell close lines together.
        near_peak = cv2.dilate(strength, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        low_ok = (strength > 0.025) & (strength > 0.2 * near_peak)
        high_ok = strength > 0.05
        z = (strength - center) / _local_noise(strength, center)
        mask = _hysteresis((z > 4.0) & low_ok, (z > 8.0) & high_ok)
        # A ridge filter integrates along thin lines, lifting faint pencil well above
        # paper grain that is only as dark per pixel.
        ridge = sato(z, sigmas=(1.5, 2.5), black_ridges=False)
        r = ridge[valid] if valid.any() else ridge.ravel()
        r_med = float(np.median(r))
        rz = (ridge - r_med) / (1.4826 * float(np.median(np.abs(r - r_med))) + 1e-6)
        mask |= _hysteresis((rz > 4.0) & low_ok, (rz > 9.0) & (strength > 0.03))
    mask &= valid
    r = int(round(close_radius))
    if r > 0:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    mask = _drop_specks(mask, min_blob_area)
    mask = _fill_holes(mask, max_hole_area)
    return Ink(strength, mask, valid)
