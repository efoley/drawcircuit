"""3D picture of a circuit from its track JSON: ``drawcircuit-3d track.json``.

The road is drawn at its elevation, with a curtain below it down to the ground
coloured by grade, so climbs and descents read at a glance. Heights are
exaggerated, since a road climbs a few tens of meters over kilometers: by
default just enough for the height range to look about 0.3 times the track's
size. Only the JSON is used, the same way any other consumer would read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import BSpline

VIEWS = ((105.0, 30.0), (150.0, 30.0))  # (azimuth, elevation) in degrees
ROAD = (0.18, 0.18, 0.2, 1.0)


def grade_colors(g: np.ndarray) -> np.ndarray:
    """RGB in 0..1: grey for flat, orange to dark red for climbs, light to dark blue for
    descents (the overlay's colours)."""
    a = np.clip(np.abs(g) / 0.10, 0, 1)[:, None]
    up = np.array([250, 170, 40]) * (1 - a) + np.array([150, 20, 30]) * a
    down = np.array([90, 190, 235]) * (1 - a) + np.array([10, 40, 140]) * a
    c = np.where((g > 0)[:, None], up, down)
    c[np.abs(g) < 0.0025] = [150, 150, 150]
    return c / 255


def auto_exaggeration(rise: float, size: float, ratio: float = 0.3) -> float:
    """A round height exaggeration that makes ``rise`` look ``ratio`` times ``size``."""
    raw = ratio * size / max(rise, 1e-9)
    return float(max(1, round(raw))) if raw < 10 else float(5 * round(raw / 5))


def render(doc: dict, *, exaggeration: float | None = None, views=VIEWS) -> Figure:
    ref, prof, scale = doc["reference"], doc.get("profile"), doc.get("scale")
    if ref is None:
        raise ValueError("a 3D view needs a circuit with a reference line")
    m, unit = (scale["m_per_px"], "m") if scale else (1.0, "px")
    line = BSpline(np.array(ref["line"]["knots"]), np.array(ref["line"]["control_points"]),
                   ref["line"]["degree"], extrapolate="periodic")
    s, u, wl, wr = np.array(ref["rows"]).T
    graded = bool(prof and prof["rows"])
    _, z, grade = np.array(prof["rows"]).T if graded else (s, np.zeros_like(s), np.zeros_like(s))

    p, d1 = line(u), line(u, 1)
    t = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    n = np.column_stack([t[:, 1], -t[:, 0]])  # left normal on the page (y down)
    height = doc["frame"]["height"]
    world = lambda q: np.column_stack([q[:, 0] * m, (height - q[:, 1]) * m])  # y up
    center, left, right = world(p), world(p + n * wl[:, None]), world(p - n * wr[:, None])
    z = z * m
    nxt = np.roll(np.arange(len(s)), -1)
    road = [np.array([[*left[i], z[i]], [*left[j], z[j]], [*right[j], z[j]], [*right[i], z[i]]])
            for i, j in enumerate(nxt)]
    curtain = [np.array([[*center[i], 0], [*center[j], 0], [*center[j], z[j]], [*center[i], z[i]]])
               for i, j in enumerate(nxt)]
    colors = np.column_stack([grade_colors((grade + grade[nxt]) / 2), np.full(len(s), 0.45)])
    rise = max(float(np.ptp(z)), 1e-6)
    lo = np.vstack([left, right]).min(axis=0)
    hi = np.vstack([left, right]).max(axis=0)
    if exaggeration is None:
        exaggeration = auto_exaggeration(rise, float(np.max(hi - lo))) if graded else 1.0

    fig = Figure(figsize=(9 * len(views), 7.5), dpi=150)
    for k, (azim, elev) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d", computed_zorder=False)
        # One collection, so road and curtain faces are depth-sorted together.
        faces = (curtain if graded else []) + road
        face_colors = np.vstack(([colors] if graded else []) + [np.tile(ROAD, (len(road), 1))])
        ax.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, edgecolors="none"))
        ax.add_collection3d(Line3DCollection(
            [np.column_stack([np.vstack([e, e[:1]]), np.append(z, z[0]) + 0.002 * rise]) for e in (left, right)],
            colors=[(0.1, 0.1, 0.12)], linewidths=0.6))
        ax.plot(center[:, 0], center[:, 1], np.zeros_like(z), color="0.6", lw=0.8)  # ground shadow

        start = np.array([[*left[0], z[0]], [*right[0], z[0]]])
        ax.plot(*start.T, color="white", lw=3)
        ax.plot(*start.T, color="black", lw=1.2, ls=(0, (1, 1)))
        above = 0.12 * rise if graded else 0.0
        ax.text(*center[0], z[0] + above, "start", fontsize=8, ha="center", va="bottom", zorder=10)
        if prof:
            lap = len(s) * ref["spacing_px"]
            for sec in prof["sections"]:
                i = int(np.argmin(np.abs(s - (sec["s_start_px"] + sec["length_px"] / 2) % lap)))
                color = grade_colors(np.array([sec["grade"] or 0.0]))[0] * 0.75
                ax.text(center[i, 0], center[i, 1], z[i] + above, str(sec["index"]), color=color,
                        fontsize=11, fontweight="bold", ha="center", va="bottom", zorder=10)

        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], rise * exaggeration if graded else 1.0), zoom=1.15)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(0, z.max() * 1.02 if graded else 1.0)
        ax.set_xlabel(f"x ({unit})", labelpad=8)
        ax.set_ylabel(f"y ({unit})", labelpad=8)
        ax.set_zlabel(f"elevation ({unit})", labelpad=4)
        ax.tick_params(labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.97, 0.97, 0.97, 1))
        ax.view_init(elev=elev, azim=azim)

    if graded:
        fig.legend(handles=[Patch(color=grade_colors(np.array([sec["grade"]]))[0],
                                  label=f"{sec['index']}  {100 * sec['grade']:+.1f}%  "
                                        f"{sec['length_px'] * m:.0f} {unit}{'  (solved)' if sec['solved'] else ''}")
                            for sec in prof["sections"]],
                   loc="lower center", ncol=len(prof["sections"]), fontsize=9, frameon=False,
                   title="section, grade, length", title_fontsize=9)
    title = f"{len(s) * ref['spacing_px'] * m:.0f} {unit} lap"
    if scale:
        lo, hi = (doc["track_width_px"][k] * m for k in ("min", "max"))
        road = f"{lo:.0f} m" if hi - lo < 0.5 else f"{lo:.0f}–{hi:.0f} m"
        title = f"{scale['length_m'] / 1000:.1f} km lap, {road} road"
    if graded:
        title += f", {np.ptp(z):.0f} {unit} from lowest to highest point (height exaggerated ×{exaggeration:g})"
    fig.suptitle(title, fontsize=12)
    return fig


def _view(text: str) -> tuple[float, float]:
    azim, elev = (float(v) for v in text.split(","))
    return azim, elev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drawcircuit-3d", description=__doc__.splitlines()[0])
    ap.add_argument("track", type=Path, help="track JSON written by drawcircuit")
    ap.add_argument("-o", "--output", type=Path, help="PNG to write (default: <track>.3d.png)")
    ap.add_argument("--exaggeration", type=float,
                    help="height exaggeration (default: the height range looks 0.3 times the track's size)")
    ap.add_argument("--view", type=_view, action="append", metavar="AZIM,ELEV",
                    help="viewpoint in degrees; repeat for several side by side (default: 105,30 and 150,30)")
    args = ap.parse_args(argv)
    try:
        fig = render(json.loads(args.track.read_text()), exaggeration=args.exaggeration,
                     views=args.view or VIEWS)
    except ValueError as e:
        print(f"drawcircuit-3d: {e}", file=sys.stderr)
        return 2
    name = args.track.name.removesuffix(".json").removesuffix(".track")
    out = args.output or args.track.with_name(name + ".3d.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
