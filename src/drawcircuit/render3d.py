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


def render(doc: dict, *, name: str | None = None, exaggeration: float | None = None, views=VIEWS) -> Figure:
    if doc.get("version") != 2:
        raise ValueError("this reads drawcircuit track files of format version 2 (in metres)")
    ref, prof = doc["reference"], doc.get("profile")
    if ref is None:
        raise ValueError("a 3D view needs a circuit with a reference line")
    unit = "m"
    line = BSpline(np.array(ref["line"]["knots"]), np.array(ref["line"]["control_points"]),
                   ref["line"]["degree"], extrapolate="periodic")
    s, u, wl, wr = np.array(ref["rows"]).T
    graded = bool(prof and prof["rows"])
    _, z, grade = np.array(prof["rows"]).T if graded else (s, np.zeros_like(s), np.zeros_like(s))

    p, d1 = line(u), line(u, 1)
    t = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])  # left normal (y up)
    center, left, right = p, p + n * wl[:, None], p - n * wr[:, None]
    base = min(0.0, float(z.min()))  # curtains hang down to the lowest point of the lap
    nxt = np.roll(np.arange(len(s)), -1)
    road = [np.array([[*left[i], z[i]], [*left[j], z[j]], [*right[j], z[j]], [*right[i], z[i]]])
            for i, j in enumerate(nxt)]
    curtain = [np.array([[*center[i], base], [*center[j], base], [*center[j], z[j]], [*center[i], z[i]]])
               for i, j in enumerate(nxt)]
    colors = np.column_stack([grade_colors((grade + grade[nxt]) / 2), np.full(len(s), 0.45)])
    rise = max(float(np.ptp(z)), 1e-6)
    lo = np.vstack([left, right]).min(axis=0)
    hi = np.vstack([left, right]).max(axis=0)
    if exaggeration is None:
        exaggeration = auto_exaggeration(rise, float(np.max(hi - lo))) if graded else 1.0
    lap = len(s) * ref["spacing_m"]

    # Whole-lap views, then a close-up of each place where the lap crosses itself.
    panels = [(view, None) for view in views]
    crossing_of = {}
    zhi_all = float(z.max())
    half = 15 * doc["track_width_m"]["median"]
    for c in (prof or {}).get("crossings", [])[:2]:
        crossing_of[len(panels)] = c
        at = np.array(c["point"], dtype=float)
        i = int(np.argmin(np.abs(s - c["s_under_m"])))
        d = center[nxt[i]] - center[i - 1]  # seen from the side of the lower road
        panels.append(((float(np.degrees(np.arctan2(d[1], d[0]))) - 90.0, 15.0), (at - half, at + half)))
    fig = Figure(figsize=(9 * len(panels), 7.5), dpi=150)
    for k, ((azim, elev), window) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d", computed_zorder=False)
        if window is None:
            keep, (a, b), zoom_z = np.ones(len(s), bool), (lo, hi), exaggeration
        else:
            a, b = window
            inside = np.all((center >= a) & (center <= b), axis=1)
            keep = inside & inside[nxt]
            zoom_z = 3.0
        idx = np.nonzero(keep)[0]
        # One collection, so road and curtain faces are depth-sorted together.
        faces = ([curtain[i] for i in idx] if graded else []) + [road[i] for i in idx]
        face_colors = np.vstack(([colors[idx]] if graded else []) + [np.tile(ROAD, (len(idx), 1))])
        ax.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, edgecolors="none"))
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        ax.add_collection3d(Line3DCollection(
            [np.column_stack([e[np.append(r, nxt[r[-1]])], z[np.append(r, nxt[r[-1]])] + 0.002 * rise])
             for e in (left, right) for r in runs if len(r)],
            colors=[(0.1, 0.1, 0.12)], linewidths=0.6))
        for r in runs:
            if len(r):
                ax.plot(center[r, 0], center[r, 1], np.full(len(r), base), color="0.6", lw=0.8)  # ground shadow

        above = 0.12 * rise if graded and window is None else 0.03 * rise
        if keep[0]:  # the start/finish line, a flag pole beside it and the direction of travel
            ends = np.linspace(0.0, 1.0, 9)
            for j in range(8):
                piece = left[0] + np.outer(ends[j:j + 2], right[0] - left[0])
                ax.plot(piece[:, 0], piece[:, 1], [z[0]] * 2, color="black" if j % 2 else "white", lw=4,
                        solid_capstyle="butt", zorder=11)
            pole = 0.35 * (zhi_all - base) if graded else 1.0
            ax.plot([left[0][0]] * 2, [left[0][1]] * 2, [z[0], z[0] + pole], color="black", lw=1.5, zorder=11)
            ax.text(left[0][0], left[0][1], z[0] + pole, " start / finish", fontsize=9, fontweight="bold",
                    ha="left", va="bottom", zorder=12)
            # An arrow along the road, drawn flat (a 3D quiver tilts its head into the exaggerated height).
            ahead = (center[1] - center[0]) / np.linalg.norm(center[1] - center[0])
            side = np.array([-ahead[1], ahead[0]])
            size = 0.06 * float(np.max(b - a))
            tip = center[0] + size * ahead
            for piece in ([center[0], tip], [tip - 0.35 * size * (ahead - 0.6 * side), tip],
                          [tip - 0.35 * size * (ahead + 0.6 * side), tip]):
                piece = np.array(piece)
                ax.plot(piece[:, 0], piece[:, 1], [z[0]] * 2, color="black", lw=1.8, zorder=12)
        for sec in (prof or {}).get("sections", []):
            inside = idx[(s[idx] - sec["s_start_m"]) % lap < sec["length_m"]]
            if len(inside) == 0:
                continue
            target = (sec["s_start_m"] + sec["length_m"] / 2) % lap
            i = inside[np.argmin(np.abs((s[inside] - target + lap / 2) % lap - lap / 2))]
            color = grade_colors(np.array([sec["grade"] or 0.0]))[0] * 0.75
            ax.text(center[i, 0], center[i, 1], z[i] + above, str(sec["index"]), color=color,
                    fontsize=11, fontweight="bold", ha="center", va="bottom", zorder=10)

        zlo, zhi = (base, float(z.max())) if window is None else (base, float(z[idx].max()) if len(idx) else 1.0)
        ax.set_box_aspect((b[0] - a[0], b[1] - a[1], (zhi - zlo) * zoom_z if graded else 1.0),
                          zoom=1.15 if window is None else 1.0)
        ax.set_xlim(a[0], b[0])
        ax.set_ylim(a[1], b[1])
        ax.set_zlim(zlo, zhi * 1.02 if graded else 1.0)
        ax.set_xlabel(f"x ({unit})", labelpad=8)
        ax.set_ylabel(f"y ({unit})", labelpad=8)
        if graded:
            ax.set_zlabel(f"elevation ({unit})", labelpad=4)
        else:  # a flat lap: no height to show
            ax.set_zticks([])
        ax.tick_params(labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.97, 0.97, 0.97, 1))
        ax.view_init(elev=elev, azim=azim)
        if window is not None:
            c = crossing_of[k]
            i, j = (int(np.argmin(np.abs(s - c[key]))) for key in ("s_over_m", "s_under_m"))
            x, y = c["point"]
            ax.plot([x, x], [y, y], [z[j], z[i]], color=(0.8, 0.1, 0.1), lw=2, marker="_", ms=10, zorder=12)
            ax.text(x, y, (z[i] + z[j]) / 2, f"  {c['clearance_m']:.1f} m", color=(0.8, 0.1, 0.1),
                    fontsize=11, fontweight="bold", zorder=12)
            ax.set_title(f"where the lap crosses itself: {c['clearance_m']:.1f} m between the roads "
                         f"(height ×{zoom_z:g})", fontsize=10)

    if graded:
        fig.legend(handles=[Patch(color=grade_colors(np.array([sec["grade"]]))[0],
                                  label=f"{sec['index']}  {100 * sec['grade']:+.1f}%  "
                                        f"{sec['length_m']:.0f} {unit}{'  (solved)' if sec['solved'] else ''}")
                            for sec in prof["sections"]],
                   loc="lower center", ncol=min(len(prof["sections"]), 7), fontsize=9, frameon=False,
                   title="section, grade, length", title_fontsize=9)
    lo, hi = (doc["track_width_m"][k] for k in ("min", "max"))
    road = f"{lo:.0f} m" if hi - lo < 0.5 else f"{lo:.0f}–{hi:.0f} m"
    title = f"{doc['length_m'] / 1000:.1f} km lap, {road} road"
    if name:
        title = f"{name}: {title}"
    if doc.get("scale_from") != "length":
        title += " (scale assumed)"
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
    ap.add_argument("--name", help="title for the picture (default: the file's name)")
    ap.add_argument("--exaggeration", type=float,
                    help="height exaggeration (default: the height range looks 0.3 times the track's size)")
    ap.add_argument("--view", type=_view, action="append", metavar="AZIM,ELEV",
                    help="viewpoint in degrees; repeat for several side by side (default: 105,30 and 150,30)")
    args = ap.parse_args(argv)
    name = args.track.name.removesuffix(".json").removesuffix(".track")
    try:
        fig = render(json.loads(args.track.read_text()), name=args.name or name, exaggeration=args.exaggeration,
                     views=args.view or VIEWS)
    except ValueError as e:
        print(f"drawcircuit-3d: {e}", file=sys.stderr)
        return 2
    out = args.output or args.track.with_name(name + ".3d.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
