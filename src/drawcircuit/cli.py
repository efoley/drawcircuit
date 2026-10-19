"""Command line interface: ``drawcircuit sketch.jpg``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import export, profile, render, route
from .layout import TrackError
from .pipeline import Params, process
from .track import Track


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="drawcircuit",
        description="Turn a photo or scan of a hand-drawn race track into B-spline track edges.",
    )
    ap.add_argument("image", type=Path, help="photo or scan of the drawing")
    ap.add_argument("-o", "--output", type=Path, help="JSON output (default: <image>.track.json)")
    ap.add_argument("--overlay", type=Path, help="overlay PNG (default: <image>.overlay.png)")
    ap.add_argument("--no-overlay", action="store_true", help="do not write the overlay PNG")
    ap.add_argument("--debug-dir", type=Path, help="write intermediate stage images here")
    ap.add_argument("--mode", choices=["auto", "edges", "centerline"], default="auto",
                    help="what was drawn: both road edges, or a single centerline (default: auto)")
    topo = ap.add_mutually_exclusive_group()
    topo.add_argument("--circuit", dest="closed", action="store_const", const=True,
                      help="force a closed circuit")
    topo.add_argument("--segment", dest="closed", action="store_const", const=False,
                      help="never close a stroke into a loop")
    ap.add_argument("--direction", choices=["ccw", "cw"], default="ccw",
                    help="travel direction for circuits, as seen on the page (default: ccw)")
    ap.add_argument("--track-width", type=float, help="road width in frame px for centerline drawings")
    ap.add_argument("--length-m", type=float,
                    help="real length in meters of the lap (along the reference line) or of the road; "
                         "sets the scale")
    ap.add_argument("--track-width-m", type=float,
                    help="road width in meters for centerline drawings (needs --length-m)")
    ap.add_argument("--grades",
                    help="grade of each section between tick marks, comma-separated in travel order "
                         "starting after the first tick past the start line, e.g. "
                         "'1-2%%,5%%,descent,flat' (descent/climb/? get the grade that closes the lap)")
    ap.add_argument("--route", type=Path, metavar="JSON",
                    help="follow these waypoints (working-frame px, in travel order) along the ink instead "
                         "of joining strokes automatically: {\"closed\": true, \"points\": [[x, y], "
                         "[x, y, \"free\"], ...]}; a free leg is drawn straight, e.g. through a tunnel")
    ap.add_argument("--section-width-m", action="append", default=[], metavar="K=W",
                    help="road width W in meters for section K of a centerline circuit, instead of "
                         "--track-width-m; repeat for more sections")
    ap.add_argument("--width-taper", type=float, default=5.0,
                    help="a change of road width is spread over this many times its size (default 5)")
    ap.add_argument("--move-tick", type=int, metavar="K",
                    help="move tick K along the lap until the grades close it, instead of solving a "
                         "descent/climb grade: e.g. '--grades ...,-3%%,flat --move-tick 7' makes a 3%% "
                         "descent as long as the climb needs")
    ap.add_argument("--under", type=int, metavar="K",
                    help="where the lap crosses itself, section K passes underneath (a tunnel or "
                         "underpass): the open grades change so it is --clearance-m below the other road")
    ap.add_argument("--clearance-m", type=float, default=6.0,
                    help="height between the two roads where the lap crosses itself (default 6)")
    ap.add_argument("--min-inside-radius-m", type=float, default=6.0,
                    help="round the corners of a centerline circuit until the inside edge turns no tighter "
                         "than this, in meters (needs --track-width-m; default 6, 0 keeps them as drawn)")
    ap.add_argument("--max-corner-turn-deg", type=float, default=200.0,
                    help="report corners that turn through more than this (default 200)")
    ap.add_argument("--vertical-radius-m", type=float,
                    help="radius of the vertical curves that ease each grade change, in meters "
                         "(needs --length-m; default 1000, 0 for abrupt changes)")
    ap.add_argument("--no-paper", action="store_true",
                    help="do not look for and straighten a sheet of paper in the photo")
    ap.add_argument("--working-size", type=int, default=1600,
                    help="long side of the working frame in px; output coordinates use this frame")
    ap.add_argument("--tol", type=float, help="max spline deviation from the drawn line, px (default 2.5)")
    ap.add_argument("--cp-spacing", type=float, help="control point spacing on straights, px (default 60)")
    ap.add_argument("--smooth", type=float, default=0.05, help="control polygon smoothing (default 0.05)")
    ap.add_argument("--max-gap", type=float, help="longest pencil break to bridge, px (default 120)")
    ap.add_argument("--ink-threshold", type=float, help="manual ink threshold 0..1 for faint drawings")
    ap.add_argument("--samples", type=int, default=256, help="sampled points per curve in the JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.track_width is not None and args.track_width_m is not None:
        ap.error("give the road width in px (--track-width) or in meters (--track-width-m), not both")
    for name in ("track_width_m", "vertical_radius_m", "section_width_m"):
        if getattr(args, name) not in (None, []) and args.length_m is None:
            ap.error(f"--{name.replace('_', '-')} needs --length-m to set the scale")
    if args.section_width_m and args.track_width_m is None:
        ap.error("--section-width-m needs --track-width-m for the other sections")
    try:
        section_widths = {int(k): float(v) for k, v in (s.split("=") for s in args.section_width_m)}
    except ValueError:
        ap.error("--section-width-m takes SECTION=WIDTH, e.g. 11=20")
    params = Params(
        working_size=args.working_size, find_paper=not args.no_paper, mode=args.mode,
        closed=args.closed, direction=args.direction, track_width=args.track_width,
        ink_threshold=args.ink_threshold, tol=args.tol, cp_spacing=args.cp_spacing,
        smooth=args.smooth, max_gap=args.max_gap, length_m=args.length_m,
        track_width_m=args.track_width_m,
        grades=args.grades.split(",") if args.grades else None, vertical_radius_m=args.vertical_radius_m,
        move_tick=args.move_tick, section_widths_m=section_widths or None, width_taper=args.width_taper,
        route=route.load(args.route) if args.route else None, under=args.under,
        clearance_m=args.clearance_m, min_inside_radius_m=args.min_inside_radius_m,
        max_corner_turn_deg=args.max_corner_turn_deg,
    )
    try:
        result = process(args.image, params)
    except (TrackError, ValueError) as e:
        print(f"drawcircuit: {e}", file=sys.stderr)
        return 2

    stem = args.image.with_suffix("")
    out = args.output or stem.with_name(stem.name + ".track.json")
    out.write_text(json.dumps(export.result_to_dict(result, args.samples), indent=1))
    if not args.no_overlay:
        overlay_path = args.overlay or stem.with_name(stem.name + ".overlay.png")
        cv2.imwrite(str(overlay_path), render.overlay(result))
    if args.debug_dir:
        render.write_debug(result, args.debug_dir)

    t = result.track
    world = export.World(t)
    m = world.m
    print(f"{t.kind}: {'two drawn edges' if t.source == 'edges' else 'drawn centerline'}"
          f" -> edges {', '.join(t.edges)}")
    for name, s in t.edges.items():
        print(f"  {name:6s} {len(s.unique_control_points):4d} control points, length {s.length() * m:7.1f} m, "
              f"fit error max {s.max_error * m:.2f} m")
    lo, hi = t.width["min"] * m, t.width["max"] * m
    road = f"{lo:.1f} m" if hi - lo < 0.05 else f"{lo:.1f} to {hi:.1f} m"
    given = "from the lap length" if t.scale_from == "length" else "assumed from the road width"
    print(f"  {t.length() * m:.0f} m long, road {road} wide (scale {given})")
    if t.reference is not None:
        r = t.reference
        print(f"  reference line {r.line.length() * m:.1f} m, min radius {r.min_radius * m:.1f} m, "
              f"1 - w*kappa >= {r.min_factor:.2f}, within {r.max_deviation * m:.1f} m of the midline")
    if t.rounded:
        print(f"  rounded {len(t.rounded)} corner{'s' if len(t.rounded) != 1 else ''} so the inside edge turns no "
              f"tighter than {args.min_inside_radius_m:g} m (the line moved up to "
              f"{max(c.shift for c in t.rounded) * m:.1f} m)")
    if t.profile is not None:
        _print_profile(t)
    for w in t.warnings:
        print(f"  warning: {world.text(w)}")
    print(f"wrote {out}")
    return 0


def _print_profile(t: Track) -> None:
    p = t.profile
    scale, unit = t.m_per_px, "m"
    n = len(p.ticks)
    print(f"  {n} tick mark{'s' if n != 1 else ''} -> {len(p.sections)} section"
          f"{'s' if len(p.sections) != 1 else ''} in travel order"
          f"{' (section 1 starts at the first tick after the start line)' if n else ''}:")
    for k in np.nonzero(p.moved)[0]:
        print(f"  tick {k + 1} moved {p.moved[k] * scale:+.0f} {unit} along the lap from where it is drawn, "
              "so the grades close the lap")
    varies = max(sec.width for sec in p.sections) - min(sec.width for sec in p.sections) > 0.02 * t.width["median"]
    for sec in p.sections:
        line = f"    {sec.index:2d} {sec.length * scale:7.0f} {unit}"
        if varies:
            line += f"  {sec.width * scale:5.1f} {unit} wide"
        if p.graded:
            line += f"  {100 * sec.grade:+6.2f}%  {sec.label}{'  (solved to close the lap)' if sec.solved else ''}"
        print(line)
    if p.graded and p.vertical_radius > 0 and p.eases.max() > 0:
        eased = p.eases[p.eases > 0] * scale
        print(f"  grade changes eased on a {p.vertical_radius * scale:.0f} {unit} vertical radius, "
              f"over {eased.min():.0f} to {eased.max():.0f} {unit} around each tick")
    for c in p.crossings if p.graded else []:
        total = t.reference.length
        over, below = (profile.section_at(p.sections, s, total).index for s in (c.s_over, c.s_under))
        x, y = export.World(t).point(c.point)
        print(f"  crossing at ({x:.0f}, {y:.0f}) m: section {over} passes "
              f"{c.clearance * scale:.1f} {unit} over section {below}")
    if p.graded:
        z = np.round(p.z * scale, 1) + 0.0  # no "-0.0"
        print(f"  elevation {z.min():+.1f} to {z.max():+.1f} {unit} relative to the start line, "
              f"total climb {np.sum(np.clip(np.diff(np.append(z, z[0])), 0, None)):.1f} {unit}")
    else:
        print(f"  pass --grades with {len(p.sections)} values in this order for an elevation profile")


if __name__ == "__main__":
    sys.exit(main())
