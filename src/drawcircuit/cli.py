"""Command line interface: ``drawcircuit sketch.jpg``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import export, render
from .layout import TrackError
from .pipeline import Params, process


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
    args = _parser().parse_args(argv)
    params = Params(
        working_size=args.working_size, find_paper=not args.no_paper, mode=args.mode,
        closed=args.closed, direction=args.direction, track_width=args.track_width,
        ink_threshold=args.ink_threshold, tol=args.tol, cp_spacing=args.cp_spacing,
        smooth=args.smooth, max_gap=args.max_gap,
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
    print(f"{t.kind}: {'two drawn edges' if t.source == 'edges' else 'drawn centerline'}"
          f" -> edges {', '.join(t.edges)}")
    for name, s in t.edges.items():
        print(f"  {name:6s} {len(s.unique_control_points):4d} control points, length {s.length():7.1f} px, "
              f"fit error max {s.max_error:.2f} px")
    print(f"  track width ~{t.width['median']:.0f} px (min {t.width['min']:.0f}, max {t.width['max']:.0f})")
    if t.reference is not None:
        r = t.reference
        print(f"  reference line {r.line.length():.1f} px, min radius {r.min_radius:.1f} px, "
              f"1 - w*kappa >= {r.min_factor:.2f}, within {r.max_deviation:.1f} px of the midline")
    for w in t.warnings:
        print(f"  warning: {w}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
