# drawcircuit

Turn a photo or scan of a hand-drawn race track into B-spline curves for the
two track edges.

- Works with pencil or pen on plain paper. It copes with uneven lighting, hard
  shadows, gaps in the line, and a sheet photographed at an angle on a table.
- Tells a **full circuit** (closed loop) apart from a **segment** (open piece of road).
- Accepts either drawing style:
  - **both edges drawn** (two lines): fits each edge directly
  - **one line drawn** (the centerline): fits it and offsets it by a road width
- Writes cubic B-splines (knots plus control points) as JSON, plus an overlay
  PNG for checking the result.
- For circuits, also writes a smooth **reference line** with the road width to
  either side measured along its normal: the input a racing-line optimiser
  works from (x, y, w_left, w_right).

![circuit example](examples/complex_circuit.overlay.png)
![street circuit example](examples/street_circuit.overlay.png)
![segment example](examples/s_curve_photo.overlay.png)

`examples/` holds the test drawings with their JSON and overlays: two pencil
photos with shadows (`s_curve_photo`, and `hand_drawn_circuit` whose road
deliberately bulges to twice its width on one side), and six clean digital
circuits with a narrow road (`complex_circuit`, `flowing_circuit`, `street_circuit`,
`hairpin_circuit`, `s_bend_circuit` and `diagonal_circuit`, the last four with sharp corners).

## Install and run

```sh
uv sync
uv run drawcircuit sketch.jpg              # writes sketch.track.json and sketch.overlay.png
uv run drawcircuit sketch.jpg --debug-dir dbg   # also writes every intermediate stage
```

The CLI prints what it found:

```
circuit: two drawn edges -> edges outer, inner
  outer   223 control points, length  8048.7 px, fit error max 3.21 px
  inner   223 control points, length  7938.0 px, fit error max 2.85 px
  track width ~17 px (min 15, max 19)
  reference line 7990.5 px, min radius 18.3 px, 1 - w*kappa >= 0.55, within 0.5 px of the midline
```

From Python:

```python
from drawcircuit import process, Params

result = process("sketch.jpg", Params())
track = result.track
track.kind            # "circuit" or "segment"
track.source          # "edges" (two lines drawn) or "centerline" (one line drawn)
outer = track.edges["outer"]       # circuits: "outer"/"inner", segments: "left"/"right"
outer.sample(500)                  # (500, 2) points
outer(0.25), outer(0.25, nu=1)     # position and derivative at a parameter value
outer.knots, outer.control_points, outer.degree
track.left_edge, track.right_edge  # e.g. "inner", "outer": which edge is on the left in the direction of travel
ref = track.reference              # circuits only, else None
ref.line                           # the reference line, a Spline
ref.s, ref.u, ref.w_left, ref.w_right   # one row every 2 px of arc length
```

## Output format (`*.track.json`)

```jsonc
{
  "type": "circuit",            // or "segment"
  "source": "edges",            // "edges": two edges drawn; "centerline": one line drawn and offset
  "direction": "ccw",           // circuits only: travel direction as seen on the page
  "left_edge": "inner",         // the edge on the left in the direction of travel
  "right_edge": "outer",
  "frame": {                    // coordinate system of every point below
    "width": 1600, "height": 1236, "units": "px", "y_axis": "down",
    "paper_detected": true,     // the sheet was found in the photo and straightened
    "input_to_frame": [[...]]   // 3x3 homography, input image px -> frame px
  },
  "track_width_px": {"median": 70.2, "min": 67.9, "max": 73.4},
  "edges": {
    "outer": {                  // circuits: outer/inner, segments: left/right
      "degree": 3,
      "periodic": true,         // true for circuits
      "domain": [0.0, 1.0],
      "knots": [...],           // len(knots) == len(control_points) + degree + 1
      "control_points": [[x, y], ...],
      "length_px": 3391.7,
      "fit_error_px": {"rms": 0.5, "max": 1.1},
      "samples": [[x, y], ...]  // the curve sampled evenly in parameter
    },
    "inner": {...}
  },
  "centerline": null,           // the fitted centerline when source == "centerline"
  "reference": {                // circuits only, else null
    "line": {...},              // closed spline, same fields as an edge
    "smoothing_sigma_px": 7.68, // Gaussian smoothing applied to the true midline
    "max_midline_deviation_px": 2.39,
    "min_radius_px": 15.3,
    "min_1_minus_w_kappa": 0.3678, // min over the lap of 1 - w_inside * |curvature|
    "spacing_px": 1.999817,   // lap length / number of rows, about 2 px
    "columns": ["s_px", "u", "w_left", "w_right"],
    "rows": [[0.0, 0.0, 10.15, 10.282], [2.0, 0.0004195, 10.185, 10.287], ...]
  },
  "warnings": []
}
```

**The reference line.** Racing-line optimisers describe the road as a smooth
reference line plus the width to either side of it. `reference.line` is a
closed spline between the edges that starts beside their start and runs in
the direction of travel. Each row gives the arc length `s_px` along the line, its
parameter `u`, and the distances `w_left`, `w_right` (both positive) from the
point `line(u)` to the `left_edge` and `right_edge`, measured along the line's
normal. So `line(u) + w_left * n` lies on the left edge and `line(u) - w_right * n`
on the right edge, where `n` is the unit normal pointing to the left.

The line is the true midline (the points equally far from both edges),
smoothed just enough that `1 - w_inside * |curvature| >= 0.3` everywhere, where
`w_inside` is the width on the side the line turns towards. Where that factor
reaches 0, the normals of neighbouring points cross inside the road. Sharp
corners need the smoothing: it widens them and moves the line towards their
inside. The edges are left exactly as drawn.

**Evaluating the splines.** Each spline is a standard B-spline: any de Boor
evaluator gives the curve for u in `domain`, which is [0, 1]. The parameter is
roughly proportional to arc length. Knots are not evenly spaced: they are
packed into corners and spread out on straights.

- **Segments (open curves)** use clamped knots, so they start and end exactly at
  their first and last control points.
- **Circuits (closed curves)** use a periodic knot vector, where `t[i + n] = t[i] + 1`
  for `n` distinct control points. The last `degree` control points repeat the
  first ones, which makes the curve continuous in position, tangent and
  curvature where the loop joins.

`scipy.interpolate.BSpline(t, c, k)` evaluates both kinds directly; for
circuits pass `extrapolate="periodic"`.

**Conventions.**

- Circuits run counter-clockwise on the page by default (`--direction cw` to flip).
  Seen in the direction of travel, the inner edge is then on the left.
  `left_edge` and `right_edge` name the edges on each side.
- Both circuit edges start side by side, at the top-most point of the outer edge.
- Segments start at the end nearer the top-left of the page. `left` and `right`
  are relative to that direction of travel.
- Coordinates are pixels of the working frame: the straightened sheet (or the
  whole image), scaled so its long side is `--working-size` (default 1600).
  The y axis points down. If you need y up, flip it as `y' = height - y`.

## How it works

1. **Page.** If the photo shows a sheet on a darker surface, the sheet's four
   corners are found and the image is straightened. A sunlit patch between
   shadows is not mistaken for the sheet: the sheet must be convex, have four
   corners, and be clearly brighter than its surroundings.
2. **Ink.** The local paper brightness is estimated with a morphological
   closing. This removes thin dark lines but keeps the sharp edges of shadows.
   Each pixel's ink strength is how much darker it is than its local paper.
   Thresholds are set in units of the *local* noise level, because shadows are
   noisier. A ridge filter adds faint pencil lines, since it gathers evidence
   along the line the way the eye does.
3. **Skeleton.** The ink is thinned to a one-pixel skeleton and short spurs are
   pruned. The skeleton is split into paths at junctions and at sharp corners.
4. **Strokes.** Paths are joined end to end, cheapest join first. The cost
   favors short gaps and straight continuation. A long gap is bridged only if
   both ends line up along a smooth arc and the pieces on either side are real
   lines rather than small marks. This bridges pencil skips, passes straight
   through crossings (a start/finish line, a figure-8), and ignores scribbles.
   A stroke whose ends join becomes a loop.
5. **Layout.** Long strokes are candidate track lines. Two strokes that run
   alongside each other at a steady distance count as the edges: two nested
   loops make a circuit, two open lines make a segment. Otherwise the longest
   stroke is taken as a centerline.
6. **B-splines.** Each line is fitted with a penalized least-squares cubic
   B-spline. Knot density follows the curve's turning (measured over ~8 px so
   hand jitter does not count): `--cp-spacing` apart on straights plus 4 per
   radian of turning. Fitting starts sparse and adds knots until 98% of the
   line is within `--tol` pixels, so a looser tolerance gives fewer control
   points. For a centerline, the edges are offset by
   `--track-width` along the normal and then fitted.
7. **Reference line.** For circuits, the true midline is traced as the zero
   level of (distance to outer edge − distance to inner edge). It is smoothed
   with a Gaussian of growing width (from 2 px, ×1.4 per step) until the
   condition above holds, then fitted. The widths are found by casting rays
   along the line's normals onto the edges.

## Options

| option | default | meaning |
|---|---|---|
| `--mode auto\|edges\|centerline` | auto | what was drawn |
| `--circuit` / `--segment` | detect | force a closed circuit / never close a stroke into a loop |
| `--track-width PX` | 3.5% of the diagonal | road width for centerline drawings |
| `--direction ccw\|cw` | ccw | travel direction for circuits |
| `--tol PX` | 2.5 | how closely the spline follows the drawn line (bigger = smoother, fewer control points) |
| `--cp-spacing PX` | 60 | control point spacing on straights |
| `--smooth X` | 0.05 | control polygon smoothing penalty |
| `--max-gap PX` | 120 | longest break in a line that can be bridged |
| `--ink-threshold X` | auto | manual ink threshold (0..1) if detection misbehaves |
| `--no-paper` | | do not look for a sheet to straighten |
| `--working-size PX` | 1600 | long side of the output frame |
| `--samples N` | 256 | sampled points per curve in the JSON |
| `--debug-dir DIR` | | write the stage images (`1_page` … `6_result`) |

## Drawing tips

- Draw on plain, unlined paper. Firm pencil or pen works best, but faint lines
  usually work too.
- Draw each edge as one continuous line where you can. Small skips are fine.
- Labels, arrows and start/finish lines are fine. Keep labels from touching the
  track lines.
- Leave about a pencil width of space between the two edges. Where the edges
  touch, the program has to guess which line is which.
- Photograph from roughly above. Shadows are fine. A darker table around the
  sheet lets the sheet be found and straightened.

## Limitations

- Figure-8 tracks work when drawn as a centerline. Drawn with two edges, the
  edges cross at the bridge and the drawing is misread as a centerline circuit
  (with a warning). Use `--mode edges` to get an error instead of a wrong guess.
- A segment drawn as a closed outline (both edges plus the two end caps) is
  taken as one closed line, so it is misread as a centerline circuit. Leave
  the ends of a segment open.
- A centerline's road width is a single constant (`--track-width`).
- Only circuits get a reference line. A figure-8 does not get one, since its
  edges cross.
- Output is in pixels. Real-world scale is not inferred from the paper size.

## Development

```sh
uv run pytest                                  # unit + end-to-end tests on synthetic sketches
uv run python -m drawcircuit.synth out_dir     # write synthetic example sketches
```

`drawcircuit.synth` renders pencil-like sketches with known ground truth:
wobbly strokes with gaps, pencil grain, uneven light, hard shadows, scribbles,
start lines, and perspective photos of a sheet on a table. The tests check
classification and fit accuracy against that ground truth.
