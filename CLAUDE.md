# drawcircuit

Turns a photo of a hand-drawn race track into B-spline track edges, a
reference line with road widths and, when grades are marked, an elevation
profile, as JSON for the racer game. README.md covers usage, the JSON format
and how the pipeline works.

## This program is driven by an LLM

Assume an LLM (you) runs drawcircuit for the user, with the photo in front of it:

- **The code measures**: anything that needs pixel accuracy, such as tracing
  the lines, fitting splines, the reference line, where tick marks cross the
  line, the scale and the elevation.
- **You read and decide**: handwritten grades and labels, the direction of
  travel, what a mark means, and what the user wants changed ("make the
  descent 3%", "this section is 20 m wide"). Pass these as options
  (`--grades`, `--direction`, `--move-tick`, `--length-m`, `--track-width-m`,
  `--section-width-m`, `--under`, ...).
- Do not build OCR, label parsing or other heuristics for things you can read
  off the photo. Prefer explicit, general options you can fill in, and output
  you can check.

## Making a circuit from a photo

1. Copy the photo into the working area (`examples/` if it should be kept,
   downscaled to about 2000 px on the long side) and run
   `uv run drawcircuit photo.jpg --length-m L --track-width-m W`. Ask for the
   lap length and road width, or reuse what the user gave last time.
2. Look at the overlay (`photo.overlay.png`): the line should be traced as
   drawn, and every tick should have a black bar. A bar where nothing was
   meant as a tick (an erased mark, a stray line) goes with `--drop-tick X,Y`.
   Sections are numbered in travel order from the start line. If the tracing goes wrong (a tunnel, a
   crossing, ticks at corners), give the route instead: read waypoints off
   `--debug-dir`'s `1_page.png` (a grid helps) into a `--route` file, with
   `"free"` legs through a tunnel, and check the traced line again.
3. Read the grades and the direction off the photo. Positive grades climb in
   the direction of travel, so the order the grades are written in gives
   `--direction`, unless an arrow shows it. A start line marked on the drawing
   goes in as `--start X,Y` (read off `1_page.png` or the overlay). Rerun with
   `--grades` in section order.
4. Check the printed sections and the overlay against the photo, and render
   `uv run drawcircuit-3d photo.track.json`. Where the lap crosses itself,
   pass the section that goes underneath with `--under`.
5. Tell the user what you read and assumed: the direction, labels that were
   hard to read, how you took a range such as `1-2%`, and any solved or
   moved sections. At a real scale corners are rounded to a 6 m inside edge;
   pass on the warnings about hairpins that could not be rounded and corners
   that turn more than 200° (spirals), which only the user can redraw.

## Conventions

- racer (the line optimiser) and racer-game read the JSON (`reference`,
  `left_edge` / `right_edge`, `profile`). Agree format changes with them first;
  keep them additive within a version, and bump the version when one breaks.
- The JSON is in world metres (format version 2): x right and y up on the
  drawing, z up, origin at the start/finish point, and nothing about the image.
  Inside the code everything is in working-frame px with y down, and
  `export.World` converts; keep pixels out of the file, warnings included
  (write lengths as "N px" and points as "(x, y) px" so the export converts them).
- `uv run pytest` runs the tests. The `*_debug/` folders are not committed.
