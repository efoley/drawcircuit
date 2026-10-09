"""End-to-end checks on synthetic pencil sketches with known ground truth."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drawcircuit import Params, TrackError, export, geometry, process, synth
from drawcircuit.cli import main

from test_bspline import de_boor

CASES = {
    "circuit_edges": dict(kind="circuit", source="edges", seed=1, scribbles=3),
    "circuit_centerline": dict(kind="circuit", source="centerline", seed=2),
    "segment_edges": dict(kind="segment", source="edges", seed=3),
    "segment_centerline": dict(kind="segment", source="centerline", seed=4),
    "photo_on_table": dict(kind="circuit", source="edges", seed=5, photo=True, retrace=True),
    "s_curve_hard_shadows": dict(kind="segment", source="edges", seed=6, shape="s", size=(1200, 1600),
                                 width=120.0, shadows=True),
    "many_gaps": dict(kind="circuit", source="edges", seed=7, gaps=8),
    "start_line": dict(kind="circuit", source="edges", seed=12, start_line=True),
    "start_line_with_gaps": dict(kind="segment", source="edges", seed=1028, gaps=5, start_line=True,
                                 darkness=0.17, width=64.22),
    "faint_shadows": dict(kind="circuit", source="edges", seed=14, darkness=0.22, shadows=True),
    "faint_s_curve": dict(kind="segment", source="edges", seed=18, darkness=0.2, shadows=True, shape="s",
                          size=(1200, 1600), width=120.0),
    "figure_eight": dict(kind="circuit", source="centerline", seed=16, shape="8"),
    "narrow_road": dict(kind="circuit", source="edges", seed=10, width=35.0),
    "scribble_near_end": dict(kind="segment", source="edges", seed=5032, gaps=9, scribbles=5,
                              darkness=0.47, width=77.13),
}


def _to_frame(result, pts):
    return cv2.perspectiveTransform(pts[None].astype(np.float64), result.homography)[0]


@pytest.mark.parametrize("name", CASES)
def test_sketch(name):
    sk = synth.make(**CASES[name])
    result = process(sk.image, Params())
    t = result.track
    assert (t.kind, t.source) == (sk.kind, sk.source)
    assert set(t.edges) == ({"outer", "inner"} if sk.kind == "circuit" else {"left", "right"})

    closed = sk.kind == "circuit"
    if sk.source == "edges":
        truth = [_to_frame(result, e) for e in sk.edges]
        curves = list(t.edges.values())
    else:
        truth = [_to_frame(result, sk.centerline)]
        curves = [t.centerline]
    for s in curves:
        assert s.periodic == closed
        d = min((geometry.nearest_distances(s.sample(1500), g, dst_closed=closed) for g in truth),
                key=np.median)
        assert np.median(d) < 1.5 and np.percentile(d, 95) < 3.5, (np.median(d), np.percentile(d, 95))
    if sk.source == "edges" and not result.paper_detected:  # synthetic pages are already 1600 px
        assert abs(t.width["median"] - sk.width) < 0.1 * sk.width


def test_orientation_conventions():
    result = process(synth.make(kind="circuit", source="edges", seed=1).image, Params(direction="ccw"))
    outer, inner = result.track.edges["outer"], result.track.edges["inner"]
    assert geometry.signed_area(outer.sample(500)) < 0  # counter-clockwise on the page (y down)
    assert geometry.signed_area(inner.sample(500)) < 0
    assert abs(geometry.signed_area(outer.sample(500))) > abs(geometry.signed_area(inner.sample(500)))
    # both edges start side by side, at the top of the outer edge
    assert np.linalg.norm(outer(0.0) - inner(0.0)) < 1.5 * result.track.width["max"]
    cw = process(synth.make(kind="circuit", source="edges", seed=1).image, Params(direction="cw"))
    assert geometry.signed_area(cw.track.edges["outer"].sample(500)) > 0


def test_segment_left_is_left_of_travel():
    result = process(synth.make(kind="segment", source="edges", seed=3).image, Params())
    left, right = result.track.edges["left"], result.track.edges["right"]
    u = np.linspace(0.3, 0.7, 20)
    normal = geometry.left_normals(right(u, 1))
    assert np.all(np.sum((left(u) - right(u)) * normal, axis=1) > 0)


def test_forced_topology_and_mode():
    img = synth.make(kind="circuit", source="edges", seed=1).image
    assert process(img, Params(mode="centerline")).track.source == "centerline"
    seg = synth.make(kind="segment", source="centerline", seed=4).image
    with pytest.raises(TrackError):
        process(seg, Params(mode="edges"))


def test_blank_page_raises():
    blank = np.full((900, 1200, 3), 235, np.uint8)
    with pytest.raises(TrackError):
        process(blank, Params())


def test_json_export_is_self_consistent():
    result = process(synth.make(kind="circuit", source="edges", seed=1).image, Params())
    doc = json.loads(json.dumps(export.result_to_dict(result, samples=64)))
    assert doc["type"] == "circuit" and doc["source"] == "edges" and doc["direction"] == "ccw"
    for edge in doc["edges"].values():
        t, c, k = edge["knots"], edge["control_points"], edge["degree"]
        assert len(t) == len(c) + k + 1
        a, b = edge["domain"]
        u = np.linspace(a, b, 64, endpoint=not edge["periodic"])
        ours = np.array([de_boor(t, c, k, x) for x in u])
        np.testing.assert_allclose(ours, edge["samples"], atol=0.01)


def test_cli_writes_outputs(tmp_path):
    img = tmp_path / "sketch.jpg"
    cv2.imwrite(str(img), synth.make(kind="segment", source="edges", seed=3).image)
    assert main([str(img), "--debug-dir", str(tmp_path / "debug")]) == 0
    doc = json.loads((tmp_path / "sketch.track.json").read_text())
    assert doc["type"] == "segment" and set(doc["edges"]) == {"left", "right"}
    assert (tmp_path / "sketch.overlay.png").exists()
    assert (tmp_path / "debug" / "5_strokes.png").exists()


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name, kind, width", [
    ("s_curve_photo.png", "segment", 129.0),     # pencil photo with hard shadows
    ("complex_circuit.png", "circuit", 17.0),    # crisp digital lines, narrow road, many hairpins
    ("flowing_circuit.png", "circuit", 19.0),
    ("street_circuit.png", "circuit", 20.0),     # right-angle corners
    ("hairpin_circuit.png", "circuit", 21.0),    # square corners and a 180-degree stub
    ("hand_drawn_circuit.png", "circuit", 34.0), # pencil photo in shadow, with a deliberate bulge
])
def test_real_examples(name, kind, width):
    result = process(EXAMPLES / name, Params())
    t = result.track
    assert (t.kind, t.source) == (kind, "edges")
    assert abs(t.width["median"] - width) < 0.1 * width
    for s in t.edges.values():
        assert s.rms_error < 1.5
