"""The reference line between a circuit's edges and the widths measured along its normal."""

from pathlib import Path

import numpy as np
import pytest

from drawcircuit import Params, geometry, process, synth
from drawcircuit.reference import MIN_FACTOR

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _rebuilt_edge_error(track, side: str) -> float:
    """Farthest distance from line +- w * normal to the named edge."""
    ref = track.reference
    p, normal = ref.line(ref.u), geometry.left_normals(ref.line(ref.u, 1))
    if side == "left":
        pts, edge = p + normal * ref.w_left[:, None], track.edges[track.left_edge]
    else:
        pts, edge = p - normal * ref.w_right[:, None], track.edges[track.right_edge]
    return float(geometry.nearest_distances(pts, edge.sample(40000), dst_spacing=0.1, dst_closed=True).max())


def _check(track):
    ref = track.reference
    assert ref is not None
    assert ref.min_factor >= MIN_FACTOR
    assert np.all(ref.w_left > 0) and np.all(ref.w_right > 0)
    np.testing.assert_allclose(np.diff(ref.s), 2.0, atol=0.01)
    assert ref.u[0] == 0.0 and np.all(np.diff(ref.u) > 0) and ref.u[-1] < 1.0
    assert _rebuilt_edge_error(track, "left") < 0.1
    assert _rebuilt_edge_error(track, "right") < 0.1


def test_drawn_edges():
    sk = synth.make(kind="circuit", source="edges", seed=1)
    result = process(sk.image, Params())
    t = result.track
    _check(t)
    assert (t.left_edge, t.right_edge) == ("inner", "outer")
    assert abs(np.median(t.reference.w_left + t.reference.w_right) - sk.width) < 0.03 * sk.width
    d = geometry.nearest_distances(t.reference.line.sample(1500), sk.centerline, dst_closed=True)
    assert np.median(d) < 1.5
    # starts beside the edges' start and runs the same way round
    start = (t.edges["outer"](0.0) + t.edges["inner"](0.0)) / 2
    assert np.linalg.norm(t.reference.line(0.0) - start) < 2.0
    assert geometry.signed_area(t.reference.line.sample(500)) < 0


def test_clockwise_puts_the_outer_edge_on_the_left():
    t = process(synth.make(kind="circuit", source="edges", seed=1).image, Params(direction="cw")).track
    assert (t.left_edge, t.right_edge) == ("outer", "inner")
    _check(t)


def test_drawn_centerline():
    t = process(synth.make(kind="circuit", source="centerline", seed=2).image, Params()).track
    _check(t)
    half = t.width["median"] / 2
    assert np.abs(np.median(t.reference.w_left) - half) < 0.5
    assert np.abs(np.median(t.reference.w_right) - half) < 0.5


def test_segments_have_no_reference():
    t = process(synth.make(kind="segment", source="edges", seed=3).image, Params()).track
    assert t.reference is None
    assert (t.left_edge, t.right_edge) == ("left", "right")


@pytest.mark.parametrize("name", ["hairpin_circuit.png", "street_circuit.png", "hand_drawn_circuit.png"])
def test_sharp_corners(name):
    """Right-angle corners tighter than the road is wide: smoothing must widen them."""
    t = process(EXAMPLES / name, Params()).track
    _check(t)
    assert t.reference.max_deviation < 0.5 * t.width["median"]
