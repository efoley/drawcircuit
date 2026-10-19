"""Tick marks across a centerline, the sections between them, grades and real-world scale."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drawcircuit import Params, TrackError, export, geometry, process, profile, skeleton, synth
from drawcircuit.profile import parse_grade
from drawcircuit.reference import MIN_FACTOR

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CLIMB = dict(direction="cw", length_m=4000.0, track_width_m=10.0,
             grades=["1-2%", "5%", "7%", "9%", "3%", "descent", "flat"])


@pytest.mark.parametrize("text, grade, sign", [
    ("5%", 0.05, 0), ("5", 0.05, 0), (" -3 % ", -0.03, 0), ("+2.5%", 0.025, 0), ("1-2%", 0.015, 0),
    ("1 to 3", 0.02, 0), ("-1-2%", -0.015, 0), ("flat", 0.0, 0), ("Descent", None, -1),
    ("climb", None, 1), ("?", None, 0),
])
def test_parse_grade(text, grade, sign):
    g, s = parse_grade(text)
    assert s == sign
    assert g == pytest.approx(grade) if grade is not None else g is None


def test_parse_grade_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_grade("steep")


def test_stub_at_a_corner_is_not_followed():
    """A tick drawn on one side of the line, right where the line turns sharply."""
    mask = np.zeros((500, 500), np.uint8)
    cv2.polylines(mask, [np.int32([[100, 300], [300, 300], [340, 160]])], False, 1, 5)
    cv2.line(mask, (300, 300), (321, 328), 1, 5)
    skel = skeleton.skeleton_from_mask(mask.astype(bool), spur_length=12)
    paths = skeleton.pieces(skel, min_length=10, bridge_length=15, stub_length=60)
    assert len(paths) == 1
    ends = sorted(map(tuple, np.round(paths[0].points[[0, -1]])))
    assert np.allclose(ends, [(100, 300), (340, 160)], atol=8)


def _sketch(seed=2, photo=False, ticks=6):
    return synth.make(kind="circuit", source="centerline", seed=seed, ticks=ticks, photo=photo)


@pytest.mark.parametrize("seed, photo", [(2, False), (3, True), (11, False)])
def test_ticks_are_found_where_drawn(seed, photo):
    sk = _sketch(seed, photo)
    result = process(sk.image, Params())
    prof = result.track.profile
    truth = cv2.perspectiveTransform(np.array(sk.ticks)[None].astype(float), result.homography)[0]
    assert len(prof.ticks) == len(truth)
    d = np.linalg.norm(truth[:, None] - prof.points[None], axis=2)
    assert d.min(axis=1).max() < 8.0
    assert len(prof.sections) == len(truth) and not prof.graded


def test_no_ticks_without_marks():
    t = process(synth.make(kind="circuit", source="centerline", seed=2, scribbles=4).image, Params()).track
    assert t.profile is None and t.m_per_px is None


def test_sections_tile_the_lap_in_travel_order():
    t = process(_sketch().image, Params()).track
    prof, total = t.profile, t.reference.length
    assert np.all(np.diff(prof.ticks) > 0) and 0 < prof.ticks[0] and prof.ticks[-1] < total
    assert sum(sec.length for sec in prof.sections) == pytest.approx(total)
    for a, b in zip(prof.sections, prof.sections[1:]):
        assert a.s_end == pytest.approx(b.s_start)
    assert prof.sections[-1].s_end == pytest.approx(prof.sections[0].s_start)  # wraps past the start


def test_scale_and_metric_road_width():
    t = process(_sketch().image, Params(length_m=3000, track_width_m=12)).track
    assert t.reference.length * t.m_per_px == pytest.approx(3000)
    assert t.width["median"] * t.m_per_px == pytest.approx(12, rel=0.005)
    assert t.reference.min_factor >= MIN_FACTOR


def test_grades_close_the_lap():
    grades = ["3%", "flat", "-2", "?", "1-2%", "4%"]
    t = process(_sketch().image, Params(grades=grades)).track
    prof = t.profile
    assert [sec.label for sec in prof.sections] == grades
    assert [sec.solved for sec in prof.sections] == [False, False, False, True, False, False]
    assert sum(sec.grade * sec.length for sec in prof.sections) == pytest.approx(0, abs=1e-6)
    assert prof.z[0] == pytest.approx(0.0)
    # elevation follows the grades between rows, and is continuous from section to section
    ds = np.diff(t.reference.s)
    np.testing.assert_allclose(np.diff(prof.z), (prof.grade[1:] + prof.grade[:-1]) / 2 * ds, atol=1e-3)
    for a, b in zip(prof.sections, prof.sections[1:]):
        assert a.z_end == pytest.approx(b.z_start)
    assert prof.sections[-1].z_end == pytest.approx(prof.sections[0].z_start)


def test_grade_changes_are_eased():
    grades = ["3%", "flat", "-2", "?", "1-2%", "4%"]
    t = process(_sketch().image, Params(length_m=3000, grades=grades, vertical_radius_m=800)).track
    prof, ref = t.profile, t.reference
    radius = 800 / t.m_per_px
    assert prof.vertical_radius == pytest.approx(radius)
    # the grade changes no faster than 1 / radius, all the way round the lap
    ds = np.diff(np.append(ref.s, ref.length))
    assert np.all(np.abs(np.diff(np.append(prof.grade, prof.grade[0]))) <= ds / radius + 1e-9)
    # each curve is as long as the grade change needs, and centred on its tick
    change = [b.grade - a.grade for a, b in zip(np.roll(prof.sections, 1), prof.sections)]
    np.testing.assert_allclose(prof.eases, radius * np.abs(change))
    for tick, sec, dg, ease in zip(prof.ticks, prof.sections, change, prof.eases):
        mid = np.interp(tick, ref.s, prof.grade)
        assert mid == pytest.approx(sec.grade - dg / 2, abs=2 / radius)
        assert np.interp(tick + ease / 2 + 2, ref.s, prof.grade) == pytest.approx(sec.grade)
    # away from the ticks the elevation is that of the straight grades
    abrupt = process(_sketch().image, Params(length_m=3000, grades=grades, vertical_radius_m=0)).track.profile
    assert abrupt.eases.max() == 0
    near = np.min(np.abs((ref.s[:, None] - prof.ticks[None] + ref.length / 2) % ref.length - ref.length / 2)
                  - prof.eases[None] / 2, axis=1) < 0
    np.testing.assert_allclose(prof.z[~near], abrupt.z[~near], atol=1e-6)
    assert np.abs(prof.z - abrupt.z).max() == pytest.approx(np.max(np.abs(change) * prof.eases / 8), rel=0.05)


def test_short_sections_limit_the_easing():
    t = process(_sketch().image, Params(length_m=3000, grades=["9%", "-9%", "9", "-9", "9", "-9"],
                                        vertical_radius_m=20000)).track
    prof = t.profile
    for k, ease in enumerate(prof.eases):
        assert ease <= min(prof.sections[k - 1].length, prof.sections[k].length) + 1e-9
    assert any("too short" in w for w in t.warnings)


def test_metric_lengths_need_a_scale():
    with pytest.raises(ValueError):
        process(_sketch().image, Params(vertical_radius_m=500))


def test_wrong_number_of_grades():
    with pytest.raises(TrackError, match="6 sections"):
        process(_sketch().image, Params(grades=["flat"] * 5))


def test_grades_that_do_not_close_are_reported():
    t = process(_sketch().image, Params(grades=["1%"] * 6)).track
    assert any("do not close" in w for w in t.warnings)


def test_json_export():
    result = process(_sketch().image, Params(length_m=3000, track_width_m=12,
                                             grades=["3%", "flat", "-2", "?", "1-2%", "4%"]))
    doc = json.loads(json.dumps(export.result_to_dict(result, samples=64)))
    assert doc["scale"]["length_m"] == pytest.approx(3000)
    assert doc["scale"]["track_width_m"] == pytest.approx(12, rel=0.005)
    prof = doc["profile"]
    assert prof["columns"] == ["s_px", "z_px", "grade"]
    rows, ref_rows = np.array(prof["rows"]), np.array(doc["reference"]["rows"])
    np.testing.assert_allclose(rows[:, 0], ref_rows[:, 0])
    assert [s["index"] for s in prof["sections"]] == [1, 2, 3, 4, 5, 6]
    assert [t["s_px"] for t in prof["ticks"]] == [s["s_start_px"] for s in prof["sections"]]
    assert prof["sections"][3]["solved"] and prof["sections"][3]["label"] == "?"


def test_climb_example():
    """The pencil photo: a one-line circuit with seven ticks and the grades written beside it.
    One tick is drawn on one side only, where the line turns sharply."""
    t = process(EXAMPLES / "climb_circuit.jpg", Params(**CLIMB)).track
    assert (t.kind, t.source) == ("circuit", "centerline")
    assert t.warnings == []
    assert t.reference.min_factor >= MIN_FACTOR
    assert t.reference.length * t.m_per_px == pytest.approx(4000)
    assert t.width["median"] * t.m_per_px == pytest.approx(10, rel=0.005)
    prof = t.profile
    assert len(prof.ticks) == 7
    assert np.linalg.norm(prof.points - [1249, 492], axis=1).min() < 5  # the one-sided tick
    grades = [round(100 * sec.grade, 1) for sec in prof.sections]
    assert grades[:5] == [1.5, 5.0, 7.0, 9.0, 3.0] and grades[6] == 0.0
    assert -5.0 < grades[5] < -3.5 and prof.sections[5].solved
    assert 35 < prof.z.max() * t.m_per_px < 45


def test_moving_a_tick_closes_the_lap():
    """A 3% descent instead of a solved one: the tick at its end moves until the climb is undone."""
    params = dict(CLIMB, grades=["1-2%", "5%", "7%", "9%", "3%", "-3%", "flat"], move_tick=7)
    t = process(EXAMPLES / "climb_circuit.jpg", Params(**params)).track
    prof, m = t.profile, t.m_per_px
    assert t.warnings == []
    assert [sec.solved for sec in prof.sections] == [False] * 7
    assert sum(sec.grade * sec.length for sec in prof.sections) == pytest.approx(0, abs=1e-6)
    climb = sum(sec.grade * sec.length for sec in prof.sections[:5])
    assert prof.sections[5].length == pytest.approx(climb / 0.03)
    assert np.count_nonzero(prof.moved) == 1 and 300 < prof.moved[6] * m < 420
    drawn = process(EXAMPLES / "climb_circuit.jpg", Params(**CLIMB)).track.profile
    assert prof.ticks[6] == pytest.approx(drawn.ticks[6] + prof.moved[6])
    np.testing.assert_allclose(prof.ticks[:6], drawn.ticks[:6])
    assert abs(prof.z).max() * m < 40 and prof.z.max() * m > 35


@pytest.mark.parametrize("grades, tick, message", [
    (["3%", "flat", "-2", "?", "1-2%", "4%"], 4, "every section"),
    (["3%", "flat", "-2", "-2", "1-2%", "4%"], 4, "same grade"),
    (["3%", "flat", "-2", "-1", "1-2%", "4%"], 9, "no tick 9"),
    (["30%", "flat", "-2", "-1", "1-2%", "4%"], 4, "past a neighbouring tick"),
])
def test_moving_a_tick_needs_a_solution(grades, tick, message):
    with pytest.raises(TrackError, match=message):
        process(_sketch().image, Params(grades=grades, move_tick=tick))


@pytest.mark.parametrize("params", [
    dict(length_m=3000, track_width_m=12, grades=["3%", "flat", "-2", "?", "1-2%", "4%"]),
    dict(),  # ticks but no grades or scale: a flat lap in px
])
def test_3d_view(tmp_path, params):
    from drawcircuit import render3d

    result = process(_sketch().image, Params(**params))
    track = tmp_path / "sketch.track.json"
    track.write_text(json.dumps(export.result_to_dict(result, samples=32)))
    assert render3d.main([str(track), "--view", "30,20"]) == 0
    image = cv2.imread(str(tmp_path / "sketch.3d.png"))
    assert image is not None and image.shape[0] > 200


@pytest.mark.parametrize("rise, size, expected", [(39, 1300, 10), (65, 1000, 5), (2, 1000, 150), (500, 100, 1)])
def test_auto_exaggeration(rise, size, expected):
    from drawcircuit.render3d import auto_exaggeration

    assert auto_exaggeration(rise, size) == expected


def test_switchback_example():
    """A second pencil photo: nine ticks, two flat sections, and a long solved descent."""
    grades = ["2%", "6%", "5%", "7%", "5%", "flat", "descent", "flat", "1%"]
    t = process(EXAMPLES / "switchback_circuit.jpg",
                Params(direction="cw", length_m=5000, track_width_m=10, grades=grades)).track
    prof = t.profile
    assert len(prof.ticks) == 9
    assert t.reference.min_factor >= MIN_FACTOR
    assert sum(sec.grade * sec.length for sec in prof.sections) == pytest.approx(0, abs=1e-6)
    assert prof.sections[6].solved and -4.5 < 100 * prof.sections[6].grade < -3.5
    assert 75 < np.ptp(prof.z) * t.m_per_px < 90


def test_section_widths():
    """One section 20 m wide in a 10 m road, with smooth tapers at its ticks."""
    params = Params(length_m=3000, track_width_m=10, section_widths_m={3: 20}, width_taper=5)
    t = process(_sketch().image, params).track
    ref, prof, m = t.reference, t.profile, t.m_per_px
    assert t.reference.length * m == pytest.approx(3000)
    widths = [sec.width * m for sec in prof.sections]
    assert widths[2] == pytest.approx(20, abs=0.2)
    assert all(w == pytest.approx(10, abs=0.2) for k, w in enumerate(widths) if k != 2)
    # the width changes smoothly: 10 m over about 50 m, centred on each tick
    w = (ref.w_left + ref.w_right) * m
    ds = np.diff(np.append(ref.s, ref.length)) * m
    rate = np.abs(np.diff(np.append(w, w[0]))) / ds
    assert rate.max() < 10 / 50 * np.pi / 2 * 1.2
    for tick in prof.ticks[2:4]:
        near = np.abs((ref.s - tick + ref.length / 2) % ref.length - ref.length / 2) * m
        assert w[near < 1].mean() == pytest.approx(15, abs=1.0)
        beyond = w[(near > 35) & (near < 45)]  # past the 25 m half-taper, give or take the tick's position
        assert np.all((np.abs(beyond - 10) < 0.5) | (np.abs(beyond - 20) < 0.5))
    assert t.width["min"] * m == pytest.approx(10, abs=0.2) and t.width["max"] * m == pytest.approx(20, abs=0.2)


@pytest.mark.parametrize("params, error", [
    (dict(length_m=3000, track_width_m=10, section_widths_m={9: 20}), TrackError),
    (dict(length_m=3000, section_widths_m={3: 20}), ValueError),
    (dict(track_width_m=10, section_widths_m={3: 20}), ValueError),
])
def test_section_widths_errors(params, error):
    with pytest.raises(error):
        process(_sketch().image, Params(**params))


def test_wide_straight_example():
    """A 2 km pencil circuit whose start straight is 20 m wide and the rest 10 m."""
    grades = ["flat", "12%", "flat", "-5%", "flat", "2%", "-2%", "flat", "12%", "-4%", "flat"]
    t = process(EXAMPLES / "wide_straight_circuit.jpg",
                Params(direction="cw", length_m=2000, track_width_m=10, section_widths_m={11: 20},
                       grades=grades, move_tick=11, vertical_radius_m=100)).track
    prof, m = t.profile, t.m_per_px
    assert len(prof.ticks) == 11
    assert t.warnings == ["a corner is tighter than half the track width (19 px); "
                          "the inside edge will fold over itself there"]
    # street-style climbs: the 12% kicks hold their grade for all but a few meters
    for sec in (prof.sections[1], prof.sections[8]):
        inside = (t.reference.s - sec.s_start) % t.reference.length < sec.length
        held = np.abs(prof.grade[inside] - 0.12) < 1e-4
        assert held.mean() > 0.75
    assert t.reference.min_factor >= MIN_FACTOR
    assert prof.sections[10].width * m == pytest.approx(20, abs=0.3)
    # the descent runs on past the drawn tick, onto the wide straight, until the lap closes at 4%
    assert 100 < prof.moved[10] * m < 150 and prof.sections[9].grade == -0.04
    assert prof.sections[0].width * m == pytest.approx(10, abs=0.3)
    # the start line is on the wide straight
    assert (t.reference.w_left[0] + t.reference.w_right[0]) * m == pytest.approx(20, abs=0.3)


TUNNEL = dict(direction="cw", length_m=8000, track_width_m=10, vertical_radius_m=300,
              grades="flat,15%,10%,descent,1%,flat,-1%,flat,1%,3%,4%,flat,2%,6%,8%,5%,9%,2%,flat,"
                     "-3%,-2%,flat,-7%,flat,8%,12%,descent".split(","))


def _tunnel(**kw):
    from drawcircuit import route

    return process(EXAMPLES / "tunnel_circuit.jpg",
                   Params(route=route.load(EXAMPLES / "tunnel_circuit.route.json"), **{**TUNNEL, **kw})).track


def test_tunnel_example():
    """An 8 km lap traced along waypoints, which passes under itself through a tunnel."""
    t = _tunnel(under=4)
    prof, m = t.profile, t.m_per_px
    assert (t.kind, t.source) == ("circuit", "centerline")
    assert len(prof.ticks) == 27
    assert t.reference.min_factor >= MIN_FACTOR
    assert t.reference.length * m == pytest.approx(8000)
    # the reference rows measure each road's own width, not the other road's at the crossing
    assert np.all(np.abs((t.reference.w_left + t.reference.w_right) * m - 10) < 1.0)
    [c] = prof.crossings
    assert c.clearance * m == pytest.approx(6.0, abs=0.1)
    assert profile.section_at(prof.sections, c.s_under, t.reference.length).index == 4
    # both descents stay descents; the one through the tunnel is steeper to make the room
    assert prof.sections[3].solved and prof.sections[26].solved
    assert prof.sections[3].grade < prof.sections[26].grade < 0
    assert sum(sec.grade * sec.length for sec in prof.sections) == pytest.approx(0, abs=1e-6)
    assert not any("passes only" in w for w in t.warnings)


def test_crossing_without_room_is_reported():
    t = _tunnel()
    [c] = t.profile.crossings
    assert c.clearance * t.m_per_px < 1.0
    assert any("passes only" in w and "--under" in w for w in t.warnings)


def test_route_through_a_figure_eight():
    """Waypoints along a centerline that crosses itself: the line goes straight through the
    crossing, and the reference line measures widths to its own edges."""
    from drawcircuit import route

    sk = synth.make(kind="circuit", source="centerline", seed=16, shape="8")
    truth = geometry.resample(sk.centerline, count=24, closed=True)
    rt = route.Route(truth, np.zeros(len(truth), bool), True)
    t = process(sk.image, Params(route=rt, length_m=3000, track_width_m=12)).track
    assert t.reference is not None and t.reference.min_factor >= MIN_FACTOR
    d = geometry.nearest_distances(t.centerline.sample(2000), sk.centerline, dst_closed=True)
    assert np.percentile(d, 95) < 3.0
    w = (t.reference.w_left + t.reference.w_right) * t.m_per_px
    assert np.percentile(np.abs(w - 12), 99) < 0.5


def test_route_point_off_the_drawing():
    from drawcircuit import route

    sk = synth.make(kind="circuit", source="centerline", seed=2)
    pts = geometry.resample(sk.centerline, count=12, closed=True)
    pts[3] += [80, 80]
    with pytest.raises(TrackError, match="route point 4"):
        process(sk.image, Params(route=route.Route(pts, np.zeros(12, bool), True)))
