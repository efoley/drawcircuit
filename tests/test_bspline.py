import numpy as np

from drawcircuit import bspline


def de_boor(t, c, k, x):
    """Plain de Boor evaluation, independent of scipy, as a consumer of the JSON would do it."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    n = len(c)
    i = int(np.clip(np.searchsorted(t, x, side="right") - 1, k, n - 1))
    d = [c[j + i - k].copy() for j in range(k + 1)]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            lo, hi = t[j + i - k], t[j + 1 + i - r]
            alpha = (x - lo) / (hi - lo)
            d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


def circle(r=300.0, n=2000, noise=0.0, seed=0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([800 + r * np.cos(th), 600 + r * np.sin(th)])
    return pts + np.random.default_rng(seed).normal(0, noise, pts.shape)


def test_closed_fit_is_accurate_and_c2_across_the_seam():
    s = bspline.fit(circle(noise=0.5), closed=True, tol=1.0)
    assert s.periodic and s.max_error < 2.5
    radius = np.linalg.norm(s.sample(1000) - [800, 600], axis=1)
    assert np.abs(radius - 300).max() < 1.0
    a, b = s.domain
    for nu in (0, 1, 2):
        np.testing.assert_allclose(s(a, nu), s(b, nu), rtol=1e-6, atol=1e-6)
    # periodic layout: the last `degree` control points repeat the first ones
    np.testing.assert_allclose(s.control_points[-3:], s.control_points[:3])


def test_open_fit_hits_the_ends():
    x = np.linspace(100, 1500, 1500)
    pts = np.column_stack([x, 600 + 200 * np.sin(x / 250)])
    s = bspline.fit(pts, closed=False)
    a, b = s.domain
    assert np.linalg.norm(s(a) - pts[0]) < 1.5
    assert np.linalg.norm(s(b) - pts[-1]) < 1.5
    assert s.max_error < 2.5


def test_tolerance_controls_control_point_count():
    # a hand-drawn-like loop: smooth wobble of a few px plus pixel jitter
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = 300 + 3 * np.sin(23 * th) + 2 * np.sin(41 * th + 1)
    pts = np.column_stack([800 + r * np.cos(th), 600 + r * np.sin(th)])
    pts += np.random.default_rng(1).normal(0, 0.4, pts.shape)
    loose = bspline.fit(pts, closed=True, tol=6.0)
    tight = bspline.fit(pts, closed=True, tol=1.0)
    assert len(loose.unique_control_points) < 40 < len(tight.unique_control_points)
    assert tight.rms_error < 1.0


def test_de_boor_matches_for_open_and_periodic():
    for closed, pts in ((True, circle()), (False, circle()[:900])):
        s = bspline.fit(pts, closed=closed)
        u = s.params(50)
        ours = np.array([de_boor(s.knots, s.control_points, s.degree, x) for x in u])
        np.testing.assert_allclose(ours, s(u), atol=1e-6)
