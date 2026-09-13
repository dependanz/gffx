"""Independent arithmetic for the original deformation learning chapter."""
import pytest
from tools.testing_dashboard.animation_lesson import CHAPTER, worked_examples


def test_animation_lesson_worked_examples_match_hand_arithmetic():
    """The teaching examples have checkable answers independent of animation kernels."""
    actual = worked_examples()
    expected = {"unit_corrective": (3., 2., 2.), "negative_corrective": (-3., 2., 5.),
                "translated_child": (2., 3., 0.), "triangle_centroid": (1., 1., 0.),
                "quarter_turn_x": (0., 1., 0.), "corrective_bytes": 48220800}
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert 2*actual["corrective_bytes"] > 64*1024**2


def test_animation_lesson_explains_composition_and_limits():
    """Readers can distinguish deformation math, model assets and unsupported promises."""
    assert CHAPTER["id"] == "learn/animation-deformation"
    assert CHAPTER["evidence"] == ["animation/contract"]
    for phrase in ("Worked example", "Try it yourself", "Answer", "gffx boundary",
                   "bind", "not a simulation", "first-order", "inverse"):
        assert phrase in CHAPTER["body"]
    assert r"\sum_{k}" in CHAPTER["body"] and r"\sum_{j}" in CHAPTER["body"]
