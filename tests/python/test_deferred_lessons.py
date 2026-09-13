"""Exact rational checks for the original renderer learning chapter."""
from fractions import Fraction
from tools.testing_dashboard.deferred_lessons import CHAPTERS,worked_examples


def test_deferred_lesson_perspective_example_matches_rational_arithmetic():
    raw=[Fraction(1,3)/w for w in (1,2,4)]
    expected=[value/sum(raw) for value in raw]
    actual=worked_examples()['perspective_weights']
    assert all(abs(a-float(b))<1e-15 for a,b in zip(actual,expected))
    assert expected==[Fraction(4,7),Fraction(2,7),Fraction(1,7)]


def test_deferred_lesson_premultiplied_example_and_boundaries():
    values=worked_examples()
    actual=tuple(c+(1-values['coverage']) for c in values['premultiplied_red'])
    assert actual==values['red_over_white']
    chapter=CHAPTERS[0]
    assert chapter['id']=='learn/deferred-rendering'
    for phrase in ('zero active normal','candidate','NDC units','recomputation','not a path tracer'):
        assert phrase in chapter['body']
