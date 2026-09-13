"""The new M1 mathematics chapters are discoverable and fingerprinted."""
from tools.testing_dashboard import build_lessons, textbook


def test_m1_lessons_are_registered_in_dependency_order():
    ids = [chapter['id'] for chapter in textbook.CHAPTERS]
    assert {'learn/image-compositing', 'learn/image-sampling',
            'learn/deferred-rendering', 'learn/animation-deformation'} <= set(ids)
    visited = set()
    for chapter in textbook.CHAPTERS:
        assert set(chapter['prerequisites']) <= visited
        visited.add(chapter['id'])


def test_m1_textbook_introduction_separates_library_progress_from_consumers():
    intro = build_lessons.articles()['learn/index']['html']
    assert 'native deformation' in intro
    assert 'consumer implementations remain separate' in intro
    assert 'not a physics solver' in intro
    assert 'not the planned animation' not in intro


def test_m1_lesson_fingerprint_covers_modular_chapters(tmp_path, monkeypatch):
    names = ('build_lessons.py', 'mathml.py', 'textbook.py',
             'image_lessons.py', 'deferred_lessons.py', 'animation_lesson.py')
    for name in names:
        (tmp_path / name).write_text(name, encoding='utf-8')
    monkeypatch.setattr(build_lessons, 'ROOT', tmp_path)
    original = build_lessons.fingerprint()
    for name in names[-3:]:
        (tmp_path / name).write_text(name + ' revised', encoding='utf-8')
        changed = build_lessons.fingerprint()
        assert changed != original
        original = changed
