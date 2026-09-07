"""The mini-textbook is authored explanation, not an automatic support claim."""
import importlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[2]

def book():
    return importlib.import_module('tools.testing_dashboard.textbook')

def test_textbook_has_ordered_complete_chapters_and_acyclic_prerequisites():
    chapters=book().CHAPTERS
    assert len(chapters)==14
    seen=set()
    for chapter in chapters:
        assert chapter['id'] not in seen
        assert set(chapter['prerequisites'])<=seen
        seen.add(chapter['id'])
    assert 'learn/shading/energy-conservation' in seen

@pytest.mark.parametrize('slug',['vectors','transforms','triangles','topology','barycentric',
    'cameras','rasterization','textures','proximity','sampling','autodiff','numerics','animation'])
def test_textbook_chapter_is_substantial_reproducible_and_mathematically_structured(slug):
    build=importlib.import_module('tools.testing_dashboard.build_lessons')
    identifier='learn/'+slug
    article=build.articles()[identifier]
    assert json.loads((build.ROOT/'static/learn'/build.asset_name(identifier)).read_text(encoding='utf-8'))==article
    text=re.sub('<[^>]+>',' ',article['html'])
    assert len(text.split())>=300
    for phrase in ('Worked example','Try it yourself','Answer','gffx boundary','Follow the evidence'):
        assert phrase in article['html']
    formulas=re.findall(r'<math\b.*?</math>',article['html'],re.S)
    assert len(formulas)>=2
    for formula in formulas:ET.fromstring(formula)
    assert len(article['html'])<100000

def test_textbook_all_links_resolve_and_assets_are_fixed_not_user_paths(tmp_path):
    build=importlib.import_module('tools.testing_dashboard.build_lessons')
    catalog=importlib.import_module('tools.testing_dashboard.catalog')
    Index=importlib.import_module('tools.testing_dashboard.model').Index
    index=Index(tmp_path/'book.sqlite')
    valid={p['id'] for p in catalog.PAGES}
    articles=build.articles()
    assert len(articles)==15
    assert len({build.asset_name(identifier) for identifier in articles})==len(articles)
    for identifier,article in articles.items():
        page=index.page(identifier)
        assert page['kind']=='article' and page['evidence'] is None and page['counts']=={}
        assert page['article_html']==article['html']
        assert Index(tmp_path/'book.sqlite').page(identifier)==page
        for href in re.findall(r'href="([^"]+)"',article['html']):
            if href.startswith('/testing/'):
                assert parse_qs(urlsplit(href).query)['page'][0] in valid
        with pytest.raises(KeyError):index.page(identifier,run='not-an-experiment')
    for identifier in ('learn/../../secret','learn/unknown','energy-conservation'):
        with pytest.raises(KeyError):build.asset_name(identifier)

def test_textbook_worked_examples_match_independent_arithmetic():
    examples=book().worked_examples()
    expected={'dot':0.,'triangle_area':3.,'barycentric_x':.6,'barycentric_y':1.5,
      'pixel_x':570.,'pixel_y':115.,'perspective_attribute':1/3,
      'bilinear':.3125,'segment_x':1.,'segment_distance_squared':4.,
      'sample_weight':.5,'loss_gradient':-2.,'freefall_height':1.95095}
    assert examples.keys()==expected.keys()
    for key,value in expected.items():assert examples[key]==pytest.approx(value,abs=1e-12)

def test_textbook_index_lists_reading_order_and_implementation_boundaries():
    build=importlib.import_module('tools.testing_dashboard.build_lessons')
    body=build.articles()['learn/index']['html']
    for chapter in book().CHAPTERS:assert chapter['title'] in body
    assert 'not a claim that every topic is implemented' in body
    assert 'Future chapters' in body

def test_textbook_catalog_refresh_preserves_dates_but_updates_titles_and_parents(tmp_path):
    model=importlib.import_module('tools.testing_dashboard.model')
    index=model.Index(tmp_path/'book.sqlite')
    nodeid='tests/python/test_textbook.py::test_textbook_worked_examples_match_independent_arithmetic'
    tid=model.test_id(nodeid)
    with index._db() as db:
        db.execute("UPDATE inspection_pages SET title='Old title',created_on='2020-01-01' WHERE id='learn/shading/energy-conservation'")
        # A reporter running the previous catalog initially sees a new test as unassigned.
        db.execute("INSERT INTO runs(id,kind,state) VALUES('old','tests','running')")
        db.execute("INSERT INTO tests(run,id,nodeid) VALUES('old',?,?)",(tid,nodeid))
        index._test_page(db,nodeid)
        db.execute("UPDATE inspection_pages SET parent='infrastructure/unassigned' WHERE id=?",('test/'+tid,))
    model.Index(tmp_path/'book.sqlite')
    with index._db() as db:
        article=db.execute("SELECT title,created_on FROM inspection_pages WHERE id='learn/shading/energy-conservation'").fetchone()
        test=db.execute('SELECT parent FROM inspection_pages WHERE id=?',('test/'+tid,)).fetchone()
    assert article['title'].startswith('13 ·') and article['created_on']=='2020-01-01'
    assert test['parent']=='dashboard/learning-checks'
