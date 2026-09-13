"""Dependency-free educational pages, not material implementation acceptance."""
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

def module(name):
    return importlib.import_module('tools.testing_dashboard.'+name)

def test_mathml_subset_has_structural_fractions_scripts_and_operators():
    render=module('mathml').render
    root=ET.fromstring(render(r'\frac{x_i^2}{\sqrt{\alpha}} + \sum_{k=0}^{9} a_k'))
    tags=[e.tag.rsplit('}',1)[-1] for e in root.iter()]
    for tag in ('math','mfrac','msubsup','msqrt','munderover','annotation'): assert tag in tags
    assert root.tag=='{http://www.w3.org/1998/Math/MathML}math'
    assert list(root.iter())[-1].text==r'\frac{x_i^2}{\sqrt{\alpha}} + \sum_{k=0}^{9} a_k'

def test_mathml_subset_supports_matrices_alignment_cases_and_text():
    render=module('mathml').render
    for env in ('matrix','pmatrix','bmatrix','aligned','cases'):
        root=ET.fromstring(render(r'\begin{'+env+r'}a&b\\c&\text{otherwise}\end{'+env+'}'))
        ns={'m':'http://www.w3.org/1998/Math/MathML'}
        assert len(root.findall('.//m:mtr',ns))==2
        assert len(root.findall('.//m:mtd',ns))==4
        assert root.find('.//m:mtext',ns).text=='otherwise'

@pytest.mark.parametrize('source',[r'\input{secret}',r'\href{x}{y}',r'\unknown',r'\frac{x}',
    'x^^2','x_1_2','{x',r'\begin{matrix}a&b\\c\end{matrix}',r'\begin{cases}a\end{matrix}',
    r'\left(x',r'\right)', 'x&y', 'x'*8193, '{'*50+'x'+'}'*50, r'\vec'*50+' x'],
    ids=['input','href','unknown','fraction','script','duplicate','group','ragged','environment','left','right','alignment','length','depth','accents-depth'])
def test_mathml_invalid_or_excessive_input_fails_explicitly(source):
    with pytest.raises(ValueError):module('mathml').render(source)

def test_mathml_text_is_escaped_not_executable_markup():
    rendered=module('mathml').render(r'\text{<img src=x onerror=alert(1)>}')
    assert '<img' not in rendered
    root=ET.fromstring(rendered)
    assert root.find('.//{http://www.w3.org/1998/Math/MathML}mtext').text=='<img src=x onerror=alert(1)>'

def test_mathml_styles_use_core_compatible_identifier_characters():
    root=ET.fromstring(module('mathml').render(r'\mathbf{N}+\mathrm{R}'))
    tokens=root.findall('.//{http://www.w3.org/1998/Math/MathML}mi')
    assert tokens[0].text=='𝐍'
    assert tokens[1].get('mathvariant')=='normal'

def test_learning_article_is_generated_reproducibly_and_has_provenance():
    build=module('build_lessons')
    actual=json.loads((ROOT/'tools/testing_dashboard/static/learn/energy-conservation.json').read_text(encoding='utf-8'))
    assert actual==build.article()
    for text in ('<math','<annotation','1.42787','1.00003','not a public','shading/material-contract'):
        assert text in actual['html']
    assert '<script' not in actual['html'] and '<iframe' not in actual['html']
    assert len(actual['html'].encode())<100_000

def test_learning_pages_survive_restart_without_test_execution(tmp_path):
    Index=module('model').Index
    identifier='learn/shading/energy-conservation'
    index=Index(tmp_path/'index.sqlite')
    page=index.page(identifier)
    assert page['kind']=='article' and page['state']=='article'
    assert page['evidence'] is None and page['counts']=={} and page['total']==0
    assert page['article_html'] and page['updated_on']=='2026-09-07'
    # The catalog is intentionally paginated; adding pages must not make this
    # restart check depend on a lesson staying among the first fifty entries.
    found=[]
    offset=0
    while True:
        directory=index.directory(offset=offset,limit=7)
        found.extend(p['id'] for p in directory['items'])
        offset+=len(directory['items'])
        if offset>=directory['total']:break
        assert directory['items'], 'pagination stopped before the declared total'
    assert len(found)==len(set(found))==directory['total']
    assert identifier in found
    assert Index(tmp_path/'index.sqlite').page(identifier)==page
    assert 'article_html' not in next(p for p in index.pages() if p['id']==identifier)
    assert index.page('shading/material-contract')['learning_article']==identifier
    with pytest.raises(KeyError):index.page(identifier,run='invented')
