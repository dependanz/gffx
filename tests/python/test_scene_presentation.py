"""The scene demonstrations and explanations must use the real public interface."""
import importlib
import inspect
from pathlib import Path
import torch


def test_scene_demo_calls_public_flatten_and_is_bounded(monkeypatch):
    from gffx.torch.scene import Scene
    demo=importlib.import_module('tools.testing_dashboard.scene_demo')
    original=Scene.flatten;calls=[]
    def observed(self,**kw):calls.append(kw);return original(self,**kw)
    monkeypatch.setattr(Scene,'flatten',observed)
    result=demo.payload()
    assert len(calls)>=3 and len(result['frames'])==12
    assert result['vertices'].shape==(24,3) and result['faces'].shape==(36,3)
    assert result['image'].shape==(192,192,3) and result['image'].dtype==torch.uint8
    assert result['image'].max()>0
    assert result['fit_losses'][-1]<result['fit_losses'][0]
    assert len(result['fit_losses'])==8


def test_scene_every_test_has_authored_explanation():
    from tools.testing_dashboard.explanations import explain
    for file in ('test_scene.py','test_scene_integration.py','test_scene_presentation.py'):
        import ast
        tree=ast.parse((Path(__file__).parent/file).read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'):
                row=explain('tests/python/'+file+'::'+node.name)
                assert row['authored'],node.name


def test_scene_textbook_and_catalog_agree():
    from tools.testing_dashboard.textbook import CHAPTERS
    from tools.testing_dashboard.catalog import PAGES
    assert any(c['id']=='learn/scenes' for c in CHAPTERS)
    ids={p['id'] for p in PAGES}
    assert {'scene/contract','scene/integration','scene/hierarchy-demo'}<=ids
