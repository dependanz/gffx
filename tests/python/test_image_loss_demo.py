"""The fitting page must distinguish measurements from amplified display maps."""
import importlib
import pytest
import torch


def test_fitting_panels_preserve_target_before_after_and_background():
    demo=importlib.import_module('tools.testing_dashboard.image_loss_demo')
    report=dict(indices=[129],result=dict(target=[[.1,.2,.3]],passes=[
        dict(values=[[.2,.3,.4]],gradient=[[[.001,.002,.003]]*32]*32),
        dict(values=[[.15,.25,.35]],gradient=[[[.001,.002,.003]]*32]*32)]))
    images=demo.panels(report)
    assert images['images'].shape==(128,384,3)
    assert not bool(images['images'][0].any())
    from tools.testing_dashboard.image_reference_demo import display_rgb
    for column,values in enumerate(([.1,.2,.3],[.2,.3,.4],[.15,.25,.35])):
        assert torch.equal(images['images'][1,1+128*column],display_rgb(torch.tensor(values,dtype=torch.float64)))
    assert images['gradient'].shape==(32,32,3)
    assert images['gradient_scale']==pytest.approx(.003)


def test_fitting_publisher_has_finite_explained_artifacts_without_intercepting_live_tests(monkeypatch):
    from unittest.mock import MagicMock
    import gffx.dashboard
    from tools.testing_dashboard import reporter
    demo=importlib.import_module('tools.testing_dashboard.image_loss_demo')
    live_send=reporter.Reporter.send;events=[];fake=MagicMock()
    fake.send.side_effect=lambda event,**values:events.append((event,values)) or True
    image=torch.zeros((128,384,3),dtype=torch.uint8)
    rows=[dict(model=m,pixels=6536,indices=[],supervisor={},result=dict(passes=[dict(loss=.1),dict(loss=.09)])) for m in ('classic','principled')]
    with monkeypatch.context() as patches:
        patches.setattr(reporter,'Reporter',lambda *a,**kw:fake)
        patches.setattr(gffx.dashboard,'connect',lambda *a,**kw:MagicMock())
        patches.setattr(demo,'panels',lambda r:dict(images=image,gradient=torch.zeros((32,32,3),dtype=torch.uint8),gradient_scale=.01))
        assert live_send is reporter.Reporter.__dict__.get('send',live_send)
        demo.publish(rows,'a'*64,'unused','unit')
    assert reporter.Reporter.send is live_send
    artifacts=next(v['artifacts'] for event,v in events if event=='demo')
    assert len(artifacts)==10 and sum(a['kind']=='image' for a in artifacts)==4
    assert all(a['label'] for a in artifacts)
    assert events[-1]==('finish',dict(exit_code=0))
