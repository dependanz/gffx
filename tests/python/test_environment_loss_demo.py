"""Sparse environment checks must not be presented as complete images."""
import importlib
import pytest
import torch


def test_environment_swatches_label_two_pixels_without_inventing_an_image():
    demo=importlib.import_module('tools.testing_dashboard.environment_loss_demo')
    gradient=torch.zeros((32,32,3),dtype=torch.float64);gradient[2,3]=.002
    result=dict(inputs=dict(target=[[.1,.2,.3],[.4,.5,.6]]),passes=[
        dict(offsets=[dict(values=[[.2,.3,.4],[.5,.6,.7]],gradient=gradient.tolist())]*2),
        dict(offsets=[dict(values=[[.15,.25,.35],[.45,.55,.65]],gradient=gradient.tolist())]*2)])
    pictures=demo.panels(result)
    assert pictures['swatches'].shape==(128,192,3) and pictures['gradient'].shape==(32,32,3)
    from tools.testing_dashboard.image_reference_demo import display_rgb
    assert torch.equal(pictures['swatches'][0,0],display_rgb(torch.tensor([.1,.2,.3],dtype=torch.float64)))
    assert torch.equal(pictures['swatches'][64,128],display_rgb(torch.tensor([.45,.55,.65],dtype=torch.float64)))
    assert pictures['gradient_scale']==pytest.approx(.002)


def test_environment_demo_publication_stays_finite_and_preserves_live_reporter(monkeypatch):
    from unittest.mock import MagicMock
    import gffx.dashboard
    from tools.testing_dashboard import reporter
    demo=importlib.import_module('tools.testing_dashboard.environment_loss_demo')
    real=reporter.Reporter;send=real.send;events=[];fake=MagicMock()
    fake.send.side_effect=lambda event,**kw:events.append((event,kw)) or True
    reports=[dict(model=m,environment=e,indices=[1,2],supervisor={},result=dict(calls=2,directions=256,
        passes=[dict(offsets=[dict(split=s,loss=x,metrics={}) for s in (.5,.25)]) for x in (.1,.09)]))
        for m in ('classic','principled') for e in ('uniform','affine')]
    with monkeypatch.context() as p:
        p.setattr(reporter,'Reporter',lambda *a,**kw:fake)
        p.setattr(gffx.dashboard,'connect',lambda *a,**kw:MagicMock())
        p.setattr(demo,'panels',lambda r:dict(swatches=torch.zeros((128,192,3),dtype=torch.uint8),gradient=torch.zeros((32,32,3),dtype=torch.uint8),gradient_scale=.1))
        demo.publish(reports,'a'*64,'unused','unit')
        assert real.send is send
    artifacts=next(kw['artifacts'] for event,kw in events if event=='demo')
    assert len(artifacts)==18 and sum(x['kind']=='image' for x in artifacts)==8
    assert all(x['label'] for x in artifacts)
    assert events[-1]==('finish',dict(exit_code=0))
