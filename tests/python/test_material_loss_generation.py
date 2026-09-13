"""Same-work grazing allocation, separate from the physical model or query curve."""
import pytest
torch = pytest.importorskip('torch')


def test_generation_allocation_preserves_work_and_caps():
    from tools.material_loss_resource import generation_orders
    assert generation_orders(.0001, .1, 280, 32) == (140, 64)
    assert generation_orders(.1**2/16, .1, 280, 32) == (280, 32)
    assert generation_orders(1., .001, 280, 32) == (280, 32)
    for radial, azimuth in ((192,32),(280,32),(512,64)):
        for c in (0.,1e-9,.1,1.):
            x,y=generation_orders(c,.1,radial,azimuth)
            assert x*y == radial*azimuth and 16<=x<=512 and 16<=y<=512
    for radial,azimuth in ((16,32),(33,32),(280,512),(1024,32)):
        with pytest.raises(ValueError):generation_orders(.1,.1,radial,azimuth)


def test_conditional_generator_dispatches_actual_knots(monkeypatch):
    from tools import material_loss_resource as resource
    calls=[]
    def fake(c,r,*,resolution,azimuth_resolution):
        calls.append((float(c),float(r),resolution,azimuth_resolution))
        return torch.tensor([.8,0. if c==0 else .2],dtype=torch.float64)
    monkeypatch.setattr(resource,'endpoints',fake)
    result=resource.generate(roughness_nodes=2,cosine_nodes=17,resolution=32,
                             azimuth_resolution=16,grazing_azimuth=True)
    assert len(calls)==34 and len(result.losses)==2
    assert any(n==16 for _,_,n,_ in calls) and any(n==32 for _,_,n,_ in calls)
    for c,r,n,a in calls:
        assert (n,a)==((16,32) if c<r*r/16 else (32,16))


def test_allocation_boundary_and_grazing_regressions(tmp_path, request):
    import json
    from tools.material_loss_resource import generation_pair
    old=torch.get_num_threads();torch.set_num_threads(1)
    rows=[]
    try:
        for r in (.00231,.0137,.0371,.287,.891):
            for c in (.003*r*r, r*r/16*(1-.001),r*r/16,r*r/16*(1+.001),1.):
                rough=torch.tensor(r,dtype=torch.float64)
                a=generation_pair(c,rough,resolution=280,azimuth_resolution=32)
                b=generation_pair(c,rough,resolution=512,azimuth_resolution=64)
                for f in (0,1):
                    rows.append(dict(roughness=r,cosine=c,f0=f,error=abs(float(a[f]-b[f])),
                                     limit=1e-12 if b[f]==0 else .0001*float(b[f])))
    finally:torch.set_num_threads(old)
    (tmp_path/'allocation-boundary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    from tools.testing_dashboard.pytest_plugin import Evidence
    Evidence(request.node.nodeid).record('allocation-boundary',dict(rows=rows))
    assert all(row['error']<=row['limit'] for row in rows),rows
