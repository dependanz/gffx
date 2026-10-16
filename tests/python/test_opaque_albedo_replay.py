"""Explicit albedo replay keeps canonical pixel carries and owns no lasting cache."""
import os,torch,pytest
from test_opaque_full_original import original
from gffx.torch import render


def bits(t):return t.detach().cpu().contiguous().numpy().tobytes()


def reference(s,c,region,g,gm,initial):
    a=s.prims[1].parameters['albedo'];a.requires_grad_();carry=initial.clone();rows=[];masks=[]
    y,x,h,w=region
    for yy in range(h):
        for xx in range(w):
            out=render.opaque_window(s,c,(y+yy,x+xx,1,1),cull_mode=2)
            grad,=torch.autograd.grad((out['colour'],out['mask']),a,(g[:,yy:yy+1,xx:xx+1],gm[:,yy:yy+1,xx:xx+1]))
            carry.add_(grad);rows.append(out['colour'].detach());masks.append(out['mask'].detach())
    return dict(colour=torch.cat(rows,2).reshape(1,h,w,3),mask=torch.cat(masks,2).reshape(1,h,w),albedo_gradient=carry)


@pytest.mark.parametrize('region',[(64,62,1,4),(0,0,2,2),(10,59,1,1),(15,48,1,1),(117,67,1,1)])
def test_replay_matches_canonical_pixels_and_ordered_carry(region):
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];_,_,h,w=region
    g=a.new_tensor([.2,-.3,.7]).expand(1,h,w,3).contiguous();gm=a.new_full((1,h,w),.1);initial=a.new_tensor([-.0,5e-324,-.2])
    expected=reference(s,c,region,g,gm,initial);before=[bits(x) for x in (a,g,gm,initial)]
    actual=render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial,cull_mode=2)
    assert all(bits(actual[k])==bits(v) for k,v in expected.items())
    assert before==[bits(x) for x in (a,g,gm,initial)]
    assert all(not x.requires_grad for x in actual.values())


def test_replay_prepares_fixed_geometry_once_and_does_not_persist(monkeypatch):
    from gffx.torch import _opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];calls=[];old=scene._project
    def project(*args,**kw):calls.append(1);return old(*args,**kw)
    monkeypatch.setattr(scene,'_project',project)
    for _ in range(2):render.opaque_albedo_replay(s,c,(64,62,1,4),a.new_ones((1,1,4,3)),a.new_zeros((1,1,4)))
    assert len(calls)==2


@pytest.mark.parametrize('bad',['geometry_gradient','cotangent_shape','nonfinite_carry','too_many_pixels'])
def test_replay_refuses_unsupported_targets_or_invalid_inputs(bad):
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];region=(64,62,1,4);g=a.new_ones((1,1,4,3));gm=a.new_zeros((1,1,4));initial=a.new_zeros(3)
    if bad=='geometry_gradient':s.prims[0].vertices.requires_grad_()
    elif bad=='cotangent_shape':g=g[:,:,:1]
    elif bad=='nonfinite_carry':initial[0]=float('nan')
    else:region=(0,0,3,128);g=a.new_ones((1,3,128,3));gm=a.new_zeros((1,3,128))
    with pytest.raises((ValueError,NotImplementedError)):render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)


def test_replay_refuses_input_mutation_before_result(monkeypatch):
    from gffx.torch import _opaque_stream as stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];old=stream._evaluate;calls=[]
    def mutate(*args,**kw):
        result=old(*args,**kw)
        if not calls:
            with torch.no_grad():a.add_(.01)
        calls.append(1);return result
    monkeypatch.setattr(stream,'_evaluate',mutate)
    with pytest.raises(RuntimeError,match='modified'):render.opaque_albedo_replay(s,c,(64,62,1,4),a.new_ones((1,1,4,3)),a.new_zeros((1,1,4)))


def test_replay_public_facade_and_partition_carries():
    import gffx.render as facade
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo']
    # The empty-to-covered transition exercises zeros and material contributions
    # in one rectangle; splitting calls must not insert a chunk reduction.
    region=(10,57,1,4);g=a.new_tensor([-.0,-5e-324,.7]).expand(1,1,4,3).contiguous();gm=a.new_zeros((1,1,4));initial=a.new_tensor([-.0,5e-324,-.2])
    expected=reference(s,c,region,g,gm,initial)
    whole=facade.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)
    carry=initial
    for x in range(4):
        part=facade.opaque_albedo_replay(s,c,(10,57+x,1,1),g[:,:,x:x+1],gm[:,:,x:x+1],initial_gradient=carry)
        carry=part['albedo_gradient']
        for k in ('colour','mask'):assert bits(part[k])==bits(whole[k][:,:,x:x+1])
    assert bits(carry)==bits(whole['albedo_gradient'])==bits(expected['albedo_gradient'])
