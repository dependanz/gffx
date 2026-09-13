"""Bounded pointer/shape audit; hooks retain metadata, never extra tensor graphs."""
import json
import torch
from test_deferred import api
from test_deferred_recompute import fields,resource


def test_deferred_budget_constants_stay_borrowed_through_recomputed_backward(record_property):
    d=api();n,v,l,a,r,f=fields();budget=resource()
    def key(value):return (str(value.device),value.untyped_storage().data_ptr())
    tables={key(value):value.untyped_storage().nbytes() for value in budget._data.values()}
    versions={name:value._version for name,value in budget._data.items()}
    assert all(not value.requires_grad and value.grad is None for value in budget._data.values())
    saved={};events=0
    def pack(value):
        nonlocal events
        events+=1
        # Only immutable metadata is retained by the audit, not a tensor handle.
        saved[key(value)]=dict(bytes=value.untyped_storage().nbytes(),shape=list(value.shape),
                               requires_grad=value.requires_grad)
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
        output=d._principled_recompute(n,v,l,dict(albedo=a,roughness=r,f0=f,budget=budget))
        gradient,=torch.autograd.grad(output.sum(),r)
    assert torch.isfinite(gradient) and gradient!=0
    assert all(value._version==versions[name] and not value.requires_grad and value.grad is None
               for name,value in budget._data.items())
    assert {key(value):value.untyped_storage().nbytes() for value in budget._data.values()}==tables
    borrowed={key_:value for key_,value in saved.items() if key_ in tables}
    assert all(not value['requires_grad'] for value in borrowed.values())
    scratch=3*budget.workspace(6,backward=True)+8*2*192
    nonresource=sum(value['bytes'] for key_,value in saved.items() if key_ not in tables)
    assert nonresource<=scratch
    record_property('storage_audit',json.dumps(dict(saved_events=events,
        resource_unique_bytes=sum(tables.values()),borrowed_saved_resource_bytes=sum(v['bytes'] for v in borrowed.values()),
        nonresource_unique_saved_bytes=nonresource,material_scratch_allowance_bytes=scratch,
        scope='saved storage metadata across two-point recomputation; not total transient or process peak'),sort_keys=True))
