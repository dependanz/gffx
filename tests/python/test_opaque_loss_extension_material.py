"""Independent high-precision material identity for branch residual composition."""
import torch
from decimal import Decimal as D,localcontext
from tools.opaque_shading_intervals import Jet
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_loss_extension import LossExtension
from tools.principled_material_oracle import material_pair

def test_material_extension_plus_residual_contains_independent_value_and_albedo_derivative():
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    data={k:v.tolist() for k,v in budget._data.items()};curve=PreparedPrincipled(data,1.)
    params=[.2,.1,1., .1,.05,2., 1.,1.,2., .2,.4,.6, 1.,.04,.04,.04]
    def inputs():return [[Jet(v,tangent=1 if start+j==9 else 0) for j,v in enumerate(params[start:start+3])] for start in (0,3,6,9,13)]
    extension=LossExtension(curve);extension.radiance(*inputs())
    # Force adjacent smooth extensions. A zero-residual-only implementation fails.
    extension.indices=[max(0,extension.indices[0]-1),max(0,extension.indices[1]-1)]
    smooth=extension.radiance(*inputs());delta=extension.correction_radiance(*inputs())
    assert any(not q[0,0].zero for q in delta)
    with localcontext() as ctx:
        ctx.prec=90;p=list(map(D.from_float,params));cos=sum(a*b for a,b in zip(p[:3],p[6:9]))/(sum(a*a for a in p[:3])*sum(a*a for a in p[6:9])).sqrt()
        reference=material_pair(data,p);moved=list(p);moved[9]+=1;other=material_pair(data,moved)
        for i,(a,b) in enumerate(zip(smooth,delta)):
            total=a+b;value=reference[i]*cos;derivative=(other[i]-reference[i])*cos
            assert total[0,0].lo<=value<=total[0,0].hi
            assert total[0,1].lo<=derivative<=total[0,1].hi
