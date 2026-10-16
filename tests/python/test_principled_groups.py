"""Approved renderer-internal grouping; original public/raw caps remain distinct."""
import pytest,torch
from test_principled_pairs_operation import fixture
from test_principled_native_operation import raw


def grouped_fixture(points,lights):
 budget,_,args=fixture();result=list(args)
 for k in range(2,12):
  index=torch.arange(points)%2
  result[k]=args[k][index].contiguous()
 result[8]=result[8].expand(points,lights,3).clone()
 return budget,result


def same(a,b):assert torch.equal(raw(a),raw(b))


@pytest.mark.parametrize('points,lights',[(1,1),(5,3),(7,5),(128,8)])
@pytest.mark.parametrize('derivatives',[False,True])
def test_grouped_native_matches_ordered_four_point_calls(points,lights,derivatives):
 _,args=grouped_fixture(points,lights)
 expected=[];partials=[]
 for left in range(0,points,4):
  current=args[:2]+[x[left:left+4] for x in args[2:]]
  y,j=torch.ops.gffx.shading_principled_pairs(*current,False,derivatives)
  expected.append(y);partials.append(j)
 actual,jac=torch.ops.gffx.shading_principled_groups(*args,False,derivatives)
 same(actual,torch.cat(expected))
 if derivatives:
  same(jac,torch.cat(partials));cot=(torch.arange(points*lights*3,dtype=torch.float64).reshape(points,lights,3)%13-6)/17
  pg,lg=torch.ops.gffx.shading_principled_groups_backward(jac,cot)
  rows=[torch.ops.gffx.shading_principled_pairs_backward(jac[i:i+4],cot[i:i+4]) for i in range(0,points,4)]
  same(pg,torch.cat([x[0] for x in rows]));same(lg,torch.cat([x[1] for x in rows]))
 else:assert jac.numel()==0


def test_grouped_native_cuda_exact_and_existing_caps():
 _,args=grouped_fixture(128,8);cot=(torch.arange(128*8*3,dtype=torch.float64).reshape(128,8,3)%17-8)/19
 y,j=torch.ops.gffx.shading_principled_groups(*args,False,True);pg,lg=torch.ops.gffx.shading_principled_groups_backward(j,cot)
 stream=torch.cuda.Stream()
 with torch.cuda.stream(stream):
  cy,cj=torch.ops.gffx.shading_principled_groups(*(x.cuda() for x in args),False,True)
  cp,cl=torch.ops.gffx.shading_principled_groups_backward(cj,cot.cuda())
 stream.synchronize()
 for a,b in zip((y,j,pg,lg),(cy,cj,cp,cl)):same(a,b)
 for device in ('cpu','cuda'):
  for points,lights in ((129,1),(128,9)):
   _,bad=grouped_fixture(points,lights)
   with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_groups(*(x.to(device) for x in bad),False,True)
  _,five=grouped_fixture(5,1)
  with pytest.raises(RuntimeError):torch.ops.gffx.shading_principled_pairs(*(x.to(device) for x in five),False,True)
