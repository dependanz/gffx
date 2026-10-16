"""Independent ray/triangle coverage, separate from shared-native image buffers."""
import json,math,os,time,hashlib
from pathlib import Path
import pytest,torch


def test_independent_ray_mask_closed_square_and_culling():
    from tools.deferred_edge_oracle import hard_mask
    vertices=[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];faces=[[0,1,2],[0,2,3]]
    opts=dict(focal=4.,principal=(2.,2.),height=4,width=4)
    assert hard_mask(vertices,faces,**opts)==[[True]*4 for _ in range(4)]
    assert hard_mask(vertices,[f[::-1] for f in faces],**opts)==[[False]*4 for _ in range(4)]
    shifted=[[x+.5,y,z] for x,y,z in vertices]
    assert hard_mask(shifted,faces,**opts)==[[False,True,True,True] for _ in range(4)]


def test_deferred_fixed_head_independent_coverage_cpu_cuda(record_property):
    from tools.deferred_edge_oracle import hard_mask
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from gffx.torch import render
    entry=load_entry(Path(os.environ['GFFX_REFERENCE_CORPORA']),'synthetic_face_m1',allow_development=True)
    data=json.loads(entry['files']['scene.json']);deadline=time.perf_counter()+90
    def check():
        if time.perf_counter()>deadline:raise TimeoutError('independent head coverage deadline')
    expected=torch.tensor(hard_mask(data['vertices'],data['triangles'],focal=64/math.tan(math.radians(20)),principal=(64.,64.),height=128,width=128,checkpoint=check),dtype=torch.bool)
    assert bool(expected.any()) and not bool(expected.all())
    # Exact full mask implies exact silhouette edge membership, not a tolerant IoU.
    for device in ('cpu','cuda'):
        check()
        with torch.no_grad():
            model,camera=create_scene(data,'classic',device=device)
            actual=render(model,camera,outputs=('mask',))['mask'][0].cpu()
        assert torch.equal(actual,expected.to(actual.dtype))
        altered=actual.clone();altered[0,0]=1-altered[0,0]
        assert not torch.equal(altered,expected.to(actual.dtype))
    record_property('independent_head_coverage',json.dumps({'pixels':expected.numel(),'covered':int(expected.sum()),'oracle':'80-digit Decimal ray-plane plus 3D half-space membership','devices':['cpu','cuda'],'exact_masks':True,'oracle_sha256':hashlib.sha256(Path('tools/deferred_edge_oracle.py').read_bytes()).hexdigest(),'scene_sha256':hashlib.sha256(entry['files']['scene.json']).hexdigest()}))
