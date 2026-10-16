"""Research shared CPU/CUDA byte controls; not installed-wheel qualification."""
import importlib.util,os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize('curved',[False,True])
def test_streamed_cuda_matches_cpu_samples_and_first_derivatives(curved):
    from tools.opaque_batch_cuda_probe import Probe
    spec=importlib.util.spec_from_file_location('batch_geometry_tests',Path(__file__).with_name('test_opaque_batch_geometry.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if curved:
        vertices=[]
        for i in range(7):
            x=(i%3)*.25+.0625;y=(i//3)*.25+.0625;vertices.extend([[x,y,0],[x+.125,y,0],[x,y+.125,0]])
        v=np.array(vertices);f=np.arange(21,dtype=np.int32).reshape(7,3);c=np.tile([-.25,.75,.65,.5,.5,1.5],(7,1))
    else:v,f,c=m.wedge()
    gpu=Probe()
    try:
        for cursor,ids,q,r in m.batches(v,f,c):
            ii,qq,rr,_=gpu.call(v,f,c,cursor)
            assert rr[-1]==0 and rr[:8].tolist()==r
            assert ii.tobytes()==ids.tobytes() and qq.tobytes()==q.tobytes()
            g=(np.arange(3200,dtype=np.float64).reshape(1,800,4)-1700)/4096
            for seed in range(v.size):
                code,value,expected=m.derivative(v,f,c,g,cursor=cursor,vertex_seed=seed);assert code==0
                _,_,receipt,actual=gpu.call(v,f,c,cursor,seed=seed,gradient=g)
                assert receipt[-1]==0 and receipt[:8].tolist()==expected
                assert actual.tobytes()==np.array([value]).tobytes()
            if curved:
                for seed in range(c.size):
                    code,value,expected=m.derivative(v,f,c,g,cursor=cursor,cut_seed=seed);assert code==0
                    _,_,receipt,actual=gpu.call(v,f,c,cursor,cut_seed=seed,gradient=g)
                    assert receipt[-1]==0 and receipt[:8].tolist()==expected
                    assert actual.tobytes()==np.array([value]).tobytes()
    finally:gpu.close()
