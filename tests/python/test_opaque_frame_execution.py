"""Installed public windows connected to a bounded complete-frame host store."""
import json,os,importlib.util,hashlib,time
from pathlib import Path
import torch,pytest

def fixture(size=3):
    spec=importlib.util.spec_from_file_location('window_cases',Path(__file__).with_name('test_opaque_window.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.fixture(size,os.environ.get('GFFX_KNOT_DEVICE','cpu'))

def test_frame_executor_refuses_mutation_cancellation_and_corrupt_storage(tmp_path):
    from tools.opaque_frame_execution import execute,replay
    from tools.opaque_frame_store import Reader
    s,c,v=fixture(1)
    with pytest.raises(TimeoutError):execute(s,c,tmp_path/'cancelled',identity={'test':'cancel'},cancel=lambda:True)
    assert not (tmp_path/'cancelled'/'frame.json').exists()
    receipt=execute(s,c,tmp_path/'frame',identity={'test':'mutation'})
    with torch.no_grad():v.add_(.00001)
    with pytest.raises(RuntimeError,match='input|mutation'):replay(s,c,receipt)
    s,c,v=fixture(1);receipt=execute(s,c,tmp_path/'corrupt',identity={'test':'corrupt'})
    p=Path(receipt['root'])/'colour.f64';raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='hash'):replay(s,c,receipt)

def test_complete_frame_execution_with_ordered_storage_and_vjp(tmp_path,record_property):
    from tools.opaque_frame_execution import execute,replay
    from gffx.torch import render
    import numpy as np
    size=int(os.environ.get('GFFX_FRAME_SIZE','3'));s,c,v=fixture(size);device=v.device.type
    root=Path(os.environ.get('GFFX_FRAME_STORE_ROOT',str(tmp_path/'frame')))
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    import sys
    plans=[]
    def trace(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='render' and frame.f_code.co_filename.endswith('_opaque_stream.py'):
            config=frame.f_locals.get('config')
            if config:
                row={'reserve_bytes':config['reserve_bytes'],**config['workload']}
                if row not in plans:plans.append(row)
    prior=sys.getprofile();sys.setprofile(trace);start=time.perf_counter()
    try:
        receipt=execute(s,c,root,identity={'fixture':'texture-roughness-square','size':size,'wheel':os.environ['GFFX_OPAQUE_WHEEL_SHA']});grads=replay(s,c,receipt);seconds=time.perf_counter()-start
    finally:sys.setprofile(prior)
    # Finite test only: compare all bytes to the existing resident public API.
    resident=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=2)
    for key in ('colour','mask'):assert (root/(key+'.f64')).read_bytes()==resident[key].detach().cpu().numpy().tobytes()
    expected,=torch.autograd.grad((resident['colour']*v.new_tensor([.2,-.3,.7])).sum()+resident['mask'].sum()*.1,v)
    assert grads[0].cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()
    peak=None
    if device=='cuda':torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();assert peak<67108864
    record_property('frame_execution',json.dumps(dict(receipt=receipt,plans=plans,device=device,extent=[size,size],pixels=size*size,complete=True,render_executed=True,measured_seconds=seconds,exact_resident_pixels_and_vjp=True,gradient=grads[0].cpu().tolist(),cuda_peak_allocated_bytes=peak,scope='Complete finite two-triangle textured Principled scene through public global-camera windows and host-owned disk storage; not original mesh or1080p qualification.')))
