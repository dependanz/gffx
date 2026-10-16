"""Bounded complete curved-scene frames,ordered host storage and first VJP."""
import importlib.util,json,os,time,hashlib
from pathlib import Path
import torch

def test_complete_curved_frame_storage_replay_and_integrated_reference(tmp_path,record_property):
    from tools.opaque_frame_execution import execute,replay
    from gffx.torch import render
    spec=importlib.util.spec_from_file_location('curved_pixel_cases',Path(__file__).with_name('test_opaque_curved_pixels.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    size=int(os.environ['GFFX_FRAME_SIZE']);dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');s,c,n=m.fixture(size,dev)
    root=Path(os.environ.get('GFFX_FRAME_STORE_ROOT',str(tmp_path/'frame')))
    if dev=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();receipt=execute(s,c,root,identity={'fixture':'curved-horizon-single-triangle','size':size,'wheel':os.environ['GFFX_OPAQUE_WHEEL_SHA']});gradients=replay(s,c,receipt);seconds=time.perf_counter()-start
    record_property('curved_frame_execution_observed',json.dumps(dict(device=dev,size=size,measured_seconds=seconds,receipt=receipt,gradient=gradients[0].cpu().tolist())))
    resident=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=2)
    for key in ('colour','mask'):assert (root/(key+'.f64')).read_bytes()==resident[key].detach().cpu().numpy().tobytes()
    expected,=torch.autograd.grad((resident['colour']*n.new_tensor([.2,-.3,.7])).sum()+resident['mask'].sum()*.1,n)
    assert gradients[0].cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()
    ref=json.loads(Path(os.environ['GFFX_CURVED_PIXEL_REFERENCE']).read_bytes())
    for value in resident['colour'].sum((0,1,2)).tolist():m.bounds(value/(size*size),ref['result']['value'][0])
    assert abs(resident['mask'].sum().item()/(size*size)-.32)<1e-12
    # Normal-z integral derivative,with identical RGB channels and fixed cotangent.
    tangent=sum(float(x) for x in gradients[0][:,2])/(size*size*.6);m.bounds(tangent,ref['result']['derivative'][0])
    peak=None
    if dev=='cuda':torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();assert peak<67108864
    record_property('curved_frame',json.dumps(dict(receipt=receipt,device=dev,pixels=size*size,size=size,complete=True,render_executed=True,measured_seconds=seconds,exact_resident_pixels_and_vjp=True,frame_integral_matches_independent_reference=True,gradient=gradients[0].cpu().tolist(),cuda_peak_allocated_bytes=peak,scope='Complete finite curved-horizon scene and parameter gradient,not the10000-face original mesh or1080p qualification.')))
