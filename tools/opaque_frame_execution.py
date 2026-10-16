"""Host-owned bounded complete-frame execution through the installed public window API.

Disk storage, deadlines and cancellation live here, outside the runtime library.
Finite qualification driver: batch1 float64, fixed cotangents, canonical pixel VJP.
"""
from pathlib import Path
import hashlib,json,math,time
import torch
from tools.opaque_frame_store import Writer,Reader,CHANNELS,MAX_PIXELS

def _snapshot(scene,camera):
    from gffx.torch._opaque_stream import _snapshot as capture
    tensors=[];tree=capture((scene,camera,None),tensors,{})
    return tensors,repr(tree)

def _versions(tensors):
    return [(id(t),t._version,t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),t.requires_grad) for t in tensors]

def _guard(scene,camera,versions,structure):
    tensors,current=_snapshot(scene,camera)
    if current!=structure or _versions(tensors)!=versions:raise RuntimeError('frame input mutation or structure change')
    return tensors

def _input_hashes(tensors):
    rows=[]
    for t in tensors:
        h=hashlib.sha256()
        # Indexed flat chunks avoid a full contiguous copy of a strided input.
        for offset in range(0,t.numel(),4096):
            linear=torch.arange(offset,min(offset+4096,t.numel()),device=t.device);indices=[]
            for extent in reversed(t.shape):indices.append(linear%extent);linear=linear//extent
            values=t[tuple(reversed(indices))] if t.ndim else t.reshape(1)
            h.update(values.detach().cpu().contiguous().numpy().tobytes())
        rows.append(dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),sha256=h.hexdigest()))
    return rows

def _check(deadline,cancel):
    if time.perf_counter()>deadline or (cancel is not None and cancel()):raise TimeoutError('frame deadline or cancellation')

def execute(scene,camera,root,*,identity,cotangent=(.2,-.3,.7,.1),seconds=120,cancel=None,cull_mode=None,window_pixels=1,request_pixels=None,albedo_replay=False):
    from gffx.torch import render
    if cull_mode is not None and (type(cull_mode) is not int or cull_mode not in (render.CULL_NONE,render.CULL_BACK,render.CULL_FRONT)):raise ValueError('explicit supported cull mode required')
    if type(window_pixels) is not int or not 1<=window_pixels<=16:raise ValueError('bounded forward window pixels required')
    if request_pixels is not None and (type(request_pixels) is not int or not 1<=request_pixels<=256):raise ValueError('bounded host request pixels required')
    if type(albedo_replay) is not bool:raise ValueError('explicit boolean albedo replay profile required')
    if albedo_replay:
        from gffx.torch._opaque_albedo_replay import _profile
        target=_profile(scene,camera)
        tensors,_=_snapshot(scene,camera)
        active=[t for t in tensors if t.requires_grad]
        if len(active)!=1 or active[0] is not target:raise ValueError('albedo replay requires exactly one albedo gradient target')
    mode=render.CULL_NONE if cull_mode is None else cull_mode
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=120:raise ValueError('bounded frame deadline')
    if camera.batch_size!=1 or camera.fx.dtype!=torch.float64 or camera.image_height*camera.image_width>MAX_PIXELS:raise ValueError('finite batch1 float64 frame required')
    if len(cotangent)!=4 or any(not math.isfinite(x) for x in cotangent):raise ValueError('four finite fixed cotangents required')
    deadline=time.perf_counter()+seconds;tensors,structure=_snapshot(scene,camera);versions=_versions(tensors);input_hashes=_input_hashes(tensors);_check(deadline,cancel)
    pinned=dict(application=identity,inputs=input_hashes,structure_sha256=hashlib.sha256(structure.encode()).hexdigest(),profile='public-window-pixel-replay-v1')
    if cull_mode is not None:pinned.update(profile='public-window-pixel-replay-v2',cull_mode=mode)
    if window_pixels!=1:pinned.update(profile='public-window-pixel-replay-v3',cull_mode=mode,window_pixels=window_pixels)
    if request_pixels is not None:pinned.update(profile='public-window-pixel-replay-v4',cull_mode=mode,window_pixels=window_pixels,request_pixels=request_pixels)
    if albedo_replay:pinned.update(profile='public-window-albedo-replay-v5',cull_mode=mode,window_pixels=window_pixels,request_pixels=request_pixels or 256)
    w,h=camera.image_width,camera.image_height;writer=Writer(root,w,h,pinned,block_pixels=256);pending={k:bytearray() for k in CHANNELS};start=0;count=0
    import struct
    try:
        offset=0
        while offset<w*h:
            _check(deadline,cancel);_guard(scene,camera,versions,structure);y,x=divmod(offset,w)
            n=min(request_pixels or window_pixels,w-x,256-count)
            with torch.no_grad():out=render.opaque_window(scene,camera,(y,x,1,n),cull_mode=mode,**({'tile_pixels':window_pixels} if request_pixels is not None else {}))
            for k in ('colour','mask'):pending[k].extend(out[k].cpu().contiguous().numpy().tobytes())
            pending['gcolour'].extend(struct.pack('<3d',*cotangent[:3])*n);pending['gmask'].extend(struct.pack('<d',cotangent[3])*n);count+=n;offset+=n;del out
            if count==256 or offset==w*h:
                writer.append(start,count,{k:bytes(v) for k,v in pending.items()});start+=count;count=0;pending={k:bytearray() for k in CHANNELS}
        _check(deadline,cancel);_guard(scene,camera,versions,structure);receipt=writer.finish()
    finally:writer.close()
    return dict(**receipt,deadline_monotonic=deadline,identity=pinned,input_versions=versions,structure=structure,width=w,height=h,elapsed_seconds=120-(deadline-time.perf_counter()) if seconds==120 else seconds-(deadline-time.perf_counter()))

def replay(scene,camera,receipt,*,seconds=120,cancel=None):
    from gffx.torch import render
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=120:raise ValueError('bounded replay deadline')
    deadline=min(time.perf_counter()+seconds,receipt['deadline_monotonic']);tensors=_guard(scene,camera,receipt['input_versions'],receipt['structure']);_check(deadline,cancel)
    if _input_hashes(tensors)!=receipt['identity']['inputs']:raise RuntimeError('frame input bytes changed')
    profile=receipt['identity'].get('profile')
    if profile=='public-window-pixel-replay-v1' and 'cull_mode' not in receipt['identity']:mode=render.CULL_NONE
    elif profile in ('public-window-pixel-replay-v2','public-window-pixel-replay-v3'):
        if profile.endswith('v3') and (type(receipt['identity'].get('window_pixels')) is not int or not 2<=receipt['identity']['window_pixels']<=16):raise ValueError('invalid frame window identity')
        mode=receipt['identity'].get('cull_mode')
        if type(mode) is not int or mode not in (render.CULL_NONE,render.CULL_BACK,render.CULL_FRONT):raise ValueError('invalid frame cull identity')
    elif profile in ('public-window-pixel-replay-v4','public-window-albedo-replay-v5'):
        for key,cap in (('window_pixels',16),('request_pixels',256)):
            value=receipt['identity'].get(key)
            if type(value) is not int or not 1<=value<=cap:raise ValueError('invalid bounded host request identity')
        mode=receipt['identity'].get('cull_mode')
        if type(mode) is not int or mode not in (render.CULL_NONE,render.CULL_BACK,render.CULL_FRONT):raise ValueError('invalid frame cull identity')
    else:raise ValueError('unsupported frame visibility identity')
    inputs=[t for t in tensors if t.requires_grad];accumulated=[torch.zeros_like(t) for t in inputs]
    explicit=profile=='public-window-albedo-replay-v5'
    if explicit:
        from gffx.torch._opaque_albedo_replay import _profile
        target=_profile(scene,camera)
        if len(inputs)!=1 or inputs[0] is not target:raise ValueError('albedo replay target identity differs')
    reader=Reader(receipt['root'],receipt['manifest_sha256'],receipt['identity']);import numpy as np
    for offset,count,values in reader.blocks():
        if explicit:
            i=0
            while i<count:
                _check(deadline,cancel);_guard(scene,camera,receipt['input_versions'],receipt['structure']);y,x=divmod(offset+i,receipt['width'])
                n=min(count-i,receipt['width']-x,receipt['identity']['request_pixels'])
                cot={}
                for k,channels in (('colour',3),('mask',1)):
                    raw=values['g'+k][i*channels*8:(i+n)*channels*8]
                    shape=(1,1,n,3) if k=='colour' else (1,1,n)
                    cot[k]=target.new_tensor(np.frombuffer(raw,dtype='<f8').copy()).reshape(shape)
                out=render.opaque_albedo_replay(scene,camera,(y,x,1,n),cot['colour'],cot['mask'],initial_gradient=accumulated[0],cull_mode=mode)
                for k,channels in (('colour',3),('mask',1)):
                    if out[k].cpu().contiguous().numpy().tobytes()!=values[k][i*channels*8:(i+n)*channels*8]:raise ValueError('replayed frame pixels differ')
                accumulated[0]=out['albedo_gradient'];i+=n;del out,cot
            continue
        for i in range(count):
            _check(deadline,cancel);_guard(scene,camera,receipt['input_versions'],receipt['structure']);y,x=divmod(offset+i,receipt['width'])
            out=render.opaque_window(scene,camera,(y,x,1,1),cull_mode=mode);outputs=[];cotangents=[]
            for k,channels in (('colour',3),('mask',1)):
                saved=values[k][i*channels*8:(i+1)*channels*8]
                if out[k].detach().cpu().contiguous().numpy().tobytes()!=saved:raise ValueError('replayed frame pixels differ')
                if out[k].requires_grad:
                    g=values['g'+k][i*channels*8:(i+1)*channels*8];cot=out[k].new_tensor(np.frombuffer(g,dtype='<f8').copy()).reshape(out[k].shape);outputs.append(out[k]);cotangents.append(cot)
            gradients=torch.autograd.grad(outputs,inputs,cotangents,allow_unused=True) if outputs and inputs else [None]*len(inputs)
            for a,g in zip(accumulated,gradients):
                if g is not None:a.add_(g)
            del out,outputs,cotangents,gradients
    _check(deadline,cancel);_guard(scene,camera,receipt['input_versions'],receipt['structure']);return accumulated
