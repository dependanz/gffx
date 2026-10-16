"""One complete overflowing pixel with bounded, serial first-VJP replay.

Only the streaming dispatcher invokes this after a native quadrature refusal.
No partial pixel escapes. Host supervision owns the unchanged worker deadline;
all failed, counting, forward and reverse calls run inside that same worker.
"""
import torch
from ._opaque_quadrature import _batch_work

class _Work:
    def __init__(self):self.value=0
    def __call__(self,value):
        self.value+=value
        if self.value>2000000:raise OverflowError('opaque pixel aggregate clipping work exhausted')

def _versions(tensors):return tuple(t._version for t in tensors)

def _evaluate(config,tensors,region,cursors,first,prepared=None,shared=None):
    from ._opaque_stream import _restore_forward
    from ._opaque_scene import shade
    s,c,b=_restore_forward(config,tensors,shared)
    return shade(s,c,outputs=('colour','mask'),cull_mode=config['cull'],background=b,workspace_limit_bytes=config['limit'],_region=region,_workload=config['workload'],_batch_cursors=cursors,_batch_first=first,_prepared_geometry=prepared,_shared_geometry=shared)

def _advance(cursors,receipts,*,rule=0):
    from ._opaque_quadrature import _batch_limits
    batch_cap,block_cap,slot_cap=_batch_limits(rule)
    if len(cursors)!=len(receipts):raise RuntimeError('opaque batch receipt count')
    result=[]
    for cursor,r in zip(cursors,receipts):
        if cursor is None:
            if r is not None:raise RuntimeError('completed opaque frame resumed')
            result.append(None);continue
        if len(r)!=8 or any(type(v)!=int for v in r):raise RuntimeError('opaque batch receipt type')
        begin,index,slots,blocks=cursor
        if any(type(v)!=int for v in cursor) or not 0<=index<batch_cap or not 0<=begin<=block_cap or not 0<=slots<=index*800 or not 0<=blocks<=index*32:raise RuntimeError('opaque batch cursor profile mismatch')
        if r[1] not in (0,1) or not 0<=r[2]<=800 or not 0<=r[3]<=32 or not 0<=r[6]<=r[3] or r[0]!=begin+r[6] or r[4]!=slots+r[2] or r[5]!=blocks+r[3] or r[4]>slot_cap or r[5]>block_cap or r[7]<0 or ((r[2]==0)!=(r[6]==0)) or ((r[3]==0)!=(r[6]==0)):
            raise RuntimeError('opaque batch receipt prefix/extent mismatch')
        if not r[1] and (index==batch_cap-1 or not r[6]):raise OverflowError('opaque pixel finite continuation exhausted')
        result.append(None if r[1] else (r[0],index+1,r[4],r[5]))
    return tuple(result)

class _Pixel(torch.autograd.Function):
    @staticmethod
    def forward(ctx,config,region,shared,*tensors):
        from ._opaque_stream import _restore_forward
        ctx.set_materialize_grads(False);ctx.save_for_backward(*tensors)
        ctx.config=config;ctx.region=region;ctx.receipts=[];ctx.work=_batch_work.get() or _Work()
        like=tensors[config['like']];batch=config['shape'][0];carry=like.new_zeros((batch,4))
        cursors=tuple((0,0,0,0) for _ in range(batch));versions=_versions(tensors);prepared={}
        token=_batch_work.set(ctx.work)
        try:
            for index in range(16):
                if _versions(tensors)!=versions:raise RuntimeError('opaque pixel inputs modified during forward')
                result=_evaluate(config,tensors,region,cursors,index==0,prepared,shared);receipts=result['_receipts'];next_=_advance(cursors,receipts)
                values=result['_contributions']
                if not bool(torch.isfinite(values).all()):raise ValueError('nonfinite opaque batch contributions')
                # Carry individual samples, in native order. No batch-subtotal
                # reassociation, threshold or dropping zero-weight boundary nodes.
                for b,r in enumerate(receipts):
                    if r is not None:
                        carry[b].copy_(torch.ops.gffx.shading_ordered_rgba(carry[b],values[b,:r[2]].contiguous()))
                ctx.receipts.append((cursors,receipts));cursors=next_
                del result,values
                if all(c is None for c in cursors):break
            else:raise OverflowError('incomplete opaque pixel')
            if _versions(tensors)!=versions:raise RuntimeError('opaque pixel inputs modified during forward')
            rgb=carry[:,:3].reshape(batch,1,1,3);mask=carry[:,3].reshape(batch,1,1)
            _,_,background=_restore_forward(config,tensors,shared)
            if background is not None:rgb=rgb+(1-mask[...,None])*background.reshape(-1,1,1,3)
            if not bool(torch.isfinite(rgb).all() & torch.isfinite(mask).all()):raise ValueError('nonfinite complete opaque pixel')
            return rgb,mask
        finally:_batch_work.reset(token)
    @staticmethod
    def backward(ctx,gcolour,gmask):
        if torch.is_grad_enabled():raise RuntimeError('opaque pixel supports first derivatives only')
        saved=ctx.saved_tensors;needs=ctx.needs_input_grad[3:];ids=[i for i,n in enumerate(needs) if n]
        accumulated=[torch.zeros_like(saved[i]) if i in ids else None for i in range(len(saved))]
        token=_batch_work.set(ctx.work)
        try:
            for index,(cursors,receipts) in enumerate(ctx.receipts):
                temporary=[v.detach().requires_grad_(n) for v,n in zip(saved,needs)]
                with torch.enable_grad():
                    result=_evaluate(ctx.config,temporary,ctx.region,cursors,index==0)
                    if result['_receipts']!=receipts:raise RuntimeError('opaque pixel replay layout changed')
                    outputs=[];cotangents=[]
                    for name,g in (('colour',gcolour),('mask',gmask)):
                        if g is not None and result[name].requires_grad:outputs.append(result[name]);cotangents.append(g.contiguous())
                    gradients=torch.autograd.grad(outputs,[temporary[i] for i in ids],cotangents,allow_unused=True) if outputs else [None]*len(ids)
                for i,g in zip(ids,gradients):
                    if g is not None:accumulated[i].add_(g)
                del temporary,result,outputs,cotangents,gradients
            return (None,None,None,*accumulated)
        finally:_batch_work.reset(token)

def pixel(config,tensors,region,shared=None):
    rgb,mask=_Pixel.apply(config,region,shared,*tensors)
    return {'colour':rgb,'mask':mask}
