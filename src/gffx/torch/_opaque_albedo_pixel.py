"""Call-local fixed-albedo continuation with exact sample and batch carries."""
import torch


@torch.no_grad()
def replay(config,tensors,region,shared,cotangent):
    from . import _opaque_pixel as pixel
    from ._opaque_quadrature import _batch_work
    from ._opaque_stream import _restore_forward
    from ._opaque_scene import shade
    if config['shape'][0]!=1 or shared is None or 'constant_albedo' not in shared:raise ValueError('explicit one-pixel albedo continuation required')
    scene,camera,background=_restore_forward(config,tensors,shared)
    if background is not None:raise ValueError('explicit albedo continuation background')
    like=tensors[config['like']];rgba=like.new_zeros((1,4));gradient=like.new_zeros(3)
    cursors=((0,0,0,0),);versions=pixel._versions(tensors);prepared={};work=pixel._Work();token=_batch_work.set(work)
    try:
        for index in range(16):
            if pixel._versions(tensors)!=versions:raise RuntimeError('opaque pixel inputs modified during native replay')
            result=shade(scene,camera,outputs=('colour','mask'),cull_mode=config['cull'],background=None,workspace_limit_bytes=config['limit'],_region=region,_workload=config['workload'],_batch_cursors=cursors,_batch_first=index==0,_prepared_geometry=prepared,_shared_geometry=shared,_albedo_cotangent=cotangent)
            receipts=result['_receipts'];next_=pixel._advance(cursors,receipts)
            if '_albedo_gradient' not in result:
                # Retain charged work across the attempted batch and complete
                # canonical fallback. The old ctx keeps only its existing work
                # counter and original inputs; no prepared geometry is saved.
                del result,prepared
                with torch.enable_grad():return pixel.pixel(config,tensors,region,shared)
            values=result['_contributions'];g=result['_albedo_gradient']
            if not bool(torch.isfinite(values).all() & torch.isfinite(g).all()):raise ValueError('nonfinite native continuation contributions')
            receipt=receipts[0]
            if receipt is not None:rgba[0].copy_(torch.ops.gffx.shading_ordered_rgba(rgba[0],values[0,:receipt[2]].contiguous()))
            # Identical to the retained _Pixel.backward ascending batch carry.
            gradient.add_(g);cursors=next_;del result,values,g
            if all(c is None for c in cursors):break
        else:raise OverflowError('incomplete opaque pixel')
        if pixel._versions(tensors)!=versions:raise RuntimeError('opaque pixel inputs modified during native replay')
        if not bool(torch.isfinite(rgba).all() & torch.isfinite(gradient).all()):raise ValueError('nonfinite complete opaque pixel')
        return dict(colour=rgba[:,:3].reshape(1,1,1,3),mask=rgba[:,3].reshape(1,1,1),_albedo_gradient=gradient)
    finally:_batch_work.reset(token)
