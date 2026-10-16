"""Native quadrature geometry; shading and its acceptance remain separate."""
import math,torch
from contextvars import ContextVar
_batch_work=ContextVar("opaque_batch_work",default=None)
def _batch_limits(rule):
    if type(rule)!=int or rule not in (0,1,2,3,4):raise ValueError('opaque batch rule must be DEFAULT(0), SH_BILINEAR25(1) , SH_BILINEAR25_STREAM52(2) or SH_BILINEAR25_STREAM64(3)')
    calls=80 if rule==4 else (64 if rule==3 else (52 if rule==2 else 16))
    return calls,calls*32,calls*800

def _charge_batch(receipt):
    charge=_batch_work.get()
    if charge is not None:charge(receipt[7])

from ._common import check_vertices,check_faces,check_same_device,materialize,translate_native_error
class _Quadrature(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v,f,cuts,h,w,ox,oy,limit,transport,forward_limit):
        try:owners,samples=(torch.ops.gffx.opaque_material_quadrature if transport else torch.ops.gffx.opaque_quadrature)(v,f,cuts,h,w,ox,oy,forward_limit)
        except RuntimeError as e:
            translated=translate_native_error(e)
            if isinstance(translated,OverflowError):raise OverflowError('opaque quadrature: '+str(translated)) from None
            raise translated from None
        ctx.transport=transport;ctx.save_for_backward(v,f,cuts);ctx.options=(h,w,ox,oy,limit);ctx.mark_non_differentiable(owners);return owners,samples
    @staticmethod
    def backward(ctx,unused,g):
        if torch.is_grad_enabled():raise RuntimeError('opaque quadrature supports first derivatives only')
        v,f,c=ctx.saved_tensors
        try:gv,gc=(torch.ops.gffx.opaque_material_quadrature_backward if ctx.transport else torch.ops.gffx.opaque_quadrature_backward)(v,f,c,materialize(g),*ctx.options)
        except RuntimeError as e:raise translate_native_error(e) from None
        return gv,None,gc,None,None,None,None,None,None,None

class _Batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v,f,c,ox,oy,cursor,limit,rule):
        try:owners,samples,receipt=torch.ops.gffx.opaque_material_batch_rule(v,f,c,ox,oy,*cursor,limit,rule)
        except RuntimeError as e:raise translate_native_error(e) from None
        _charge_batch(receipt)
        ctx.save_for_backward(v,f,c);ctx.options=(ox,oy,*cursor,limit,rule);ctx.receipt=tuple(receipt)
        ctx.mark_non_differentiable(owners);return owners,samples,tuple(receipt)
    @staticmethod
    def backward(ctx,unused,g,unused_receipt):
        if torch.is_grad_enabled():raise RuntimeError('opaque material batch supports first derivatives only')
        v,f,c=ctx.saved_tensors
        try:gv,gc,receipt=torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,materialize(g),*ctx.options,((1 if ctx.needs_input_grad[0] else 0)|(2 if ctx.needs_input_grad[2] else 0)))
        except RuntimeError as e:raise translate_native_error(e) from None
        _charge_batch(receipt)
        if tuple(receipt[:7])!=ctx.receipt[:7]:raise RuntimeError('opaque batch replay layout changed')
        return gv,None,gc,None,None,None,None,None

def _checked(pixel_vertices,faces,cuts,height,width,*,origin=(0.,0.),workspace_limit_bytes=67108864,curves=None,edge=None,material_transport=False,batch_cursor=None,batch_rule=0,boundary_stream=False):
    """Return face owners[P,800] and barycentric/area samples[P,800,4].

    P=height*width; each pixel has32 charged blocks and800 node slots.
    Unused owners are -1 and samples zero. Material cuts[F,3,K] are homogeneous
    corner numerators, K<=16. Optional curves[F,6] gives one homogeneous
    barycentric quadratic per face: q00,q11,q22,doubled q01,q02,q12.
    A zero row explicitly disables the curve. Linear cuts plus the curve consume
    at most16 columns. Only strictly transverse radial conic patches are admitted;
    tangencies, vertex events and uncertified topology refuse. Both curved sides
    use Gauss12 angular by capacity-counted Gauss12orGauss5 radial nodes,
    consuming288or120slots and five of the existing32charged blocks together.
    After preserving that curved choice, flat fans use outerGauss5 and a shared
    inner order5through8 chosen from remaining800-slot capacity (25..40nodes).
    Backward freezes the unseeded per-pixel choices for all derivative seeds.
    This computes bounded visible-region quadrature;
    callers must separately qualify the shader's integration error. CPU/CUDA
    float32/float64 use the same ordered geometry arithmetic.
    """
    batch_cap,block_cap,slot_cap=_batch_limits(batch_rule)
    if batch_rule and (batch_cursor is None or curves is not None):raise ValueError("SH_BILINEAR25 requires a batch without curves")
    check_vertices(pixel_vertices,'pixel_vertices');check_faces(faces)
    if not isinstance(cuts,torch.Tensor):raise TypeError('cuts must be a tensor')
    if cuts.ndim!=3 or cuts.shape[:2]!=(len(faces),3) or cuts.shape[2]>(40 if batch_rule==4 else 16):raise ValueError('cuts require [F,3,K], K<=16')
    if cuts.dtype!=pixel_vertices.dtype:raise ValueError('cut precision must match vertices')
    check_same_device(pixel_vertices,faces,cuts)
    if curves is not None:
        if not isinstance(curves,torch.Tensor) or curves.shape!=(len(faces),6):raise ValueError('curves require [F,6]')
        if curves.dtype!=pixel_vertices.dtype:raise ValueError('curve precision must match vertices')
        check_same_device(pixel_vertices,curves)
        if cuts.shape[2]>=16:raise ValueError('combined linear/curved capacity16 exceeded')
        cuts=torch.cat((cuts.reshape(len(faces),3*cuts.shape[2]),curves),-1)[:,None,:]

    if type(height)!=int or type(width)!=int or not 1<=height<=1048576 or not 1<=width<=1048576:raise ValueError('quadrature dimensions')
    if type(workspace_limit_bytes)!=int or not 1<=workspace_limit_bytes<=67108864:raise ValueError('quadrature budget')
    if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('quadrature origin')
    ox,oy=map(float,origin)
    if not all(math.isfinite(v) and abs(v)<=1048576 for v in (ox,oy)):raise ValueError('quadrature origin range')
    reserve=1048576+3*(pixel_vertices.numel()*pixel_vertices.element_size()+cuts.numel()*cuts.element_size())+faces.numel()*faces.element_size()+height*width*800*(12*pixel_vertices.element_size()+4)
    if reserve>workspace_limit_bytes:raise ValueError('quadrature conservative workspace budget exceeded')
    forward_limit=workspace_limit_bytes
    if pixel_vertices.device.type=='cuda' and height*width>1 and edge is None and batch_cursor is None and reserve+3*1048576>workspace_limit_bytes:
        # Keep the admitted one-slab path. Bound only the forward adapter's
        # allowance below its optional four-slab threshold; backward retains
        # the original independently admitted budget and full output reserve.
        output=height*width*800*(4*pixel_vertices.element_size()+4)
        forward_limit=min(workspace_limit_bytes,4*1048576+output-1)

    if batch_cursor is not None:
        if not isinstance(batch_cursor,(tuple,list)) or len(batch_cursor)!=4 or any(type(x)!=int for x in batch_cursor):raise ValueError('batch cursor requires four integers')
        begin,batch,slots,blocks=batch_cursor
        if not 0<=begin<=block_cap or not 0<=batch<batch_cap or not 0<=slots<=batch*800 or not 0<=blocks<=batch*32 or begin>blocks or ((batch==0)!=(begin==0)):raise ValueError('invalid batch cursor')
        if boundary_stream:
            if type(edge)!=int or not 0<=edge<=3:raise ValueError('boundary edge must be0..3')
            if pixel_vertices.requires_grad or cuts.requires_grad:raise NotImplementedError('boundary nodes support forward-only camera flux')
            try:ids,q,receipt=(torch.ops.gffx.opaque_boundary_batch80 if batch_rule==4 else torch.ops.gffx.opaque_boundary_batch64)(materialize(pixel_vertices),materialize(faces),materialize(cuts),ox,oy,edge,*batch_cursor,workspace_limit_bytes)
            except RuntimeError as e:raise translate_native_error(e) from None
            from ._opaque_pixel import _advance
            receipt=tuple(receipt);_advance((tuple(batch_cursor),),(receipt,),rule=batch_rule)
            _charge_boundary64(receipt,rule=batch_rule)
            return ids,q,receipt
        return _Batch.apply(materialize(pixel_vertices),materialize(faces),materialize(cuts),ox,oy,tuple(batch_cursor),workspace_limit_bytes,batch_rule)
    if edge is not None:
        if type(edge)!=int or not 0<=edge<=3:raise ValueError('boundary edge must be0..3')
        if pixel_vertices.requires_grad or cuts.requires_grad:raise NotImplementedError('boundary nodes are forward-only; camera transport supports first derivatives')
        try:return torch.ops.gffx.opaque_boundary_quadrature(materialize(pixel_vertices),materialize(faces),materialize(cuts),height,width,ox,oy,edge,workspace_limit_bytes)
        except RuntimeError as e:raise translate_native_error(e) from None
    return _Quadrature.apply(materialize(pixel_vertices),materialize(faces),materialize(cuts),height,width,ox,oy,workspace_limit_bytes,material_transport,forward_limit)

def opaque_quadrature(pixel_vertices,faces,cuts,height,width,*,origin=(0.,0.),workspace_limit_bytes=67108864,curves=None):
    return _checked(pixel_vertices,faces,cuts,height,width,origin=origin,workspace_limit_bytes=workspace_limit_bytes,curves=curves)
opaque_quadrature.__doc__=_checked.__doc__

def opaque_boundary_quadrature(pixel_vertices,faces,cuts,height,width,*,edge,origin=(0.,0.),workspace_limit_bytes=67108864,curves=None):
    """Forward-only visible pixel-edge Gauss12 nodes; edge0/1/2/3 left/right/top/bottom.

    Same layouts and caps as opaque_quadrature. Fourth channel is edge-length
    weight. Up to32 final segments; no general nonlinear integration certificate.
    Coincident silhouettes/uncertain conic roots refuse; inputs must not require
    gradients. This primitive supplies first camera transport, not higher VJPs.
    """
    return _checked(pixel_vertices,faces,cuts,height,width,origin=origin,workspace_limit_bytes=workspace_limit_bytes,curves=curves,edge=edge)


def opaque_material_quadrature(pixel_vertices,faces,cuts,height,width,*,origin=(0.,0.),workspace_limit_bytes=67108864,curves=None):
    """Material-following weighted-integral first VJP with relative boundary flux.

    Same caps as opaque_quadrature. Zero-weight edge nodes MUST be shaded.
    This is not a raw-node derivative; use only for weighted material integrals.
    It uses35volume+9edge slots per flat fan when capacity permits, otherwise
    the existing raw rule. Explicit linear shading cuts always retain that raw
    interior rule: a value exactly on a discontinuity cannot supply both traces.
    No second derivatives or general accuracy guarantee.
    """
    return _checked(pixel_vertices,faces,cuts,height,width,origin=origin,workspace_limit_bytes=workspace_limit_bytes,curves=curves,material_transport=True)

def opaque_material_quadrature_batch(pixel_vertices,faces,cuts,*,cursor=(0,0,0,0),origin=(0.,0.),workspace_limit_bytes=67108864,curves=None,rule=0):
    """One explicit single-pixel batch, returning owners, samples and receipt.

    Cursor: (next_piece,batch_index,prefix_slots,prefix_blocks). Receipt:
    (next_piece,complete,slots,blocks,total_slots,total_blocks,pieces,clip_work).
    Start at zero; continue with (r[0],batch+1,r[4],r[5]). At most16batches,
    800slots/32blocks each;12800slots/512blocks total. Candidate cap32 for the default rule;512 for explicit SH_BILINEAR25.
    Caller binds unchanged inputs, counts work, preserves accumulation order
    and exposes no pixel before completion. This stateless low-level cursor is
    not an authentication token. Backward checks saved tensor versions and
    uses the same discrete layout. Zero-weight boundary nodes must be shaded.
    Same material-integral first-VJP semantics as opaque_material_quadrature;
    rule2 uses the same SH25 nodes with an explicit52-call/1664-block/41600-slot
    allowance. All profiles retain32blocks/800slots per call. Bind the same rule
    to every continuation; the stateless cursor is not authenticated.
    rule0 retains curved288 and flat44(no cuts)/40; rule1 selects the explicit
    SH_BILINEAR25 rule and refuses curves. Both require independent shader
    checks; selecting a rule is not an accuracy certificate.
    """
    return _checked(pixel_vertices,faces,cuts,1,1,origin=origin,workspace_limit_bytes=workspace_limit_bytes,curves=curves,batch_cursor=cursor,batch_rule=rule)


def _charge_boundary64(receipt,*,rule=3):
    work=_batch_work.get()
    if work is None:return
    work(receipt[7])
    if not hasattr(work,'sh_volume'):raise RuntimeError('SH boundary stream requires whole-pixel volume accounting')
    before=getattr(work,'sh_edges',(0,0,0));now=(before[0]+1,before[1]+receipt[2],before[2]+receipt[3]);work.sh_edges=now
    if any(a+b>cap for a,b,cap in zip(work.sh_volume,now,(_batch_limits(rule)[0],_batch_limits(rule)[2],_batch_limits(rule)[1]))):raise OverflowError('SH boundary and volume exceed shared whole-pixel budget')


def opaque_boundary_batch64(pixel_vertices,faces,cuts,*,edge,cursor=(0,0,0,0),origin=(0.,0.),workspace_limit_bytes=67108864):
    """Forward-only SH64 edge continuation;32 whole segments/800 slots per call.

    Caller binds immutable inputs. Scene use charges volume and all four edges
    to one64-call/2048-block/51200-slot/2M-work budget. Legacy edge API is unchanged.
    """
    return _checked(pixel_vertices,faces,cuts,1,1,origin=origin,workspace_limit_bytes=workspace_limit_bytes,edge=edge,batch_cursor=cursor,batch_rule=3,boundary_stream=True)


def opaque_boundary_batch80(pixel_vertices,faces,cuts,*,edge,cursor=(0,0,0,0),origin=(0.,0.),workspace_limit_bytes=67108864):
    """Forward-only SH80 boundary stream charged together with volume work."""
    return _checked(pixel_vertices,faces,cuts,1,1,origin=origin,workspace_limit_bytes=workspace_limit_bytes,edge=edge,batch_cursor=cursor,batch_rule=4,boundary_stream=True)
