"""Scoped orthographic classic SH composition over existing native quadrature.

No new visibility algorithm. Texture knots are exact affine fields, packed per
face; canonical one-pixel replay owns the same native capacities and ordering.
"""
import torch

def profile(scene,camera):
    from .scene import Material,Light
    from . import material,light
    mats=[p for p in scene.prims if type(p)==Material]
    lamps=[p for p in scene.prims if type(p)==Light]
    return camera.projection_kind=='orthographic' and bool(mats) and bool(lamps) and all(p.evaluator is material.classic for p in mats) and all(p.evaluator is light.spherical_harmonics for p in lamps)

def texture_cuts(packed,pixels,selected,region,*,compact=False,cut_limit=16):
    from ._deferred import TextureSource,AttributeSource
    from . import render
    v,f=packed.vertices,packed.faces;fields=[];masks=[]
    if cut_limit not in (16,40) or (cut_limit==40 and not compact):raise ValueError("40 texture cuts require compact storage")
    ids=selected.nonzero().flatten()
    if ids.numel()>512:raise OverflowError('opaque SH candidate-face capacity512 exceeded')
    if not ids.numel():return v.new_empty((len(ids) if compact else len(f),3,0))
    for number,mat in enumerate(packed.materials):
        active=packed.face_material[ids]==number
        spec=mat.parameters.get('specular')
        if not isinstance(spec,torch.Tensor) or bool((spec!=0).any()):raise ValueError('opaque SH requires zero specular')
        for name,source in mat.parameters.items():
            if isinstance(source,AttributeSource):raise NotImplementedError('opaque SH attribute material fields are not qualified')
            if not isinstance(source,TextureSource):continue
            if name!='albedo' or source.filter!=render.FILTER_BILINEAR or source.mip_filter!=render.MIP_NEAREST or source.lod is not None or source.wrap_u!=render.WRAP_BORDER or source.wrap_v!=render.WRAP_BORDER:
                raise NotImplementedError('opaque SH texture requires explicit bilinear level0 border sampling')
            uv=packed.attributes[source.uv].values[ids]
            for axis,size in enumerate((source.width,source.height)):
                field=uv[...,axis]
                # Full face range is conservative and cannot omit a texture knot.
                # Refuse capacity rather than sampling across unpartitioned knots.
                with torch.no_grad():
                    lo,hi=_pixel_field_bounds(field,pixels[f[ids].long(),:2],region)
                    range_active=active & (lo<=hi) & torch.isfinite(lo) & torch.isfinite(hi)
                    lo=torch.where(range_active,lo,torch.zeros_like(lo));hi=torch.where(range_active,hi,torch.zeros_like(hi))
                    first=(lo*size-.5).floor().to(torch.int64)+1
                    last=(hi*size-.5).ceil().to(torch.int64)-1
                    # Border sampling is constant outside these last tent knots.
                    first=first.clamp_min(-1);last=last.clamp_max(size)
                    count=torch.where(range_active,(last-first+1).clamp_min(0),0)
                    maximum=int(count.max()) if count.numel() else 0
                if maximum>cut_limit:raise OverflowError(f'opaque SH face texture-cut capacity{cut_limit} exceeded')
                for rank in range(maximum):
                    enabled=active & (count>rank)
                    threshold=(first+rank+.5)/size
                    if compact and len(fields)==80:raise OverflowError("compact texture construction field capacity80 exceeded")
                    fields.append(torch.where(enabled[:,None],field-threshold[:,None],torch.zeros_like(field)))
                    masks.append(enabled)
    # Columns belonging to distinct materials can share a slot: no face owns
    # two materials. Within each face retain every knot in deterministic order.
    if not fields:return v.new_empty((len(ids) if compact else len(f),3,0))
    counts=torch.stack(masks).sum(0)
    if bool((counts>cut_limit).any()):raise OverflowError(f'opaque SH face texture-cut capacity{cut_limit} exceeded')
    slots=int(counts.max())
    if compact:
        values=torch.stack(fields,-1);enabled=torch.stack(masks,-1)
        ranks=torch.arange(len(fields),device=v.device).expand_as(enabled)
        order=torch.argsort(torch.where(enabled,ranks,torch.full_like(ranks,len(fields))),dim=-1,stable=True)[...,:slots]
        small=torch.gather(values,2,order[:,None,:].expand(-1,3,-1))
        return small.contiguous()
    columns=[torch.zeros_like(fields[0]) for _ in range(slots)];index=torch.zeros_like(counts)
    for field,enabled in zip(fields,masks):
        for k in range(slots):columns[k]=columns[k]+torch.where((enabled & (index==k))[:,None],field,torch.zeros_like(field))
        index=index+enabled
    small=torch.stack(columns,-1)
    return v.new_zeros((len(f),3,slots)).index_copy(0,ids,small).contiguous()


def _pixel_field_bounds(field,tri,region):
    """Outward-rounded affine range on the pixel rectangle intersected with
    the face's range. Selection only; differentiable knot fields stay intact.
    Uncertain determinant falls back to the entire face, never omits a knot.
    """
    def down(x):return torch.nextafter(x,torch.full_like(x,-float('inf')))
    def up(x):return torch.nextafter(x,torch.full_like(x,float('inf')))
    def point(x):return (x,x)
    def add(a,b):return down(a[0]+b[0]),up(a[1]+b[1])
    def neg(a):return -a[1],-a[0]
    def sub(a,b):return add(a,neg(b))
    def mul(a,b):
        x=torch.stack((a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
        return down(x.amin(0)),up(x.amax(0))
    def div(a,b):return mul(a,(down(1/b[1]),up(1/b[0])))
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    x1=sub(point(b[:,0]),point(a[:,0]));y1=sub(point(b[:,1]),point(a[:,1]))
    x2=sub(point(c[:,0]),point(a[:,0]));y2=sub(point(c[:,1]),point(a[:,1]))
    d1=sub(point(field[:,1]),point(field[:,0]));d2=sub(point(field[:,2]),point(field[:,0]))
    det=sub(mul(x1,y2),mul(x2,y1));valid=(det[0]>0)|(det[1]<0)
    det=(torch.where(valid,det[0],torch.ones_like(det[0])),torch.where(valid,det[1],torch.ones_like(det[1])))
    dx=div(sub(mul(d1,y2),mul(d2,y1)),det);dy=div(sub(mul(x1,d2),mul(x2,d1)),det)
    oy,ox,h,w=region
    xx=sub((torch.full_like(field[:,0],ox),torch.full_like(field[:,0],ox+w)),point(a[:,0]))
    yy=sub((torch.full_like(field[:,0],oy),torch.full_like(field[:,0],oy+h)),point(a[:,1]))
    interval=add(point(field[:,0]),add(mul(dx,xx),mul(dy,yy)))
    lo=field.amin(-1);hi=field.amax(-1)
    finite=torch.isfinite(interval[0])&torch.isfinite(interval[1])&valid
    roughlo=torch.where(finite,torch.maximum(lo,interval[0]),lo);roughhi=torch.where(finite,torch.minimum(hi,interval[1]),hi)
    # Extreme values of an affine field on triangle/rectangle intersection
    # occur at original vertices, edge/rectangle intersections or contained
    # rectangle corners. Enclose all possible events; ambiguous arithmetic
    # retains the wider bound, never discards a contribution.
    lower=[];upper=[];uncertain=~finite
    def offer(value,possible):
        lower.append(torch.where(possible,value[0],torch.full_like(lo,float('inf'))))
        upper.append(torch.where(possible,value[1],torch.full_like(hi,-float('inf'))))
    for j in range(3):
        pos=tri[:,j];possible=(pos[:,0]>=ox)&(pos[:,0]<=ox+w)&(pos[:,1]>=oy)&(pos[:,1]<=oy+h)
        offer(point(field[:,j]),possible)
        other=(j+1)%3;p0=tri[:,j];p1=tri[:,other]
        for axis,limits in ((0,(ox,ox+w)),(1,(oy,oy+h))):
            d=sub(point(p1[:,axis]),point(p0[:,axis]));nonzero=(d[0]>0)|(d[1]<0)
            parallel=p1[:,axis]==p0[:,axis];uncertain|=~nonzero&~parallel
            dd=(torch.where(nonzero,d[0],torch.ones_like(lo)),torch.where(nonzero,d[1],torch.ones_like(lo)))
            for boundary in limits:
                t=div(sub(point(torch.full_like(lo,boundary)),point(p0[:,axis])),dd)
                value=add(point(field[:,j]),mul(t,sub(point(field[:,other]),point(field[:,j]))))
                position=add(point(p0[:,1-axis]),mul(t,sub(point(p1[:,1-axis]),point(p0[:,1-axis]))))
                low,high=(oy,oy+h) if axis==0 else (ox,ox+w)
                possible=nonzero&(t[0]<=1)&(t[1]>=0)&(position[0]<=high)&(position[1]>=low)
                offer(value,possible)
    orientation=det[0]>0
    for px in (ox,ox+w):
        for py in (oy,oy+h):
            possible=valid.clone()
            for j in range(3):
                p0=tri[:,j];p1=tri[:,(j+1)%3]
                ex=sub(point(p1[:,0]),point(p0[:,0]));ey=sub(point(p1[:,1]),point(p0[:,1]))
                xx=sub(point(torch.full_like(lo,px)),point(p0[:,0]));yy=sub(point(torch.full_like(lo,py)),point(p0[:,1]))
                side=sub(mul(ex,yy),mul(ey,xx))
                possible &= torch.where(orientation,side[1]>=0,side[0]<=0)
            value=add(point(field[:,0]),add(mul(dx,sub(point(torch.full_like(lo,px)),point(a[:,0]))),mul(dy,sub(point(torch.full_like(lo,py)),point(a[:,1])))))
            offer(value,possible)
    boundlo=torch.stack(lower).amin(0);boundhi=torch.stack(upper).amax(0)
    uncertain|=torch.isnan(boundlo)|torch.isnan(boundhi)
    return torch.where(uncertain,roughlo,torch.maximum(roughlo,boundlo)),torch.where(uncertain,roughhi,torch.minimum(roughhi,boundhi))


def _profile(config):
    calls=config.get('sh_batch_profile',16)
    if type(calls)!=int or calls not in (16,52,64,80):raise ValueError('SH batch profile must be16,52 or64')
    return calls,4 if calls==80 else (3 if calls==64 else (2 if calls==52 else 1))

class _Replay(torch.autograd.Function):
    """Explicit16,52 or64 serial calls per pixel, shared by integrated subcells.

    Every call retains32blocks/800slots. Whole-pixel totals follow the explicit profile:512,1664,2048 or2560blocks,
    12800,41600 or51200slots and2M clipping work across forward and first-VJP replay.
    """
    @staticmethod
    def forward(ctx,config,*tensors):
        from ._opaque_pixel import _Work,_advance,_versions
        from ._opaque_quadrature import _batch_work
        config=dict(config);batch_cap,rule=_profile(config)
        reuse=config.get('sh_reuse_partition',False)
        if rule==4 and config.get('sh_storage_profile')!=80:raise ValueError('SH80 requires compact-storage configuration')
        if type(reuse)!=bool or (reuse and rule not in (2,3,4)):raise ValueError('partition reuse requires explicit SH52 orSH64')
        ctx.config=config;ctx.save_for_backward(*tensors);ctx.set_materialize_grads(False);ctx.schedule=_Schedules(config['shape'][1]*config['shape'][2],config['shape'][0],80) if rule==4 else [];ctx.works=[];ctx.retained_pool=None;ctx.replay_failed=False
        versions=_versions(tensors)
        b,h,w=config['shape'];like=tensors[config['like']];rgb=like.new_zeros((b,h,w,3));mask=like.new_zeros((b,h,w));shared={}
        for y in range(h):
            for x in range(w):
                if reuse:
                    from ._opaque_partition import Pool
                    shared['partition_pool']=Pool(config['limit']-1048576-config['workload']['estimated_bytes'])
                grid=_whole_pixel(config,tensors,y,x,shared);gy,gx=grid;scale=gy*gx;schedule=[];work=_Work();ctx.works.append(work);slots=0;blocks=0;carry=like.new_zeros((b,4))
                token=_batch_work.set(work)
                try:
                    for dy in range(gy):
                        for dx in range(gx):
                            cursor=tuple((0,0,0,0) for _ in range(b))
                            while True:
                                if len(schedule)==batch_cap:raise OverflowError('opaque SH whole-pixel call capacity exceeded')
                                if _versions(tensors)!=versions:raise RuntimeError('opaque SH inputs modified during forward')
                                result=_cell(config,tensors,y,x,dy,dx,shared,grid,cursor);receipts=result['_receipts'];next_=_advance(cursor,receipts,rule=rule)
                                slots+=sum(r[2] for r in receipts if r is not None);blocks+=sum(r[3] for r in receipts if r is not None)
                                if slots>batch_cap*800 or blocks>batch_cap*32:raise OverflowError('opaque SH whole-pixel sample/block capacity exceeded')
                                schedule.append((dy,dx,grid,cursor,receipts))
                                values=result['_contributions']/scale
                                if not bool(torch.isfinite(values).all()):raise ValueError('nonfinite opaque SH contribution')
                                for frame,r in enumerate(receipts):
                                    if r is not None:carry[frame].copy_(torch.ops.gffx.shading_ordered_rgba(carry[frame],values[frame,:r[2]].contiguous()))
                                del values
                                del result
                                if all(c is None for c in next_):break
                                cursor=next_
                finally:_batch_work.reset(token)
                work.sh_volume=(len(schedule),slots,blocks)
                ctx.schedule.append(schedule)
                from ._opaque_stream import _restore_forward
                _,_,background=_restore_forward(config,tensors,shared)
                colour=carry[:,:3];alpha=carry[:,3]
                if background is not None:colour=colour+(1-alpha[:,None])*background.reshape(-1,3)
                if not bool(torch.isfinite(colour).all() & torch.isfinite(alpha).all()):raise ValueError('nonfinite complete opaque SH pixel')
                rgb[:,y,x]=colour;mask[:,y,x]=alpha
        if _versions(tensors)!=versions:raise RuntimeError('opaque SH inputs modified during forward')
        # One spatial pixel only: retain its already budgeted pool, never a frame
        # of pools. No extra tensor clones or derivative state are allocated here.
        if reuse and h*w==1 and any(ctx.needs_input_grad[1:]):ctx.retained_pool=shared['partition_pool']
        return rgb,mask
    @staticmethod
    def backward(ctx,gcolour,gmask):
        if torch.is_grad_enabled():raise RuntimeError('opaque SH supports first derivatives only')
        from ._opaque_quadrature import _batch_work
        if ctx.replay_failed:raise RuntimeError('opaque SH replay failed; discard the graph')
        config=ctx.config;saved=ctx.saved_tensors;needs=ctx.needs_input_grad[1:];ids=[i for i,n in enumerate(needs) if n]
        accum=[torch.zeros_like(t) if n else None for t,n in zip(saved,needs)];_,h,w=config['shape']
        for y in range(h):
            for x in range(w):
                pool=None
                if config.get('sh_reuse_partition',False):
                    from ._opaque_partition import Pool
                    pool=ctx.retained_pool;ctx.retained_pool=None
                    if pool is None:pool=Pool(config['limit']-1048576-config['workload']['estimated_bytes'])
                    else:
                        try:pool.begin_replay()
                        except Exception:
                            ctx.replay_failed=True
                            raise
                token=_batch_work.set(ctx.works[y*w+x])
                try:
                    for dy,dx,grid,cursor,receipt in ctx.schedule[y*w+x]:
                        ts=[t.detach().requires_grad_(n) for t,n in zip(saved,needs)]
                        with torch.enable_grad():
                            result=_cell(config,ts,y,x,dy,dx,{'partition_pool':pool} if pool is not None else None,grid,cursor)
                            if config.get('sh_reuse_partition',False):
                                same=tuple(None if r is None else tuple(r[:7]) for r in result['_receipts'])==tuple(None if r is None else tuple(r[:7]) for r in receipt)
                            else:same=result['_receipts']==receipt
                            if not same:raise RuntimeError('opaque SH replay layout changed')
                            outputs=[];cotangents=[]
                            for key,g in (('colour',gcolour),('mask',gmask)):
                                if g is not None and result[key].requires_grad:outputs.append(result[key]);cotangents.append(g[:,y:y+1,x:x+1].contiguous()/(grid[0]*grid[1]))
                            gs=torch.autograd.grad(outputs,[ts[i] for i in ids],cotangents,allow_unused=True) if outputs else [None]*len(ids)
                        for i,g in zip(ids,gs):
                            if g is not None:accum[i].add_(g)
                        del result,outputs,cotangents,gs,ts
                except Exception:
                    ctx.replay_failed=True
                    if pool is not None:pool.poisoned=True
                    raise
                finally:_batch_work.reset(token)
        return (None,*accum)

def _whole_pixel(config,tensors,y,x,shared):
    from ._opaque_stream import _restore_forward
    from ._opaque_scene import _project,_face_bounds,_select_bounds
    s,c,_=_restore_forward(config,tensors,shared)
    if 'whole' not in shared:
        packed,_,pixels,keep,_=_project(s,c,config['cull']);shared['whole']=(keep,_face_bounds(pixels,packed.faces),packed)
    keep,bounds,packed=shared['whole'];oy,ox=config['origin'];selected=_select_bounds(keep,bounds,ox+x,oy+y,1,1)
    if int(selected.sum())>512:raise OverflowError('opaque SH whole-pixel candidate-face capacity512 exceeded')
    # Fixed finite admission order; choose a rectangular layout that
    # retain every texture knot under the existing per-face capacity. This is
    # geometric metadata only, never renderer/reference-output based selection.
    for grid in ((1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,3),(4,4)):
        gy,gx=grid;key=('admission',grid)
        if key not in shared:
            _,_,px,_,_=_project(s,_scaled_camera(c,grid),config['cull']);shared[key]=(px,_face_bounds(px,packed.faces))
        px,cell_bounds=shared[key];admitted=True
        for dy in range(gy):
            for dx in range(gx):
                region=(gy*(oy+y)+dy,gx*(ox+x)+dx,1,1)
                chosen=_select_bounds(keep,cell_bounds,region[1],region[0],1,1)
                try:cuts=texture_cuts(packed,px,chosen,region,compact=_profile(config)[1]==4,cut_limit=40 if _profile(config)[1]==4 else 16);del cuts
                except OverflowError as error:
                    if str(error)!=f'opaque SH face texture-cut capacity{40 if _profile(config)[1]==4 else 16} exceeded':raise
                    admitted=False;break
            if not admitted:break
        if admitted:return grid
    raise OverflowError('opaque SH bounded rectangular cut admission exhausted')


def _scaled_camera(camera,grid):
    gy,gx=grid
    return type(camera).from_orthographic(camera.fx*gx,camera.fy*gy,camera.cx*gx,camera.cy*gy,camera.image_height*gy,camera.image_width*gx,camera.near,camera.far,pose=camera.pose)



def _cell(config,tensors,y,x,dy,dx,shared,grid,cursor):
    from ._opaque_stream import _restore_forward
    from ._opaque_scene import shade
    from ._deferred import _camera
    s,c,b=_restore_forward(config,tensors,None if torch.is_grad_enabled() else shared);c=_scaled_camera(c,grid)
    oy,ox=config['origin'];region=(grid[0]*(oy+y)+dy,grid[1]*(ox+x)+dx,1,1)
    # Forward geometry caches belong to the scaled camera only.
    cell_shared=None if shared is None or torch.is_grad_enabled() else shared.setdefault(('cells',grid),{})
    result=shade(s,c,outputs=('colour','mask'),cull_mode=config['cull'],background=b,workspace_limit_bytes=config['limit'],_region=region,_workload=config['workload'],_shared_geometry=cell_shared,_batch_cursors=cursor,_batch_rule=_profile(config)[1],_partition_pool=None if shared is None else shared.get('partition_pool'),_partition_key=(y,x,dy,dx),_batch_first=all(v is not None and v[1]==0 for v in cursor))
    return result

def render(config,tensors,outputs):
    # Public opaque dispatch must not promote this incomplete experiment.
    raise NotImplementedError(
        'orthographic classic SH opaque rendering is not qualified: '
        'texture-boundary tangents and whole-pixel capacity remain unresolved')


def _render_candidate(config,tensors,outputs):
    """Qualification-only entry; not a supported rendering profile."""
    rgb,mask=_Replay.apply(config,*tensors)
    return {key:dict(colour=rgb,mask=mask)[key] for key in outputs}


class _Schedules:
    """Fixed-width host metadata, charged before rendering. No frame of Python tuples."""
    def __init__(self,pixels,batch,calls):
        from array import array
        self.batch=batch;self.calls=calls;self.width=4+12*batch;self.count=0
        self.rows=array('i',[0])*(pixels*calls*self.width);self.lengths=array('i',[0])*pixels
        if self.rows.itemsize!=4:raise RuntimeError('schedule requires 32-bit host integers')
    def append(self,schedule):
        if self.count>=len(self.lengths) or len(schedule)>self.calls:raise OverflowError('SH80 replay schedule capacity')
        from array import array
        for i,(dy,dx,grid,cursors,receipts) in enumerate(schedule):
            row=[dy,dx,*grid]
            for cursor,receipt in zip(cursors,receipts):row.extend([-1]*12 if cursor is None else [*cursor,*receipt])
            if len(row)!=self.width:raise ValueError('SH80 schedule batch extent')
            start=(self.count*self.calls+i)*self.width;self.rows[start:start+self.width]=array('i',row)
        self.lengths[self.count]=len(schedule);self.count+=1
    def __len__(self):return self.count
    def __getitem__(self,pixel):
        if not 0<=pixel<self.count:raise IndexError(pixel)
        result=[]
        for i in range(self.lengths[pixel]):
            start=(pixel*self.calls+i)*self.width;row=self.rows[start:start+self.width];cursors=[];receipts=[]
            for b in range(self.batch):
                part=row[4+12*b:16+12*b];cursors.append(None if part[0]<0 else tuple(part[:4]));receipts.append(None if part[0]<0 else tuple(part[4:]))
            result.append((row[0],row[1],tuple(row[2:4]),tuple(cursors),tuple(receipts)))
        return result
