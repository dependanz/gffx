"""Private optional C11/Driver-API fused forward shading; no runtime compiler."""
import ctypes
import math
import sys
from pathlib import Path
import torch

_libraries={}
_missing=set()
_MICRO_FACE_THRESHOLD=1_000_000

def pack_indirect(environment,probes,irradiance,mirrors,like):
    from ._raster_lighting import _dfg
    cubes=[environment]+[p.environment for p in probes];infos=[];images=[];offset=0;metadata=[]
    for i,cube in enumerate(cubes):
        row=[0]*32
        if cube is not None:
            row[0]=len(cube.levels)
            if row[0]>15:raise ValueError('cubemap has too many levels')
            for level,atlas in enumerate(cube.atlases):
                value=atlas.permute(0,2,3,1).contiguous().reshape(-1,3);row[1+2*level]=offset;row[2+2*level]=atlas.shape[-1];images.append(value);offset+=len(value)
        infos.append(row)
        if i:
            p=probes[i-1];pos=torch.as_tensor(p.position,device=like.device,dtype=like.dtype)
            bounds=p.bounds if p.bounds is not None else ((0,0,0),(0,0,0));metadata.append(torch.cat((pos,like.new_tensor(bounds).flatten(),like.new_tensor([int(p.bounds is not None),0,0]))))
        else:metadata.append(like.new_zeros(12))
    cube_info=torch.tensor(infos,device=like.device,dtype=torch.int32);cube_images=torch.cat(images).flatten()if images else like.new_zeros(3)
    meta=torch.stack(metadata);sh=environment.irradiance.contiguous()if environment is not None else like.new_zeros((9,3))
    gi=like.new_zeros(16)
    if irradiance is not None:
        gi[:3]=like.new_tensor(irradiance.coefficients.shape[:3]);gi[3:6]=torch.as_tensor(irradiance.bounds[0],device=like.device,dtype=like.dtype);gi[6:9]=torch.as_tensor(irradiance.bounds[1],device=like.device,dtype=like.dtype);gi[9]=irradiance.normal_bias;gi[10]=int(irradiance.interpolation=="normal")
        gi=torch.cat((gi,irradiance.coefficients.flatten()))
    mi=[];mp=[];offset=0
    for descriptor,capture in mirrors:
        image=capture['image'];h,w=image.shape[:2];mi.append(torch.cat((capture['camera'].world_to_clip().flatten(),like.new_tensor([w,h,offset,0]))));mp.append(image.reshape(-1,3));offset+=h*w
    mirror_info=torch.stack(mi)if mi else like.new_empty((0,20));mirror_pixels=torch.cat(mp).flatten()if mp else like.new_empty(0)
    return tuple(t.contiguous()for t in (cube_info,cube_images,meta,sh,gi,mirror_info,mirror_pixels,_dfg(like)))

def indirect(points,normals,view,parameters,data,mirror_ids,classic=False):
    if points.dtype!=torch.float32 or not available(points.device) or not hasattr(_libraries[points.device.type],'gffx_raster_indirect_f32'):return None
    # Legacy glossy retains its separately declared approximation in the tensor fallback.
    if classic and bool(torch.as_tensor(parameters.get('specular',0)).any()):return None
    count=len(points);fields=points.new_zeros((count,12));fields[:,:3]=parameters.get('albedo'if classic else'base_color',0)
    for key,index,default in [('roughness',3,.3),('metallic',4,0),('ior',5,1.5),('coat',6,0),('coat_roughness',7,.12)]:
        value=torch.as_tensor(parameters.get(key,default),device=points.device,dtype=points.dtype);fields[:,index]=value.reshape(-1)if value.ndim else value
    ci,images,meta,sh,gi,mi,mp,dfg=data
    tensors=[t.contiguous()for t in (points,normals,view,fields,ci,images,meta,sh,gi,mi,mp,mirror_ids,dfg)];output=torch.empty_like(points)
    fn=_libraries[points.device.type].gffx_raster_indirect_f32
    if points.is_cuda:
        with torch.cuda.device(points.device):
            stream=torch.cuda.current_stream(points.device);code=fn(1,count,len(ci),len(mi),int(classic),*[t.data_ptr()for t in tensors],output.data_ptr(),stream.cuda_stream)
            for t in [*tensors,output]:t.record_stream(stream)
    else:code=fn(0,count,len(ci),len(mi),int(classic),*[t.data_ptr()for t in tensors],output.data_ptr(),None)
    if code:raise RuntimeError('native image lighting failed: '+str(code))
    return output

def available(device='cpu'):
    device=torch.device(device).type
    if device in _libraries:return True
    if device in _missing:return False
    stem='gffx_raster_cuda' if device=='cuda' else 'gffx_raster'
    name=stem+'.dll' if sys.platform=='win32' else 'lib'+stem+('.dylib' if sys.platform=='darwin' else '.so')
    path=Path(__file__).parent.parent/name
    if not path.is_file():_missing.add(device);return False
    library=ctypes.CDLL(str(path))
    fn=library.gffx_raster_shade_v2_f32
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*11
    fn.restype=ctypes.c_int
    if device=='cuda':
        fn=library.gffx_raster_binned_f32
        fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*8
        fn.restype=ctypes.c_int
    if device=='cuda':
        fn=library.gffx_raster_shadow_bins_f32
        fn.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*7
        fn.restype=ctypes.c_int
    if hasattr(library,'gffx_raster_indirect_f32'):
        fn=library.gffx_raster_indirect_f32;fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*15;fn.restype=ctypes.c_int
    if device=='cuda' and hasattr(library,'gffx_raster_micro_f32'):
        fn=library.gffx_raster_micro_f32;fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*9;fn.restype=ctypes.c_int
    _libraries[device]=library
    return True

def pack_sources(sources,like,size):
    # Pack host scalar metadata once and stack device vectors in bulk. Thousands
    # of tiny host-to-device assignments used to serialize every area map.
    rows=[];entries=[];zero=like.new_zeros(3)
    vectors={key:[] for key in ['position','intensity','normal','spot','center','u','v','radiance']}
    for source in sources:
        row=[0.]*40;row[0]={'point':0,'spot':1,'sun':2}[source['kind']]
        row[10]=int('normal' in source);row[16]=len(entries);row[17]=len(source['maps'])
        if source['kind']=='spot':
            row[14]=math.cos(source['angle']/2);row[15]=row[14]+(1-row[14])*source['blend']
        area=source.get('area',{})
        if area:
            row[19]=1 if source['sample_index']==0 else 2;row[32]=len(entries);row[33]=area['samples']**2 if source['maps'] else 0;row[34]=area['samples']
            if source['maps'] and 'stratum' not in source['maps'][0]:
                from ._raster import _attach_stratum_meta
                _attach_stratum_meta(source['maps'][0],area,source['sample_index'])
        for key,value in [('position',source.get('position',source.get('direction',zero))),('intensity',source.get('intensity',source.get('irradiance',zero))),('normal',source.get('normal',zero)),('spot',source.get('direction',zero)),('center',area.get('center',zero)),('u',area.get('u',zero)),('v',area.get('v',zero)),('radiance',area.get('radiance',zero))]:vectors[key].append(value)
        rows.append(row);entries.extend(source['maps'])
    light_data=like.new_tensor(rows).reshape(-1,40)
    for key,start in [('position',1),('intensity',4),('normal',7),('spot',11),('center',20),('u',23),('v',26),('radiance',29)]:
        if rows:light_data[:,start:start+3]=torch.stack(vectors[key])
    if entries:
        meta=like.new_tensor([[e['camera'].near,int(e['inverse_depth'])] for e in entries])
        projection=torch.stack([e['projection'] for e in entries]);origins=torch.stack([e['origin'] for e in entries])
        zero_s=like.new_zeros(6)
        stratum=torch.stack([e.get('stratum',zero_s).to(like.dtype) for e in entries])
        maps=torch.cat((torch.stack([e['matrix'] for e in entries]).flatten(1),torch.stack([e['inverse'] for e in entries]).flatten(1),torch.stack([e['view'] for e in entries]).flatten(1),projection[:,0,0,None],projection[:,1,1,None],meta,origins,stratum,like.new_zeros((len(entries),3))),1)
        depths=[e['depth'] for e in entries];first=depths[0];storage=first.untyped_storage().data_ptr();start=first.storage_offset()
        shared=all(d.untyped_storage().data_ptr()==storage and d.storage_offset()==start+i*size*size for i,d in enumerate(depths))
        depth_data=first.as_strided((len(depths),size,size),(size*size,size,1)) if shared else torch.stack(depths)
    else:maps=like.new_empty((0,64));depth_data=like.new_empty((0,size,size))
    from ._ltc import table
    return light_data,maps,depth_data,table(like)


def shade(points,normals,geometric,view,parameters,sources,size,classic=False,curvature=None):
    """One stream-ordered launch; all tensors keep ownership until it is queued."""
    if points.dtype!=torch.float32 or not available(points.device):return None
    count=len(points)
    if not count:return torch.zeros_like(points)
    fields=points.new_zeros((count,12))
    def assign(key,start,width,default):
        value=torch.as_tensor(parameters.get(key,default),device=points.device,dtype=points.dtype)
        if width==1:fields[:,start]=value.reshape(-1) if value.ndim else value
        else:fields[:,start:start+width]=value[:,None] if value.ndim==1 and value.shape==(count,) and width==3 and key=='specular' else value
    if classic:
        assign('albedo',0,3,0);assign('specular',8,3,0);assign('exponent',11,1,1)
        kind=2 if parameters.get('two_sided',False) else 1
    else:
        assign('base_color',0,3,0)
        for key,index,default in [('roughness',3,.3),('metallic',4,0),('ior',5,1.5),('coat',6,0),('coat_roughness',7,.12)]:assign(key,index,1,default)
        kind=0
    if curvature is not None:fields[:,3 if classic else 8]=torch.as_tensor(curvature,device=points.device,dtype=points.dtype)
    tensors=[x.contiguous() for x in [points,normals,geometric,view,fields,*sources]]
    if any(t.device!=points.device or t.dtype!=points.dtype for t in tensors):raise ValueError('fused raster inputs must share float32 dtype and device')
    output=torch.empty_like(points)
    if points.is_cuda:
        with torch.cuda.device(points.device):
            stream=torch.cuda.current_stream(points.device)
            result=_libraries[points.device.type].gffx_raster_shade_v2_f32(1,count,len(sources[0]),size,kind,*[t.data_ptr() for t in tensors],output.data_ptr(),stream.cuda_stream)
            # Allocator lifetime follows the same stream as the external launch.
            for t in [*tensors,output]:t.record_stream(stream)
    else:result=_libraries[points.device.type].gffx_raster_shade_v2_f32(0,count,len(sources[0]),size,kind,*[t.data_ptr() for t in tensors],output.data_ptr(),None)
    if result:raise RuntimeError('gffx native raster shading failed: '+str(result))
    return output


def raster(ndc,faces,height,width,cull):
    """Hard single-layer raster with exact sorted conservative candidate bins."""
    if ndc.device.type!='cuda' or ndc.dtype!=torch.float32 or not available(ndc.device):return None
    if len(faces)>=_MICRO_FACE_THRESHOLD:return _micro(ndc,faces,height,width,cull,False)
    tiles_x=(width+15)//16;tiles_y=(height+15)//16
    ndc=ndc.contiguous();faces=faces.contiguous()
    counts=torch.empty(tiles_x*tiles_y,dtype=torch.int32,device=ndc.device)
    fn=_libraries['cuda'].gffx_raster_shadow_bins_f32
    with torch.cuda.device(ndc.device):
        stream=torch.cuda.current_stream(ndc.device)
        code=fn(0,len(faces),height,width,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),None,None,None,stream.cuda_stream)
        if code:raise RuntimeError('primary bin counts failed: '+str(code))
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0).to(torch.int32)))
        total=int(offsets[-1]);candidates=torch.empty(total,dtype=torch.int32,device=ndc.device)
        code=fn(2,len(faces),height,width,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),offsets.data_ptr(),candidates.data_ptr(),None,stream.cuda_stream)
        for t in [ndc,faces,counts,offsets,candidates]:t.record_stream(stream)
        if code:raise RuntimeError('primary bin fill failed: '+str(code))
    tile=torch.repeat_interleave(torch.arange(tiles_x*tiles_y,device=ndc.device),offsets[1:]-offsets[:-1],output_size=total)
    keys=tile*max(1,len(faces))+candidates
    order=torch.argsort(keys);candidates=candidates[order].contiguous()
    del tile,keys,order,counts
    indices=torch.empty((height*width,),device=ndc.device,dtype=torch.int32)
    bary=ndc.new_empty((height*width,3));depth=ndc.new_empty(height*width)
    tensors=[t.contiguous() for t in [ndc,faces,offsets,candidates,indices,bary,depth]]
    with torch.cuda.device(ndc.device):
        stream=torch.cuda.current_stream(ndc.device)
        code=_libraries['cuda'].gffx_raster_binned_f32(height,width,cull,*[t.data_ptr()for t in tensors],stream.cuda_stream)
        for t in tensors:t.record_stream(stream)
    if code:raise RuntimeError('gffx binned raster failed: '+str(code))
    return indices,bary,depth

def shadow(ndc,faces,height,width):
    """Depth-only native bins; no sort or full face/barycentric output buffers."""
    if ndc.device.type!='cuda' or ndc.dtype!=torch.float32 or not available(ndc.device):return None
    if len(faces)>=_MICRO_FACE_THRESHOLD:return _micro(ndc,faces,height,width,1,True)
    depth=ndc.new_empty((height,width));ndc=ndc.contiguous();faces=faces.contiguous()
    counts=torch.empty(((width+15)//16)*((height+15)//16),dtype=torch.int32,device=ndc.device)
    with torch.cuda.device(ndc.device):
        stream=torch.cuda.current_stream(ndc.device);fn=_libraries['cuda'].gffx_raster_shadow_bins_f32
        code=fn(0,len(faces),height,width,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),None,None,None,stream.cuda_stream)
        if code:raise RuntimeError('gffx shadow bin counts failed: '+str(code))
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0).to(torch.int32)))
        total=int(offsets[-1]);candidates=torch.empty(total,dtype=torch.int32,device=ndc.device)
        code=fn(1,len(faces),height,width,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),offsets.data_ptr(),candidates.data_ptr(),depth.data_ptr(),stream.cuda_stream)
        for t in [ndc,faces,depth,counts,offsets,candidates]:t.record_stream(stream)
    if code:raise RuntimeError('gffx depth-only shadow raster failed: '+str(code))
    return depth

def _micro(ndc,faces,h,w,cull,shadow_only):
    ndc=ndc.contiguous();faces=faces.contiguous();buckets=h*w+((w+15)//16)*((h+15)//16)
    counts=torch.empty(buckets,dtype=torch.int32,device=ndc.device);fn=_libraries['cuda'].gffx_raster_micro_f32
    with torch.cuda.device(ndc.device):
        stream=torch.cuda.current_stream(ndc.device)
        code=fn(0,len(faces),h,w,cull,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),None,None,None,None,None,stream.cuda_stream)
        if code:raise RuntimeError('micro bin counts failed: '+str(code))
        prefix=counts.cumsum(0);total=int(prefix[-1])
        if total>torch.iinfo(torch.int32).max:raise OverflowError('native raster candidate offsets exceed int32')
        offsets=torch.cat((counts.new_zeros(1),prefix.to(torch.int32)));del prefix
        candidates=torch.empty(total,dtype=torch.int32,device=ndc.device)
        code=fn(1,len(faces),h,w,cull,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),offsets.data_ptr(),candidates.data_ptr(),None,None,None,stream.cuda_stream)
        if code:raise RuntimeError('micro bin fill failed: '+str(code))
        if not shadow_only:
            bucket=torch.repeat_interleave(torch.arange(buckets,device=ndc.device),offsets[1:]-offsets[:-1],output_size=total)
            order=torch.argsort(bucket*max(1,len(faces))+candidates);candidates=candidates[order].contiguous();del bucket,order
        depth=ndc.new_empty(h*w);indices=torch.empty(h*w,dtype=torch.int32,device=ndc.device)if not shadow_only else None;bary=ndc.new_empty(h*w,3)if not shadow_only else None
        code=fn(3 if shadow_only else 2,len(faces),h,w,cull,ndc.data_ptr(),faces.data_ptr(),counts.data_ptr(),offsets.data_ptr(),candidates.data_ptr(),indices.data_ptr()if indices is not None else None,bary.data_ptr()if bary is not None else None,depth.data_ptr(),stream.cuda_stream)
        for tensor in [ndc,faces,counts,offsets,candidates,depth,indices,bary]:
            if tensor is not None:tensor.record_stream(stream)
    if code:raise RuntimeError('micro raster draw failed: '+str(code))
    return depth.reshape(h,w)if shadow_only else(indices,bary,depth)
