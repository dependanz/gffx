"""Bounded, hash-pinned conversion of the selected local FLAME data.

Legacy class names map to inert state holders; no Chumpy/SciPy code is loaded.
This is a narrow converter for approved files under an external memory watchdog,
not a general-purpose unpickler or a security boundary for arbitrary uploads.
"""
import hashlib,math,pickle,zipfile,io,struct
from pathlib import Path
import numpy as np

class _LegacyState:
    def __setstate__(self,state):
        if not isinstance(state,dict):raise ValueError('legacy dictionary state required')
        self.state=state

class _ArrayState(_LegacyState):pass
class _SparseState(_LegacyState):pass

class _StorageReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch' and name in ('LongStorage','FloatStorage'):return name
        raise ValueError('unsupported nested storage global')
    def persistent_load(self,pid):
        if not isinstance(pid,tuple) or len(pid)!=6:raise ValueError('invalid storage descriptor')
        kind,dtype,key,device,count,view=pid
        if kind!='storage' or dtype not in ('LongStorage','FloatStorage') or not isinstance(key,str) or device!='cpu' or type(count)!=int or not 0<count<=100000 or view is not None:raise ValueError('unsupported storage descriptor')
        return pid

def _storage_bytes(blob):
    # Parse the selected legacy CPU storage envelope as data; never call torch.load
    # or its unrestricted _load_from_bytes nested pickle loader.
    if not isinstance(blob,bytes) or len(blob)>1024**2:raise ValueError('storage byte cap')
    f=io.BytesIO(blob)
    try:
        fields=[_StorageReader(f).load() for _ in range(5)]
        magic,version,system,descriptor,keys=fields
        if magic!=119547037146038801333356 or version!=1001 or not isinstance(system,dict) or system.get('protocol_version')!=1001 or system.get('little_endian') is not True:raise ValueError('unsupported storage envelope')
        if not isinstance(descriptor,tuple) or len(descriptor)!=6:raise ValueError('storage descriptor missing')
        _,kind,key,_,count,_=descriptor
        if keys!=[key]:raise ValueError('storage keys')
        n=struct.unpack('<q',f.read(8))[0]
        dtype=np.dtype('<i8' if kind=='LongStorage' else '<f4')
        raw=f.read()
        if n!=count or len(raw)!=count*dtype.itemsize:raise ValueError('storage byte count')
        return np.frombuffer(raw,dtype=dtype).copy()
    except (EOFError,pickle.UnpicklingError,struct.error,TypeError) as e:raise ValueError('invalid storage encoding') from e

def _rebuild_array(storage,offset,size,stride,requires_grad,hooks):
    if not isinstance(storage,np.ndarray) or storage.ndim!=1 or type(offset)!=int or offset!=0 or not isinstance(size,tuple) or not 1<=len(size)<=3 or any(type(n)!=int or not 0<n<=100000 for n in size) or math.prod(size)!=storage.size or requires_grad is not False or hooks!={}:raise ValueError('unsupported tensor layout')
    expected=tuple(math.prod(size[i+1:]) for i in range(len(size)))
    if stride!=expected:raise ValueError('noncontiguous tensor storage')
    return storage.reshape(size)

class _Restricted(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch.storage','_load_from_bytes'):return _storage_bytes
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return _rebuild_array
        if (module,name)==('collections','OrderedDict'):return dict
        if (module,name) in [('chumpy.ch','Ch')]:return _ArrayState
        if (module,name) in [('scipy.sparse.csc','csc_matrix')]:return _SparseState
        if module in ('builtins','__builtin__') and name=='set':return set
        if module=='numpy' and name in ('ndarray','dtype'):return getattr(np,name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np._core.multiarray,name)
        raise ValueError('unsupported pickle global '+module+'.'+name)
    def persistent_load(self,pid):raise ValueError('persistent pickle references forbidden')

def verify_hash(path,expected,check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block);check()
    if h.hexdigest()!=expected:raise ValueError('source hash mismatch')

def _header(f):
    version=np.lib.format.read_magic(f)
    if version==(1,0):return np.lib.format.read_array_header_1_0(f)
    if version==(2,0):return np.lib.format.read_array_header_2_0(f)
    raise ValueError('unsupported NPY version')

def load_legacy(path,expected):
    path=Path(path)
    if path.stat().st_size>64*1024**2:raise ValueError('legacy file cap exceeded')
    verify_hash(path,expected)
    with path.open('rb') as f:
        if path.suffix=='.npy':
            shape,order,dtype=_header(f)
            if shape!=() or dtype!=np.dtype('O'):raise ValueError('scalar embedding dictionary required')
        value=_Restricted(f,encoding='latin1').load()
        if f.read(1):raise ValueError('unexpected trailing legacy data')
    if path.suffix=='.npy':
        if not isinstance(value,np.ndarray) or value.shape!=() or value.dtype!=np.dtype('O'):raise ValueError('invalid embedding wrapper')
        value=value.item()
    if not isinstance(value,dict):raise ValueError('model dictionary required')
    return value

def _array(value):
    if isinstance(value,_ArrayState):value=value.state.get('x')
    if not isinstance(value,np.ndarray) or value.dtype.kind not in 'fiu':raise ValueError('numeric array required')
    if not np.isfinite(value).all():raise ValueError('nonfinite model data')
    return value

def _dense(value,shape):
    if not isinstance(value,_SparseState):return _array(value)
    s=value.state
    if tuple(s.get('_shape',()))!=shape:raise ValueError('sparse regressor shape')
    data=_array(s['data']);idx=_array(s['indices']);ptr=_array(s['indptr'])
    if idx.dtype.kind not in 'iu' or ptr.dtype.kind not in 'iu' or ptr.shape!=(shape[1]+1,) or data.ndim!=1 or idx.shape!=data.shape or data.size>math.prod(shape):raise ValueError('invalid sparse arrays')
    if ptr[0]!=0 or ptr[-1]!=data.size or np.any(ptr[1:]<ptr[:-1]) or np.any(idx<0) or np.any(idx>=shape[0]):raise ValueError('invalid sparse indices')
    result=np.zeros(shape,dtype=np.float64)
    for j in range(shape[1]):
        indices=idx[ptr[j]:ptr[j+1]]
        if len(np.unique(indices))!=len(indices):raise ValueError('duplicate sparse index')
        result[indices,j]=data[ptr[j]:ptr[j+1]]
    return result

def geometry_arrays(d,*,shape_count=100,expression_count=50):
    if type(shape_count)!=int or not 1<=shape_count<=300 or type(expression_count)!=int or not 1<=expression_count<=100:raise ValueError('component count')
    v=_array(d['v_template']);V=len(v)
    if v.shape!=(V,3) or not 3<=V<=5023:raise ValueError('template extent')
    sd=_array(d['shapedirs']);pd=_array(d['posedirs']);w=_array(d['weights']);tree=_array(d['kintree_table']);faces=_array(d['f'])
    if sd.ndim!=3 or sd.shape[:2]!=(V,3) or sd.shape[2]<300+expression_count or pd.shape!=(V,3,36) or w.shape!=(V,5) or tree.shape!=(2,5):raise ValueError('FLAME model layout')
    if faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>10000 or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=V):raise ValueError('face indices')
    if tree.dtype.kind not in 'iu' or not np.array_equal(tree[1],np.arange(5)):raise ValueError('joint ids')
    parents=tree[0].astype(np.int64);parents[0]=-1
    if any(not 0<=parents[i]<i for i in range(1,5)):raise ValueError('joint parents')
    if np.any(w<0) or not np.allclose(w.sum(1),1,rtol=0,atol=1e-7):raise ValueError('skin weights')
    reg=_dense(d['J_regressor'],(5,V))
    if reg.shape!=(5,V):raise ValueError('joint regressor')
    blend=np.concatenate((sd[:,:,:shape_count],sd[:,:,300:300+expression_count]),-1).transpose(2,0,1)
    return {k:np.ascontiguousarray(a,dtype=np.int32 if k in ('parents','faces') else np.float64) for k,a in dict(template=v,blend_basis=blend,pose_basis=pd.transpose(2,0,1),joint_regressor=reg,parents=parents,skin_weights=w,faces=faces).items()}

def landmark_arrays(d,*,faces):
    result={}
    for prefix,shape,out in [('static',(51,),'landmark'),('dynamic',(79,17),'contour'),('full',(68,),'full_landmark')]:
        indices=_array(d[prefix+'_lmk_faces_idx']);bary=_array(d[prefix+'_lmk_bary_coords'])
        if prefix=='full' and indices.shape==(1,68) and bary.shape==(1,68,3):indices=indices[0];bary=bary[0]
        if indices.shape!=shape or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=faces) or bary.shape!=(*shape,3) or not np.allclose(bary.sum(-1,dtype=np.float64),1,atol=1e-7,rtol=0):raise ValueError('landmark table')
        result[out+'_faces']=indices.astype(np.int32);result[out+'_barycentric']=np.ascontiguousarray(bary,dtype=np.float64)
    return result

def obj_arrays(path):
    path=Path(path)
    if path.stat().st_size>4*1024**2:raise ValueError('OBJ cap')
    v=[];uv=[];faces=[];uf=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='v':v.append([float(x) for x in parts[1:4]])
        elif parts[0]=='vt':uv.append([float(x) for x in parts[1:3]])
        elif parts[0]=='f':
            if len(parts)!=4:raise ValueError('triangle OBJ required')
            parsed=[x.split('/') for x in parts[1:]]
            faces.append([int(x[0])-1 for x in parsed]);uf.append([int(x[1])-1 for x in parsed])
    v=np.array(v);uv=np.array(uv);f=np.array(faces,dtype=np.int32);ft=np.array(uf,dtype=np.int32)
    if not 3<=len(v)<=5023 or v.shape!=(len(v),3) or not 3<=len(uv)<=20000 or uv.shape!=(len(uv),2) or not 1<=len(f)<=10000 or np.any(f<0) or np.any(f>=len(v)) or np.any(ft<0) or np.any(ft>=len(uv)) or not np.isfinite(v).all() or not np.isfinite(uv).all():raise ValueError('OBJ extent/indices')
    corners=uv[ft].copy();corners[...,1]=1-corners[...,1]
    return dict(template=v,faces=f,uv=uv,uv_faces=ft,face_uv=corners)

def texture_arrays(path,expected,*,components=50,size=256,check=lambda:None):
    if type(components)!=int or not 1<=components<=200 or type(size)!=int or not 1<=size<=256:raise ValueError('texture selection cap')
    verify_hash(path,expected,check);largest=0
    with zipfile.ZipFile(path) as z:
        with z.open('mean.npy') as f:
            shape,order,dtype=_header(f)
            if len(shape)!=3 or shape[-1]!=3 or not 1<=shape[0]<=512 or shape[1]!=shape[0] or size>shape[0] or order or dtype!=np.dtype('float64'):raise ValueError('texture mean layout')
            count=math.prod(shape)*8;blob=f.read(count+1)
            if len(blob)!=count:raise ValueError('mean byte count')
            mean=np.frombuffer(blob,dtype=dtype).reshape(shape)
        H,W,_=shape;indices=np.arange(size)*H//size
        mean=mean[indices[:,None],indices[None,:],::-1].copy()/255
        if not np.isfinite(mean).all():raise ValueError('nonfinite texture mean')
        out=np.empty((size,size,3,components),dtype=np.float64)
        with z.open('tex_dir.npy') as f:
            shape,order,dtype=_header(f)
            if len(shape)!=4 or shape[:3]!=(H,W,3) or not components<=shape[3]<=200 or order or dtype!=np.dtype('float64'):raise ValueError('texture basis layout')
            row_bytes=W*3*shape[3]*8;selected={int(y):i for i,y in enumerate(indices)}
            for y in range(H):
                check();blob=f.read(row_bytes);largest=max(largest,len(blob))
                if len(blob)!=row_bytes:raise ValueError('truncated basis')
                if y in selected:
                    row=np.frombuffer(blob,dtype=dtype).reshape(W,3,shape[3]);part=row[indices,::-1,:components]
                    if not np.isfinite(part).all():raise ValueError('nonfinite selected texture basis')
                    out[selected[y]]=part/255
            if f.read(1):raise ValueError('trailing basis data')
    return dict(texture_mean=mean,texture_basis=np.ascontiguousarray(out.transpose(3,0,1,2)),largest_chunk_bytes=largest)
