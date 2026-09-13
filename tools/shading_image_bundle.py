"""Create-only, bounded image sidecars; hashes establish integrity, not accuracy.

Never execute saved source, unpickle, infer a newest revision, or promote the
parent registry's qualification. A caller must retain the exact manifest hash.
"""
from array import array
import math
from pathlib import Path
import re
import sys

import torch

from tools import corpus_registry as registry

PROFILE='classic-images-v1'
SETTINGS=dict(width=128,height=128,dtype='float64',device='cpu',near=.1,far=10.,
              focal=64/math.tan(math.radians(20)),principal_point=[64.,64.],
              camera_position=[0.,0.,3.],directional_light=[1.,1.,2.],
              irradiance=[1.,1.,1.],specular=[0.,.1],exponent=2.,
              channel_atol=1e-10,channel_rtol=1e-8,psnr_peak=1.,
              background='zero; face=-1',interpolation='perspective corrected',
              qualification='development reference; shared native raster, independent scalar light')
BUFFER_FIELDS={
    'covered':((128,128),torch.bool),'face':((128,128),torch.int32),
    **{name:((128,128,3),torch.float64) for name in ('affine','barycentric','world','normal','view')},
    'uv':((128,128,2),torch.float64),'depth':((128,128),torch.float64),
    'vertex_pixels':((1986,2),torch.float64),
    **{'albedo/'+name:((128,128,3),torch.float64) for name in ('constant','vertex','gradient','checker')},
}
IMAGE_FIELDS=tuple(model+'/'+colour for model in ('diffuse','classic','environment_diffuse')
                   for colour in ('constant','vertex','gradient','checker'))
FORMATS={'float64':('d',8,torch.float64),'int32':('i',4,torch.int32),'bool':('B',1,torch.bool)}
SOURCE_NAMES=('synthetic_face_fixture','corpus_registry','shading_independent_evaluator',
              'shading_environment_reference','shading_image_reference','shading_image_bundle',
              'shading_image_probe','material_loss_cost')
SOURCE_FILES={name+'.py.txt':Path(__file__).with_name(name+'.py') for name in SOURCE_NAMES}
SOURCE_FILES.update({name+'.py.txt':Path(__file__).parent/'testing_dashboard'/(name+'.py')
                     for name in ('image_reference_demo','presentation','camera_demo')})


def tensor_bytes(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu':raise ValueError('CPU tensor required')
    dtype=str(value.dtype).removeprefix('torch.')
    if dtype not in FORMATS:raise ValueError('unsupported binary dtype')
    if dtype=='float64' and not torch.isfinite(value).all():raise ValueError('nonfinite binary float')
    code,size,_=FORMATS[dtype]
    data=array(code,value.detach().reshape(-1).tolist())
    if data.itemsize!=size:raise ValueError('unsupported platform binary width')
    if sys.byteorder!='little':data.byteswap()
    return data.tobytes()


def tensor_from_bytes(blob,shape,dtype):
    if dtype not in FORMATS or not isinstance(shape,(list,tuple)) or not 1<=len(shape)<=3:
        raise ValueError('invalid binary descriptor')
    if any(type(x) is not int or not 0<x<=49152 for x in shape) or math.prod(shape)>49152:
        raise ValueError('binary shape limit')
    code,size,torch_type=FORMATS[dtype]
    if len(blob)!=math.prod(shape)*size:raise ValueError('binary length mismatch')
    data=array(code);data.frombytes(blob)
    if data.itemsize!=size:raise ValueError('unsupported platform binary width')
    if sys.byteorder!='little':data.byteswap()
    if dtype=='bool' and any(x not in (0,1) for x in data):raise ValueError('invalid boolean byte')
    if dtype=='float64' and any(not math.isfinite(x) for x in data):raise ValueError('nonfinite binary float')
    return torch.tensor(data.tolist(),dtype=torch_type).reshape(shape)


def flatten_buffers(buffers):
    return {name:(buffers['albedos'][name.split('/')[1]] if name.startswith('albedo/') else buffers[name])
            for name in BUFFER_FIELDS}


def _encode(values,fields):
    registry._keys(values,fields)
    chunks=[];descriptors=[];offset=0
    for name,(shape,dtype) in fields.items():
        value=values[name]
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=dtype:
            raise ValueError('array shape/dtype mismatch: '+name)
        blob=tensor_bytes(value)
        descriptors.append(dict(name=name,shape=list(shape),dtype=str(dtype).removeprefix('torch.'),offset=offset,bytes=len(blob)))
        chunks.append(blob);offset+=len(blob)
    return b''.join(chunks),descriptors


def _descriptors(descriptors,fields,file_size):
    if not isinstance(descriptors,list) or len(descriptors)!=len(fields):raise ValueError('array count mismatch')
    seen=set();offset=0
    for desc in descriptors:
        registry._keys(desc,('name','shape','dtype','offset','bytes'))
        name=desc['name']
        if not isinstance(name,str) or name not in fields or name in seen:raise ValueError('invalid/duplicate array name')
        seen.add(name);shape,dtype=fields[name]
        if not isinstance(desc['shape'],list) or any(type(x) is not int for x in desc['shape']) or desc['shape']!=list(shape):
            raise ValueError('array shape mismatch')
        if desc['dtype']!=str(dtype).removeprefix('torch.'):raise ValueError('array dtype mismatch')
        size=math.prod(shape)*FORMATS[desc['dtype']][1]
        if type(desc['offset']) is not int or type(desc['bytes']) is not int or desc['offset']!=offset or desc['bytes']!=size:
            raise ValueError('array overlap/gap/size mismatch')
        offset+=size
    if offset!=file_size:raise ValueError('array container not exactly covered')


def _parent(root):
    result=registry.load_entry(root,'synthetic_face_m1',allow_development=True)
    record=next(x for x in registry._index(root)['entries'] if x['id']=='synthetic_face_m1')
    return result,dict(id=record['id'],revision=result['manifest']['revision'],sha256=record['sha256'])


def _fields():return dict(buffers=BUFFER_FIELDS,images={x:((128,128,3),torch.float64) for x in IMAGE_FIELDS})


def write_bundle(root,buffers,images,*,producer,details):
    """Write a completed new revision. Failure leaves only a rejected partial dir."""
    parent,parent_id=_parent(root)
    payloads={};arrays={}
    for name,fields in _fields().items():
        payloads[name+'.bin'],arrays[name]=_encode(buffers if name=='buffers' else images,fields)
    sources={}
    for filename,source_path in SOURCE_FILES.items():
        blob=registry._read(source_path,registry.MAX_FILE_BYTES)
        payloads[filename]=blob;sources[filename]=registry._hash(blob)
    if payloads['synthetic_face_fixture.py.txt']!=parent['files']['generator.py.txt']:
        raise ValueError('generator source changed since parent registration')
    payloads['LICENSE.txt']=parent['files']['LICENSE.txt']
    files=[dict(path=name,bytes=len(blob),sha256=registry._hash(blob)) for name,blob in sorted(payloads.items())]
    manifest=dict(schema_version=1,status='development',profile=PROFILE,parent=parent_id,
                  settings=SETTINGS,producer=producer,details=details,arrays=arrays,
                  sources=sources,license='MIT',files=files)
    blob=registry._json_bytes(manifest)
    if len(blob)>registry.MAX_METADATA_BYTES:raise ValueError('manifest size limit')
    # Full read-side metadata preflight before any directory is created.
    _metadata(manifest)
    digest=registry._hash(blob)
    relative=f'profiles/synthetic_face_m1/{PROFILE}/{digest}/manifest.json'
    path=registry._safe(root,relative)
    path.parent.mkdir(parents=True,exist_ok=False)
    for name,data in payloads.items():
        with registry._safe(path.parent,name).open('xb') as handle:handle.write(data)
    # Exclusive manifest creation is last. Stored snapshots are never imported.
    with path.open('xb') as handle:handle.write(blob)
    return dict(path=relative,sha256=digest)


def _metadata(manifest):
    registry._keys(manifest,('schema_version','status','profile','parent','settings','producer','details','arrays','sources','license','files'))
    if type(manifest['schema_version']) is not int or manifest['schema_version']!=1 or manifest['status']!='development' or manifest['profile']!=PROFILE:
        raise ValueError('invalid image schema/status/profile')
    if manifest['settings']!=SETTINGS or manifest['license']!='MIT':raise ValueError('invalid settings/license')
    registry._keys(manifest['parent'],('id','revision','sha256'));registry._hex(manifest['parent']['sha256'])
    if not isinstance(manifest['producer'],dict) or not isinstance(manifest['details'],dict):raise ValueError('invalid provenance objects')
    expected_sources=set(SOURCE_FILES)
    registry._keys(manifest['sources'],expected_sources)
    registry._keys(manifest['arrays'],('buffers','images'))
    files=manifest['files']
    if not isinstance(files,list) or not 1<=len(files)<=32:raise ValueError('file count limit')
    seen=set();total=0
    for row in files:
        registry._keys(row,('path','bytes','sha256'));registry._hex(row['sha256'])
        name=row['path']
        if not isinstance(name,str) or name not in expected_sources|{'LICENSE.txt','buffers.bin','images.bin'} or name.casefold() in seen:
            raise ValueError('invalid/duplicate file path')
        seen.add(name.casefold())
        if type(row['bytes']) is not int or not 0<row['bytes']<=registry.MAX_FILE_BYTES:raise ValueError('file byte limit')
        total+=row['bytes']
    if total>registry.MAX_TOTAL_BYTES:raise ValueError('total byte limit')
    if seen!={x.casefold() for x in expected_sources|{'LICENSE.txt','buffers.bin','images.bin'}}:raise ValueError('missing bundle file')
    by_name={x['path']:x for x in files}
    for group,fields in _fields().items():_descriptors(manifest['arrays'][group],fields,by_name[group+'.bin']['bytes'])
    for name,digest in manifest['sources'].items():
        registry._hex(digest)
        if by_name[name]['sha256']!=digest:raise ValueError('source hash mismatch')


def load_bundle(root,*,path,sha256):
    """Explicit path+hash only. Fully validate metadata/hashes before array decode."""
    registry._hex(sha256)
    location=registry._safe(root,path)
    if not re.fullmatch(r'profiles/synthetic_face_m1/classic-images-v1/[0-9a-f]{64}/manifest\.json',path):
        raise ValueError('invalid profile path')
    blob=registry._read(location,registry.MAX_METADATA_BYTES)
    if registry._hash(blob)!=sha256:raise ValueError('manifest hash mismatch')
    manifest=registry._decode(blob);_metadata(manifest)
    parent,parent_id=_parent(root)
    if manifest['parent']!=parent_id:raise ValueError('parent identity/hash mismatch')
    payloads={}
    for row in manifest['files']:
        data=registry._read(registry._safe(location.parent,row['path']),row['bytes'])
        if len(data)!=row['bytes'] or registry._hash(data)!=row['sha256']:raise ValueError('payload bytes/hash mismatch')
        payloads[row['path']]=data
    if payloads['LICENSE.txt']!=parent['files']['LICENSE.txt'] or payloads['synthetic_face_fixture.py.txt']!=parent['files']['generator.py.txt']:
        raise ValueError('parent source/license mismatch')
    decoded={}
    for group in _fields():
        data=payloads[group+'.bin'];decoded[group]={}
        for desc in manifest['arrays'][group]:
            start=desc['offset']
            decoded[group][desc['name']]=tensor_from_bytes(data[start:start+desc['bytes']],desc['shape'],desc['dtype'])
    return dict(manifest=manifest,**decoded)
