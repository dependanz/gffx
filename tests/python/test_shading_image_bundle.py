"""Image sidecar corruption tests; synthetic arrays test storage, not rendering."""
import hashlib
import importlib
import json
from pathlib import Path
import struct

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE='fc11090311196dc175206ba96db4fb006bf2d299'


def api():return importlib.import_module('tools.shading_image_bundle')


@pytest.fixture
def bundle(tmp_path):
    from tools.corpus_registry import register_synthetic
    register_synthetic(tmp_path,ROOT,base_commit=BASE)
    m=api()
    buffers={k:torch.zeros(shape,dtype=dtype) for k,(shape,dtype) in m.BUFFER_FIELDS.items()}
    buffers['face'].fill_(-1)
    buffers['covered'][64,64]=True;buffers['face'][64,64]=0
    images={k:torch.zeros((128,128,3),dtype=torch.float64) for k in m.IMAGE_FIELDS}
    producer=dict(kind='test-storage-only',versions={},installed_files={},wheel=None)
    record=m.write_bundle(tmp_path,buffers,images,producer=producer,details={'purpose':'storage test'})
    return m,tmp_path,record,buffers,images


def rewrite(root,record,edit):
    path=root/record['path'];manifest=json.loads(path.read_bytes());edit(manifest)
    blob=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    path.write_bytes(blob)
    return dict(path=record['path'],sha256=hashlib.sha256(blob).hexdigest())


def test_bundle_roundtrip_is_exact_development_and_manifest_last(bundle):
    m,root,record,buffers,images=bundle
    result=m.load_bundle(root,**record)
    assert result['manifest']['status']=='development'
    assert result['manifest']['profile']=='classic-images-v1'
    assert result['manifest']['parent']['id']=='synthetic_face_m1'
    for group,expected in (('buffers',buffers),('images',images)):
        assert set(result[group])==set(expected)
        for key in expected:assert torch.equal(result[group][key],expected[key])
    assert (root/record['path']).name=='manifest.json'
    assert (root/'registry.json').read_text().find('not_measured')==-1
    from tools.corpus_registry import load_entry
    assert set(load_entry(root,'synthetic_face_m1',allow_development=True)['manifest']['profiles'].values())=={'not_measured'}
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):m.write_bundle(root,buffers,images,producer=result['manifest']['producer'],details={'purpose':'storage test'})
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad',['manifest_hash','payload','missing','parent','partial'])
def test_bundle_rejects_changed_or_incomplete_evidence(bundle,bad):
    m,root,record,_,_=bundle;directory=(root/record['path']).parent
    if bad=='manifest_hash':record=dict(record,sha256='0'*64)
    elif bad=='payload':(directory/'images.bin').write_bytes(b'wrong')
    elif bad=='missing':(directory/'buffers.bin').unlink()
    elif bad=='parent':(root/'registry.json').write_text('{"schema_version":1,"entries":[]}')
    else:(directory/'manifest.json').unlink()
    with pytest.raises((ValueError,KeyError)):m.load_bundle(root,**record)


@pytest.mark.parametrize('bad',['schema','status','profile','parent_hash','path','duplicate_file','size','bool_size','field','dtype','shape','bool_shape','offset','overlap','gap','trailing','source','license'])
def test_bundle_rejects_rehashed_malformed_metadata(bundle,bad):
    m,root,record,_,_=bundle
    def edit(manifest):
        field=manifest['arrays']['buffers'][0]
        if bad=='schema':manifest['schema_version']=True
        elif bad=='status':manifest['status']='accepted'
        elif bad=='profile':manifest['profile']='cycles'
        elif bad=='parent_hash':manifest['parent']['sha256']='0'*64
        elif bad=='path':manifest['files'][0]['path']='../outside.bin'
        elif bad=='duplicate_file':manifest['files'].append(manifest['files'][0])
        elif bad=='size':manifest['files'][0]['bytes']=2**40
        elif bad=='bool_size':manifest['files'][0]['bytes']=True
        elif bad=='field':field['name']='unknown'
        elif bad=='dtype':field['dtype']='pickle'
        elif bad=='shape':field['shape']=[2**40]
        elif bad=='bool_shape':field['shape'][0]=True
        elif bad=='offset':field['offset']=-1
        elif bad=='overlap':manifest['arrays']['buffers'][1]['offset']=0
        elif bad=='gap':field['offset']=1
        elif bad=='trailing':manifest['arrays']['buffers'].pop()
        elif bad=='source':manifest['sources']['shading_image_bundle.py.txt']='0'*64
        else:manifest['license']='unknown'
    changed=rewrite(root,record,edit)
    with pytest.raises(ValueError):m.load_bundle(root,**changed)


@pytest.mark.parametrize('bad',['nan','invalid_bool'])
def test_bundle_rejects_rehashed_invalid_binary_values(bundle,bad):
    m,root,record,_,_=bundle;directory=(root/record['path']).parent
    manifest=json.loads((root/record['path']).read_bytes())
    fields=manifest['arrays']['buffers']
    desc=next(x for x in fields if x['name']==('world' if bad=='nan' else 'covered'))
    blob=bytearray((directory/'buffers.bin').read_bytes())
    encoded=struct.pack('<d',float('nan')) if bad=='nan' else b'\x02'
    blob[desc['offset']:desc['offset']+len(encoded)]=encoded
    (directory/'buffers.bin').write_bytes(blob)
    def edit(data):
        next(x for x in data['files'] if x['path']=='buffers.bin')['sha256']=hashlib.sha256(blob).hexdigest()
    with pytest.raises(ValueError):m.load_bundle(root,**rewrite(root,record,edit))


def test_writer_validates_arrays_before_creating_directory(bundle):
    m,root,_,buffers,images=bundle
    before=set(root.rglob('*'));buffers['world'][0,0,0]=float('nan')
    with pytest.raises(ValueError):m.write_bundle(root,buffers,images,producer={},details={})
    assert set(root.rglob('*'))==before


def test_binary_codec_is_little_endian_and_never_uses_pickle():
    m=api();values=torch.tensor([1.5,-2.],dtype=torch.float64)
    blob=m.tensor_bytes(values)
    assert blob==struct.pack('<2d',1.5,-2.)
    assert torch.equal(m.tensor_from_bytes(blob,[2],'float64'),values)
    with pytest.raises(ValueError):m.tensor_from_bytes(b'\x02',[1],'bool')


@pytest.mark.parametrize('path',['../manifest.json','C:/outside.json','a\\b.json','CON/manifest.json'])
def test_bundle_rejects_unsafe_explicit_paths(tmp_path,path):
    with pytest.raises(ValueError):api().load_bundle(tmp_path,path=path,sha256='0'*64)
