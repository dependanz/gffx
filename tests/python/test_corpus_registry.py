"""Local development registry: integrity is not rendering admission."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = 'fc11090311196dc175206ba96db4fb006bf2d299'


def api():
    return importlib.import_module('tools.corpus_registry')


def registered(tmp_path):
    m = api()
    m.register_synthetic(tmp_path, ROOT, base_commit=BASE)
    return m


def rewrite_manifest(root, change):
    index=json.loads((root/'registry.json').read_text())
    path=root/index['entries'][0]['path']
    data=json.loads(path.read_text())
    change(data)
    blob=json.dumps(data,sort_keys=True,separators=(',',':')).encode()
    path.write_bytes(blob)
    index['entries'][0]['sha256']=hashlib.sha256(blob).hexdigest()
    (root/'registry.json').write_text(json.dumps(index))


def test_missing_entry_names_unregistered_id(tmp_path):
    m=api()
    with pytest.raises(KeyError,match='synthetic_face_m1.*unregistered'):
        m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_registration_roundtrip_preserves_development_and_profile_boundaries(tmp_path):
    m=registered(tmp_path)
    with pytest.raises(ValueError,match='development'):
        m.load_entry(tmp_path,'synthetic_face_m1')
    entry=m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)
    assert entry['manifest']['source']['commit'] is None
    assert entry['manifest']['source']['base_commit']==BASE
    assert entry['manifest']['status']=='development'
    assert entry['manifest']['profiles']=={'gffx_model_m1':'not_measured','cycles_fidelity':'not_measured'}
    scene=json.loads(entry['files']['scene.json'])
    assert len(scene['vertices'])==1986 and len(scene['triangles'])==3968
    assert entry['files']['generator.py.txt']==(ROOT/'tools/synthetic_face_fixture.py').read_bytes()
    with pytest.raises(ValueError,match='not measured'):
        m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True,require_profile='gffx_model_m1')


def test_duplicate_registration_never_overwrites_files(tmp_path):
    m=registered(tmp_path)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):m.register_synthetic(tmp_path,ROOT,base_commit=BASE)
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


@pytest.mark.parametrize('bad',['duplicate_id','duplicate_json_key','schema','missing_hash','unsafe_manifest'])
def test_registry_rejects_invalid_index(tmp_path,bad):
    m=registered(tmp_path); path=tmp_path/'registry.json'; index=json.loads(path.read_text())
    if bad=='duplicate_id':index['entries']*=2
    elif bad=='schema':index['schema_version']=2
    elif bad=='missing_hash':index['entries'][0].pop('sha256')
    elif bad=='unsafe_manifest':index['entries'][0]['path']='../manifest.json'
    if bad=='duplicate_json_key':path.write_text('{"schema_version":1,"schema_version":1,"entries":[]}')
    else:path.write_text(json.dumps(index))
    with pytest.raises(ValueError):m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


@pytest.mark.parametrize('bad',['license','source_hash','duplicate_file','negative_size','boolean_size','schema','profile','status'])
def test_manifest_rejects_invalid_semantics_after_hash_verification(tmp_path,bad):
    m=registered(tmp_path)
    def change(data):
        if bad=='license':data['license']={}
        elif bad=='source_hash':data['source']['sha256']='0'*64
        elif bad=='duplicate_file':data['files'].append(data['files'][0])
        elif bad=='negative_size':data['files'][0]['bytes']=-1
        elif bad=='boolean_size':data['files'][0]['bytes']=True
        elif bad=='schema':data['schema_version']=2
        elif bad=='profile':data['profiles']['gffx_model_m1']='accepted'
        else:data['status']='published'
    rewrite_manifest(tmp_path,change)
    with pytest.raises(ValueError):m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


@pytest.mark.parametrize('path',['../outside','C:/outside','/outside','scene.json:secret','a\\b','a/../b'])
def test_manifest_rejects_unsafe_file_paths(tmp_path,path):
    m=registered(tmp_path)
    rewrite_manifest(tmp_path,lambda d:d['files'][0].update(path=path))
    with pytest.raises(ValueError):m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


@pytest.mark.parametrize('target',['manifest','artifact'])
def test_tampered_bytes_are_not_accepted(tmp_path,target):
    m=registered(tmp_path)
    index=json.loads((tmp_path/'registry.json').read_text())
    manifest=tmp_path/index['entries'][0]['path']
    path=manifest if target=='manifest' else manifest.parent/'scene.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='(hash|size)'):
        m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_oversized_registry_is_rejected_before_json_parse(tmp_path,monkeypatch):
    m=api(); (tmp_path/'registry.json').write_bytes(b' '*(m.MAX_METADATA_BYTES+1))
    def forbidden(*args,**kwargs):pytest.fail('parsed oversized index')
    # Intercept this loader's parser, not the reporter/pytest's shared JSON module.
    monkeypatch.setattr(m,'_decode',forbidden)
    assert json.loads('{"reporting":true}')=={'reporting':True}
    with pytest.raises(ValueError,match='size'):
        m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_interrupted_index_publication_leaves_no_registration(tmp_path,monkeypatch):
    m=api()
    def fail(*args,**kwargs):raise OSError('simulated index publication failure')
    monkeypatch.setattr(m.os,'replace',fail)
    with pytest.raises(OSError,match='simulated'):
        m.register_synthetic(tmp_path,ROOT,base_commit=BASE)
    assert not (tmp_path/'registry.json').exists()
    assert not (tmp_path/'registration.lock').exists()
    assert list(tmp_path.glob('synthetic_face_m1/*/manifest.json'))
    with pytest.raises(KeyError):m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_existing_registration_lock_is_never_taken_over(tmp_path):
    m=api(); (tmp_path/'registration.lock').write_text('another writer')
    with pytest.raises(FileExistsError):m.register_synthetic(tmp_path,ROOT,base_commit=BASE)
    assert (tmp_path/'registration.lock').read_text()=='another writer'


def test_loader_never_executes_copied_generator(tmp_path):
    m=registered(tmp_path)
    def change(data):
        item=next(f for f in data['files'] if f['role']=='generator_source')
        blob=b"raise AssertionError('must never execute snapshot')\n"
        directory=next(tmp_path.glob('synthetic_face_m1/*'))
        (directory/item['path']).write_bytes(blob)
        item.update(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        data['source']['sha256']=item['sha256']
        data['generation']['generator_sha256']=item['sha256']
    rewrite_manifest(tmp_path,change)
    m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_nonfinite_json_numbers_are_rejected(tmp_path):
    m=registered(tmp_path)
    with pytest.raises(ValueError):m._decode(b'{"value":1e999}')
    index=json.loads((tmp_path/'registry.json').read_text())
    path=tmp_path/index['entries'][0]['path']
    blob=path.read_bytes().replace(b'"exposure":1.0',b'"exposure":1e999')
    assert blob!=path.read_bytes()
    path.write_bytes(blob); index['entries'][0]['sha256']=hashlib.sha256(blob).hexdigest()
    (tmp_path/'registry.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)


def test_reparse_paths_are_rejected_without_requiring_link_privileges(tmp_path,monkeypatch):
    m=registered(tmp_path); original=Path.lstat
    class Reparse:
        st_file_attributes=0x400
        st_mode=0o100000
    def stat(path,*args,**kwargs):
        return Reparse() if path.name=='scene.json' else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',stat)
    with pytest.raises(ValueError,match='reparse'):
        m.load_entry(tmp_path,'synthetic_face_m1',allow_development=True)
