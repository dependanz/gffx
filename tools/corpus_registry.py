"""Bounded local development corpus registration; never rendering admission.

Standard library only. Explicit root, create-only revisions, atomic index update.
An interrupted writer can leave an unregistered directory; never remove it here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile

MAX_METADATA_BYTES = 65536
MAX_FILE_BYTES = 8*1024*1024
MAX_TOTAL_BYTES = 16*1024*1024
MAX_FILES = 16
MAX_ENTRIES = 128


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _keys(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):
        raise ValueError('invalid metadata keys')


def _hex(value,length=64):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):
        raise ValueError('invalid hash')


def _id(value):
    if not isinstance(value,str) or len(value)>128 or not re.fullmatch('[a-z0-9]+(?:_[a-z0-9]+)*',value):
        raise ValueError('invalid entry ID')


def _no_links(path):
    for part in [*reversed(path.parents),path]:
        try:info=part.lstat()
        except FileNotFoundError:continue
        if part.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('links/reparse points are not allowed in corpus paths')


def _safe(root,relative):
    if not isinstance(relative,str) or not relative or len(relative)>256:
        raise ValueError('invalid relative path')
    parts=relative.split('/')
    reserved={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
    if any(not re.fullmatch('[A-Za-z0-9_.-]+',p) or p in ('.','..') or p.endswith('.') or p.split('.')[0].upper() in reserved for p in parts):
        raise ValueError('unsafe relative path')
    path=Path(root).absolute().joinpath(*parts)
    _no_links(path)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes corpus root')
    return path


def _read(path,limit):
    _no_links(path)
    if not path.is_file():raise ValueError('missing corpus file: '+str(path.name))
    if path.stat().st_size>limit:raise ValueError('file size limit exceeded')
    with path.open('rb') as handle:blob=handle.read(limit+1)
    if len(blob)>limit:raise ValueError('file size limit exceeded')
    return blob


def _decode(blob):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    def finite(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError('nonfinite JSON number')
        return number
    try:return json.loads(blob,object_pairs_hook=pairs,parse_constant=invalid,parse_float=finite)
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
        raise ValueError('invalid JSON metadata') from error


def _index(root):
    path=_safe(root,'registry.json')
    if not path.exists():return dict(schema_version=1,entries=[])
    index=_decode(_read(path,MAX_METADATA_BYTES))
    _keys(index,('schema_version','entries'))
    if type(index['schema_version']) is not int or index['schema_version']!=1 or not isinstance(index['entries'],list) or len(index['entries'])>MAX_ENTRIES:
        raise ValueError('unsupported registry schema or entry limit')
    seen=set()
    for entry in index['entries']:
        _keys(entry,('id','path','sha256')); _id(entry['id']); _hex(entry['sha256']); _safe(root,entry['path'])
        if entry['id'] in seen:raise ValueError('duplicate entry ID')
        seen.add(entry['id'])
    return index


def _load_manifest(root,record):
    path=_safe(root,record['path']); blob=_read(path,MAX_METADATA_BYTES)
    if _hash(blob)!=record['sha256']:raise ValueError('manifest hash mismatch')
    manifest=_decode(blob)
    _keys(manifest,('schema_version','id','revision','status','source','license','generation','scene','reference_for','renderer','comparison','profiles','files'))
    if type(manifest['schema_version']) is not int or manifest['schema_version']!=1:raise ValueError('unsupported manifest schema')
    _id(manifest['id'])
    if manifest['id']!=record['id'] or manifest['status']!='development':raise ValueError('invalid entry identity/status')
    revision=manifest['revision']
    if not isinstance(revision,str) or not re.fullmatch('dev-[0-9a-f]{16}',revision):raise ValueError('invalid development revision')
    if record['path']!=f"{manifest['id']}/{revision}/manifest.json":raise ValueError('manifest path/identity mismatch')
    if manifest['profiles']!={'gffx_model_m1':'not_measured','cycles_fidelity':'not_measured'}:
        raise ValueError('development entries cannot assert measured profiles')
    source=manifest['source']; _keys(source,('path','commit','base_commit','sha256','snapshot'))
    _hex(source['sha256']); _hex(source['base_commit'],40)
    paths={'synthetic_face_m1':'tools/synthetic_face_fixture.py','synthetic_flame_m1':'tools/synthetic_flame_fixture.py'}
    if source['commit'] is not None or source['path']!=paths.get(manifest['id']) or source['snapshot']!='generator.py.txt':
        raise ValueError('invalid development source provenance')
    license=manifest['license']; _keys(license,('id','file'))
    if license!={'id':'MIT','file':'LICENSE.txt'}:raise ValueError('missing/unsupported license')
    if manifest['renderer'] is not None or manifest['comparison'] is not None:
        raise ValueError('development asset is not a rendered reference')
    if manifest['reference_for']!=['synthetic_test_inputs']:raise ValueError('unsupported reference claim')
    generation=manifest['generation']; _keys(generation,('generator_sha256','python','platform','parameters','invocation'))
    if generation['generator_sha256']!=source['sha256'] or any(not isinstance(generation[k],str) or not generation[k] for k in ('python','platform','invocation')):
        raise ValueError('generation provenance mismatch')
    entries=manifest['files']
    if not isinstance(entries,list) or not 1<=len(entries)<=MAX_FILES:raise ValueError('file count limit')
    seen=set(); total=0
    for item in entries:
        _keys(item,('path','role','bytes','sha256')); _hex(item['sha256']); _safe(path.parent,item['path'])
        if item['path'].casefold() in seen:raise ValueError('duplicate file path')
        seen.add(item['path'].casefold())
        if type(item['bytes']) is not int or not 0<=item['bytes']<=MAX_FILE_BYTES:raise ValueError('invalid file size')
        total+=item['bytes']
        if total>MAX_TOTAL_BYTES:raise ValueError('total size limit')
    if {(x['path'],x['role']) for x in entries}!={('scene.json','scene'),('generator.py.txt','generator_source'),('LICENSE.txt','license')}:
        raise ValueError('missing source/license/scene files')
    files={}
    for item in entries:
        data=_read(_safe(path.parent,item['path']),item['bytes'])
        if len(data)!=item['bytes']:raise ValueError('artifact size mismatch')
        if _hash(data)!=item['sha256']:raise ValueError('artifact hash mismatch')
        files[item['path']]=data
    if _hash(files['generator.py.txt'])!=source['sha256']:raise ValueError('source hash mismatch')
    if not files['LICENSE.txt'].startswith(b'MIT License') or b'Permission is hereby granted' not in files['LICENSE.txt']:
        raise ValueError('missing license notice')
    scene=_decode(files['scene.json'])
    if not isinstance(scene,dict) or scene.get('id')!=manifest['id'] or scene.get('schema_version')!=1:
        raise ValueError('scene identity/schema mismatch')
    if scene.get('parameters')!=generation['parameters'] or scene.get('scene')!=manifest['scene']:
        raise ValueError('scene/generation metadata mismatch')
    return dict(manifest=manifest,files=files)


def load_entry(root,entry_id,*,allow_development=False,require_profile=None):
    """Verify a registered development asset; never execute its source snapshot."""
    _id(entry_id)
    if type(allow_development) is not bool:raise ValueError('explicit bool required')
    index=_index(root)
    record=next((row for row in index['entries'] if row['id']==entry_id),None)
    if record is None:raise KeyError(f'{entry_id}: unregistered corpus entry')
    result=_load_manifest(root,record)
    if not allow_development:raise ValueError('development entry requires explicit opt-in')
    if require_profile is not None:raise ValueError(f'{require_profile}: reference profile not measured')
    return result


def register_synthetic(root,source_root,*,base_commit,recipe="synthetic_face_m1"):
    """Run the local generator; never overwrite or execute stored snapshots/Git writes."""
    if recipe=='synthetic_face_m1':
        from tools import synthetic_face_fixture as generator
        generator_path='tools/synthetic_face_fixture.py'
    elif recipe=='synthetic_flame_m1':
        from tools import synthetic_flame_fixture as generator
        generator_path='tools/synthetic_flame_fixture.py'
    else:raise ValueError('unsupported fixed development recipe')
    _hex(base_commit,40)
    root=Path(root).absolute(); source_root=Path(source_root).absolute()
    _no_links(root)
    if Path(generator.__file__).resolve()!=(source_root/generator_path).resolve():
        raise ValueError('generator source root does not match imported module')
    source=_read(source_root/generator_path,MAX_FILE_BYTES)
    license=_read(source_root/'LICENSE',MAX_FILE_BYTES)
    if not license.startswith(b'MIT License'):raise ValueError('missing MIT license')
    scene=generator.generate(); blob=generator.encode(scene)
    if _read(source_root/generator_path,MAX_FILE_BYTES)!=source:raise ValueError('generator changed during execution')
    entry_id=scene['id']; revision='dev-'+_hash(blob+source+license)[:16]
    root.mkdir(parents=True,exist_ok=True)
    lock=_safe(root,'registration.lock')
    with lock.open('xb') as handle:handle.write(b'development corpus writer\n')
    temporary=None
    try:
        index=_index(root)
        if any(row['id']==entry_id for row in index['entries']):raise FileExistsError('entry already registered')
        if len(index['entries'])>=MAX_ENTRIES:raise ValueError('registry entry limit')
        relative=f'{entry_id}/{revision}/manifest.json'; path=_safe(root,relative)
        path.parent.mkdir(parents=True,exist_ok=False)
        files={'scene.json':('scene',blob),'generator.py.txt':('generator_source',source),'LICENSE.txt':('license',license)}
        descriptors=[]
        for name,(role,data) in files.items():
            if len(data)>MAX_FILE_BYTES:raise ValueError('generated file size limit')
            with _safe(path.parent,name).open('xb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            descriptors.append(dict(path=name,role=role,bytes=len(data),sha256=_hash(data)))
        manifest=dict(schema_version=1,id=entry_id,revision=revision,status='development',
                      source=dict(path=generator_path,commit=None,base_commit=base_commit,sha256=_hash(source),snapshot='generator.py.txt'),
                      license=dict(id='MIT',file='LICENSE.txt'),
                      generation=dict(generator_sha256=_hash(source),python=platform.python_version(),platform=platform.system(),parameters=scene['parameters'],invocation=generator.__name__+'.generate()'),
                      scene=scene['scene'],reference_for=['synthetic_test_inputs'],renderer=None,comparison=None,
                      profiles=dict(gffx_model_m1='not_measured',cycles_fidelity='not_measured'),files=descriptors)
        manifest_blob=_json_bytes(manifest)
        if len(manifest_blob)>MAX_METADATA_BYTES:raise ValueError('manifest size limit')
        with path.open('xb') as handle:
            handle.write(manifest_blob); handle.flush(); os.fsync(handle.fileno())
        record=dict(id=entry_id,path=relative,sha256=_hash(manifest_blob))
        result=_load_manifest(root,record)
        index['entries'].append(record); index_blob=_json_bytes(index)
        if len(index_blob)>MAX_METADATA_BYTES:raise ValueError('registry size limit')
        with tempfile.NamedTemporaryFile(dir=root,prefix='registry-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name); handle.write(index_blob); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,_safe(root,'registry.json')); temporary=None
        return result
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--source-root',required=True); parser.add_argument('--base-commit',required=True)
    args=parser.parse_args()
    result=register_synthetic(args.root,args.source_root,base_commit=args.base_commit)
    manifest=result['manifest']
    print(json.dumps(dict(id=manifest['id'],revision=manifest['revision'],status=manifest['status'],
                          file_bytes=sum(len(x) for x in result['files'].values()),profiles=manifest['profiles'])))


if __name__=='__main__':main()
