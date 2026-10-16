"""Windows PTX resource generation preserves exact binary payload and terminator."""
import json,importlib.util
from pathlib import Path
import pytest

@pytest.mark.parametrize('size',[1,2047,2048,2049,16385])
def test_ptx_resource_preserves_every_input_byte_and_exact_terminator(tmp_path,size):
    p=Path(__file__).resolve().parents[2]/'cmake/GffxEmbedPtxResource.py';spec=importlib.util.spec_from_file_location('embed_resource',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    raw=bytes(i%256 for i in range(size));source=tmp_path/'input.ptx';source.write_bytes(raw);payload=tmp_path/'payload.bin';rc=tmp_path/'payload.rc';header=tmp_path/'payload.h';m.generate(source,payload,rc,header)
    assert payload.read_bytes()==raw+b'\0';assert '101 RCDATA '+json.dumps(payload.resolve().as_posix()) in rc.read_text();assert '#define GFFX_PTX_RESOURCE_ID 101' in header.read_text();assert f'_size = {size}UL' in header.read_text()

def test_ptx_resource_refuses_empty_payload_and_unsafe_resource_identifier(tmp_path):
    p=Path(__file__).resolve().parents[2]/'cmake/GffxEmbedPtxResource.py';spec=importlib.util.spec_from_file_location('embed_resource',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=tmp_path/'input.ptx';source.write_bytes(b'');dest=[tmp_path/z for z in ('p.bin','p.rc','p.h')]
    with pytest.raises(ValueError):m.generate(source,*dest)
    source.write_bytes(b'x')
    for ident in (0,65536,'101'):
        with pytest.raises(ValueError):m.generate(source,*dest,resource_id=ident)
