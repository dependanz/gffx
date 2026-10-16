"""Generated PTX payload bytes and bounded C initializer token count."""
from pathlib import Path
import re,subprocess,shutil
import pytest
@pytest.mark.parametrize('size',[1,2047,2048,2049,16385])
def test_ptx_chunked_literals_preserve_every_byte_and_nul(tmp_path,size):
    raw=bytes(i%256 for i in range(size));source=tmp_path/'input.ptx';output=tmp_path/'output.h';source.write_bytes(raw)
    subprocess.run([shutil.which('cmake'),'-DGFFX_PTX_INPUT='+str(source),'-DGFFX_PTX_OUTPUT='+str(output),'-DGFFX_PTX_SYMBOL=payload','-P',str(Path(__file__).resolve().parents[2]/'cmake/GffxEmbedPtx.cmake')],check=True,capture_output=True,timeout=10)
    text=output.read_text();rows=re.findall(r'^"((?:\\x[0-9a-f]{2})+)"',text,re.M)
    assert rows,'PTX must use bounded string chunks rather than one C token per byte'
    chunks=[bytes.fromhex(s.replace('\\x','')) for s in rows]
    assert all(len(s)==2048 for s in chunks[:-1]) and len(chunks[-1])<=2048
    assembled=b''.join(chunks);assert assembled==raw+b'\0'
