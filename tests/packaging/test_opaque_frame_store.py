"""External frame bytes, completion and bounded replay; no renderer speed claim."""
import hashlib,json,struct,time,os
from pathlib import Path
import pytest

def test_frame_store_exact_ordered_roundtrip_and_no_overwrite(tmp_path):
    from tools.opaque_frame_store import Writer,Reader
    root=tmp_path/'frame';w=Writer(root,3,1,{'scene':'fixture'},block_pixels=2)
    values={'colour':struct.pack('<6d',-0.,1e-310,.2,.3,.4,.5),'mask':struct.pack('<2d',1.,.5),'gcolour':struct.pack('<6d',1.,-2.,3.,4.,-5.,6.),'gmask':struct.pack('<2d',.25,-.5)}
    w.append(0,2,values);tail={k:v[:len(v)//2] for k,v in values.items()};w.append(2,1,tail);receipt=w.finish()
    got=list(Reader(root,receipt['manifest_sha256'],{'scene':'fixture'}).blocks())
    assert [(a,b) for a,b,_ in got]==[(0,2),(2,1)]
    for name in values:assert got[0][2][name]==values[name] and got[1][2][name]==tail[name]
    with pytest.raises(FileExistsError):Writer(root,3,1,{'scene':'fixture'})

def test_frame_store_rejects_incomplete_reordered_and_oversized_writes(tmp_path):
    from tools.opaque_frame_store import Writer
    w=Writer(tmp_path/'frame',3,1,{},block_pixels=2);v={'colour':bytes(48),'mask':bytes(16),'gcolour':bytes(48),'gmask':bytes(16)}
    for start,n in ((1,2),(0,3)):
        with pytest.raises(ValueError):w.append(start,n,v)
    assert w.next_pixel==0 and all(p.stat().st_size==0 for p in w.paths.values())
    w.append(0,2,v)
    with pytest.raises(ValueError,match='incomplete'):w.finish()
    assert not (w.root/'frame.json').exists();w.close()

def test_frame_store_pins_identity_and_checks_bytes_before_replay(tmp_path):
    from tools.opaque_frame_store import Writer,Reader
    root=tmp_path/'frame';w=Writer(root,1,1,{'wheel':'abc'});v={'colour':bytes(24),'mask':bytes(8),'gcolour':bytes(24),'gmask':bytes(8)};w.append(0,1,v);r=w.finish()
    with pytest.raises(ValueError,match='identity'):Reader(root,r['manifest_sha256'],{'wheel':'other'})
    with pytest.raises(ValueError,match='manifest hash'):Reader(root,'0'*64,{'wheel':'abc'})
    with (root/'colour.f64').open('r+b') as f:f.write(b'X')
    with pytest.raises(ValueError,match='block hash'):next(Reader(root,r['manifest_sha256'],{'wheel':'abc'}).blocks())
    with (root/'colour.f64').open('ab') as f:f.write(b'X')
    with pytest.raises(ValueError,match='length'):Reader(root,r['manifest_sha256'],{'wheel':'abc'})

def test_frame_store_full_hd_storage_and_ordered_cotangent_replay(tmp_path,record_property):
    from tools.opaque_frame_store import Writer,Reader
    start=time.perf_counter();root=Path(os.environ.get('GFFX_FRAME_STORE_ROOT',str(tmp_path/'fullhd')));w=Writer(root,1920,1080,{'fixture':'constant-binary64-storage-only'},block_pixels=512)
    pixel={'colour':struct.pack('<3d',.25,.5,.75),'mask':struct.pack('<d',1.),'gcolour':struct.pack('<3d',.5,-.25,.125),'gmask':struct.pack('<d',.25)}
    for offset in range(0,1920*1080,512):
        n=min(512,1920*1080-offset);w.append(offset,n,{k:v*n for k,v in pixel.items()})
    receipt=w.finish();write_seconds=time.perf_counter()-start;start=time.perf_counter();total=0.;seen=0;peak=0
    for offset,n,values in Reader(root,receipt['manifest_sha256'],{'fixture':'constant-binary64-storage-only'}).blocks():
        assert offset==seen;seen+=n;peak=max(peak,sum(map(len,values.values())))
        for k,v in values.items():assert v==pixel[k]*n
        # Exactly representable per-pixel VJP in canonical order; no renderer invocation.
        for i in range(n):total+=.5-.25+.125+.25
    assert seen==2073600 and total==1296000. and peak<=32768
    assert sum(p.stat().st_size for p in w.paths.values())==132710400
    record_property('fullhd_storage',json.dumps(dict(receipt=receipt,pixels=seen,ordered_vjp=total,payload_peak_bytes=peak,write_seconds=write_seconds,read_replay_seconds=time.perf_counter()-start,render_executed=False,scope='Full1080p output/cotangent storage and exact synthetic ordered replay,not scene accuracy or rendering throughput.')))
