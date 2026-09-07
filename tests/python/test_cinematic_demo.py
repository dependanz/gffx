"""High-resolution presentation safety and genuine camera-recovery fixtures."""
import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_showcase_refinement_preserves_valid_triangles_and_increases_detail():
    from tools.testing_dashboard.cinematic import refine_fixture
    import torch
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    points,faces=refine_fixture(v,f)
    assert points.shape==(6,3) and faces.shape==(4,3)
    assert faces.dtype==torch.int32 and int(faces.max())<len(points)
    assert torch.isfinite(points).all()

def test_demo_highlight_is_reciprocal_and_has_conservative_energy_bound():
    from tools.testing_dashboard.cinematic import highlight_lobe
    import torch,math
    n=torch.tensor([0.,0.,1.],dtype=torch.float64)
    k=torch.arange(8192,dtype=torch.float64);z=(k+.5)/8192;phi=k*(math.pi*(3-math.sqrt(5)))
    l=torch.stack([(1-z*z).sqrt()*phi.cos(),(1-z*z).sqrt()*phi.sin(),z],-1)
    for theta in [0.,.5,1.2]:
        v=torch.tensor([math.sin(theta),0,math.cos(theta)],dtype=torch.float64)
        f=.88/math.pi+.12*highlight_lobe(n,l,v)
        assert abs(float(2*math.pi*(f*z).mean()))<=1.001
        assert torch.allclose(highlight_lobe(n,l,v),highlight_lobe(n,v,l),atol=1e-12)

def test_cinematic_camera_fit_changes_pose_and_focal_and_converges():
    from tools.testing_dashboard.cinematic import solve_shot
    import torch
    result=solve_shot()
    assert result['history'][0]['loss']>100
    assert result['history'][-1]['loss']<1e-5
    a=torch.tensor(result['history'][0]['parameters']);b=torch.tensor(result['history'][-1]['parameters'])
    assert (a[:6]-b[:6]).abs().max()>.1
    assert abs(float(a[6]-b[6]))>.05
    assert result['objective']=='known 3D landmark reprojection; not image loss'
    from tools.testing_dashboard.cinematic import shot_scene,shot_camera,render_beauty,TARGET
    s=shot_scene();reference=render_beauty(s,shot_camera(torch.tensor(TARGET),height=48,width=80)).float()
    before=render_beauty(s,shot_camera(a.float(),height=48,width=80)).float()
    after=render_beauty(s,shot_camera(b.float(),height=48,width=80)).float()
    assert (before-reference).abs().mean()>3
    assert (after-reference).abs().mean()<.5

def test_png_presentation_assets_round_trip_without_image_dependencies(tmp_path):
    from tools.testing_dashboard.presentation import png_rgb,AssetWriter
    import zlib,struct
    pixels=bytes([255,0,0,0,255,0,0,0,255,255,255,255])
    data=png_rgb(2,2,pixels)
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    chunks=[];offset=8
    while offset<len(data):
        size=struct.unpack('!I',data[offset:offset+4])[0];kind=data[offset+4:offset+8];value=data[offset+8:offset+8+size]
        assert zlib.crc32(kind+value)&0xffffffff==struct.unpack('!I',data[offset+8+size:offset+12+size])[0]
        if kind==b'IDAT':chunks.append(value)
        offset+=size+12
    assert zlib.decompress(b''.join(chunks))==b'\0'+pixels[:6]+b'\0'+pixels[6:]
    writer=AssetWriter(tmp_path,budget=1024)
    item=writer.put(data)
    assert writer.put(data)==item and writer.bytes==len(data)
    assert len(list(tmp_path.glob('*.png')))==1
    with pytest.raises(ValueError,match='budget'):AssetWriter(tmp_path/'small',budget=1).put(data)

def test_presentation_route_is_confined_and_missing_assets_are_explicit(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    from tools.testing_dashboard.presentation import AssetWriter,png_rgb
    from urllib.request import urlopen
    from urllib.error import HTTPError
    writer=AssetWriter(tmp_path/'presentation');item=writer.put(png_rgb(1,1,b'\xff\x00\x00'))
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        base='http://'+server.address
        with urlopen(base+item['url']) as response:
            assert response.headers['Content-Type']=='image/png'
            assert response.read().startswith(b'\x89PNG')
        for path in ['/testing/media/../../testing.sqlite','/testing/media/'+'0'*64+'.png']:
            with pytest.raises(HTTPError) as caught:urlopen(base+path)
            assert caught.value.code==404

def test_cinematic_composition_has_finite_distinct_lighting():
    from tools.testing_dashboard.cinematic import shot_scene,shot_camera,render_beauty,TARGET
    import torch
    scene=shot_scene();camera=shot_camera(torch.tensor(TARGET),height=48,width=80)
    a=render_beauty(scene,camera,0);b=render_beauty(scene,camera,.7)
    assert a.shape==(48,80,3) and a.dtype==torch.uint8
    assert not torch.equal(a,b)
    assert torch.equal(a,render_beauty(scene,camera,0))
    # The demo currently uses hard rasterization without near-plane clipping: do not
    # submit a stage triangle spanning behind the camera (the initial preview did).
    from tools.testing_dashboard.cinematic import INITIAL
    for p in [TARGET,INITIAL]:
        assert shot_camera(torch.tensor(p),height=48,width=80).project(scene['vertices'],in_front_only=True).valid.all()
