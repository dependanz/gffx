"""Complete-face sums require all pieces and outward numerical acceptance."""
from copy import deepcopy
from fractions import Fraction as Q
import pytest

def fixtures():
    data={'selected':{'visible_parts':[{'face':0},{'face':1},{'face':1}]},'faces':[{'face':10},{'face':11}]}
    row=lambda i:dict(status='qualified',case=dict(piece=i,motion='view_x',anchor=1),result=dict(piece=i,face=11,motion='view_x',value=['[1e-30,2e-30]']*3,derivative=['[-2e-30,-1e-30]']*3))
    return data,[row(1),row(2)]

def test_face_sum_includes_every_signed_tiny_piece_outward():
    from tools.opaque_face_certificate import compose
    data,rows=fixtures();r=compose(data,face=11,motion='view_x',certificates=rows)
    assert r['pieces']==[1,2] and r['status']=='qualified_face' and r['face']==11
    for key,exact in [('value',Q('3e-30')),('derivative',Q('-3e-30'))]:
        for lo,hi in r[key]:assert Q(lo)<=exact<=Q(hi) and (Q(lo)>0 if key=='value' else Q(hi)<0)

@pytest.mark.parametrize('bad',['missing','duplicate','motion','face','case','failed','nan','reverse','wide','sum_wide'])
def test_face_sum_refuses_incomplete_mismatched_or_weak_evidence(bad):
    from tools.opaque_face_certificate import compose
    data,rows=fixtures()
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows[1]=deepcopy(rows[0])
    if bad=='motion':rows[1]['result']['motion']='normal_tilt'
    if bad=='face':rows[1]['result']['face']=10
    if bad=='case':rows[1]['case']['piece']=1
    if bad=='failed':rows[1]['status']='unresolved'
    if bad in ('nan','reverse','wide'):rows[1]['result']['value'][0]={'nan':'[NaN,1]','reverse':'[2,1]','wide':'[0,3e-12]'}[bad]
    if bad=='sum_wide':
        for row in rows:row['result']['value'][0]='[0,1.5e-12]'
    with pytest.raises(ValueError):compose(data,face=11,motion='view_x',certificates=rows)


@pytest.mark.parametrize('change',['none','bundle_digest','input','certificate_bytes','certificate_digest','missing','duplicate','failed','sum','motion'])
def test_face_bundle_checks_input_artifact_bytes_and_recomputed_sum(tmp_path,change):
    import hashlib,json
    from tools.opaque_face_certificate import compose,load_face_bundle
    data,certs=fixtures();rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for i,r in enumerate(certs):
        p=tmp_path/(str(i)+'.json');p.write_text(json.dumps(r),encoding='utf8');rows.append(dict(path=str(p),sha256=sha(p)))
    result=compose(data,face=11,motion='view_x',certificates=certs);bundle=dict(status='qualified_face',input_sha256='a'*64,certificates=rows,result=result)
    if change=='input':bundle['input_sha256']='b'*64
    if change=='certificate_bytes':(tmp_path/'0.json').write_text('{}',encoding='utf8')
    if change=='certificate_digest':rows[0]['sha256']='0'*64
    if change=='missing':rows.pop()
    if change=='duplicate':rows[1]=rows[0]
    if change=='failed':
        certs[0]['status']='unresolved';(tmp_path/'0.json').write_text(json.dumps(certs[0]),encoding='utf8');rows[0]['sha256']=sha(tmp_path/'0.json')
    if change=='sum':bundle['result']['derivative'][0]=['0','0']
    if change=='motion':bundle['result']['motion']='normal_tilt'
    p=tmp_path/'bundle.json';p.write_text(json.dumps(bundle),encoding='utf8');digest='0'*64 if change=='bundle_digest' else sha(p)
    if change=='none':assert load_face_bundle(p,sha256=digest,data=data,input_sha256='a'*64)==result
    else:
        with pytest.raises(ValueError):load_face_bundle(p,sha256=digest,data=data,input_sha256='a'*64)


def pixel_fixtures():
    owners=[0,1,1,1,2,2,3,3,3,3,3,4,5]
    data={'selected':{'visible_parts':[{'face':i} for i in owners]},'faces':[{'face':10+i} for i in range(6)]}
    rows=[dict(status='qualified',case=dict(piece=i,motion='view_x',anchor=0),result=dict(piece=i,face=10+f,motion='view_x',value=['[1e-30,2e-30]']*3,derivative=['[-2e-30,-1e-30]']*3)) for i,f in enumerate(owners)]
    return data,rows


def test_pixel_sum_preserves_every_signed_tiny_piece():
    from tools.opaque_face_certificate import compose_pixel
    data,rows=pixel_fixtures();r=compose_pixel(data,motion='view_x',certificates=rows)
    assert r['pieces']==list(range(13)) and r['faces']==list(range(10,16)) and r['status']=='qualified_pixel'
    for key,exact in [('value',Q('19.5e-30')),('derivative',Q('-19.5e-30'))]:
        for lo,hi in r[key]:assert Q(lo)<=exact<=Q(hi) and (Q(lo)>0 if key=='value' else Q(hi)<0)


@pytest.mark.parametrize('bad',['missing','duplicate','motion','face','case','failed','nan','reverse','wide'])
def test_pixel_sum_refuses_incomplete_mismatched_or_invalid_evidence(bad):
    from tools.opaque_face_certificate import compose_pixel
    data,rows=pixel_fixtures()
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows[12]=deepcopy(rows[0])
    if bad=='motion':rows[12]['result']['motion']='normal_tilt'
    if bad=='face':rows[12]['result']['face']=10
    if bad=='case':rows[12]['case']['piece']=0
    if bad=='failed':rows[12]['status']='unresolved'
    if bad in ('nan','reverse','wide'):rows[12]['result']['value'][0]={'nan':'[NaN,1]','reverse':'[2,1]','wide':'[0,3e-12]'}[bad]
    with pytest.raises(ValueError):compose_pixel(data,motion='view_x',certificates=rows)


def test_pixel_sum_refuses_combined_uncertainty_even_when_every_face_passes():
    from tools.opaque_face_certificate import compose,compose_pixel
    data,rows=pixel_fixtures()
    for r in rows:r['result']['derivative']=['[0,1.6e-13]']*3
    for face in range(10,16):assert compose(data,face=face,motion='view_x',certificates=[r for r in rows if r['result']['face']==face])['status']=='qualified_face'
    with pytest.raises(ValueError,match='pixel reference radius'):compose_pixel(data,motion='view_x',certificates=rows)


@pytest.mark.parametrize('change',['none','bundle_digest','input','certificate_bytes','certificate_digest','missing','duplicate','failed','sum','motion'])
def test_pixel_bundle_checks_input_artifact_bytes_and_recomputed_sum(tmp_path,change):
    import hashlib,json
    from tools.opaque_face_certificate import compose_pixel,load_pixel_bundle
    data,certs=pixel_fixtures();rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for i,r in enumerate(certs):
        p=tmp_path/(str(i)+'.json');p.write_text(json.dumps(r),encoding='utf8');rows.append(dict(path=str(p),sha256=sha(p)))
    result=compose_pixel(data,motion='view_x',certificates=certs);bundle=dict(status='qualified_pixel',input_sha256='a'*64,certificates=rows,result=result)
    if change=='input':bundle['input_sha256']='b'*64
    if change=='certificate_bytes':(tmp_path/'0.json').write_text('{}',encoding='utf8')
    if change=='certificate_digest':rows[0]['sha256']='0'*64
    if change=='missing':rows.pop()
    if change=='duplicate':rows[1]=rows[0]
    if change=='failed':
        certs[0]['status']='unresolved';(tmp_path/'0.json').write_text(json.dumps(certs[0]),encoding='utf8');rows[0]['sha256']=sha(tmp_path/'0.json')
    if change=='sum':bundle['result']['derivative'][0]=['0','0']
    if change=='motion':bundle['result']['motion']='normal_tilt'
    p=tmp_path/'bundle.json';p.write_text(json.dumps(bundle),encoding='utf8');digest='0'*64 if change=='bundle_digest' else sha(p)
    if change=='none':assert load_pixel_bundle(p,sha256=digest,data=data,input_sha256='a'*64)==result
    else:
        with pytest.raises(ValueError):load_pixel_bundle(p,sha256=digest,data=data,input_sha256='a'*64)


@pytest.mark.parametrize('motion',['world_x','camera_x','world_shear_xz'])
def test_physical_pixel_sum_binds_input_and_world_source(motion):
    from tools.opaque_face_certificate import compose_pixel
    data,rows=pixel_fixtures()
    for row in rows:
        row['case']['motion']=row['result']['motion']=motion
        row.update(input_sha256='a'*64,world_profile_sha256='b'*64)
    result=compose_pixel(data,motion=motion,certificates=rows,input_sha256='a'*64,world_profile_sha256='b'*64)
    assert result['motion']==motion and result['world_profile_sha256']=='b'*64
    assert result['pieces']==list(range(13))
    for lo,hi in result['derivative']:assert Q(lo)<=Q('-19.5e-30')<=Q(hi)<0


@pytest.mark.parametrize('bad',['missing_input','missing_world','wrong_input','wrong_world','invalid_digest','packed_row','missing_piece','duplicate_piece','combined_radius'])
def test_physical_pixel_sum_refuses_unbound_or_incomplete_evidence(bad):
    from tools.opaque_face_certificate import compose_pixel
    data,rows=pixel_fixtures();kw=dict(input_sha256='a'*64,world_profile_sha256='b'*64)
    for row in rows:
        row['case']['motion']=row['result']['motion']='world_x'
        row.update(input_sha256='a'*64,world_profile_sha256='b'*64)
    if bad=='missing_input':kw.pop('input_sha256')
    if bad=='missing_world':kw.pop('world_profile_sha256')
    if bad=='wrong_input':rows[7]['input_sha256']='c'*64
    if bad=='wrong_world':rows[7]['world_profile_sha256']='c'*64
    if bad=='invalid_digest':kw['world_profile_sha256']='z'*64
    if bad=='packed_row':rows[7]['case']['motion']=rows[7]['result']['motion']='view_x'
    if bad=='missing_piece':rows.pop()
    if bad=='duplicate_piece':rows[-1]=deepcopy(rows[0])
    if bad=='combined_radius':
        for row in rows:row['result']['derivative']=['[0,1.6e-13]']*3
    with pytest.raises(ValueError):compose_pixel(data,motion='world_x',certificates=rows,**kw)


@pytest.mark.parametrize('change',['none','world_header','world_expected','certificate_world','input_header'])
def test_physical_pixel_bundle_requires_expected_source_identity(tmp_path,change):
    import hashlib,json
    from tools.opaque_face_certificate import compose_pixel,load_pixel_bundle
    data,certs=pixel_fixtures();rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for i,row in enumerate(certs):
        row['case']['motion']=row['result']['motion']='world_x';row.update(input_sha256='a'*64,world_profile_sha256='b'*64)
        p=tmp_path/(str(i)+'.json');p.write_text(json.dumps(row));rows.append(dict(path=str(p),sha256=sha(p)))
    result=compose_pixel(data,motion='world_x',certificates=certs,input_sha256='a'*64,world_profile_sha256='b'*64)
    bundle=dict(status='qualified_pixel',input_sha256='a'*64,world_profile_sha256='b'*64,certificates=rows,result=result);expected='b'*64
    if change=='world_header':bundle['world_profile_sha256']='c'*64
    if change=='world_expected':expected='c'*64
    if change=='input_header':bundle['input_sha256']='c'*64
    if change=='certificate_world':
        certs[0]['world_profile_sha256']='c'*64;(tmp_path/'0.json').write_text(json.dumps(certs[0]));rows[0]['sha256']=sha(tmp_path/'0.json')
    p=tmp_path/'bundle.json';p.write_text(json.dumps(bundle))
    if change=='none':assert load_pixel_bundle(p,sha256=sha(p),data=data,input_sha256='a'*64,world_profile_sha256=expected)==result
    else:
        with pytest.raises(ValueError):load_pixel_bundle(p,sha256=sha(p),data=data,input_sha256='a'*64,world_profile_sha256=expected)
