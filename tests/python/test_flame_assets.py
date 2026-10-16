import importlib,pickle,hashlib,zipfile
import numpy as np
import pytest

def api():return importlib.import_module('tools.flame_assets')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_restricted_legacy_numeric_roundtrip_and_hash_refusal(tmp_path):
    p=tmp_path/'model.pkl';p.write_bytes(pickle.dumps({'x':np.arange(12,dtype=np.float64).reshape(4,3)},protocol=4))
    got=api().load_legacy(p,sha(p));np.testing.assert_array_equal(got['x'],np.arange(12).reshape(4,3))
    with pytest.raises(ValueError,match='hash'):api().load_legacy(p,'0'*64)

def test_restricted_legacy_refuses_arbitrary_callable(tmp_path):
    class Unsupported:
        def __reduce__(self):return eval,('1+1',)
    p=tmp_path/'unsupported.pkl';p.write_bytes(pickle.dumps(Unsupported()))
    with pytest.raises(ValueError,match='global'):api().load_legacy(p,sha(p))

def model_dict():
    V=4;J=5;K=400
    return dict(v_template=np.arange(12,dtype=float).reshape(4,3)/10,shapedirs=np.arange(V*3*K,dtype=float).reshape(V,3,K)/10000,posedirs=np.zeros((V,3,36)),J_regressor=np.full((J,V),.25),weights=np.tile([1.,0.,0.,0.,0.],(V,1)),kintree_table=np.array([[2**32-1,0,1,1,1],[0,1,2,3,4]],dtype=np.uint32),f=np.array([[0,1,2],[0,2,3]]))

def test_geometry_conversion_explicit_expression_offset_and_tree():
    d=model_dict();got=api().geometry_arrays(d,shape_count=2,expression_count=2)
    want=np.stack([d['shapedirs'][:,:,k] for k in (0,1,300,301)])
    np.testing.assert_array_equal(got['blend_basis'],want)
    np.testing.assert_array_equal(got['parents'],[-1,0,1,1,1])
    assert got['pose_basis'].shape==(36,4,3)

@pytest.mark.parametrize('bad',['faces','weights','tree','nonfinite'])
def test_geometry_conversion_rejects_invalid_data(bad):
    d=model_dict()
    if bad=='faces':d['f'][0,0]=4
    elif bad=='weights':d['weights'][0,0]=.8
    elif bad=='tree':d['kintree_table'][0,2]=4
    else:d['shapedirs'][0,0,0]=float('nan')
    with pytest.raises(ValueError):api().geometry_arrays(d,shape_count=2,expression_count=2)

def test_streamed_texture_selection_matches_independent_channels_and_nearest(tmp_path):
    mean=np.arange(4*4*3,dtype=np.float64).reshape(4,4,3)+60
    basis=np.arange(4*4*3*5,dtype=np.float64).reshape(4,4,3,5)/100
    p=tmp_path/'texture.npz';np.savez(p,mean=mean,tex_dir=basis)
    got=api().texture_arrays(p,sha(p),components=2,size=2)
    np.testing.assert_array_equal(got['texture_mean'],mean[::2,::2,::-1]/255)
    np.testing.assert_array_equal(got['texture_basis'],basis[::2,::2,::-1,:2].transpose(3,0,1,2)/255)
    assert got['largest_chunk_bytes']<=4*3*5*8

@pytest.mark.parametrize('bad',['fortran','object','components','truncated'])
def test_streamed_texture_rejects_unsupported_or_incomplete_input(tmp_path,bad):
    mean=np.zeros((4,4,3));basis=np.zeros((4,4,3,5));n=2
    if bad=='fortran':basis=np.asfortranarray(basis)
    elif bad=='object':basis=basis.astype(object)
    elif bad=='components':n=6
    p=tmp_path/'texture.npz';np.savez(p,mean=mean,tex_dir=basis)
    if bad=='truncated':p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises((ValueError,zipfile.BadZipFile)):
        api().texture_arrays(p,sha(p),components=n,size=2)

def test_obj_preserves_face_corner_uv_order_and_rejects_quad(tmp_path):
    p=tmp_path/'mesh.obj';p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/3 2/1 3/2\n')
    got=api().obj_arrays(p)
    np.testing.assert_array_equal(got['faces'],[[0,1,2]])
    np.testing.assert_array_equal(got['face_uv'],[[[0,0],[0,1],[1,1]]])
    p.write_text(p.read_text()+'f 1/1 2/2 3/3 1/1\n')
    with pytest.raises(ValueError):api().obj_arrays(p)

def test_landmark_tables_validate_shape_weights_and_face_indices():
    d={}
    for prefix,shape in [('static',(51,)),('dynamic',(79,17)),('full',(68,))]:
        d[prefix+'_lmk_faces_idx']=np.zeros(shape,dtype=np.int64)
        d[prefix+'_lmk_bary_coords']=np.tile([.2,.3,.5],(*shape,1))
    got=api().landmark_arrays(d,faces=2);assert got['contour_faces'].shape==(79,17)
    d['static_lmk_faces_idx'][0]=2
    with pytest.raises(ValueError):api().landmark_arrays(d,faces=2)

def test_legacy_torch_embedding_decodes_as_arrays_without_torch_execution(tmp_path):
    import torch
    p=tmp_path/'embed.npy'
    np.save(p,{'dynamic':torch.arange(12,dtype=torch.int64).reshape(3,4),'bary':torch.arange(12,dtype=torch.float32).reshape(3,4)},allow_pickle=True)
    got=api().load_legacy(p,sha(p))
    np.testing.assert_array_equal(got['dynamic'],np.arange(12).reshape(3,4))
    np.testing.assert_array_equal(got['bary'],np.arange(12,dtype=np.float32).reshape(3,4))

@pytest.mark.parametrize('bad',['truncated','nested_callable','stride','offset','gradient'])
def test_legacy_torch_storage_and_tensor_reject_invalid_inputs(bad):
    import io,torch
    f=api()
    if bad in ('truncated','nested_callable'):
        buffer=io.BytesIO();torch.save(torch.arange(4).storage(),buffer,_use_new_zipfile_serialization=False)
        data=buffer.getvalue()[:-1] if bad=='truncated' else pickle.dumps(eval)
        with pytest.raises(ValueError):f._storage_bytes(data)
    else:
        with pytest.raises(ValueError):f._rebuild_array(np.arange(4),1 if bad=='offset' else 0,(2,2),(1,2) if bad=='stride' else (2,1),bad=='gradient',{})

def test_full_landmarks_accept_only_explicit_singleton_batch():
    d={}
    for prefix,shape in [('static',(51,)),('dynamic',(79,17)),('full',(1,68))]:
        d[prefix+'_lmk_faces_idx']=np.zeros(shape,dtype=np.int64)
        d[prefix+'_lmk_bary_coords']=np.tile([.2,.3,.5],(*shape,1))
    got=api().landmark_arrays(d,faces=2)
    assert got['full_landmark_faces'].shape==(68,)
    d['full_lmk_faces_idx']=np.zeros((2,68),dtype=np.int64)
    with pytest.raises(ValueError):api().landmark_arrays(d,faces=2)

def test_landmark_partition_check_uses_stored_values_without_float32_sum_error():
    d={}
    # Exact sum of these stored float32 values is within1e-7; float32 reduction exceeds it.
    values=np.array([.38456762,.37298355,.2424489],dtype=np.float32)
    assert abs(values.astype(np.float64).sum()-1)<1e-7
    for prefix,shape in [('static',(51,)),('dynamic',(79,17)),('full',(68,))]:
        d[prefix+'_lmk_faces_idx']=np.zeros(shape,dtype=np.int64)
        d[prefix+'_lmk_bary_coords']=np.tile(values,(*shape,1))
    got=api().landmark_arrays(d,faces=2)
    np.testing.assert_array_equal(got['contour_barycentric'][0,0],values.astype(np.float64))
