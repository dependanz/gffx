"""No-render cost admission on the actual original scene; never scale a crop."""
import os,json,hashlib
from pathlib import Path
import torch,numpy as np,pytest

def test_original_opaque_startup_capacity_and_missing_frame_anchors(record_property):
    from tools.corpus_registry import load_entry
    from tools.deferred_flame import create_scene
    from tools.opaque_mesh_preflight import candidate_counts
    from tools.deferred_progressive_admission import forecast
    from gffx.torch._opaque_scene import _project,_normals,_horizon_cuts,_extra_geometry_bytes
    from gffx.torch._deferred import _plan_workload
    from copy import copy
    from gffx.torch.camera import Camera
    from gffx.torch import render
    blob=load_entry(Path('D:/devspace/gffx/corpora'),'synthetic_flame_m1',allow_development=True)['files']['scene.json'];model,cam=create_scene(json.loads(blob),'principled');rows=[]
    for size in (32,64,128):
        camera=Camera.from_intrinsics(cam.fx*size/128,cam.fy*size/128,cam.cx*size/128,cam.cy*size/128,size,size,.1,10.,pose=cam.pose)
        packed,resolved,pixels,keep,clip=_project(model,camera,render.CULL_BACK)
        counts=candidate_counts(pixels.numpy(),packed.faces.numpy()[keep.numpy()],size,size)
        row=dict(width=size,height=size,max_candidates=int(counts.max()),pixels_over_cap=int((counts>32).sum()),candidate_cap=32,candidate_admitted=bool((counts<=32).all()))
        try:cuts,curves=_horizon_cuts(packed,resolved,clip,_normals(packed),keep);row['horizon']={'status':'composed; native topology still required','linear_columns':cuts.shape[-1],'curve_column':curves is not None}
        except (NotImplementedError,OverflowError) as e:row['horizon']={'status':'refused','reason':str(e)}
        proxy=copy(camera);proxy.image_height=1;proxy.image_width=800;extra=_extra_geometry_bytes(model,camera);row['opaque_geometry_reserve_bytes']=extra
        try:row['one_pixel_workload']=_plan_workload(model,proxy,1,67108864-1048576-65536-extra,('colour','mask'),None)
        except ValueError as e:row['one_pixel_workload']={'status':'refused','reason':str(e)}
        rows.append(row)
    with pytest.raises(ValueError,match='two complete anchors'):forecast([],{'profile':{'width':1920,'height':1080}})
    record_property('frame_cost',json.dumps(dict(status='not_admitted',render_executed=False,scene_sha256=hashlib.sha256(blob).hexdigest(),startup_capacity=rows,matching_complete_original_opaque_anchors=0,forecast_seconds=None,reasons=['No two complete matching original-mesh opaque anchors; previous pixel crops,small different scenes and hard/soft render timings are ineligible.','Public bounded windows now exist; original-mesh horizon/capacity/storage inventory and matching complete anchors must still admit this workload.'],resident_1080p_reserve_bytes=1048576+65536+1920*1080*4*8*3,active_limit_bytes=67108864,deadline_seconds=120,scope='Original scene startup capacity and missing-evidence refusal; no full-frame render or speed estimate.')))
