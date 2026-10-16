"""Bounded attribution of remaining eager dispatch; not performance acceptance."""
import cProfile,pstats,io,json,os,time
from pathlib import Path
import torch


def test_nativecurve_remaining_dispatch_profile(record_property):
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from gffx.torch.camera import Camera
    from gffx.torch import render
    entry=load_entry(Path('D:/devspace/gffx/corpora'),'synthetic_face_m1',allow_development=True)
    model,cam=create_scene(json.loads(entry['files']['scene.json']),'principled',device='cuda')
    camera=Camera.from_intrinsics(cam.fx/4,cam.fy/4,cam.cx/4,cam.cy/4,32,32,.1,10.,pose=cam.pose)
    profile=cProfile.Profile()
    with torch.no_grad():
        profile.enable();image=render(model,camera,outputs=('colour','mask'));torch.cuda.synchronize();profile.disable()
    assert torch.isfinite(image['colour']).all()
    stats=pstats.Stats(profile);rows=[]
    for (file,line,name),(primitive,calls,own,cumulative,callers) in stats.stats.items():
        rows.append(dict(file=file,line=line,name=name,calls=calls,own_seconds=own,cumulative_seconds=cumulative))
    rows.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
    path=Path(os.environ['GFFX_PRINCIPLED_PROFILE_ROWS']);assert not path.exists();path.write_text(json.dumps(rows,indent=2))
    record_property('profile_scope','One instrumented32x32 diagnostic; profiler timings are not renderer performance acceptance')
    record_property('profile_rows_path',str(path))
