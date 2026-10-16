"""Bounded original-scene quadratic horizon localization,without rendering."""
import json,hashlib,os,time
from pathlib import Path
from fractions import Fraction as Q
import numpy as np

def test_original_curved_horizon_localization(record_property):
    from tools.corpus_registry import load_entry
    from tools.deferred_flame import create_scene
    from tools.opaque_mesh_preflight import selected_faces
    from tools.opaque_moving_reference import moving_fragments
    from gffx.torch._opaque_scene import _project,_normals
    from gffx.torch import render
    blob=load_entry(Path('D:/devspace/gffx/corpora'),'synthetic_flame_m1',allow_development=True)['files']['scene.json']
    model,camera=create_scene(json.loads(blob),'principled')
    packed,camera,pixels,keep,clip=_project(model,camera,render.CULL_BACK)
    v=packed.vertices.detach().numpy();f=packed.faces.numpy();n=_normals(packed).detach().numpy();view=camera.pose[0,:3,3].numpy()-v[f];px=pixels.detach().numpy();cw=clip[:,3].numpy()
    rows=[];front=np.flatnonzero(keep.numpy());start=time.monotonic()
    for face in front:
        if time.monotonic()-start>90:raise TimeoutError('curved preflight90s')
        # Exact rational signs of binary input normal/view samples. This is not a
        # float tolerance or a sampled coverage certificate.
        nn=[[Q(float(z)) for z in p] for p in n[face]];vv=[[Q(float(z)) for z in p] for p in view[face]]
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        diag=[dot(nn[i],vv[i]) for i in range(3)]
        cross=[dot(nn[i],vv[j])+dot(nn[j],vv[i]) for i,j in ((0,1),(0,2),(1,2))]
        if min(diag+cross)>0 or max(diag+cross)<0:continue
        tri=px[f[face],:2];area=abs(float(np.linalg.det(np.stack((tri[1]-tri[0],tri[2]-tri[0])))))/2
        rows.append(dict(face=int(face),vertices=f[face].tolist(),corners=tri.tolist(),clip_w=cw[f[face]].tolist(),normal=n[face].tolist(),view=view[face].tolist(),diagonal=list(map(str,diag)),doubled_cross=list(map(str,cross)),projected_area=area,corner_crossing=bool(min(diag)<0<max(diag))))
    rows.sort(key=lambda r:(not r['corner_crossing'],-r['projected_area'],r['face']))
    choices=[];refusals=[];attempted=set()
    def check():
        if time.monotonic()-start>90:raise TimeoutError("curved preflight90s")
    for row in rows:
        if not row['corner_crossing']:continue
        d=list(map(Q,row['diagonal']));cross=list(map(Q,row['doubled_cross']));tri=np.array(row['corners'])
        for k,(i,j) in enumerate(((0,1),(0,2),(1,2))):
            if d[i]*d[j]>=0:continue
            coeff=[float(d[i]+d[j]-cross[k]),float(cross[k]-2*d[i]),float(d[i])]
            roots=np.roots(coeff if coeff[0] else coeff[1:])
            t=next(float(z.real) for z in roots if abs(z.imag)<1e-12 and 0<z.real<1)
            beta=np.zeros(3);beta[i]=1-t;beta[j]=t
            weights=beta*np.array(row['clip_w']);weights/=weights.sum();where=weights@tri
            x,y=map(int,np.floor(where));ff=f[front];ids=front[selected_faces(px,ff,x,y)]
            if not (0<=x<128 and 0<=y<128) or len(ids)>32:continue
            vertices=sorted(set(f[ids].ravel().tolist()));index={a:b for b,a in enumerate(vertices)}
            local=px[vertices].copy();local[:,0]-=x;local[:,1]-=y
            faces=[[index[int(z)] for z in q] for q in f[ids]]
            if (x,y) in attempted:continue
            if len(attempted)==8:break
            attempted.add((x,y))
            from tools.opaque_shading_intervals import Unresolved
            try:parts,cert=moving_fragments(local.tolist(),faces,[[0,0,0]]*len(vertices),checkpoint=check)
            except Unresolved as e:
                refusals.append(dict(origin=[x,y],face=row['face'],reason=str(e)));continue
            found=[p for p in parts if int(ids[p['face']])==row['face']]
            if not found:continue
            choices.append(dict(face=row['face'],origin=[x,y],candidate_faces=ids.tolist(),root_selection_only=[t,where.tolist()],screen_vertices=local.tolist(),faces=faces,original_vertex_ids=vertices,visible_parts=parts,certificate=cert,selected_face_parts=found))
            break
        if choices:break
    assert rows
    Path(os.environ['GFFX_CURVED_PREFLIGHT']+'.localization.json').write_text(json.dumps(dict(faces=rows,refusals=refusals,choices=choices),indent=2,default=str),encoding='utf-8')
    assert choices
    result=dict(status='localized',render_executed=False,scene_sha256=hashlib.sha256(blob).hexdigest(),retained_faces=len(front),refusals=refusals,quadratic_uncertain_faces=len(rows),corner_crossing_faces=sum(r['corner_crossing'] for r in rows),faces=rows,selected=choices[0],seconds=time.monotonic()-start,scope='Exact rational whole-face Bernstein signs and depth-ordered pixel fragments; approximate edge root selects only the probe,not a reference or admission.')
    path=Path(os.environ['GFFX_CURVED_PREFLIGHT']);path.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8');record_property('curved_preflight',json.dumps({k:v for k,v in result.items() if k not in ('faces','selected')},default=str))
