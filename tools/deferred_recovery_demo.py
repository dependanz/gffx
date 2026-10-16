"""Fixed finite albedo recovery through the installed renderer; no consumer code."""
import json,math,time,hashlib,ctypes
from pathlib import Path

def solve(fixtures,output):
    import torch
    from gffx.torch import scene,render
    from gffx.torch.camera import Camera
    from gffx.torch._deferred import _plan_workload
    from tools.corpus_registry import load_entry
    from tools.deferred_head import create_scene
    from tools import shading_independent_evaluator as scalar,principled_direct_images as reference
    from tools.testing_dashboard.presentation import png_rgb
    from tools.testing_dashboard.image_showcase import _FONT
    from tools.shading_cuda_closeout import digest
    assert digest(scalar.__file__)==reference.SCALAR_HASH
    output.mkdir();report=output/'report.json';deadline=time.perf_counter()+90
    row=dict(status='working',steps=20,learning_rate=12.,size=64,target_parameters=[.2,.4,.6],initial_parameters=[.72,.12,.25],losses=[],source_sha256=digest(__file__),reference_sha256=reference.SCALAR_HASH)
    def save():report.write_text(json.dumps(row,indent=2),encoding='utf-8')
    def check():
        if time.perf_counter()>deadline:raise TimeoutError('90-second finite recovery deadline')
    try:
        entry=load_entry(fixtures,'synthetic_face_m1',allow_development=True);data=json.loads(entry['files']['scene.json']);row['scene_sha256']=hashlib.sha256(entry['files']['scene.json']).hexdigest()
        with torch.no_grad():
            base,cam=create_scene(data,'classic')
            camera=Camera.from_intrinsics(cam.fx*.5,cam.fy*.5,cam.cx*.5,cam.cy*.5,64,64,.1,10.,pose=cam.pose)
            row['geometry_plan']=_plan_workload(base,camera,1,67108864,('normal','world_position','mask'))
            geometry=render(base,camera,outputs=('normal','world_position','mask'))
        albedo=torch.tensor(row['initial_parameters'],dtype=torch.float64,requires_grad=True)
        old=next(p for p in base.prims if type(p)==scene.Material)
        replacement=scene.Material(old.name,old.evaluator,dict(old.parameters,albedo=albedo))
        model=scene.Scene([replacement if p is old else p for p in base.prims])
        row['fit_plan']=_plan_workload(model,camera,1,67108864,('colour','mask'))
        fixed=[camera.fx,camera.fy,camera.cx,camera.cy,camera.pose]
        for prim in model.prims:
            if type(prim)==scene.Mesh:fixed.extend([prim.vertices,prim.faces,*[v.values for v in prim.attributes.values()]])
            if isinstance(prim,scene.Material):fixed.extend(v for v in prim.parameters.values() if isinstance(v,torch.Tensor) and v is not albedo)
        frozen=[(v._version,v.detach().clone()) for v in fixed]
        mask=geometry['mask'][0].bool();constant=torch.zeros((64,64,3),dtype=torch.float64);slope=torch.zeros_like(constant)
        light=[1.,1.,2.];light=[v/math.sqrt(6) for v in light]
        for y,x in mask.nonzero().tolist():
            check();n=geometry['normal'][0,y,x].tolist();point=geometry['world_position'][0,y,x].tolist();view=[-point[0],-point[1],3-point[2]]
            norm=math.hypot(*n);cosine=max(0.,math.fsum(a/norm*b for a,b in zip(n,light)))
            zero=scalar.classic(n,view,light,albedo=[0.,0.,0.],specular=.1,exponent=2.)
            one=scalar.classic(n,view,light,albedo=[1.,1.,1.],specular=.1,exponent=2.)
            constant[y,x]=torch.tensor([v*cosine for v in zero],dtype=torch.float64)
            slope[y,x]=torch.tensor([(a-b)*cosine for a,b in zip(one,zero)],dtype=torch.float64)
        target=constant+slope*torch.tensor(row['target_parameters'],dtype=torch.float64)
        max_error=0.;initial=None
        for iteration in range(21):
            check();actual=render(model,camera,outputs=('colour','mask'))['colour'][0]
            independent=constant+slope*albedo.detach()
            torch.testing.assert_close(actual,independent,atol=1e-10,rtol=1e-9)
            max_error=max(max_error,float((actual.detach()-independent).abs().max()))
            loss=(actual[mask]-target[mask]).square().mean();row['losses'].append(float(loss.detach()))
            if iteration==0:initial=actual.detach().clone()
            if iteration==20:final=actual.detach().clone();break
            gradient,=torch.autograd.grad(loss,albedo)
            assert torch.isfinite(gradient).all()
            with torch.no_grad():albedo.sub_(12*gradient)
            assert bool(((albedo>=0)&(albedo<=1)&torch.isfinite(albedo)).all())
            save()
        row.update(initial_mse=row['losses'][0],final_mse=row['losses'][-1],parameters=albedo.detach().tolist(),max_parameter_error=float((albedo.detach()-albedo.new_tensor(row['target_parameters'])).abs().max()),max_independent_model_error=max_error,covered_pixels=int(mask.sum()),fixed_inputs_unchanged=all(v._version==version and torch.equal(v,snapshot) for v,(version,snapshot) in zip(fixed,frozen)))
        assert row['fixed_inputs_unchanged'] and row['final_mse']<.01*row['initial_mse'] and row['max_parameter_error']<.03
        row['arrays']={}
        for name,array in [('target',target),('initial',initial),('final',final)]:
            array=array.contiguous();blob=ctypes.string_at(array.data_ptr(),array.numel()*array.element_size());file=output/(name+'.f64');file.open('xb').write(blob);row['arrays'][name]=dict(path=str(file),sha256=hashlib.sha256(blob).hexdigest(),shape=[64,64,3],bytes=len(blob))
        board=torch.tensor([13,20,32],dtype=torch.uint8).expand(820,1160,3).clone()
        glyphs=dict(_FONT,**{'5':'11111 10000 10000 11110 00001 00001 11110','6':'01110 10000 10000 11110 10001 10001 01110','7':'11111 00001 00010 00100 01000 01000 01000','8':'01110 10001 10001 01110 10001 10001 01110','9':'01110 10001 10001 01111 00001 00001 01110','.':'00000 00000 00000 00000 00000 00110 00110',':':'00000 00110 00110 00000 00110 00110 00000'})
        def text(x,y,value,scale=2):
            value=value.upper();assert all(c==' ' or c in glyphs for c in value),value
            for i,c in enumerate(value):
                if c==' ':continue
                for yy,line in enumerate(glyphs[c].split()):
                    for xx,on in enumerate(line):
                        if on=='1':board[y+yy*scale:y+(yy+1)*scale,x+(i*6+xx)*scale:x+(i*6+xx+1)*scale]=board.new_tensor([219,230,241])
        def panel(value,x,y,title,caption,error=False):
            if error:
                q=((value-target).abs().amax(-1)*20).clamp(0,1);rgb=torch.stack((q,q*.25,q*.65),-1)
            else:
                linear=value.clamp(0,1);rgb=torch.where(linear<=.0031308,12.92*linear,1.055*linear.pow(1/2.4)-.055)
            text(x,y,title,3);pixels=(rgb*255).round().to(torch.uint8).repeat_interleave(4,0).repeat_interleave(4,1);board[y+35:y+291,x:x+256]=pixels;text(x,y+306,caption,2)
        text(38,24,'FIT A MATERIAL THROUGH THE RENDERER',4);text(38,68,'Fixed head / independent scalar target / 20 fresh gradient steps',2)
        panel(target,38,116,'TARGET','Independent material values')
        panel(initial,416,116,'START',f'MSE {row["initial_mse"]:.5g}')
        panel(final,794,116,'FITTED',f'MSE {row["final_mse"]:.5g}')
        panel(initial,38,472,'START ERROR','Linear max-channel error x20',True)
        panel(final,416,472,'FITTED ERROR','Same fixed error scale x20',True)
        text(794,484,'RGB ALBEDO',3)
        for i,(want,got) in enumerate(zip(row['target_parameters'],row['parameters'])):text(794,534+36*i,f'{want:.4f} / {got:.4f}',2)
        text(794,664,'Target / recovered',2);text(794,708,'Geometry and light fixed.',2);text(794,744,'64 x 64 CPU example.',2)
        blob=png_rgb(1160,820,ctypes.string_at(board.data_ptr(),board.numel()));sha=hashlib.sha256(blob).hexdigest();file=output/(sha+'.png');file.open('xb').write(blob)
        row.update(asset=dict(url='/testing/media/'+sha+'.png',sha256=sha,bytes=len(blob)),asset_path=str(file),status='passed',scope='Actual installed-renderer albedo recovery with independent scalar Classic target on native geometry. Fixed64-square CPU example;20 fresh gradient steps. Display sRGB and nearest enlargement; both error panels fixed x20 linear scale. No consumer or large/all-parameter fitting claim.')
        check();save();return row
    except Exception as error:row.update(status='failed',error=str(error));save();raise
