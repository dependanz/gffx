"""Reconstruct saved nine-parameter recovery images once; never rerun the optimizer."""
import argparse,ast,hashlib,importlib.util,json,os,xml.etree.ElementTree as ET
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel','recovery-report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);a=p.parse_args()
    if not a.worker:return cost.supervise(a.fixtures,a.output,worker_module='tools.image_recovery_gallery',worker_args=['--wheel',str(a.wheel),'--sha256',a.sha256,'--recovery-report',str(a.recovery_report)])
    row={'status':'working','worker_pid':os.getpid()};cost.write_report(a.output,row)
    try:
        import torch
        from tools.testing_dashboard.image_showcase import _FONT, _text
        from tools.testing_dashboard.presentation import png_rgb
        glyphs=dict(_FONT, **{'5':'11111 10000 10000 11110 00001 00001 11110','6':'01110 10000 10000 11110 10001 10001 01110','7':'11111 00001 00010 00100 01000 01000 01000','8':'01110 10001 10001 01110 10001 10001 01110','9':'01110 10001 10001 01111 00001 00001 01110','.':'00000 00000 00000 00000 00000 00110 00110',':':'00000 00110 00110 00000 00110 00110 00000'})
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256);row['driver_sha256']=artifact.digest(__file__)
        saved=json.loads(a.recovery_report.read_text(encoding='utf-8'));assert saved['status']=='passed' and saved['wheel']['sha256']==a.sha256
        source=Path('tests/python/test_image_full_recovery.py');assert artifact.digest(source)==saved['tests'][source.as_posix()]
        xml=a.recovery_report.with_suffix('.xml');root=ET.parse(xml).getroot()
        found=[{p.attrib['name']:p.attrib['value'] for p in case.findall('properties/property')} for case in root.iter('testcase')]
        props=next(p for p in found if 'recovered_parameters' in p)
        params=ast.literal_eval(props['recovered_parameters'])
        spec=importlib.util.spec_from_file_location('saved_recipe',source);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
        fixed=recipe.sources();before=recipe._fixed_identity(fixed)
        with torch.no_grad():
            target=torch.tensor(recipe.scalar_reference(fixed,recipe.TARGET),dtype=torch.float64)
            initial=recipe.evaluate(fixed,torch.tensor(recipe.INITIAL,dtype=torch.float64))
            final=recipe.evaluate(fixed,torch.tensor(params,dtype=torch.float64))
        assert before==recipe._fixed_identity(fixed)
        mse=[float((v-target).square().mean()) for v in (initial,final)]
        assert mse==[float(props['initial_mse']),float(props['final_fresh_mse'])],(mse,props)
        row.update(recovery_report_sha256=artifact.digest(a.recovery_report),recovery_xml_sha256=artifact.digest(xml),test_sha256=artifact.digest(source),mse=mse,parameters=params,closure_evaluations_saved=int(props['closure_evaluations_each_repeat']))
        folder=a.output.with_suffix('');folder.mkdir()
        board=torch.tensor([13,20,32],dtype=torch.uint8).expand(960,1280,3).clone()
        def text(x,y,s,size=22,fill=(219,230,241)):
            scale=max(1,size//8);s=s.upper()
            assert all(c in glyphs or c==' ' for c in s),s
            for i,c in enumerate(s):
                if c==' ':continue
                for yy,line in enumerate(glyphs[c].split()):
                    for xx,on in enumerate(line):
                        if on=='1':board[y+yy*scale:y+(yy+1)*scale,x+(i*6+xx)*scale:x+(i*6+xx+1)*scale]=board.new_tensor(fill)
        text(40,24,'RECOVER THE LAYER',38);text(40,78,'Nine coupled parameters. Fixed source pixels. Actual gffx outputs.',24)
        def panel(t,x,y,label,caption,error=False):
            if error:
                q=((t-target)[...,:3].abs().amax(-1)*20).clamp(0,1);rgb=torch.stack((q,q*.25,q*.65),-1)
            else:
                linear=t[...,:3].clamp(0,1);rgb=torch.where(linear<=.0031308,12.92*linear,1.055*linear.pow(1/2.4)-.055)
            u=(rgb*255).round().to(torch.uint8).contiguous();image=u.repeat_interleave(9,0).repeat_interleave(9,1)
            text(x,y,label,25);board[y+40:y+328,x:x+288]=image;text(x,y+350,caption,19)
        panel(target,40,134,'Independent target','Scalar equations / 32 x 32')
        panel(initial,480,134,'Before fitting',f'MSE {mse[0]:.6g}')
        panel(final,920,134,'Recovered image',f'MSE {mse[1]:.6g}')
        panel(initial,40,538,'Before: absolute error','Linear max-channel error x20',True)
        panel(final,480,538,'After: same error scale','Same fixed linear scale x20',True)
        text(920,542,'Target / recovered',24)
        for i,(name,t,v) in enumerate(zip(recipe.NAMES,recipe.TARGET,params)):
            text(920,588+i*27,f'{name:10s} {t: .4f} / {v: .4f}',18)
        text(920,845,'79 saved closure evaluations.',18)
        text(920,875,'Image recovery is not unique',18)
        text(920,900,'parameter identification.',18)
        import ctypes
        blob=png_rgb(1280,960,ctypes.string_at(board.data_ptr(),board.numel()));sha=hashlib.sha256(blob).hexdigest();file=folder/(sha+'.png');file.write_bytes(blob)
        row['asset']={'url':'/testing/media/'+sha+'.png','sha256':sha,'bytes':len(blob)};row['asset_path']=str(file)
        row['scope']='Current installed-wheel reevaluation of saved recovered parameters and original scalar target. No optimizer rerun. Display alone uses sRGB encoding and nearest pixel enlargement; error uses fixed x20 linear scale. Original numerical arrays and tolerances unchanged.'
        row['status']='passed';cost.write_report(a.output,row)
        manifest={'version':1,'title':'Recover the layer — nine parameters','page':'image/full-recovery-demo','images':[{'label':'Target, fit and fixed-scale error','caption':row['scope'],'asset':row['asset'],'path':str(file)}],'provenance':{'producer_report':str(a.output),'producer_report_sha256':None,'wheel_sha256':a.sha256},'scope':row['scope']}
        # Supervisor appends resource measurements after exit; manifest is finalized separately.
        with (folder/'manifest-pending.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    except Exception as e:row.update(status='stopped',error=str(e));raise
    finally:cost.write_report(a.output,row)
    return 0
if __name__=='__main__':raise SystemExit(main())
