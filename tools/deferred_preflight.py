"""Pure shape-only conservative buffer charges for prospective deferred profiles.

No torch import, tensors, renderer or allocation probe. Passing this partial estimate
does NOT admit a render: geometry, sources, lighting and shader scratch are absent.
The formula is the candidate planner's buffer component, not measured memory.
"""
import argparse
import json


def buffer_charge(height,width,*,batch=1,factor=1,element_bytes=8,
                       geometric_outputs=True,soft=False,limit=67108864):
    for name,value in (('height',height),('width',width),('batch',batch),
                       ('factor',factor),('element_bytes',element_bytes),('limit',limit)):
        if type(value) is not int or value<=0:
            raise ValueError(name+' must be a positive integer')
    if element_bytes not in (4,8):raise ValueError('only float32/float64 planner profiles')
    if type(geometric_outputs) is not bool or type(soft) is not bool:
        raise ValueError('output and coverage switches must be booleans')
    base=batch*height*width
    extra=base if geometric_outputs and (factor>1 or soft) else 0
    result=(base*factor*factor+extra)*(256*element_bytes+32)
    return dict(height=height,width=width,batch=batch,supersample=factor,
                dtype='float64' if element_bytes==8 else 'float32',soft=soft,
                geometric_outputs=geometric_outputs,conservative_buffer_charge_bytes=result,
                rgb_mask_output_bytes=base*4*element_bytes,
                workspace_limit_bytes=limit,
                disposition='refused_by_buffer_charge' if result>limit else 'full_preflight_required',
                measured=False,admitted=False)


def target_profiles():
    """Explicit resolution targets; no invented FLAME topology/sample counts."""
    result=[]
    for name,height,width in (('synthetic_face_m1',128,128),
                              ('FLAME-scale target',256,256),
                              ('FLAME-scale target',512,512),
                              ('FLAME-scale target',1080,1920)):
        for soft in (False,True):
            row=buffer_charge(height,width,soft=soft)
            row.update(fixture=name,topology='1986 vertices / 3968 triangles' if height==128 else 'not specified',
                       omitted=['input/resource storage','packed geometry','attribute sources',
                                'selection masks','lighting graphs','material scratch','process overhead'])
            result.append(row)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(dict(status='planning_only',profiles=target_profiles()),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
