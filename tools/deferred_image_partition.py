"""Explicit caller-side crop-camera measurement helper, not a public renderer mode.

Keeps the original pixel-centre grid, checks every tile before executing any, and
reserves complete outputs plus concatenation storage. Large backward is refused;
this helper never claims that retaining tile graphs solves full-image fitting.
"""

def plan_partitioned(model,camera,*,tile_rows,outputs=('colour','mask'),soft=0.,workspace_limit_bytes=67108864):
    import torch
    from gffx.torch import render
    from gffx.torch.camera import Camera
    from gffx.torch._deferred import _plan_workload
    if type(tile_rows) is not int or tile_rows<1 or tile_rows&(tile_rows-1) or camera.image_height%tile_rows:raise ValueError('power-of-two exact row partitions required')
    channels={'colour':3,'mask':1,'depth':1,'world_position':3,'face_index':1,'normal':3,'uv':2,'barycentric':3,'material_index':1,'uv_present':1,'vertex_colour':3,'colour_present':1}
    if not outputs or set(outputs)-set(channels):raise ValueError('image output channels required')
    retained=2*camera.batch_size*camera.image_height*camera.image_width*sum(channels[k] for k in outputs)*8
    allowance=workspace_limit_bytes-retained
    if allowance<=0:raise ValueError('complete output storage exceeds allowance')
    plans=[];cameras=[]
    for row in range(0,camera.image_height,tile_rows):
        crop=Camera.from_intrinsics(camera.fx,camera.fy,camera.cx,camera.cy-row,tile_rows,camera.image_width,camera.near,camera.far,pose=camera.pose)
        plan=_plan_workload(model,crop,1,allowance,outputs,soft=soft)
        if plan['backward'] and camera.image_height*camera.image_width>256:raise ValueError('large retained tile graphs are not admitted')
        plans.append(dict(row=row,height=tile_rows,retained_output_bytes=retained,allowance=allowance,estimate=plan));cameras.append(crop)
    return cameras,plans


def render_partitioned(model,camera,*,tile_rows,outputs=('colour','mask'),soft=0.,workspace_limit_bytes=67108864,checkpoint=lambda:None):
    import torch
    from gffx.torch import render
    cameras,plans=plan_partitioned(model,camera,tile_rows=tile_rows,outputs=outputs,soft=soft,workspace_limit_bytes=workspace_limit_bytes)
    chunks=[]
    for crop,plan in zip(cameras,plans):
        checkpoint();chunks.append(render(model,crop,outputs=outputs,soft=soft,workspace_limit_bytes=plan['allowance']))
    return {key:torch.cat([part[key] for part in chunks],dim=1) for key in outputs},plans
