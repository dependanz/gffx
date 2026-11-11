"""Clip primary triangles while preserving material and face-corner attributes."""
from dataclasses import replace
import torch

def clip(packed,camera,plane=None):
    original=torch.arange(len(packed.faces),device=packed.faces.device)
    view=camera.world_to_view()[0]
    planes=[(-view[2,:3],-view[2,3]-camera.near*1.00001)]
    if plane is not None:
        p,n=plane;planes.append((n,-(n*p).sum()+1e-6))
    for normal,offset in planes:
        v=packed.vertices;f=packed.faces
        vertex_distance=(v*normal).sum(-1)+offset
        distance=torch.index_select(vertex_distance,0,f.reshape(-1)).reshape(-1,3);inside=distance>=0;count=inside.sum(-1)
        if bool((count==3).all()):continue
        generated=[];attributes={key:[] for key in packed.attributes};ids=[]
        for number in (1,2):
            selected=torch.nonzero(count==number).flatten()
            if not len(selected):continue
            good=inside[selected];index=(good if number==1 else ~good).long().argmax(-1);j=(index+1)%3;k=(index+2)%3;b=torch.arange(len(index),device=v.device)
            ds=distance[selected];t1=ds[b,index]/(ds[b,index]-ds[b,j]);t2=ds[b,index]/(ds[b,index]-ds[b,k])
            basis=torch.eye(3,device=v.device,dtype=v.dtype).expand(len(index),3,3)
            a0,a1,a2=basis[b,index],basis[b,j],basis[b,k]
            q1=a0+t1[:,None]*(a1-a0);q2=a0+t2[:,None]*(a2-a0)
            triangles=[torch.stack((a0,q1,q2),1)]if number==1 else [torch.stack((a1,a2,q2),1),torch.stack((a1,q2,q1),1)]
            for weights in triangles:
                generated.append(weights@v[f[selected].long()]);ids.append(selected)
                for key,attribute in packed.attributes.items():attributes[key].append(weights@attribute.values[selected])
        kept=count==3;selected=torch.cat(ids)if ids else original.new_empty(0)
        extra=torch.cat(generated).reshape(-1,3)if generated else v.new_empty((0,3))
        faces=torch.arange(len(extra),device=f.device,dtype=f.dtype).reshape(-1,3)+len(v)
        new_attributes={key:replace(a,values=torch.cat((a.values[kept],torch.cat(attributes[key])if attributes[key]else a.values[:0])))for key,a in packed.attributes.items()}
        original=torch.cat((original[kept],original[selected]))
        packed=replace(packed,vertices=torch.cat((v,extra)),faces=torch.cat((f[kept],faces)),face_material=torch.cat((packed.face_material[kept],packed.face_material[selected])),attributes=new_attributes)
    return packed,original
