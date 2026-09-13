"""Experimental tensor-resident CPU float64 loss curves; no public gffx API.

Vectorizes the frozen bounded-cubic recipe. Resource construction is eager;
queries neither rebuild constants, read files nor integrate numerical quadrature.
Workspace values are conservative planning estimates pending allocator calibration.
"""
from __future__ import annotations
import copy
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import torch
from tools.material_loss_smooth import SmoothLossResource, bounded_slopes, controls, _bezier

MAX_BYTES = 8*1024**2
WORKSPACE_BYTES = 64*1024**2
MODEL = 'ggx-correlated-smith-schlick-reciprocal-matte-v1'
REPRESENTATION = 'bounded-cubic-loss-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate resource key')
        value[key] = item
    return value


def _validate_envelope(value):
    keys = {'format_version','model','representation','dtype','domain','generator',
            'evidence_id','payload_sha256','payload'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('resource envelope keys')
    if (type(value['format_version']) is not int or value['format_version'] != 1
            or value['model'] != MODEL or value['representation'] != REPRESENTATION
            or value['dtype'] != 'float64'):
        raise ValueError('resource format/model/representation/dtype')
    if not isinstance(value['evidence_id'], str) or not 1 <= len(value['evidence_id']) <= 256:
        raise ValueError('resource evidence_id')
    generator = value['generator']
    if not isinstance(generator, dict) or set(generator) != {'source_sha256','policy'}:
        raise ValueError('resource generator')
    digest = generator['source_sha256']
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                               or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('generator source hash')
    if not isinstance(generator['policy'], str) or not 1 <= len(generator['policy']) <= 128:
        raise ValueError('generator policy')
    payload = value['payload']
    if not isinstance(payload, dict) or set(payload) != {'roughness_nodes','cosine_nodes','losses'}:
        raise ValueError('resource payload')
    r,c,v = (payload[key] for key in ('roughness_nodes','cosine_nodes','losses'))
    if (not all(isinstance(x,list) for x in (r,c,v)) or not 2<=len(r)<=129
            or not 2<=len(c)<=513 or len(v)!=len(r)
            or any(not isinstance(row,list) or len(row)!=len(c) for row in v)
            or any(not isinstance(pair,list) or len(pair)!=2 for row in v for pair in row)):
        raise ValueError('resource node limits or shape')
    if any(type(x) not in (float,int) or not math.isfinite(x)
           for x in r+c+[x for row in v for pair in row for x in pair]):
        raise ValueError('resource finite numeric data')
    if (not all(0<x<=1 for x in r) or not all(b>a for a,b in zip(r,r[1:]))
            or c[0]!=0 or c[-1]!=1 or not all(b>a for a,b in zip(c,c[1:]))):
        raise ValueError('resource axes')
    if value['domain'] != {'roughness':[r[0],r[-1]],'cosine':[0,1],'f0':[0,1]}:
        raise ValueError('resource domain metadata')
    if any(not (0<row[0][0]<=1 and row[0][1]==0)
           or any(not (0<pair[1]<=pair[0]<=1) for pair in row[1:]) for row in v):
        raise ValueError('resource positive ordered endpoint losses')
    if value['payload_sha256'] != _digest(payload):
        raise ValueError('resource payload hash')
    if len(_canonical(value)) > MAX_BYTES:
        raise ValueError('resource byte limit')


class TensorLossResource:
    """Private immutable-by-interface constants; only scalar/exact-shape sharing."""
    def __setattr__(self, name, value):
        raise AttributeError('resource is immutable')

    @classmethod
    def from_resource(cls, resource, *, evidence_id, generator=None):
        # Explicit conversion only. No original tuple or tensor storage is retained.
        payload = {key: torch.tensor(getattr(resource,key),dtype=torch.float64).tolist()
                   for key in ('roughness_nodes','cosine_nodes','losses')}
        return cls.from_envelope(dict(format_version=1,model=MODEL,
            representation=REPRESENTATION,dtype='float64',
            domain=dict(roughness=[payload['roughness_nodes'][0],payload['roughness_nodes'][-1]],cosine=[0,1],f0=[0,1]),
            generator=generator or dict(source_sha256=None,policy='caller-provided; generation not attested'),
            evidence_id=evidence_id,payload_sha256=_digest(payload),payload=payload))

    @classmethod
    def from_envelope(cls, value):
        _validate_envelope(value)
        p=value['payload']
        ref=SmoothLossResource(*(torch.tensor(p[k],dtype=torch.float64)
                                 for k in ('roughness_nodes','cosine_nodes','losses')))
        obj=object.__new__(cls)
        data={field.name:torch.tensor(getattr(ref,field.name),dtype=torch.float64)
              for field in fields(ref)}
        data['_weights']=data['_weights'].reshape(-1,4)
        data['_rlog']=data['roughness_nodes'].log()
        data['_clog']=data['cosine_nodes'][1:].log()
        object.__setattr__(obj,'_data',MappingProxyType(data))
        object.__setattr__(obj,'_metadata',copy.deepcopy({k:v for k,v in value.items() if k!='payload'}))
        return obj

    @classmethod
    def load(cls,path):
        with Path(path).open('rb') as stream:
            blob=stream.read(MAX_BYTES+1)
        if len(blob)>MAX_BYTES: raise ValueError('resource byte limit')
        return cls.from_envelope(json.loads(blob,object_pairs_hook=_unique))

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def to_envelope(self):
        value=self.metadata
        value['payload']={key:self._data[key].tolist() for key in ('roughness_nodes','cosine_nodes','losses')}
        return value

    def inspect(self):
        return dict(metadata=self.metadata,tensors={k:v.clone() for k,v in self._data.items()},
                    persistent_tensor_bytes=sum(v.numel()*v.element_size() for v in self._data.values()))

    def workspace(self,count,*,backward):
        if type(count) is not int or not 0<=count<=32 or type(backward) is not bool:
            raise ValueError('workspace batch limit or mode')
        return 8*count*len(self._data['cosine_nodes'])*(384 if backward else 160)

    def _validate(self, owner, parameters, limit):
        if type(limit) is not int or not 0<=limit<=WORKSPACE_BYTES:
            raise ValueError('workspace limit must be integer bytes within64 MiB')
        for name,x in parameters.items():
            if not isinstance(x,torch.Tensor) or x.dtype!=torch.float64:
                raise TypeError(name+' requires float64 tensor')
            if x.device.type!='cpu': raise ValueError(name+' requires CPU device')
            if x.shape!=owner.shape and x.ndim!=0: raise ValueError(name+' field shape')
            if not bool(torch.isfinite(x).all()): raise ValueError(name+' finite input')
            lo,hi=(self._metadata['domain']['roughness'] if name=='roughness' else (0,1))
            if not bool(((x>=lo)&(x<=hi)).all()): raise ValueError(name+' outside resource domain')
        needed=self.workspace(owner.numel(),backward=torch.is_grad_enabled() and any(x.requires_grad for x in parameters.values()))
        if needed>limit: raise ValueError('workspace budget exceeded before curve allocation')

    def _curve(self,r):
        d=self._data; cn=d['cosine_nodes']; x=d['_rlog']
        i=(torch.searchsorted(d['roughness_nodes'],r.contiguous(),right=True)-1).clamp(max=len(x)-2)
        h=x[i+1]-x[i]; u=(r.log()-x[i])/h
        def interpolate(y,m,trailing):
            width=h.reshape((-1,)+(1,)*trailing)
            control=torch.stack((y[i],y[i]+width*m[i]/3,y[i+1]-width*m[i+1]/3,y[i+1]))
            return _bezier(control,u.reshape((-1,)+(1,)*trailing)).exp()
        values=interpolate(d['_logs'],d['_slopes'],2)
        horizon=interpolate(d['_hlogs'],d['_hslopes'],0)
        horizon=torch.stack((horizon,horizon*0),dim=-1)
        angular=values.transpose(0,1)
        slopes=bounded_slopes(d['_clog'],angular,ordered_pairs=True)
        b=controls(d['_clog'],angular,slopes).permute(2,0,1,3)
        return cn,values,horizon,b

    def _pairs(self,c,r,*,direction,mean):
        cn,values,horizon,b=self._curve(r)
        pair=average=None
        if direction:
            linear=(1-c[:,None]/cn[1])*horizon+(c[:,None]/cn[1])*values[:,0]
            if len(cn)==2:
                pair=linear
            else:
                # Safe inactive log branch preserves exact horizon gradients.
                active=c>=cn[1]; safe=torch.where(active,c,cn[1])
                i=(torch.searchsorted(cn,safe.contiguous(),right=True)-1).clamp(max=len(cn)-2)
                u=(safe.log()-cn[i].log())/(cn[i+1].log()-cn[i].log())
                selected=b[torch.arange(len(r)),i-1].transpose(0,1)
                cubic=_bezier(selected,u[:,None])
                pair=torch.where(active[:,None],cubic,linear)
        if mean:
            width=self._data['_clog'][1:]-self._data['_clog'][:-1]
            average=cn[1].square()*(horizon+2*values[:,0])/3
            average=average+(2*cn[None,2:,None,None].square()*width[None,:,None,None]
                              *b*self._data['_weights'][None,:,:,None]).sum(dim=(1,2))
        for value,is_mean in ((pair,False),(average,True)):
            if value is not None and not bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()
                                             and (value[:,0]>=value[:,1]).all()
                                             and ((value>0).all() if is_mean else True)):
                raise ValueError('bounded curve arithmetic is not representable')
        return pair,average

    def _query(self,owner,r,f,*,direction,mean,workspace_limit_bytes):
        parameters=dict(roughness=r,f0=f)
        if direction: parameters['cosine']=owner
        if not isinstance(owner,torch.Tensor): raise TypeError('query owner must be tensor')
        self._validate(owner,parameters,workspace_limit_bytes)
        shape=owner.shape
        r,f=r.expand(shape).reshape(-1),f.expand(shape).reshape(-1)
        if owner.numel()==0:
            empty=(owner+r.reshape(shape)+f.reshape(shape))*0
            return empty if not (direction and mean) else dict(loss=empty,energy=1-empty,mean_loss=empty,mean_energy=1-empty)
        p,a=self._pairs(owner.reshape(-1) if direction else None,r,direction=direction,mean=mean)
        mix=lambda x:((1-f)*x[:,0]+f*x[:,1]).reshape(shape)
        if direction and mean:
            loss,average=mix(p),mix(a)
            return dict(loss=loss,energy=1-loss,mean_loss=average,mean_energy=1-average)
        return mix(p if direction else a)

    def evaluate(self,cosine,roughness,f0,*,workspace_limit_bytes=WORKSPACE_BYTES):
        return self._query(cosine,roughness,f0,direction=True,mean=True,workspace_limit_bytes=workspace_limit_bytes)

    def directional_loss(self,cosine,roughness,f0,*,workspace_limit_bytes=WORKSPACE_BYTES):
        return self._query(cosine,roughness,f0,direction=True,mean=False,workspace_limit_bytes=workspace_limit_bytes)

    def mean_loss(self,roughness,f0,*,workspace_limit_bytes=WORKSPACE_BYTES):
        return self._query(roughness,roughness,f0,direction=False,mean=True,workspace_limit_bytes=workspace_limit_bytes)

    def integrate_for_test(self,roughness,f0,*,order=32):
        """Independent segment Gauss integration, never called by a query."""
        from tools.material_budget_reference import _resolution
        from tools.shading_energy_probe import _gauss_unit
        _resolution(order,maximum=64)
        self._validate(roughness,dict(roughness=roughness,f0=f0),WORKSPACE_BYTES)
        if roughness.ndim or f0.ndim: raise ValueError('test integrator is scalar only')
        cn,values,horizon,b=self._curve(roughness.reshape(1))
        values,horizon,b=values[0],horizon[0],b[0]
        u,w=_gauss_unit(order)
        first_curve=(1-u[:,None])*horizon+u[:,None]*values[0]
        first=(2*cn[1].square()*u[:,None]*first_curve*w[:,None]).sum(0)
        width=cn[2:].log()-cn[1:-1].log()
        cosine=cn[1:-1,None]*torch.exp(width[:,None]*u)
        basis=torch.stack(((1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3),dim=0)
        curve=torch.einsum('ikf,kq->iqf',b,basis)
        pair=first+(2*cosine[...,None].square()*width[:,None,None]*curve*w[None,:,None]).sum(dim=(0,1))
        return (1-f0)*pair[0]+f0*pair[1]
