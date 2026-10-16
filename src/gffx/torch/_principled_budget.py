"""Eager float64 principled loss resources with distinct CPU/CUDA profiles.

Port of the frozen bounded-cubic recipe, independent of development tools.
The exact shipped table has a finite CPU-float64 acceptance profile. Unknown
research data and material use require separate explicit opt-ins. Scratch estimates are planning
bounds, not allocator measurements or OS reservations.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
import torch
from ._loss_curve import bounded_slopes, bernstein_weights, controls, _bezier
from . import _budget_catalog
from ._reference_channels import multiply
from ._curve_math import roughness_interpolate, sum_last, log_unit

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


def _number(value):
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


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
    if any(not _number(x)
           for x in r+c+[x for row in v for pair in row for x in pair]):
        raise ValueError('resource finite numeric data')
    if (not all(0<x<=1 for x in r) or not all(b>a for a,b in zip(r,r[1:]))
            or c[0]!=0 or c[-1]!=1 or not all(b>a for a,b in zip(c,c[1:]))):
        raise ValueError('resource axes')
    domain = value['domain']
    if (not isinstance(domain, dict) or set(domain) != {'roughness', 'cosine', 'f0'}
            or any(not isinstance(v, list) or len(v) != 2 or
                   any(not _number(x) for x in v)
                   for v in domain.values())
            or domain != {'roughness':[r[0],r[-1]],'cosine':[0,1],'f0':[0,1]}):
        raise ValueError('resource domain metadata')
    if any(not (0<row[0][0]<=1 and row[0][1]==0)
           or any(not (0<pair[1]<=pair[0]<=1) for pair in row[1:]) for row in v):
        raise ValueError('resource positive ordered endpoint losses')
    if value['payload_sha256'] != _digest(payload):
        raise ValueError('resource payload hash')
    if len(_canonical(value)) > MAX_BYTES:
        raise ValueError('resource byte limit')



def _decode_blob(blob):
    def invalid_constant(value):
        raise ValueError('nonfinite JSON number')
    try:
        return json.loads(blob, object_pairs_hook=_unique, parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('resource encoding or JSON nesting') from error


def _lane(device, dtype, *, allow_unverified=False):
    if not isinstance(dtype, torch.dtype):
        raise TypeError('dtype must be a torch dtype')
    if not isinstance(device, (str, torch.device)):
        raise TypeError('device must be a string or torch device')
    try:
        target = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError('unsupported resource device') from error
    if dtype==torch.float64 and target.type=='cuda' and allow_unverified:
        if not torch.cuda.is_available():raise ValueError('CUDA resource device unavailable')
        index=torch.cuda.current_device() if target.index is None else target.index
        if not 0<=index<torch.cuda.device_count():raise ValueError('CUDA resource device index unavailable')
        return torch.device('cuda',index)
    if target.type != 'cpu' or target.index not in (None, 0) or dtype != torch.float64:
        raise ValueError('resource supports CPU float64 only; target lane is not qualified')
    return torch.device('cpu')


class _NativeLossCurve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,axis,coefficients,u):
        if axis.requires_grad or coefficients.requires_grad:
            raise ValueError('native curve axes/coefficients must be immutable resource constants')
        value,partial=torch.ops.gffx.shading_loss_curve(axis,coefficients,u)
        ctx.save_for_backward(partial)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient):
        partial,=ctx.saved_tensors
        return None,None,torch.ops.gffx.shading_loss_curve_backward(partial,gradient.contiguous())


class _NativeLossQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cn,logs,weights,coefficients,u,cosine,owners):
        if any(t.requires_grad for t in (cn,logs,weights,coefficients)):
            raise ValueError('native query coefficients must be immutable resource constants')
        value,partial=torch.ops.gffx.shading_loss_curve(logs,coefficients,u)
        query,mean=torch.ops.gffx.shading_loss_query(cn,logs,weights,value,partial,cosine,owners)
        ctx.save_for_backward(query,mean,owners)
        return query[:,:2].contiguous(),mean[:,:2].contiguous()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient,mean_gradient):
        query,mean,owners=ctx.saved_tensors
        gu,gc=torch.ops.gffx.shading_loss_query_backward(query,mean,owners,gradient.contiguous(),mean_gradient.contiguous())
        return None,None,None,None,gu,gc,None


class PrincipledBudget:
    """Immutable-by-public-interface budget; admission derives from exact identity."""
    __slots__ = ('_data', '_metadata', '_envelope')

    def __init__(self, *args, **kwargs):
        raise TypeError('use PrincipledBudget.from_envelope, load or bundled')

    def __setattr__(self, name, value):
        raise AttributeError('resource is immutable')

    @classmethod
    def bundled(cls, resource_id, *, device, dtype):
        if not isinstance(resource_id, str):
            raise TypeError('resource_id must be a string')
        if resource_id == _budget_catalog.CUDA_RESOURCE_ID:
            target = _lane(device, dtype, allow_unverified=True)
            if target.type != 'cuda':
                raise ValueError('the named CUDA resource requires a CUDA float64 target')
            blob = _budget_catalog.read_bundle(resource_id, byte_limit=MAX_BYTES)
            obj = cls.from_envelope(_decode_blob(blob), device=target, dtype=dtype,
                                    allow_unverified=True)
            admitted = _budget_catalog.cuda_acceptance(obj._envelope, device=target.type, dtype='float64')
            if admitted is None:
                raise ValueError('named CUDA bundle has no matching physical acceptance')
            metadata = obj.metadata
            metadata.update(admission='accepted', resource_id=admitted[0],
                acceptance_profile_id=admitted[1], acceptance_record_sha256=admitted[2])
            object.__setattr__(obj, '_metadata', metadata)
            return obj
        _lane(device, dtype)
        blob = _budget_catalog.read_bundle(resource_id, byte_limit=MAX_BYTES)
        return cls.from_envelope(_decode_blob(blob), device=device, dtype=dtype)

    @classmethod
    def from_envelope(cls, envelope, *, device, dtype, allow_unverified=False):
        if type(allow_unverified) is not bool:
            raise TypeError('allow_unverified must be a bool')
        _validate_envelope(envelope)
        target=_lane(device,dtype,allow_unverified=allow_unverified)
        admitted = _budget_catalog.acceptance(envelope, device=target.type, dtype='float64')
        if admitted is None and not allow_unverified:
            raise ValueError('resource is not in an accepted catalog; explicit allow_unverified required')
        payload = envelope['payload']
        data = {key: torch.tensor(payload[key], dtype=torch.float64, device='cpu')
                for key in ('roughness_nodes', 'cosine_nodes', 'losses')}
        rn, cn, values = (data[k] for k in ('roughness_nodes', 'cosine_nodes', 'losses'))
        logs, hlogs = log_unit(values[:, 1:]), log_unit(values[:, 0, 0])
        data.update(_rlog=log_unit(rn), _clog=log_unit(cn[1:]), _logs=logs, _hlogs=hlogs,
                    _slopes=bounded_slopes(log_unit(rn), logs, ordered_pairs=True),
                    _hslopes=bounded_slopes(log_unit(rn), hlogs),
                    _weights=bernstein_weights(log_unit(cn[2:])-log_unit(cn[1:-1])))
        if any(not bool(torch.isfinite(x).all()) for x in data.values()):
            raise ValueError('resource coefficients are not representable')
        # Constants are constructed in the validated float64 coefficient recipe.
        # Only this explicit eager construction transfers them; queries stay local.
        if target.type!='cpu':data={key:value.to(device=target) for key,value in data.items()}
        obj = object.__new__(cls)
        persistent = sum(x.numel()*x.element_size() for x in data.values())
        metadata = copy.deepcopy({k: v for k, v in envelope.items() if k != 'payload'})
        metadata.update(admission='accepted' if admitted else 'experimental',
                        resource_id=admitted[0] if admitted else None,
                        acceptance_profile_id=admitted[1] if admitted else None,
                        acceptance_record_sha256=admitted[2] if admitted else None,
                        execution_dtype='float64', execution_device=str(target),
                        supported_query_entries=32, persistent_tensor_bytes=persistent,
                        workspace_accounting_version=1)
        object.__setattr__(obj, '_data', MappingProxyType(data))
        object.__setattr__(obj, '_envelope', copy.deepcopy(envelope))
        object.__setattr__(obj, '_metadata', metadata)
        return obj

    @classmethod
    def load(cls, path, *, device, dtype, allow_unverified=False):
        if type(allow_unverified) is not bool:
            raise TypeError('allow_unverified must be a bool')
        _lane(device,dtype,allow_unverified=allow_unverified)
        with Path(path).open('rb') as stream:
            blob = stream.read(MAX_BYTES+1)
        if len(blob) > MAX_BYTES:
            raise ValueError('resource byte limit')
        return cls.from_envelope(_decode_blob(blob), device=device, dtype=dtype,
                                 allow_unverified=allow_unverified)

    def to(self, *, device, dtype, allow_unverified=False):
        if type(allow_unverified) is not bool:raise TypeError('allow_unverified must be a bool')
        _lane(device,dtype,allow_unverified=allow_unverified)
        return type(self).from_envelope(self._envelope, device=device, dtype=dtype,
                                       allow_unverified=True)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def inspect(self):
        return dict(metadata=self.metadata, tensors={k: v.clone() for k, v in self._data.items()},
                    persistent_tensor_bytes=self._metadata['persistent_tensor_bytes'])

    def to_envelope(self):
        return copy.deepcopy(self._envelope)

    def workspace(self,query_entries,*,backward):
        count = query_entries
        if type(count) is not int or type(backward) is not bool:
            raise TypeError('workspace count must be int and backward must be bool')
        if not 0<=count<=32:
            raise ValueError('workspace batch limit or mode')
        return 8*count*len(self._data['cosine_nodes'])*(384 if backward else 160)

    def _validate(self, owner, parameters, limit):
        if type(limit) is not int:
            raise TypeError('workspace limit must be integer bytes')
        if not 0<=limit<=WORKSPACE_BYTES:
            raise ValueError('workspace limit must be integer bytes within64 MiB')
        for name,x in parameters.items():
            if not isinstance(x,torch.Tensor) or x.dtype!=torch.float64:
                raise TypeError(name+' requires float64 tensor')
            if x.device!=self._data['roughness_nodes'].device: raise ValueError(name+' requires resource device')
            if x.shape!=owner.shape and x.ndim!=0: raise ValueError(name+' field shape')
            if not bool(torch.isfinite(x).all()): raise ValueError(name+' finite input')
            lo,hi=(self._metadata['domain']['roughness'] if name=='roughness' else (0,1))
            if not bool(((x>=lo)&(x<=hi)).all()): raise ValueError(name+' outside resource domain')
        needed=self.workspace(owner.numel(),backward=torch.is_grad_enabled() and any(x.requires_grad for x in parameters.values()))
        if needed>limit: raise ValueError('workspace budget exceeded before curve allocation')

    def _prepare_curve(self,r,*,with_width=False):
        # Only selected immutable controls are packed. The native call owns no
        # resource cache and the derivative stays in this call's autograd graph.
        d=self._data; cn=d['cosine_nodes']; x=d['_rlog']; n=len(cn)-1
        i=(torch.searchsorted(d['roughness_nodes'],r.contiguous(),right=True)-1).clamp(max=len(x)-2)
        h=x[i+1]-x[i]; u=(log_unit(r)-x[i])/h
        pieces=[]
        for y,m,trailing in ((d['_logs'],d['_slopes'],2),(d['_hlogs'],d['_hslopes'],0)):
            width=h.reshape((-1,)+(1,)*trailing)
            three=width.new_tensor(3.)
            pieces.append(torch.stack((y[i],y[i]+width*m[i]/three,y[i+1]-width*m[i+1]/three,y[i+1]),dim=-1).reshape(len(r),-1,4))
        coefficients=torch.cat(pieces,dim=1)
        return (cn,coefficients,u,h) if with_width else (cn,coefficients,u)

    def _curve(self,r):
        cn,coefficients,u=self._prepare_curve(r);n=len(cn)-1
        value=_NativeLossCurve.apply(self._data['_clog'],coefficients,u)
        return cn,value[:,:2*n].reshape(len(r),n,2),value[:,2*n:2*n+2],value[:,2*n+2:].reshape(len(r),n-1,4,2)

    def _native_pairs(self,r,c,owners):
        cn,coefficients,u=self._prepare_curve(r)
        pair,average=_NativeLossQuery.apply(cn[1:],self._data['_clog'],self._data['_weights'],
                                           coefficients,u,c.contiguous(),owners.to(dtype=torch.int32).contiguous())
        for value,is_mean in ((pair,False),(average,True)):
            if not bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()
                        and (value[:,0]>=value[:,1]).all()
                        and ((value>0).all() if is_mean else True)):
                raise ValueError('bounded curve arithmetic is not representable')
        return pair,average

    def _curve_reference(self,r):
        d=self._data; cn=d['cosine_nodes']; x=d['_rlog']
        i=(torch.searchsorted(d['roughness_nodes'],r.contiguous(),right=True)-1).clamp(max=len(x)-2)
        h=x[i+1]-x[i]; u=(log_unit(r)-x[i])/h
        def interpolate(y,m,trailing):
            width=h.reshape((-1,)+(1,)*trailing)
            three=width.new_tensor(3.)
            control=torch.stack((y[i],y[i]+width*m[i]/three,y[i+1]-width*m[i+1]/three,y[i+1]))
            return roughness_interpolate(control,u.reshape((-1,)+(1,)*trailing))
        values=interpolate(d['_logs'],d['_slopes'],2)
        horizon=interpolate(d['_hlogs'],d['_hslopes'],0)
        horizon=torch.stack((horizon,horizon*0),dim=-1)
        angular=values.transpose(0,1)
        slopes=bounded_slopes(d['_clog'],angular,ordered_pairs=True)
        b=controls(d['_clog'],angular,slopes).permute(2,0,1,3)
        return cn,values,horizon,b

    def _pairs(self,c,r,*,direction,mean):
        cosine=c if direction else r.new_empty((0,))
        owners=torch.arange(len(cosine),dtype=torch.int32,device=r.device)
        pair,average=self._native_pairs(r,cosine,owners)
        return pair if direction else None,average if mean else None

    def _curve_pairs(self,curve,c,*,direction,mean,indices=None):
        cn,values,horizon,b=curve
        pair=average=None
        if direction:
            if indices is None:
                indices=torch.arange(len(values),device=values.device)
            # Subtract close endpoint values before division. The equivalent
            # two-product form cancels enormous horizon derivative terms.
            linear=horizon[indices]+(c[:,None]/cn[1])*(values[indices,0]-horizon[indices])
            if len(cn)==2:
                pair=linear
            else:
                # Safe inactive log branch preserves exact horizon gradients.
                active=c>=cn[1]; safe=torch.where(active,c,cn[1])
                i=(torch.searchsorted(cn,safe.contiguous(),right=True)-1).clamp(max=len(cn)-2)
                logs=self._data['_clog']
                u=(log_unit(safe)-logs[i-1])/(logs[i]-logs[i-1])
                selected=b[indices,i-1].transpose(0,1)
                cubic=_bezier(selected,u[:,None])
                pair=torch.where(active[:,None],cubic,linear)
        if mean:
            width=self._data['_clog'][1:]-self._data['_clog'][:-1]
            average=cn[1].square()*(horizon+2*values[:,0])/cn.new_tensor(3.)
            terms=(2*cn[None,2:,None,None].square()*width[None,:,None,None]
                   *b*self._data['_weights'][None,:,:,None])
            average=average+sum_last(terms.permute(0,3,1,2).reshape(len(values),2,-1))
        for value,is_mean in ((pair,False),(average,True)):
            if value is not None and not bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()
                                             and (value[:,0]>=value[:,1]).all()
                                             and ((value>0).all() if is_mean else True)):
                raise ValueError('bounded curve arithmetic is not representable')
        return pair,average

    def _material_losses(self,view_cosine,light_cosine,roughness,f0,indices):
        """Reuse one current-graph curve per active point, never across calls.

        Private material path: the caller has validated domains, point ownership
        and conservative expanded-channel workspace before allocating any curve.
        F0 mixes the two endpoint losses; it does not alter curve construction.
        Only four angular controls are gathered per light, not the entire curve.
        """
        owners=torch.cat((torch.arange(len(roughness),device=roughness.device),indices))
        query,mean=self._native_pairs(roughness,torch.cat((view_cosine,light_cosine)),owners)
        view,incident=query[:len(roughness)],query[len(roughness):]
        def mix(pair,f):
            return multiply(1-f,pair[:,0,None])+multiply(f,pair[:,1,None])
        return mix(view,f0),mix(mean,f0),mix(incident,f0[indices])

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

    def directional_loss(self,cosine,roughness,f0,*,workspace_limit_bytes=WORKSPACE_BYTES):
        return self._query(cosine,roughness,f0,direction=True,mean=False,workspace_limit_bytes=workspace_limit_bytes)

    def mean_loss(self,roughness,f0,*,workspace_limit_bytes=WORKSPACE_BYTES):
        return self._query(roughness,roughness,f0,direction=False,mean=True,workspace_limit_bytes=workspace_limit_bytes)
