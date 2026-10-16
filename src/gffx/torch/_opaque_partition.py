"""Qualification-only caller-owned CPU/CUDA forward partitions and checked VJP prefixes."""
import torch
from ._common import translate_native_error

class Partition:
 def __init__(self,v,f,c,origin,rule,budget):
  if type(rule)!=int or rule not in (2,3,4) or v.device.type not in ('cpu','cuda') or v.dtype!=torch.float64:raise ValueError('partition reuse requires explicit float64 SH52,SH64 orSH80')
  if any(not t.is_contiguous() for t in (v,f,c)):raise ValueError('partition reuse requires contiguous inputs')
  self.used_bytes=(2560 if rule==4 else (2048 if rule==3 else 1664))*17*8+2*sum(t.numel()*t.element_size() for t in (v,f,c))+4096
  if self.used_bytes+1048576>budget:raise OverflowError('partition retained storage budget exceeded')
  self.inputs=tuple(t.detach().clone() for t in (v,f,c));self.origin=tuple(origin);self.rule=rule;self._bound_rule=rule;self.cursor=(0,0,0,0);self.finished=False
  try:self.rows,meta=(torch.ops.gffx.opaque_partition_prepare80 if rule==4 else (torch.ops.gffx.opaque_partition_prepare64 if rule==3 else torch.ops.gffx.opaque_partition_prepare))(v,f,c,*origin,budget)
  except RuntimeError as e:raise translate_native_error(e) from None
  self.count,self.build_work=meta;self.version=self.rows._version;self.replaying=False
  self.deriv_cursor=(0,0,0,0);self.derivatives=None;self.prefix=(0,0);self.poisoned=False
 def emit(self,v,f,c,cursor,rule,origin):
  if self.finished or tuple(cursor)!=self.cursor or rule!=self.rule or tuple(origin)!=self.origin:raise ValueError('partition continuation/profile/origin changed')
  self.check_inputs(v,f,c)
  begin,batch,slots,blocks=cursor
  try:ids,q=(torch.ops.gffx.opaque_partition_emit80 if self.rule==4 else (torch.ops.gffx.opaque_partition_emit64 if self.rule==3 else torch.ops.gffx.opaque_partition_emit))(self.rows,self.count,begin)
  except RuntimeError as e:raise translate_native_error(e) from None
  per_region=28 if self.rule==4 else 25;n=min(800//per_region,self.count-begin);end=begin+n;self.finished=end==self.count
  receipt=(end,int(self.finished),per_region*n,n,slots+per_region*n,blocks+n,n,self.build_work if batch==0 and not self.replaying else 0)
  self.cursor=(end,batch+1,receipt[4],receipt[5]);return ids,q,receipt

 def check_inputs(self,v,f,c):
  if self.rule!=self._bound_rule:raise ValueError('partition profile changed')
  if self.poisoned:raise ValueError('partition owner failed; discard it')
  if self.rows._version!=self.version:raise ValueError('partition storage modified')
  for actual,snapshot in zip((v,f,c),self.inputs):
   if actual.shape!=snapshot.shape or actual.dtype!=snapshot.dtype or actual.device!=snapshot.device or not actual.is_contiguous() or not torch.equal(actual.detach().view(torch.uint8),snapshot.view(torch.uint8)):raise ValueError('partition inputs changed')

 def backward(self,v,f,c,g,cursor,receipt,derivatives,budget):
  from ._opaque_quadrature import _charge_batch
  self.check_inputs(v,f,c)
  if tuple(cursor)!=self.deriv_cursor or (self.derivatives is not None and derivatives!=self.derivatives):raise ValueError('derivative continuation/mask changed')
  self.derivatives=derivatives
  try:
   gv,gc,actual,prefix=(torch.ops.gffx.opaque_material_prefix_backward80 if self.rule==4 else (torch.ops.gffx.opaque_material_prefix_backward64 if self.rule==3 else torch.ops.gffx.opaque_material_prefix_backward))(v,f,c,g,*self.origin,*cursor,budget,derivatives,*self.prefix)
   _charge_batch(actual)
   if tuple(actual[:7])!=tuple(receipt[:7]):raise RuntimeError('prefix replay layout changed')
  except Exception:
   self.poisoned=True
   raise
  self.prefix=tuple(prefix);self.deriv_cursor=(actual[0],cursor[1]+1,actual[4],actual[5])
  return gv,gc

class _Cached(torch.autograd.Function):
 @staticmethod
 def forward(ctx,v,f,c,entry,cursor,rule,origin,budget):
  from ._opaque_quadrature import _charge_batch
  ids,q,receipt=entry.emit(v,f,c,cursor,rule,origin);_charge_batch(receipt)
  ctx.save_for_backward(v,f,c);ctx.entry=entry;ctx.cursor=tuple(cursor);ctx.receipt=receipt;ctx.budget=budget;ctx.mark_non_differentiable(ids)
  return ids,q,receipt
 @staticmethod
 def backward(ctx,unused,g,unused_receipt):
  if torch.is_grad_enabled():raise RuntimeError('partition supports first derivatives only')
  from ._common import materialize
  v,f,c=ctx.saved_tensors;mask=(1 if ctx.needs_input_grad[0] else 0)|(2 if ctx.needs_input_grad[2] else 0)
  gv,gc=ctx.entry.backward(v,f,c,materialize(g),ctx.cursor,ctx.receipt,mask,ctx.budget)
  return gv,None,gc,None,None,None,None,None

class Pool:
 def __init__(self,budget):self.budget=budget;self.used_bytes=0;self.entries={};self.claimed=False;self.poisoned=False
 def begin_replay(self):
  # Validate every entry before rewinding any; a failed claim cannot be retried.
  if self.claimed or self.poisoned:raise ValueError('partition pool already claimed or failed')
  self.claimed=True
  try:
   for entry in self.entries.values():
    if not entry.finished or entry.derivatives is not None or entry.deriv_cursor!=(0,0,0,0) or entry.prefix!=(0,0):raise ValueError('partition forward ownership incomplete')
    entry.check_inputs(*entry.inputs)
   for entry in self.entries.values():
    entry.cursor=(0,0,0,0);entry.finished=False;entry.replaying=True
  except Exception:
   self.poisoned=True
   raise
 def emit(self,v,f,c,origin,rule,cursor,key):
  from ._opaque_quadrature import _charge_batch
  if self.poisoned:raise ValueError('partition pool failed; discard it')
  if key not in self.entries:
   if self.claimed:raise ValueError('partition replay key changed')
   if tuple(cursor)!=(0,0,0,0):raise ValueError('partition continuation without preparation')
   entry=Partition(v,f,c,origin,rule,self.budget-self.used_bytes);self.used_bytes+=entry.used_bytes;self.entries[key]=entry
  entry=self.entries[key]
  if torch.is_grad_enabled() and (v.requires_grad or c.requires_grad):return _Cached.apply(v,f,c,entry,tuple(cursor),rule,origin,self.budget-self.used_bytes)
  result=entry.emit(v,f,c,cursor,rule,origin);_charge_batch(result[2]);return result
