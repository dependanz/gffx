"""Research continuation consumer. Host-owned, in-process; not a security token.

Native receipts never make a partial pixel acceptable. This guard binds actual
input bytes and context, accounts prefix replay, and carries each weighted sample
in order across batches. It does not implement shading, autograd or allocation
admission; production needs the existing workload planner and worker supervisor.
"""
import hashlib,json,time
import numpy as np

class PixelStream:
    def __init__(self,inputs,context,*,deadline,clock=time.perf_counter):
        self.inputs=inputs;self.context=context;self.deadline=deadline;self.clock=clock
        self.identity=self._identity();self.cursor=(0,0,0,0);self.work=0
        self._carry=np.zeros(4,dtype=np.float64);self._pending=None;self._state='open'
    def _identity(self):
        h=hashlib.sha256();h.update(json.dumps(self.context,sort_keys=True,allow_nan=False).encode())
        for name,a in sorted(self.inputs.items()):
            if not isinstance(a,np.ndarray) or not a.flags.c_contiguous or a.dtype.hasobject:raise ValueError('contiguous numeric inputs required')
            h.update(json.dumps([name,a.dtype.str,a.shape]).encode());h.update(memoryview(a).cast('B'))
        return h.digest()
    def _check(self):
        if self._state!='open':raise RuntimeError('stream '+self._state)
        if self.clock()>self.deadline:raise TimeoutError('whole-worker deadline')
        if self._identity()!=self.identity:raise RuntimeError('stream inputs/context changed')
    def begin(self):
        try:
            self._check()
            if self._pending is not None:raise RuntimeError('a batch is already pending')
            self._pending=self.cursor;return self.cursor
        except Exception:self._state='failed';raise
    def consume(self,cursor,receipt,owners,samples,contributions):
        try:
            self._check()
            if self._pending is None or cursor!=self._pending or cursor!=self.cursor:raise RuntimeError('unexpected or repeated cursor')
            if len(receipt)!=8 or any(type(v)!=int for v in receipt):raise ValueError('integer receipt required')
            next_,done,slots,blocks,total_slots,total_blocks,emitted,work=receipt
            begin,batch,prefix_slots,prefix_blocks=cursor
            if done not in (0,1) or not 0<=slots<=800 or not 0<=blocks<=32 or not 0<=emitted<=blocks or work<0:raise ValueError('receipt extent')
            if next_!=begin+emitted or total_slots!=prefix_slots+slots or total_blocks!=prefix_blocks+blocks:raise ValueError('receipt prefix mismatch')
            if total_slots>12800 or total_blocks>512 or (not done and (batch==15 or not emitted)):raise RuntimeError('finite continuation exhausted')
            if (slots==0)!=(emitted==0) or (blocks==0)!=(emitted==0):raise ValueError('inconsistent empty batch')
            if owners.shape!=(1,800) or owners.dtype!=np.int32 or samples.shape!=(1,800,4) or samples.dtype!=np.float64:raise ValueError('sample layout')
            if np.any(owners[0,:slots]<0) or np.any(owners[0,slots:]!=-1) or np.any(samples[0,slots:]!=0) or not np.isfinite(samples).all():raise ValueError('invalid sample payload')
            if contributions.shape!=(slots,4) or contributions.dtype!=np.float64 or not np.isfinite(contributions).all():raise ValueError('contribution layout/finiteness')
            self.work+=work
            if self.work>2000000:raise RuntimeError('aggregate clipping work exhausted')
            for value in contributions:self._carry=self._carry+value
            if not np.isfinite(self._carry).all():raise ValueError('nonfinite accumulated pixel')
            self._check();self.cursor=(next_,batch+1,total_slots,total_blocks);self._pending=None
            if done:self._state='complete'
        except Exception:self._state='failed';raise
    def cancel(self):self._state='cancelled';self._pending=None
    def result(self):
        if self._state!='complete':raise RuntimeError('incomplete or '+self._state+' pixel')
        return self._carry.copy()
