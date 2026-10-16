"""Prospective fixed-box boundary diagnostics; never an adaptive certificate."""
import math,time
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import gauss5
from tools.opaque_transverse_peano import _error
from tools.opaque_smooth_peano import ordinary_bounds,partitioned_ordinary


class DiagnosticStopped(RuntimeError):
    def __init__(self,reason,report):
        super().__init__(reason);self.report=report


def diagnose(correction,boxes,*,prepare=lambda box:None,max_inputs=2048,max_seconds=20,clock=time.perf_counter,checkpoint=lambda:None,partition_smooth=False):
    if type(partition_smooth)!=bool:raise ValueError('Boolean partition selector required')
    if type(max_inputs)!=int or not 1<=max_inputs<=2048:raise ValueError('input cap1..2048')
    if not math.isfinite(max_seconds) or not 0<max_seconds<=20:raise ValueError('time cap(0,20] seconds')
    if not isinstance(boxes,(tuple,list)) or not 1<=len(boxes)<=3:raise ValueError('one to three fixed boxes')
    parsed=[]
    for box in boxes:
        if len(box)!=2:raise ValueError('two coordinates')
        pair=tuple(map(I,box))
        if any(v.lo<0 or v.hi>1 or v.lo>=v.hi for v in pair):raise ValueError('positive subintervals of unit square')
        parsed.append(pair)
    started=clock();report=dict(status='working',full_piece_certificate=False,input_calls=0,input_calls_by_order={},input_scope='correction inputs; fixed preparation separately timed',witnesses=0,boxes=[],max_inputs=max_inputs,max_seconds=max_seconds)
    original_inputs=correction.inputs;original_checkpoint=correction.checkpoint
    def stop(reason):
        report['status']='stopped';report['reason']=reason;raise DiagnosticStopped(reason,report)
    def check():
        checkpoint();original_checkpoint()
        elapsed=clock()-started;report['seconds']=elapsed
        if elapsed>=max_seconds:stop('diagnostic deadline')
    def inputs(*args):
        check()
        if report['input_calls']>=max_inputs:stop('diagnostic input cap')
        report['input_calls']+=1
        order=max(v.order for v in args);report['input_calls_by_order'][order]=report['input_calls_by_order'].get(order,0)+1
        return original_inputs(*args)
    correction.inputs=inputs;correction.checkpoint=check
    try:
        nodes,weights=gauss5()
        for box in parsed:
            check();preparation_start=clock();prepare(box);check();record=dict(box=box,preparation_seconds=clock()-preparation_start,orientations=[]);report['boxes'].append(record)
            h=[I(v.hi)-I(v.lo) for v in box]
            for axis in (0,1):
                check();start=clock();witnesses=[]
                def capture(detail):
                    check()
                    if len(witnesses)>=16:stop('diagnostic witness cap16')
                    report['witnesses']+=1;witnesses.append(detail)
                rows=correction.integrated_bounds(box,axis,diagnostic=capture)
                flat=[w for row in rows for w in row.get('localized_jumps',[])]
                if len(flat)!=len(witnesses):raise ValueError('diagnostic witness accounting mismatch')
                for detail,witness in zip(witnesses,flat):
                    one=dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[witness])
                    detail['uniform']=_error(one,box[axis],h[1-axis],False)[0]
                    detail['integrated']=_error(one,box[axis],h[1-axis],True)[0]
                uniform=[_error(row,box[axis],h[1-axis],False)[0] for row in rows]
                evaluated=[_error(row,box[axis],h[1-axis],True) for row in rows]
                integrated=[pair[0] for pair in evaluated];integrated_parts=[pair[1] for pair in evaluated]
                comparisons=[ordinary_bounds(row,box[axis],h[1-axis]) for row in rows]
                integrated_ranges=[row.get('third_range') for row in rows];conditioned_ranges=[]
                conditioned=[I(0)]*len(rows)
                conditioned_parts=[{key:I(0) for key in ('ordinary','global_jumps','localized_jumps','signed_correction')} for row in rows]
                integrated_seconds=clock()-start;conditioned_start=clock()
                # For the opposite tensor ordering the continuously integrated
                # axis is this axis, and the other is Gauss-conditioned on it.
                for node,weight in zip(nodes,weights):
                    check();line=list(box);line[axis]=I(box[axis].lo)+h[axis]*node
                    conditioned_rows=correction.axis_bounds(line,1-axis)
                    if len(conditioned_rows)!=len(rows):raise ValueError('diagnostic channel count changed')
                    conditioned_ranges.append([row.get('third_range') for row in conditioned_rows])
                    for k,row in enumerate(conditioned_rows):
                        value,detail=_error(row,box[1-axis],h[axis]*weight,False)
                        conditioned[k]=conditioned[k]+value
                        for key,v in detail.items():conditioned_parts[k][key]=conditioned_parts[k][key]+v
                        part=ordinary_bounds(row,box[1-axis],h[axis]*weight)
                        for key in ('unanchored','centered','effective'):comparisons[k][key]=comparisons[k][key]+part[key]
                partition=None
                if partition_smooth:
                    if len(rows)!=2:raise ValueError('material two-channel diagnostic required')
                    def third(sub,active):
                        args=[Jet(sub[j],order=3) for j in (0,1)];args[active]=Jet.variable(sub[active],order=3)
                        return correction.ordinary(*args)[1][3,0]*6
                    partition_start=clock();before_calls=report['input_calls']
                    partition=partitioned_ordinary(third,box,axis,checkpoint=check)
                    partition.update(seconds=clock()-partition_start,input_calls=report['input_calls']-before_calls)
                record['orientations'].append(dict(partitioned_ordinary=partition,integrated_axis=axis,witnesses=witnesses,uniform=uniform,integrated=integrated,conditioned=conditioned,tensor_error=[a+b for a,b in zip(integrated,conditioned)],seconds=clock()-start,integrated_seconds=integrated_seconds,conditioned_seconds=clock()-conditioned_start,components=dict(integrated=integrated_parts,conditioned=conditioned_parts),ordinary_comparison=[{key:row[key] for key in ('unanchored','centered','effective')} for row in comparisons],third_ranges=dict(integrated=integrated_ranges,conditioned=conditioned_ranges)))
        check();report['status']='diagnostic_only';return report
    finally:
        correction.inputs=original_inputs;correction.checkpoint=original_checkpoint
