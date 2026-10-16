"""Direct unit-square independent outward reference,without a triangle wrapper.

Same tensor Gauss5/order10 proof as opaque_shading_intervals,with identity
parameter mapping. Curve and triangle Jacobians remain explicit field factors.
No renderer or sampled convergence criterion; separately named offline limits are explicit.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
import math,time
from tools.opaque_shading_intervals import I,Jet,ZERO,Unresolved


def cover_accounting(rows,*,reference_profile="standard",whole_width_limit="1e-12"):
    """Summarize only the current disjoint leaf cover;at most five leaf details."""
    from tools.opaque_reference_profile import profile
    limits=profile(reference_profile)
    if not isinstance(rows,list) or not 1<=len(rows)<=limits.cells:raise ValueError('current leaf count exceeds reference profile')
    count=len(rows[0]['widths'])
    if not 1<=count<=4 or any(len(r['widths'])!=count for r in rows):raise ValueError('matching channel count')
    complete=all(r['correction_error_components'] is not None for r in rows)
    channels=[]
    for k in range(count):
        ordinary=I(0);boundary=I(0);base=I(0);correction=I(0)
        for r in rows:
            base=base+I(r['base_widths'][k][0])
            if r['correction_widths']:correction=correction+I(r['correction_widths'][k][0])
            if complete:
                for axis in r['correction_error_components']:
                    ordinary=ordinary+I(*axis[k]['ordinary'])
                    boundary=boundary+I(*axis[k]['global_jumps'])+I(*axis[k]['localized_jumps'])
        encode=lambda v:[str(v.lo),str(v.hi)]
        transport=sum((I(r.get('tangent_transport_widths',['0']*count)[k]) for r in rows),I(0))
        channels.append(dict(tangent_transport_width_sum=encode(transport),base_width_sum=encode(base),correction_width_sum=encode(correction),ordinary_radius=encode(ordinary) if complete else None,boundary_radius=encode(boundary) if complete else None))
    top=sorted(rows,key=lambda r:max(D(v) for channel in r['widths'] for v in channel),reverse=True)[:5]
    limit=I(whole_width_limit)
    if limit.lo<=0:raise ValueError('positive diagnostic whole width limit')
    summary=[dict(chart=r.get('chart',0),box=r.get('box'),depth=r['depth'],widths=r['widths'],base_widths=r['base_widths'],correction_widths=r['correction_widths'],tangent_transport_widths=r.get('tangent_transport_widths')) for r in rows]
    counts=dict(whole_width_limit=str(limit.lo),leaves_wider_than_whole_limit=sum(any(D(v)>limit.lo for ch in r['widths'] for v in ch) for r in rows),is_local_acceptance_rule=False)
    return dict(leaf_widths=summary,diagnostic_counts=counts,leaves=len(rows),components_complete=complete,channels=channels,largest_leaves=top,scope='Only current retained disjoint leaves;parent and failed-domain rows excluded. Component sums are conservative radii,not measured image errors.')


def integrate_square(field,*,tolerance='1e-12',max_cells=256,max_depth=10,checkpoint=lambda:None,correction=None,prepare=None,correction_integrator=None,tangent_correction_integrator=None,refinement_field=None,experimental_rule=None,seed_shape=(1,1),reference_profile="standard",progress=lambda row:None,domain_split=None,chart_cover=None,remainder_partition=None,tensor_rule=None,initial_leaves=None,disjoint_roots=False):
    """Bound1..4 channels and first tangents over[0,1]^2,including field Jacobians.

    Smooth extension plus separately bounded order-zero residual; tensor Gauss5
    remainder uses outward order10 coefficients in each coordinate. The caller
    certifies field smoothness on each accepted box and owns its physical map.
    chart_cover selects a fresh field/callback set per mapped unit square.
    The caller must certify complete disjoint physical coverage and Jacobians.
    disjoint_roots starts independent physical pieces at depth zero, charging
    actual evaluations only. It cannot use inherited partition depths or schedules.
    No per-chart accuracy decision or budget reset is made.
    Optional tangent_correction_integrator returns one interval per channel for
    the integrated jump transport over the supplied parameter box. Its bounds
    participate in the same global derivative-radius criterion as smooth terms.
    """
    # A chart cover is a caller-certified geometric partition. The engine checks
    # its full binary-tree depth accounting, not its geometric correctness.
    # Every chart uses the same retained-leaf sum, queue, clock and work counter.
    from tools.opaque_reference_profile import profile
    limits=profile(reference_profile)
    if type(disjoint_roots)!=bool or disjoint_roots and (chart_cover is None or initial_leaves is not None):raise ValueError('disjoint roots require a fresh explicit chart cover')
    if chart_cover is not None:
        if field is not None or any(v is not None for v in (prepare,correction,correction_integrator,tangent_correction_integrator,refinement_field,domain_split,experimental_rule,remainder_partition,tensor_rule)) or seed_shape!=(1,1):
            raise ValueError('chart cover owns fields and callbacks; qualified per-chart rule only')
        allowed={'field','depth','prepare','correction','correction_integrator','tangent_correction_integrator','refinement_field','domain_split','remainder_partition','tensor_rule'}
        if not isinstance(chart_cover,(list,tuple)) or not 1<=len(chart_cover)<=limits.cells:raise ValueError('nonempty bounded chart cover required')
        for row in chart_cover:
            if not isinstance(row,dict) or set(row)-allowed or type(row.get('depth'))!=int or not 0<=row['depth']<=max_depth or not callable(row.get('field')):
                raise ValueError('chart field and inherited depth required')
            if any(k not in ('field','depth','remainder_partition','tensor_rule') and v is not None and not callable(v) for k,v in row.items()):raise ValueError('chart callbacks must be callable')
            if row.get('tensor_rule') is not None:
                from tools.opaque_tensor_smooth import validate as validate_tensor
                validate_tensor(row['tensor_rule'])
                if row.get('remainder_partition') is not None:raise ValueError('directional rule owns its remainder')
            if row.get('remainder_partition') is not None:
                from tools.opaque_local_smooth import validate
                validate(row['remainder_partition'])
            if row.get('correction') is not None and row.get('correction_integrator') is not None:raise ValueError('choose one chart correction contract')
        if disjoint_roots:
            if any(r['depth']!=0 for r in chart_cover):raise ValueError('disjoint root depth must be zero')
            if len(chart_cover)>max_cells:raise ValueError('disjoint roots exceed work budget')
        else:
            if sum((Q(1,2**r['depth']) for r in chart_cover),Q(0))!=1:raise ValueError('complete binary-tree chart depths required')
            if 2*len(chart_cover)-1>max_cells:raise ValueError('chart tree exceeds work budget')
    if initial_leaves is not None and chart_cover is None:raise ValueError('fresh schedule requires a certified chart cover')
    if domain_split is not None and not callable(domain_split):raise ValueError('domain split callback required')
    tol=I(tolerance)
    if tol.lo<=0 or tol.hi>D('1e-6'):raise ValueError('positive tolerance at most1e-6 required')
    if type(max_cells)!=int or not 1<=max_cells<=limits.cells or type(max_depth)!=int or not 0<=max_depth<=limits.depth:raise ValueError('quadrature work limits')
    if experimental_rule not in (None,"gauss8"):raise ValueError("qualified experimental rule required")
    if remainder_partition is not None:
        from tools.opaque_local_smooth import validate
        validate(remainder_partition)
        if experimental_rule is not None:raise ValueError('localized remainder requires standard Gauss5')
    if tensor_rule is not None:
        from tools.opaque_tensor_smooth import validate as validate_tensor
        validate_tensor(tensor_rule)
        if experimental_rule is not None or remainder_partition is not None:raise ValueError('directional rule owns its remainder')
    derivative_boxes=0;quadrature_samples=0
    start=time.monotonic();gauss_points=8 if experimental_rule else 5;work_charge=3 if experimental_rule else 1;regions=0
    if type(seed_shape)!=tuple or len(seed_shape)!=2 or any(type(n)!=int or n<1 or n>limits.cells or n&(n-1) for n in seed_shape):raise ValueError('two exact power-of-two seed counts required')
    nx,ny=seed_shape;seed_depth=(nx.bit_length()-1)+(ny.bit_length()-1)
    if nx*ny*work_charge>max_cells or seed_depth>max_depth:raise ValueError('seed cover exceeds work or depth budget')

    if gauss_points==3:
        root=(I(Q(3,5)).sqrt())/2;nodes=[I('.5')-root,I('.5'),I('.5')+root];weights=[I(Q(5,18)),I(Q(4,9)),I(Q(5,18))]
    elif gauss_points==5:
        a=(I(5)+2*I(Q(10,7)).sqrt()).sqrt()/6;b=(I(5)-2*I(Q(10,7)).sqrt()).sqrt()/6
        nodes=[I('.5')-a,I('.5')-b,I('.5'),I('.5')+b,I('.5')+a]
        u=(I(322)-13*I(70).sqrt())/1800;v=(I(322)+13*I(70).sqrt())/1800
        weights=[u,v,I(Q(64,225)),v,u]
    elif gauss_points==8:
        from tools.opaque_gauss8 import rule
        nodes,weights=rule(checkpoint=checkpoint)
    else:raise ValueError('qualified Gauss rule must be3,5or8')
    order=10 if gauss_points==8 else 2*gauss_points
    remainder_factor=I(Q(math.factorial(gauss_points)**4,(order+1)*math.factorial(order)**2))
    if gauss_points==8:
        from tools.opaque_high_peano import _mass
        data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
        remainder_factor=I(math.factorial(order))*I(_mass(order,data))
    enclosures=[];pending=[(Q(i,nx),Q(i+1,nx),Q(j,ny),Q(j+1,ny),seed_depth,0) for i in range(nx-1,-1,-1) for j in range(ny-1,-1,-1)];total=None;count=0;accepted=0;max_seen=0;last=None;leaves=[];trace=[];complete=None;leaf_evidence={}
    if chart_cover is not None:
        pending=[(Q(0),Q(1),Q(0),Q(1),row['depth'],i) for i,row in reversed(list(enumerate(chart_cover)))]
        count=0 if disjoint_roots else len(chart_cover)-1  # Partition internal nodes only; all roots charged on evaluation.
        if initial_leaves is not None:
            from tools.opaque_refinement_schedule import prepare_schedule
            pending,count=prepare_schedule(chart_cover,initial_leaves,max_cells=max_cells,max_depth=max_depth,checkpoint=checkpoint)
    def check():
        checkpoint()
        if time.monotonic()-start>=limits.reference_seconds:raise Unresolved("local deadline",cells=count,last=last,last_complete_enclosure=complete)
    channel_count=None
    def evaluate(s,t,fn=None,weighted=True):
        nonlocal channel_count
        check()
        values=(field if fn is None else fn)(s,t)
        check()
        if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(q,Jet) for q in values):raise ValueError('field returns1..4 matching jets')
        if fn is None:
            if channel_count is None:channel_count=len(values)
            elif len(values)!=channel_count:raise Unresolved('matching channel count across complete cover required')
        return values
    rules={tensor_rule} if chart_cover is None else {r.get('tensor_rule') for r in chart_cover}
    directional={};quarter_weights={}
    for selected in sorted(v for v in rules if v is not None):
        from tools.opaque_tensor_smooth import rule as directional_rule,quarter_factors
        points=12 if selected in ('gauss3x12-local4','gauss4x12-local4') else 8
        directional[selected]=directional_rule(points=points,radial=4 if selected.startswith('gauss4x') else 3,checkpoint=check)
        if selected in ('gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4'):
            quarter_weights[selected]=quarter_factors(points=points,checkpoint=check)
    radial_weights=None
    if tensor_rule=='gauss3x8-local4x4' or chart_cover is not None and any(r.get('tensor_rule')=='gauss3x8-local4x4' for r in chart_cover):
        from tools.opaque_tensor_smooth import radial_quarter_factors
        radial_weights=radial_quarter_factors(checkpoint=check)
    used_rules=set()
    while True:
        check()
        if not pending:
            total=[[ZERO,ZERO] for _ in enclosures[0][1]]
            for _,bounds,_,_ in enclosures:
                for k,row in enumerate(bounds):
                    for j in range(2):total[k][j]=total[k][j]+row[j]
            complete=dict(quadrature_samples=quadrature_samples,derivative_boxes=derivative_boxes,complete_cover=True,cells=count,leaves=len(enclosures),value=[[str(row[0].lo),str(row[0].hi)] for row in total],derivative=[[str(row[1].lo),str(row[1].hi)] for row in total],widths=[[str(v.width) for v in row] for row in total])
            complete['error_accounting']=cover_accounting([leaf_evidence[box] for box,_,_,_ in enclosures],reference_profile=reference_profile,whole_width_limit=str((2*tol).lo))
            progress(dict(cells=count,leaves=len(enclosures),widths=[[str(v.width) for v in row] for row in total],seconds=time.monotonic()-start))
            if all(v.width<=2*tol.lo for row in total for v in row):break
            # Fixed leaves cannot shrink under this schedule. Interval widths
            # add even when their centers have opposite signs. Exact rational
            # endpoint subtraction avoids a rounded upper width causing refusal.
            fixed=[[sum((Q(bounds[k][j].hi)-Q(bounds[k][j].lo)
                         for box,bounds,_,_ in enclosures if box[4]>=max_depth),Q(0))
                    for j in range(2)] for k in range(len(total))]
            if any(width>2*Q(tol.lo) for row in fixed for width in row):
                raise Unresolved('depth budget: irreducible depth-limited width',cells=count,
                    fixed_widths=[[str(v) for v in row] for row in fixed],
                    last=last,last_complete_enclosure=complete)
            eligible=[i for i,(box,_,_,_) in enumerate(enclosures) if box[4]<max_depth]
            if not eligible:raise Unresolved('depth budget',cells=count,last=last,last_complete_enclosure=complete)
            index=max(eligible,key=lambda i:enclosures[i][3])
            (a,b,c,d,depth,chart_id),_,split_s,_=enclosures.pop(index)
            del leaf_evidence[(a,b,c,d,depth,chart_id)]
            if split_s:
                mid=(a+b)/2;pending.extend([(mid,b,c,d,depth+1,chart_id),(a,mid,c,d,depth+1,chart_id)])
            else:
                mid=(c+d)/2;pending.extend([(a,b,mid,d,depth+1,chart_id),(a,b,c,mid,depth+1,chart_id)])
        try:checkpoint()
        except Unresolved as e:raise Unresolved(e.details['reason'],cells=count,accepted=accepted,last=last,trace=trace,last_complete_enclosure=complete) from e
        if time.monotonic()-start>=limits.reference_seconds:raise Unresolved('local deadline',cells=count,last=last,last_complete_enclosure=complete)
        a,b,c,d,depth,chart_id=pending[-1]
        active_rule=tensor_rule if chart_cover is None else chart_cover[chart_id].get('tensor_rule')
        work_charge=2 if active_rule in ('gauss3x12-local4','gauss4x12-local4') else 3 if experimental_rule else 1
        if count+work_charge>max_cells:raise Unresolved('cell budget',cells=count,accepted=accepted,last=last,trace=trace,last_complete_enclosure=complete)
        pending.pop();count+=work_charge;regions+=1;max_seen=max(max_seen,depth);hs=I(b-a);ht=I(d-c)
        split_s=b-a>=d-c
        if chart_cover is not None:
            active=chart_cover[chart_id];field=active['field'];prepare=active.get('prepare');correction=active.get('correction')
            correction_integrator=active.get('correction_integrator');tangent_correction_integrator=active.get('tangent_correction_integrator')
            refinement_field=active.get('refinement_field');domain_split=active.get('domain_split');remainder_partition=active.get('remainder_partition');tensor_rule=active.get('tensor_rule')
        axis_s,axis_t=(nodes,weights),(nodes,weights);order_s=order_t=order;factor_s=factor_t=remainder_factor
        if tensor_rule is not None:axis_s,axis_t,(order_s,order_t),(factor_s,factor_t)=directional[tensor_rule]
        used_rules.add(tensor_rule if tensor_rule else 'gauss'+str(gauss_points)+'x'+str(gauss_points))
        try:
            if prepare is not None:
                prepare((a+b)/2,(c+d)/2,(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi)))
            accum=None
            for ns,ws in zip(*axis_s):
                for nt,wt in zip(*axis_t):
                    quadrature_samples+=1
                    if experimental_rule:checkpoint()
                    vals=evaluate(Jet(I(a)+hs*ns),Jet(I(c)+ht*nt))
                    if accum is None:accum=[[ZERO,ZERO] for _ in vals]
                    for k,q in enumerate(vals):
                        for j in range(2):accum[k][j]=accum[k][j]+q[0,j]*ws*wt*hs*ht
            if remainder_partition is None:
                if tensor_rule=='gauss3x8-local4x4':
                    dss=[evaluate(Jet.variable(I(I(a+(b-a)*Q(k,4)).lo,I(a+(b-a)*Q(k+1,4)).hi),order=order_s),Jet(I(I(c).lo,I(d).hi),order=order_s)) for k in range(4)]
                    derivative_boxes+=3
                else:
                    ds=evaluate(Jet.variable(I(I(a).lo,I(b).hi),order=order_s),Jet(I(I(c).lo,I(d).hi),order=order_s))
                if tensor_rule in ('gauss3x8-local2','gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4'):
                    # Exact symmetry halves the L1 Peano mass; narrower derivative
                    # domains add no quadrature samples and keep the global scale.
                    parts=4 if tensor_rule in ('gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4') else 2;dts=[]
                    for i in range(parts):
                        lo,hi=c+(d-c)*Q(i,parts),c+(d-c)*Q(i+1,parts)
                        dts.append(evaluate(Jet(I(I(a).lo,I(b).hi),order=order_t),Jet.variable(I(I(lo).lo,I(hi).hi),order=order_t)))
                    derivative_boxes+=1+parts
                else:
                    dt=evaluate(Jet(I(I(a).lo,I(b).hi),order=order_t),Jet.variable(I(I(c).lo,I(d).hi),order=order_t));derivative_boxes+=2
            else:
                from tools.opaque_local_smooth import remainder
                localized=remainder(evaluate,(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi)),remainder_partition,checkpoint=check);derivative_boxes+=sum(remainder_partition)
            bounds=[];error_s=D(0);error_t=D(0)
            for k in range(len(accum)):
                channel=[]
                for j in range(2):
                    if remainder_partition is None:
                        if tensor_rule in ('gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4'):
                            weighted=sum((I(row[k][order_t,j].bound)*mass for row,mass in zip(dts,quarter_weights[tensor_rule])),I(0))
                        else:
                            tbound=sum((I(row[k][order_t,j].bound) for row in dts),I(0))/2 if tensor_rule=='gauss3x8-local2' else dt[k][order_t,j].bound
                            weighted=I(tbound)*factor_t
                        radial=sum((I(row[k][order_s,j].bound)*mass for row,mass in zip(dss,radial_weights)),I(0)) if tensor_rule=='gauss3x8-local4x4' else I(ds[k][order_s,j].bound)*factor_s
                        es=ht*hs**(order_s+1)*radial;et=hs*ht**(order_t+1)*weighted
                        error_s=max(error_s,es.hi);error_t=max(error_t,et.hi);error=es+et
                        channel.append(accum[k][j]+I(error.hi.copy_negate(),error.hi))
                    else:
                        es,et=localized[0][k][j],localized[1][k][j]
                        error_s=max(error_s,es.width);error_t=max(error_t,et.width)
                        channel.append(accum[k][j]+es+et)
                bounds.append(channel)
            base_widths=[[str(v.width) for v in q] for q in bounds];correction_widths=[];correction_parts=None
            if correction_integrator is not None:
                if correction is not None:raise ValueError("choose one correction contract")
                delta=correction_integrator((I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi)))
                if len(delta)!=len(bounds):raise ValueError("correction channel count")
                components=getattr(correction_integrator,'error_breakdown',None)
                if components is not None:correction_parts=[[{k:[str(v.lo),str(v.hi)] for k,v in row.items()} for row in axis] for axis in components]
                for k,q in enumerate(delta):
                    correction_widths.append([str(q.width),"0"])
                    bounds[k][0]=bounds[k][0]+q
            elif correction is not None:
                # The correction needs only interval value/first-tangent bounds,
                # not high spatial derivatives. The field owns its physical
                # Jacobian; integrate the parameter-box measure exactly.
                delta=evaluate(Jet(I(I(a).lo,I(b).hi)),Jet(I(I(c).lo,I(d).hi)),correction,False)
                if len(delta)!=len(bounds):raise ValueError('correction channel count')
                mass=Jet((b-a)*(d-c))
                for k,q in enumerate(delta):
                    residual=q*mass
                    correction_widths.append([str(residual[0,j].width) for j in range(2)])
                    for j in range(2):bounds[k][j]=bounds[k][j]+residual[0,j]
            tangent_widths=[]
            if tangent_correction_integrator is not None:
                # A certified *integral* of distributional boundary transport,
                # already including physical Jacobians. Do not multiply by box
                # area again or approximate it by pointwise shader derivatives.
                flux=tangent_correction_integrator((I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi)))
                if not isinstance(flux,(list,tuple)) or len(flux)!=len(bounds) or any(not isinstance(q,I) for q in flux):
                    raise ValueError('tangent correction channel/type mismatch')
                for k,q in enumerate(flux):
                    bounds[k][1]=bounds[k][1]+q;tangent_widths.append(str(q.width))
            check()
            split_s=error_s>=error_t
            if correction_widths and max(D(v) for row in correction_widths for v in row)>max(D(v) for row in base_widths for v in row):
                # This only chooses work order. Acceptance still uses the complete
                # outward enclosure, never the hint or an estimated error.
                hint=getattr(correction_integrator,'preferred_split_axis',None)
                split_s=(hint==0) if type(hint)==int and hint in (0,1) else b-a>=d-c
            # A correction's derivative may dominate even when its colour
            # uncertainty is tiny. A declared axis guides subdivision only;
            # the unchanged complete sum is the sole acceptance criterion.
            tangent_hint=getattr(tangent_correction_integrator,'preferred_split_axis',None)
            if type(tangent_hint)==int and tangent_hint in (0,1) and tangent_widths:
                other=max([D(v) for row in base_widths+correction_widths for v in row],default=D(0))
                if max(D(v) for v in tangent_widths)>other:split_s=tangent_hint==0
            priority=max(v.width for row in bounds for v in row)
            guided=False
            if refinement_field is not None:
                try:
                    checkpoint()
                    gs=evaluate(Jet.variable(I(I(a).lo,I(b).hi),order=order),Jet(I(I(c).lo,I(d).hi),order=order),refinement_field)
                    checkpoint()
                    gt=evaluate(Jet(I(I(a).lo,I(b).hi),order=order),Jet.variable(I(I(c).lo,I(d).hi),order=order),refinement_field)
                    if len(gs)!=len(bounds) or len(gt)!=len(bounds):raise ValueError('guidance channel count')
                    es=max((ht*hs**(order+1)*q[order,j].bound*remainder_factor).hi for q in gs for j in range(2))
                    et=max((hs*ht**(order+1)*q[order,j].bound*remainder_factor).hi for q in gt for j in range(2))
                    base_hint=(2*(I(es)+I(et))).hi
                    residual=max((D(v) for row in correction_widths for v in row),default=D(0))
                    proposed_split=es>=et
                    if residual>base_hint:
                        hint=getattr(correction_integrator,'preferred_split_axis',None)
                        proposed_split=(hint==0) if type(hint)==int and hint in (0,1) else b-a>=d-c
                    proposed_priority=(I(base_hint)+I(residual)).hi
                    if proposed_priority>=priority:
                        priority=proposed_priority;split_s=proposed_split;guided=True
                except Unresolved:raise
                except ValueError:
                    # A heuristic has no authority to reject or admit a bound.
                    # Missing guidance keeps the already computed default order.
                    pass
            target=(2*tol*hs*ht).lo
            last=dict(widths=[[str(v.width) for v in q] for q in bounds],base_widths=base_widths,correction_widths=correction_widths,correction_error_components=correction_parts,depth=depth,refinement_priority=str(priority),guided=guided)
            if tangent_correction_integrator is not None:last['tangent_transport_widths']=tangent_widths
            okay=True  # Acceptance uses the summed enclosure,not a local quota.
        except Unresolved as e:
            raise Unresolved(e.details['reason'],cells=count,accepted=accepted,last=last,trace=trace,last_complete_enclosure=complete) from e
        except ValueError as e:
            okay=False;last=dict(domain=str(e),depth=depth)
            if str(e) in ('bounded angular segment count','bounded angular crossing count'):
                hint=getattr(correction_integrator,'domain_split_axis',None)
                if type(hint)==int and hint in (0,1):split_s=hint==0

        last['box']=[str(z) for z in (a,b,c,d)]
        if chart_cover is not None:last['chart']=chart_id
        if len(trace)<32:trace.append(dict(cell=count,okay=okay,split_s=split_s,**last))
        if okay:
            enclosures.append(((a,b,c,d,depth,chart_id),bounds,split_s,priority))
            leaf_evidence[(a,b,c,d,depth,chart_id)]=last
            accepted=len(enclosures)
        else:
            if depth==max_depth:raise Unresolved('depth budget',cells=count,last=last,last_complete_enclosure=complete)
            point=None
            if domain_split is not None:
                check();proposal=domain_split((a,b,c,d),last['domain']);check()
                if proposal is not None:
                    valid=isinstance(proposal,tuple) and len(proposal)==2
                    if valid:
                        axis,point=proposal;valid=type(axis)==int and axis in (0,1) and type(point) in (Q,int)
                    if valid:
                        lo,hi=(a,b) if axis==0 else (c,d);valid=lo<point<hi
                    if not valid:raise Unresolved('invalid domain split',cells=count,last=last,last_complete_enclosure=complete)
                    split_s=axis==0;point=Q(point);last['domain_cut']=[axis,str(point)]
            if split_s:
                mid=(a+b)/2 if point is None else point;pending.extend([(mid,b,c,d,depth+1,chart_id),(a,mid,c,d,depth+1,chart_id)])
            else:
                mid=(c+d)/2 if point is None else point;pending.extend([(a,b,mid,d,depth+1,chart_id),(a,b,c,mid,depth+1,chart_id)])
    check()
    return dict(quadrature_rules=sorted(used_rules),quadrature_samples=quadrature_samples,derivative_boxes=derivative_boxes,chart_count=1 if chart_cover is None else len(chart_cover),chart_leaves=[dict(chart=q[0][5],box=[str(z) for z in q[0][:4]],depth=q[0][4]) for q in enclosures],reference_profile=limits.record(),error_accounting=complete['error_accounting'],value=[q[0] for q in total],derivative=[q[1] for q in total],cells=count,regions=regions,work_charge=max(2 if q in ('gauss3x12-local4','gauss4x12-local4') else 3 if experimental_rule else 1 for q in used_rules),accepted=accepted,max_depth=max_seen,leaves=[[str(z) for z in q[0][:4]] for q in enclosures],tolerance=str(tol.lo),method=('outward Decimal60 + summed-enclosure directional Gauss3/4x8/12 with radial order6/8 and transverse10' if any(q.startswith(('gauss3x','gauss4x')) for q in used_rules) else 'outward Decimal60 + summed-enclosure adaptive tensor Gauss'+str(gauss_points)+' + derivative order'+str(order)),gauss_points=None if any(q.startswith(('gauss3x','gauss4x')) for q in used_rules) else gauss_points,seconds=time.monotonic()-start)
