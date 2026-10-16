from fractions import Fraction as Q
import json,os,time
from pathlib import Path
from tools.opaque_shading_intervals import I,Jet,Unresolved

def test_smooth_original_bound_diagnostic(record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    start=time.monotonic();rows=[]
    def check():
        if time.monotonic()-start>=60:raise Unresolved('smooth local bound diagnostic deadline')
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
        for index in (46,45,44,43,42):
            row=charts[index];row['prepare'](Q(1,2),Q(1,2),(I(0,1),I(0,1)));directions=[]
            for axis in (0,1):
                levels=[]
                for segments in (1,2,4):
                    terms=[];started=time.monotonic()
                    for i in range(segments):
                        args=[Jet(I(0,1),order=10),Jet(I(0,1),order=10)];args[axis]=Jet.variable(I(Q(i,segments).numerator/Q(i,segments).denominator,Q(i+1,segments).numerator/Q(i+1,segments).denominator),order=10)
                        check();values=row['field'](*args);terms.append([[str(q[10,j]) for j in (0,1)] for q in values])
                    levels.append(dict(segments=segments,coefficients=terms,seconds=time.monotonic()-started))
                directions.append(dict(axis=axis,levels=levels))
            rows.append(dict(chart=index,directions=directions))
    result=dict(rows=rows,seconds=time.monotonic()-start,scope='Diagnostic derivative enclosures only; no new piece certificate.')
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf8');record_property('smooth_bounds',str(path))


import pytest
@pytest.mark.parametrize('parts',[(1,4),(4,1),(2,2)])
@pytest.mark.parametrize('power',[0,9,10,11,15])
def test_local_smooth_exact_polynomial_and_motion(parts,power):
    from tools.opaque_square_reference import integrate_square
    scale=Q(1,2**60)
    def field(x,y):
        return [(x**power)*(y**power)*(1+x.lift(0,tangent=Q(-2,3))),scale*(x**power)*(1+y**power)*(1+x.lift(0,tangent=Q(3,7)))]
    result=integrate_square(field,tolerance='1e-12',remainder_partition=parts)
    for value,gradient,want,factor in zip(result['value'],result['derivative'],[Q(1,(power+1)**2),scale*(1+Q(1,power+1))/(power+1)],[Q(-2,3),Q(3,7)]):
        exact=I(want);derivative=I(want*factor)
        assert value.lo<=exact.lo<=exact.hi<=value.hi
        assert gradient.lo<=derivative.lo<=derivative.hi<=gradient.hi
        assert value.lo>0


def test_local_smooth_kernel_masses_match_exact_total_and_symmetry():
    from tools.opaque_local_smooth import masses
    import math
    factor=I(Q(math.factorial(5)**4,11*math.factorial(10)**2))
    for parts in (1,2,4,8):
        row=masses(parts);total=sum(row,I(0))
        assert len(row)==parts and all(q.lo>=0 for q in row)
        assert total.lo<=factor.lo<=factor.hi<=total.hi
        assert total.width<I('1e-45').lo
        for a,b in zip(row,reversed(row)):assert a.lo<=b.hi and b.lo<=a.hi
    for bad in (0,3,16,True):
        with pytest.raises(ValueError):masses(bad)


@pytest.mark.parametrize('parts',[(1,4),(4,1)])
def test_local_smooth_sharp_rational_has_independent_integral(parts):
    from tools.opaque_square_reference import integrate_square
    def field(x,y):
        a,b=(x,y) if parts[0]>1 else (y,x)
        return [(1+b)/(2-a-a.lift(0,tangent=1))]
    result=integrate_square(field,remainder_partition=parts,tolerance='1e-12')
    value=I(2).ln()*Q(3,2);derivative=I(Q(3,4))
    assert result['value'][0].lo<=value.lo<=value.hi<=result['value'][0].hi
    assert result['derivative'][0].lo<=derivative.lo<=derivative.hi<=result['derivative'][0].hi


def test_local_smooth_explicit_contract_and_deadline():
    from tools.opaque_square_reference import integrate_square
    f=lambda x,y:[x+y]
    for bad in [(3,1),(1,16),[1,4],(True,1)]:
        with pytest.raises(ValueError):integrate_square(f,remainder_partition=bad)
    with pytest.raises(ValueError):integrate_square(f,remainder_partition=(1,4),experimental_rule='gauss8')
    calls=[]
    def check():
        calls.append(1)
        if len(calls)>60:raise Unresolved('closed-control deadline')
    with pytest.raises(Unresolved,match='closed-control deadline'):integrate_square(f,remainder_partition=(1,4),checkpoint=check)


def test_prepared_polynomial_reuses_transverse_rows_without_endpoint_change():
    from tools.opaque_angle_light import PreparedIntervalPolynomial,_evaluate
    from tools.opaque_arb_jets import backend
    import tools.opaque_shading_intervals as scalar
    coeff={(0,0,0):Q(1,3),(1,1,0):Q(-3,7),(2,0,0):Q(1,2**60),(4,2,0):Q(13,17),(2,1,1):Q(7,11)}
    p=PreparedIntervalPolynomial(coeff)
    with backend(profile='ball'):
        for prec in (60,75,60):
            scalar.PRECISION=prec
            for y in (I(0,1),I(-2,3),I(Q(1,3))):
                before=p.row_evaluations
                for x in (I(0,1),I(-1,1),I(Q(1,2**60))):
                    a,b=p(x,y),_evaluate(coeff,x,y);assert (a.lo,a.hi)==(b.lo,b.hi)
                assert p.row_evaluations==before+1
        for i in range(70):p(I(1),I(Q(i,100)))
        assert len(p.entries)==64 and p.row_hits>=18


@pytest.mark.parametrize('index',[46,63])
def test_local_smooth_original_comparison(index,record_property):
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    results=[];start=time.monotonic()
    def check():
        if time.monotonic()-start>=85:raise Unresolved('local comparison deadline')
    with backend(profile='ball'):
        for parts in (None,(1,4),(1,8)):
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
            row=charts[index];started=time.monotonic()
            try:
                result=integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1 if index==46 else 256-cover.cells+1,max_depth=10-row['depth'],checkpoint=check,remainder_partition=parts)
                result=dict(status='pass',basis=result)
            except Unresolved as error:result=dict(status='unresolved',failure=error.details)
            results.append(dict(partition=parts,result=result,seconds=time.monotonic()-started))
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(chart=index,rows=results,seconds=time.monotonic()-start),default=str,indent=2),encoding='utf8');record_property('localized_bound_comparison',str(path))
    assert len(results)==3
    if index==63:assert all(r['result']['status']=='pass' for r in results)


def test_local_smooth_arb_exact_controls():
    from tools.opaque_arb_jets import backend
    with backend(profile='ball'):
        test_local_smooth_kernel_masses_match_exact_total_and_symmetry()
        for parts in [(1,4),(4,1),(2,2)]:
            for power in (0,9,10,11,15):test_local_smooth_exact_polynomial_and_motion(parts,power)
        for parts in [(1,4),(4,1)]:test_local_smooth_sharp_rational_has_independent_integral(parts)


def test_complete_cover_counts_all_contributors_without_local_acceptance():
    from tools.opaque_square_reference import cover_accounting
    rows=[dict(chart=i,box=['0','1','0','1'],depth=4,widths=[['1e-15','2e-12' if i<7 else '3e-15']],base_widths=[['1e-15','1e-12' if i<7 else '1e-15']],correction_widths=[['0','0']],correction_error_components=None) for i in range(12)]
    result=cover_accounting(rows)
    assert len(result['leaf_widths'])==12 and len(result['largest_leaves'])==5
    assert result['diagnostic_counts']['leaves_wider_than_whole_limit']==7
    assert result['diagnostic_counts']['whole_width_limit']=='1E-12'
    assert result['diagnostic_counts']['is_local_acceptance_rule'] is False


def test_transverse_reuse_controlled_cost(record_property,monkeypatch):
    from tools.opaque_angle_light import PreparedIntervalPolynomial
    from tools.opaque_original_motion_reference import angular_material_cover
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    candidate=PreparedIntervalPolynomial.__call__;rows=[];start=time.monotonic()
    def old(self,t,s):
        # Retain the same context preparation; force a transverse-row miss.
        self.entries.clear();return candidate(self,t,s)
    def check():
        if time.monotonic()-start>=60:raise Unresolved('paired reuse cost deadline')
    with backend(profile='ball'):
        for cached in (False,True,True,False):
            monkeypatch.setattr(PreparedIntervalPolynomial,'__call__',candidate if cached else old)
            p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,checkpoint=check)
            row=charts[46];started=time.monotonic()
            try:integrate_square(row['field'],prepare=row['prepare'],correction_integrator=row['correction_integrator'],tangent_correction_integrator=row['tangent_correction_integrator'],tolerance='5e-13',max_cells=1,checkpoint=check)
            except Unresolved as e:result=e.details['last_complete_enclosure']
            rows.append(dict(cached=cached,seconds=time.monotonic()-started,values=result['value'],derivatives=result['derivative'],cells=result['cells']))
    assert all((r['values'],r['derivatives'])==(rows[0]['values'],rows[0]['derivatives']) for r in rows)
    path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,scope='A-B-B-A same-cell comparison with identical outward endpoints; cache-disabled control includes lookup overhead, no full-piece speed claim.'),indent=2),encoding='utf8');record_property('controlled_reuse_cost',str(path))


def test_localized_chart_connection_and_no_implicit_budget_reset():
    from tools.opaque_original_motion_reference import angular_material_cover,angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tools.opaque_square_reference import integrate_square
    from tests.packaging.test_opaque_motion_tangent import frozen
    with pytest.raises(ValueError):angular_piece_reference({},smooth_partition=(1,4))
    with backend(profile='ball'):
        p,curve,cover,charts=angular_material_cover(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,smooth_partition=(1,4))
        selected=[i for i,c in enumerate(charts) if c.get('remainder_partition')]
        assert 46 in selected and 63 not in selected and 68 not in selected
        assert len(charts)==72 and cover.cells==143
        with pytest.raises(Unresolved) as failure:integrate_square(None,chart_cover=charts,max_cells=143,max_depth=10,checkpoint=lambda:(_ for _ in ()).throw(Unresolved('shared construction deadline')))
        assert 'shared construction deadline' in str(failure.value)


def test_localized_fresh_complete_original_piece(record_property):
    from tools.opaque_original_motion_reference import angular_piece_reference
    from tools.opaque_arb_jets import backend
    from tests.packaging.test_opaque_motion_tangent import frozen
    failure=None;rows=[]
    with backend(profile='ball'):
        try:result=angular_piece_reference(frozen(),tight_discriminant=True,algebraic_inverse=True,mapped_transport=True,exit_angle=True,unsquared_inverse=True,physical_angle=True,smooth_partition=(1,4),progress=rows.append)
        except ValueError as error:failure=str(error);result=dict(status='unresolved',failure=getattr(error,'details',str(error)))
    result['progress']=rows;path=Path(os.environ['GFFX_FULL_REFERENCE']);assert not path.exists();path.write_text(json.dumps(result,default=str,indent=2),encoding='utf8');record_property('localized_whole_piece',str(path))
    assert failure is None,result
    assert result['basis']['chart_count']==72 and result['basis']['cells']<=256
    assert all(v.width<=I('2e-12').lo for v in result['value']+result['derivative'])
