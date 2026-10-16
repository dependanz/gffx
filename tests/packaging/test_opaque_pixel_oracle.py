"""Analytical exact-area controls for the independent opaque visibility oracle."""
from fractions import Fraction as Q
import pytest


def render(v,f,colour=None,**kw):
    from tools.opaque_pixel_oracle import integrate
    return integrate(v,f,[(1,1,1)]*len(f) if colour is None else colour,**kw)


def square():
    return [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]


@pytest.mark.parametrize('faces',[[(0,1,2),(0,2,3)],[(0,1,3),(1,2,3)],[(2,1,0),(3,2,0)]])
def test_square_diagonal_and_winding_have_exact_unit_coverage(faces):
    r=render(square(),faces,height=1,width=1)
    assert r['mask']==((Q(1),),)
    assert r['colour']==(((Q(1),Q(1),Q(1)),),)


def test_subdivision_and_interior_vertex_motion_do_not_change_opaque_union():
    for centre in [(Q(1,2),Q(1,2),0),(Q(1,3),Q(2,5),0)]:
        r=render(square()+[centre],[(0,1,4),(1,2,4),(2,3,4),(3,0,4)],height=1,width=1)
        assert r['mask']==((Q(1),),)


def test_clipped_triangle_and_subpixel_triangle_areas_are_exact():
    assert render([(0,0,0),(2,0,0),(0,2,0)],[(0,1,2)],height=1,width=1)['mask']==((Q(1),),)
    r=render([(Q(1,16),Q(1,16),0),(Q(3,16),Q(1,16),0),(Q(1,16),Q(3,16),0)],[(0,1,2)],height=1,width=1)
    assert r['mask']==((Q(1,128),),)


def test_moving_silhouette_has_exact_area_and_translation_derivative():
    def area(right):
        v=[(-1,-1,0),(right,-1,0),(right,2,0),(-1,2,0)]
        return render(v,[(0,1,2),(0,2,3)],height=1,width=1)['mask'][0][0]
    for p in [Q(1,7),Q(1,3),Q(7,8)]:
        assert area(p)==p
        for eps in [Q(1,256),Q(1,1024)]:
            assert (area(p+eps)-area(p-eps))/(2*eps)==1


def test_overlapping_surfaces_choose_the_front_colour_and_preserve_full_alpha():
    v=square()+[(x,y,-1) for x,y,z in square()]
    r=render(v,[(0,1,2),(0,2,3),(4,5,6),(4,6,7)],[(1,0,0)]*2+[(0,0,1)]*2,height=1,width=1)
    assert r['mask']==((Q(1),),)
    assert r['colour']==(((Q(0),Q(0),Q(1)),),)


def test_partial_occlusion_and_background_are_integrated_before_compositing():
    v=[(0,0,0),(Q(1,2),0,0),(Q(1,2),1,0),(0,1,0)]
    r=render(v,[(0,1,2),(0,2,3)],[(1,0,0)]*2,height=1,width=1,background=(0,0,1))
    assert r['mask']==((Q(1,2),),)
    assert r['colour']==(((Q(1,2),Q(0),Q(1,2)),),)


def test_intersecting_depth_planes_have_analytical_colour_and_derivative():
    # Both triangles cover the whole pixel; z_A=x-t crosses z_B=0 at x=t.
    def red(t):
        v=[(-1,-1,-1-t),(3,-1,3-t),(-1,3,-1-t),(-1,-1,0),(3,-1,0),(-1,3,0)]
        r=render(v,[(0,1,2),(3,4,5)],[(1,0,0),(0,0,1)],height=1,width=1)
        assert r['mask']==((Q(1),),)
        assert r['colour'][0][0]==(t,Q(0),1-t)
        return r['colour'][0][0][0]
    t=Q(2,5);eps=Q(1,512)
    assert (red(t+eps)-red(t-eps))/(2*eps)==1


def test_coplanar_tie_is_earlier_face_and_degenerate_triangle_has_zero_area():
    v=[(-1,-1,0),(3,-1,0),(-1,3,0)]
    r=render(v,[(0,1,2),(0,1,2)],[(1,0,0),(0,0,1)],height=1,width=1)
    assert r['colour']==(((Q(1),Q(0),Q(0)),),)
    assert render(v,[(0,0,1)],height=1,width=1)['mask']==((Q(0),),)


def test_pixel_windows_partition_the_same_global_integral():
    v=[(Q(1,4),Q(1,4),0),(Q(7,4),Q(1,4),0),(Q(7,4),Q(7,4),0),(Q(1,4),Q(7,4),0)]
    whole=render(v,[(0,1,2),(0,2,3)],height=2,width=2)
    for y in range(2):
        for x in range(2):
            part=render(v,[(0,1,2),(0,2,3)],height=1,width=1,origin=(x,y))
            assert part['mask'][0][0]==whole['mask'][y][x]==Q(9,16)
            assert part['colour'][0][0]==whole['colour'][y][x]


@pytest.mark.parametrize('case',['nan','index','extent','channels','complexity','fraction'])
def test_invalid_or_unbounded_input_is_rejected(case):
    v=square(); f=[(0,1,2)]; colours=[(1,1,1)]; kw=dict(height=1,width=1)
    if case=='nan':v[0]=(float('nan'),0,0)
    if case=='index':f=[(0,1,99)]
    if case=='extent':kw['height']=10000
    if case=='channels':colours=[(1,1)]
    if case=='complexity':f*=33;colours*=33
    if case=='fraction':v[0]=(Q(1,2**4096),0,0)
    with pytest.raises((ValueError,TypeError)):render(v,f,colours,**kw)


def test_checkpoint_cancellation_is_not_ignored():
    def stop():raise TimeoutError('test cancellation')
    with pytest.raises(TimeoutError,match='cancellation'):
        render(square(),[(0,1,2),(0,2,3)],height=1,width=1,checkpoint=stop)


def test_single_pixel_perturbation_is_detected_by_exact_reference():
    r=render(square(),[(0,1,2),(0,2,3)],height=1,width=1)
    assert r['mask']!=((Q(1)-Q(1,2**52),),)
