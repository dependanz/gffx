"""Test-only head recipe; no licensed assets or persistent corpus admission."""
import hashlib
import importlib
import json
import math
from collections import Counter

import pytest


def module():
    return importlib.import_module('tools.synthetic_face_fixture')


def test_fixture_matches_frozen_recipe_and_counts():
    scene = module().generate()
    assert len(scene['vertices']) == 1986 and len(scene['triangles']) == 3968
    assert scene['vertices'][0] == [0., .9, 0.]
    assert scene['vertices'][-1] == [0., -.9, 0.]
    x,y,z = scene['vertices'][1 + 15*64]
    assert x == pytest.approx(0., abs=1e-15) and y == pytest.approx(0., abs=1e-15)
    assert z == pytest.approx(.55+.12*math.exp(-(.05/.22)**2), abs=1e-14)
    assert all(math.isfinite(x) for p in scene['vertices'] for x in p)


def test_fixture_is_connected_closed_outward_and_nondegenerate():
    scene = module().generate()
    vertices, faces = scene['vertices'], scene['triangles']
    edges = Counter()
    adjacent = [set() for _ in vertices]
    for face in faces:
        assert len(set(face)) == 3
        a,b,c = [vertices[i] for i in face]
        u,v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        assert sum(x*x for x in cross) > 1e-16
        assert sum(cross[i]*(a[i]+b[i]+c[i]) for i in range(3)) > 0
        for a,b in zip(face, face[1:]+face[:1]):
            edges[a,b] += 1
            adjacent[a].add(b)
    assert all(count == 1 and edges[b,a] == 1 for (a,b),count in edges.items())
    assert len(vertices)-len(edges)//2+len(faces) == 2
    seen, todo = set(), [0]
    while todo:
        node=todo.pop()
        if node not in seen:
            seen.add(node); todo.extend(adjacent[node]-seen)
    assert len(seen) == len(vertices)


def test_corner_uvs_keep_welded_geometry_and_unwrapped_seam():
    scene = module().generate()
    seen = {}
    for face, uvs in zip(scene['triangles'],scene['corner_uvs']):
        assert len(uvs) == 3
        assert max(p[0] for p in uvs)-min(p[0] for p in uvs) <= 1/64+1e-15
        for index, uv in zip(face,uvs):
            assert all(0 <= x <= 1 for x in uv)
            seen.setdefault(index,set()).add(tuple(uv))
    assert (0.,1/32) in seen[1] and (1.,1/32) in seen[1]


def test_texture_patterns_and_vertex_fields_are_explicit():
    scene = module().generate()
    textures=scene['textures']
    assert set(textures) == {'constant','gradient','checker'}
    assert all(len(value)==32 and all(len(row)==32 for row in value) for value in textures.values())
    assert textures['constant'][0][0] == [.2,.4,.6]
    assert textures['gradient'][0][0] == [.2+.5/64,.2+.5/64,.25]
    assert textures['checker'][0][0] == [.2]*3 and textures['checker'][0][4] == [.7]*3
    assert len(scene['vertex_rgb']) == 1986


def test_canonical_serialization_is_repeatable_independent_and_finite():
    m = module()
    a,b=m.generate(),m.generate()
    blob=m.encode(a)
    assert blob == m.encode(b) and json.loads(blob) == a
    assert m.sha256(a) == hashlib.sha256(blob).hexdigest()
    a['vertices'][0][0]=1.
    assert b['vertices'][0][0] == 0.
    a['vertices'][0][0]=float('nan')
    with pytest.raises(ValueError): m.encode(a)


def test_fixture_carries_declared_camera_light_and_material_profiles():
    scene=module().generate()['scene']
    assert scene['camera']==dict(position=[0.,0.,3.],target=[0.,0.,0.],up=[0.,1.,0.],vertical_fov_degrees=40.,width=128,height=128)
    assert scene['directional_light']==dict(direction=[1.,1.,2.],irradiance=[1.,1.,1.])
    assert scene['linear_rgb'] is True and scene['exposure']==1. and scene['background']==[0.,0.,0.]
    assert scene['materials']['classic']['exponent']==2.
    assert scene['materials']['principled']['roughness']==1.
    assert scene['materials']['principled']['budget_resource_id'] is None
