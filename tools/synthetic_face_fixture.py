"""First-party, test-only synthetic head. Pure in-memory generation; no runtime API.

CORPORA_V0_1.md owns recipe v1. No downloads, external assets or implicit writes.
Serialized artifacts require explicit corpus registration outside this repository.
"""
import hashlib
import json
import math


def generate():
    """Return a fresh v1 fixture with welded geometry and face-corner UVs."""
    latitudes, sectors = 32, 64
    vertices = [[0., .9, 0.]]
    for i in range(1, latitudes):
        theta = i*math.pi/latitudes
        for j in range(sectors):
            phi = j*2*math.pi/sectors
            x = .65*math.sin(theta)*math.sin(phi)
            y = .9*math.cos(theta)
            z = .55*math.sin(theta)*math.cos(phi)
            z += .12*math.sin(theta)**2*max(math.cos(phi), 0.)**4*math.exp(-(x/.16)**2-((y-.05)/.22)**2)
            vertices.append([x, y, z])
    south = len(vertices)
    vertices.append([0., -.9, 0.])
    faces, uvs = [], []

    def ring(i, j):
        return 1+(i-1)*sectors+j % sectors

    def triangle(indices, coords):
        faces.append(indices)
        uvs.append(coords)

    for j in range(sectors):
        u, next_u = j/sectors, (j+1)/sectors
        triangle([0, ring(1,j), ring(1,j+1)], [[(u+next_u)/2,0.],[u,1/latitudes],[next_u,1/latitudes]])
    for i in range(1, latitudes-1):
        for j in range(sectors):
            a,b,c,d = ring(i,j),ring(i+1,j),ring(i+1,j+1),ring(i,j+1)
            u,w,v,q = j/sectors,(j+1)/sectors,i/latitudes,(i+1)/latitudes
            triangle([a,b,c], [[u,v],[u,q],[w,q]])
            triangle([a,c,d], [[u,v],[w,q],[w,v]])
    for j in range(sectors):
        u,w = j/sectors,(j+1)/sectors
        triangle([ring(31,j),south,ring(31,j+1)], [[u,31/32],[(u+w)/2,1.],[w,31/32]])
    textures = dict(constant=[], gradient=[], checker=[])
    for y in range(32):
        textures['constant'].append([[.2,.4,.6] for _ in range(32)])
        textures['gradient'].append([[.2+.5*(x+.5)/32,.2+.5*(y+.5)/32,.25] for x in range(32)])
        textures['checker'].append([[.7 if (x//4+y//4)%2 else .2]*3 for x in range(32)])
    return dict(schema_version=1, id='synthetic_face_m1', recipe_version=1,
                parameters=dict(latitude_intervals=32, longitude_sectors=64,
                                radii=[.65,.9,.55], nose_amplitude=.12,
                                nose_x_width=.16, nose_y_width=.22, nose_y_center=.05,
                                texture_width=32, texture_height=32),
                vertices=vertices, triangles=faces, corner_uvs=uvs,
                vertex_rgb=[[.2+.4*(x/.65+1)/2,.2+.4*(y/.9+1)/2,.25] for x,y,z in vertices],
                textures=textures,
                scene=dict(camera=dict(position=[0.,0.,3.],target=[0.,0.,0.],up=[0.,1.,0.],
                                       vertical_fov_degrees=40.,width=128,height=128),
                           directional_light=dict(direction=[1.,1.,2.],irradiance=[1.,1.,1.]),
                           linear_rgb=True,exposure=1.,background=[0.,0.,0.],
                           materials=dict(classic=dict(specular=.1,exponent=2.),
                                          principled=dict(roughness=1.,f0=[.04]*3,budget_resource_id=None))))


def encode(scene):
    """Canonical JSON bytes; no on-disk artifact or admission is implied."""
    return json.dumps(scene, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256(scene):
    return hashlib.sha256(encode(scene)).hexdigest()
