"""Reproducible, offline conversion of pinned upstream data into small demo fixtures.

Run with the directory holding bunny.tar.gz, TeapotGeometry.js and three-LICENSE.txt.
No source code from the downloaded JavaScript is executed. No archive paths are extracted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

OUTPUT = Path(__file__).parent / 'demo_assets'


def build(source):
    expected={'bunny.tar.gz':'a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84',
              'TeapotGeometry.js':'31ce94a5efe7ff4dbda886b258bcab5a16e2d0e329c9f041eddf3007d2783401'}
    for name,digest in expected.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Upstream digest mismatch: '+name)
    OUTPUT.mkdir(exist_ok=True)
    with tarfile.open(source/'bunny.tar.gz') as archive:
        member='bunny/reconstruction/bun_zipper_res3.ply'
        lines=archive.extractfile(member).read().decode('ascii').splitlines()
    end=lines.index('end_header')
    nv=int(next(s.split()[-1] for s in lines[:end] if s.startswith('element vertex ')))
    nf=int(next(s.split()[-1] for s in lines[:end] if s.startswith('element face ')))
    vertices=[list(map(float,s.split()[:3])) for s in lines[end+1:end+1+nv]]
    faces=[list(map(int,s.split()[1:])) for s in lines[end+1+nv:end+1+nv+nf]]
    assert len(faces)==nf and all(len(f)==3 for f in faces)
    emit('bunny',vertices,faces,'https://graphics.stanford.edu/data/3Dscanrep/',
         'Stanford University Computer Graphics Laboratory. Research use and free redistribution with acknowledgement; commercial use requires permission. See README.md. Not the gffx code license.',
         {'archive_sha256':expected['bunny.tar.gz'],'member':member,'conversion':'XYZ and triangle indices only; no decimation or repair'})
    text=(source/'TeapotGeometry.js').read_text(encoding='utf-8')
    def numeric(name):
        body=re.search(r'const '+name+r' = \[(.*?)\];',text,re.S).group(1)
        body=re.sub(r'/\*.*?\*/','',body,flags=re.S)
        return [float(v.replace(' ','')) for v in body.split(',') if v.strip()]
    patches=list(map(int,numeric('teapotPatches')));coords=numeric('teapotVertices')
    points=[coords[i:i+3] for i in range(0,len(coords),3)]
    assert len(patches)==512 and len(points)==290
    vertices=[];faces=[];steps=6
    def weights(t):return [(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3]
    for offset in range(0,len(patches),16):
        base=len(vertices);patch=[points[i] for i in patches[offset:offset+16]]
        for i in range(steps+1):
            for j in range(steps+1):
                a=weights(i/steps);b=weights(j/steps)
                x,y,z=[sum(a[u]*b[v]*patch[u*4+v][k] for u in range(4) for v in range(4)) for k in range(3)]
                vertices.append([round(x,8),round(z,8),round(-y,8)])  # Y up; proper rotation.
        for i in range(steps):
            for j in range(steps):
                a=base+i*(steps+1)+j;b=a+steps+1
                for f in ((a,b,a+1),(a+1,b,b+1)):
                    p,q,r=[vertices[k] for k in f]
                    u=[q[k]-p[k] for k in range(3)];v=[r[k]-p[k] for k in range(3)]
                    cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                    if sum(x*x for x in cross)>1e-18:faces.append(f)
    emit('teapot',vertices,faces,'https://github.com/mrdoob/three.js/blob/r180/examples/jsm/geometries/TeapotGeometry.js',
         'MIT; Copyright 2010-2025 three.js authors. See three-LICENSE.txt.',
         {'source_sha256':expected['TeapotGeometry.js'],'conversion':'32 bicubic patches, 6 segments per direction; drop zero-area triangles; rotate Z-up to Y-up; no seam welding'})
    # Preserve the complete upstream license as an asset alongside the derived data.
    (OUTPUT/'three-LICENSE.txt').write_bytes((source/'three-LICENSE.txt').read_bytes())


def emit(name,vertices,faces,url,license_note,provenance):
    data=dict(vertices=vertices,faces=faces,source_url=url,license_note=license_note,provenance=provenance)
    target=OUTPUT/(name+'.json')
    target.write_text(json.dumps(data,separators=(',',':'))+'\n',encoding='utf-8')
    print(name,len(vertices),'vertices',len(faces),'triangles',target.stat().st_size,'bytes',hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    build(parser.parse_args().source)
