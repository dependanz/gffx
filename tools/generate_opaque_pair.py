"""Generate the internal flat SH52 paired-tangent specialization.

The scalar headers remain authoritative. This mechanical specialization shares
primal geometry, never tangent accumulation. It is admitted only for linear
SH52 cuts; incompatible support provenance or capacity falls back with charged
work. Run --check after editing any input. No build-time Python dependency.
"""
from pathlib import Path
import argparse, hashlib, re, shutil, tempfile
INPUTS = ('opaque_polygon.h','opaque_batch_cursor.h','opaque_quadrature_scalar.h','opaque_exact_orientation.h','opaque_batch_scalar.h')
def generate(core):
 with tempfile.TemporaryDirectory() as temporary:
  p=Path(temporary)
  for name in INPUTS: shutil.copyfile(core/name,p/name)
  f=p/'opaque_polygon.h';s=f.read_text().replace('#define MAX_PARTS 128','#define MAX_PARTS 64').replace('double v,d;','double v,d,e;')
  s=s.replace('a.d+b.d};','a.d+b.d,a.e+b.e};').replace('-a.d};','-a.d,-a.e};').replace('a.d*b.v+a.v*b.d};','a.d*b.v+a.v*b.d,a.e*b.v+a.v*b.e};').replace('(b.v*b.v)};','(b.v*b.v),(a.e*b.v-a.v*b.e)/(b.v*b.v)};')
  s=s.replace('int canonical_supports;','int canonical_supports; int pair_conflict;')
  start=s.index('GFFX_OPAQUE_LOCAL int support_line(')
  s=s[:start]+'''GFFX_OPAQUE_LOCAL int pair_support_same(Line a,Line b,Ctx *ctx){
   if(a.a.v!=b.a.v||a.b.v!=b.b.v||a.c.v!=b.c.v)return 0;
   int x=a.a.d==b.a.d&&a.b.d==b.b.d&&a.c.d==b.c.d;
   int y=a.a.e==b.a.e&&a.b.e==b.b.e&&a.c.e==b.c.e;
   if(x!=y){ctx->pair_conflict=1;ctx->failed=1;return 0;}
   return x&&y;
  }
  '''+s[start:]
  s=s.replace('if(q.a.v==l.a.v&&q.a.d==l.a.d&&q.b.v==l.b.v&&q.b.d==l.b.d&&q.c.v==l.c.v&&q.c.d==l.c.d)return i;','if(pair_support_same(q,l,ctx))return i;if(ctx->failed)return -1;')
  s=s.replace('GFFX_OPAQUE_LOCAL int support_same(Line a,Line b){\n    return a.a.v==b.a.v&&a.a.d==b.a.d&&a.b.v==b.b.v&&a.b.d==b.b.d&&a.c.v==b.c.v&&a.c.d==b.c.d;\n}', 'GFFX_OPAQUE_LOCAL int support_same(Line a,Line b,Ctx *ctx){return pair_support_same(a,b,ctx);}')
  s=s.replace('support_same(a,l)||support_same(negative(a),l)','support_same(a,l,ctx)||support_same(negative(a),l,ctx)')
  s=s.replace('    if(!isfinite(hit->x.d)||!isfinite(hit->y.d))ctx->failed=1;', '''    if(!isfinite(hit->x.d)||!isfinite(hit->y.d))ctx->failed=1;
      {double ra=-((a.a.e*hit->x.v+a.b.e*hit->y.v)+a.c.e),rb=-((b.a.e*hit->x.v+b.b.e*hit->y.v)+b.c.e);
       hit->x.e=(ra*b.b.v-a.b.v*rb)/det;hit->y.e=(a.a.v*rb-ra*b.a.v)/det;
       if(!isfinite(hit->x.e)||!isfinite(hit->y.e))ctx->failed=1;}
  ''')
  # Batch quadrature always has support metadata; keep non-support helper coherent too.
  s=s.replace('p->p[p->n-1].y.d==q.y.d)', 'p->p[p->n-1].y.d==q.y.d && p->p[p->n-1].x.e==q.x.e && p->p[p->n-1].y.e==q.y.e)')
  s=s.replace('r.p[0].y.d==r.p[r.n-1].y.d)', 'r.p[0].y.d==r.p[r.n-1].y.d && r.p[0].x.e==r.p[r.n-1].x.e && r.p[0].y.e==r.p[r.n-1].y.e)')
  f.write_text(s)
  f=p/'opaque_quadrature_scalar.h';s=f.read_text()
  s=s.replace('xyz[i][k].d=seed==3*(int64_t)f[i]+k?1:0;', 'xyz[i][k].d=seed==3*(int64_t)f[i]+k?1:0;xyz[i][k].e=seed>=0&&seed%3==0&&seed+1==3*(int64_t)f[i]+k?1:0;')
  s=s.replace('D b={r,a.d/(2*r)}','D b={r,a.d/(2*r),a.e/(2*r)}')
  s=s.replace('*contract=0.;*curves=0;', '*contract=0.;contract[1]=0.;*curves=0;')
  s=s.replace('!isfinite(q[t].d)', '!isfinite(q[t].d)||!isfinite(q[t].e)').replace('!isfinite(q[c].d)', '!isfinite(q[c].d)||!isfinite(q[c].e)')
  s=s.replace('if(gradient)*contract+=q[t].d*read_real(gradient,4*index+t);','if(gradient){*contract+=q[t].d*read_real(gradient,4*index+t);contract[1]+=q[t].e*read_real(gradient,4*index+t);}')
  s=s.replace('if(gradient)*contract+=q[c].d*read_real(gradient,4*index+c);','if(gradient){*contract+=q[c].d*read_real(gradient,4*index+c);contract[1]+=q[c].e*read_real(gradient,4*index+c);}')
  a=s.index('                        double ra=-((la.a.d');b=s.index('\n                    }',a)
  block=s[a:b];s=s[:b]+'\n                        {\n'+block.replace('.d','.e')+'\n                        }'+s[b:]
  s=s.replace('if(!isfinite(*contract))ctx->failed=1;', 'if(!isfinite(*contract)||!isfinite(contract[1]))ctx->failed=1;')
  # Unused non-batch entry is excluded: this prototype is the flat SH52 first-VJP only.
  s=s[:s.index('GFFX_OPAQUE_LOCAL int opaque_quadrature_execute(')]
  f.write_text(s)
  f=p/'opaque_batch_scalar.h';s=f.read_text().replace('signbit(a.d)==signbit(b.d);','signbit(a.d)==signbit(b.d)&&a.e==b.e&&signbit(a.e)==signbit(b.e);')
  s=s.replace('double value;Ctx ctx={0,initial_work};','double value[2];Ctx ctx={0,initial_work};').replace(',&value,&ctx',',value,&ctx')
  s=s.replace('if(derivatives&1)for(j=0;j<nv;++j)for(k=0;k<3;++k)', 'if(derivatives&1)for(j=0;j<nv;++j)for(k=0;k<3;k+=2)')
  s=s.replace('write_real(out,seed,value);','write_real(out,seed,value[0]);if(k==0)write_real(out,seed+1,value[1]);').replace('write_real(other,seed,value);','write_real(other,seed,value[0]);')
  s=s.replace('if(ctx.failed||quad_batch_finish(&batch)||count!=batch.slots)return 2;','if(ctx.failed||quad_batch_finish(&batch)||count!=batch.slots){receipt[7]=ctx.work;return 2;}')
  f.write_text(s)
  source='\n'.join((p/name).read_text() for name in INPUTS)
  source=re.sub(r'^#include "opaque_[^"]+"\s*$', '', source, flags=re.M)
  functions=set(re.findall(r'GFFX_OPAQUE_(?:LOCAL|CLIP_LOCAL|SEPARATION_LOCAL)\s+\w+\s+(\w+)\s*\(',source))
  types=set(re.findall(r'typedef\s+struct\s*\{.*?\}\s*(\w+)\s*;',source,re.S))
  macros=set(re.findall(r'^#define\s+(\w+)',source,re.M)) - {'GFFX_OPAQUE_LOCAL','GFFX_OPAQUE_CLIP_LOCAL','GFFX_OPAQUE_SEPARATION_LOCAL','GFFX_OPAQUE_PRECLEARED'}
  names=functions|types|macros
  source=re.sub(r'\b('+'|'.join(sorted(names,key=len,reverse=True))+r')\b',lambda m:'pair_'+m[0],source)
  hashes='\n'.join(' * '+name+' '+hashlib.sha256((core/name).read_bytes()).hexdigest() for name in INPUTS)
  return '/* GENERATED by tools/generate_opaque_pair.py; do not edit.\n'+hashes+'\n */\n#ifndef GFFX_OPAQUE_PAIR_SCALAR_H\n#define GFFX_OPAQUE_PAIR_SCALAR_H\n'+source+'\n#endif\n'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 core=Path(__file__).resolve().parents[1]/'native/core';target=core/'opaque_pair_scalar.h';expected=generate(core)
 if args.check:
  if target.read_text()!=expected:raise SystemExit('stale paired specialization; run tools/generate_opaque_pair.py')
 else:target.write_text(expected)
if __name__=='__main__':main()
