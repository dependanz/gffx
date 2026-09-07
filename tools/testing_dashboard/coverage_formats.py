"""Bounded standard-library coverage import; reports never execute repository commands."""
import hashlib
from pathlib import Path

LAYERS={'python':'Python','native':'Native C/C++','javascript':'JavaScript/TypeScript',
        'cuda':'CUDA device code','shader':'GPU shaders','other':'Other instrumented code'}
EXTENSIONS={'.py':'python','.c':'native','.h':'native','.cpp':'native','.hpp':'native',
            '.cc':'native','.cxx':'native','.cu':'cuda','.cuh':'cuda',
            '.js':'javascript','.mjs':'javascript','.ts':'javascript','.tsx':'javascript',
            '.wgsl':'shader','.glsl':'shader','.vert':'shader','.frag':'shader'}


def inventory(root, paths):
    """Explicit in-repo source inventory, not an executable-line denominator."""
    root=Path(root).resolve();result={}
    if len(paths)>10000: raise ValueError('inventory file limit')
    for value in sorted(set(paths)):
        path=(root/value).resolve()
        if not path.is_relative_to(root): raise ValueError('inventory path escapes repository')
        layer=EXTENSIONS.get(path.suffix.lower())
        if layer and path.is_file():result.setdefault(layer,[]).append(path.relative_to(root).as_posix())
    return result


def repository_inventory(root):
    import subprocess
    root=Path(root).resolve()
    result=subprocess.run(['git','-c','safe.directory='+str(root),'-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True)
    return inventory(root,result.stdout.decode('utf-8').strip('\0').split('\0'))


def lcov_report(raw, *, layer):
    if layer not in LAYERS: raise ValueError('unknown coverage layer')
    if len(raw.encode('utf-8'))>8*1024*1024: raise ValueError('coverage input exceeds 8 MiB')
    files=[];current=None;seen=set()
    def integer(value):
        n=int(value)
        if n<0 or n>2**53-1: raise ValueError('invalid coverage count')
        return n
    for line in raw.splitlines():
        key,_,value=line.partition(':')
        if key=='SF':
            value=value.replace('\\','/')
            if current is not None or not value or len(value)>4096 or value in seen: raise ValueError('invalid/duplicate LCOV source')
            current={'path':value.replace('\\','/'),'lines':{},'branch_hits':{},'function_hits':[]};seen.add(value)
        elif key=='end_of_record':
            if current is None: raise ValueError('missing LCOV source')
            hits=current['lines'];branches=current['branch_hits'];functions=current['function_hits']
            files.append({'path':current['path'],'covered':sum(n>0 for n in hits.values()),'total':len(hits),
                          'missing_lines':sorted(n for n,h in hits.items() if not h),
                          'branches':len(branches) if branches else None,
                          'covered_branches':sum(n>0 for n in branches.values()) if branches else None,
                          'functions':len(functions) if functions else None,
                          'covered_functions':sum(n>0 for n in functions) if functions else None})
            current=None
        elif key in ('DA','BRDA','FNDA'):
            if current is None: raise ValueError('missing LCOV source')
            values=value.split(',')
            if key=='DA':
                number,hits=map(integer,values[:2])
                if number<1 or number in current['lines']: raise ValueError('invalid/duplicate LCOV line')
                current['lines'][number]=hits
            elif key=='BRDA':
                if len(values)!=4 or integer(values[0])<1: raise ValueError('invalid LCOV branch')
                branch=tuple(values[:3])
                if branch in current['branch_hits']: raise ValueError('duplicate LCOV branch')
                current['branch_hits'][branch]=0 if values[3]=='-' else integer(values[3])
            else:
                hits,name=value.split(',',1)
                # Node emits separate same-named methods (e.g. constructors); each FNDA
                # entry corresponds to a function entry, not a globally unique name.
                current['function_hits'].append(integer(hits))
        if len(files)>10000: raise ValueError('coverage file limit')
    if current is not None or not files: raise ValueError('incomplete/empty LCOV report')
    return {'source':'LCOV','layer':layer,'files':files,'covered':sum(f['covered'] for f in files),
            'total':sum(f['total'] for f in files),'artifact_sha256':hashlib.sha256(raw.encode()).hexdigest()}
