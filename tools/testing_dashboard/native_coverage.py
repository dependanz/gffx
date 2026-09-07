"""GFFX Linux/GCC CPU fixture coverage profile; not a release or CUDA build."""
import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess


def collect(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    commands=[]
    def run(command):
        result=subprocess.run(list(map(str,command)),cwd=output,text=True,capture_output=True)
        commands.append(dict(command=list(map(str,command)),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        (output/'commands.json').write_text(json.dumps(commands,indent=2))
        if result.returncode:raise RuntimeError(result.stderr or str(command))
    block=re.search(r'add_library\(gffx_core SHARED(.*?)\)',(root/'CMakeLists.txt').read_text(),re.S).group(1)
    run(['gcc','--version']);run(['gcov','--version'])
    objects=[]
    for i,source in enumerate(block.split()):
        obj=output/f'core-{i}.o';objects.append(obj)
        run(['gcc','--coverage','-O0','-g','-fPIC','-std=c11','-DGFFX_BUILDING_LIBRARY=1',
             '-DGFFX_PACKAGE_VERSION="0.2.0.dev0"','-I'+str(root/'include'),'-I'+str(root/'native/core'),
             '-I'+str(root/'native/cuda'),'-c',root/source,'-o',obj])
    run(['gcc','--coverage','-shared',*objects,'-ldl','-lm','-o',output/'libgffx_core.so'])
    fixtures=sorted((root/'tests/geometry').glob('test_*.c'))+[root/'tests/io/test_ply.c',root/'tests/sweep/test_crosscutting.c']
    for source in fixtures:
        binary=output/source.stem
        run(['gcc','-std=c11','-I'+str(root/'include'),source,'-L'+str(output),'-lgffx_core',
             '-Wl,-rpath,'+str(output),'-lm','-o',binary]);run([binary])
    for obj in objects:run(['gcov','--json-format','--branch-probabilities',obj.with_suffix('.gcda')])
    files={}
    for report in output.glob('*.gcov.json.gz'):
        with gzip.open(report,'rt') as stream:data=json.load(stream)
        for item in data['files']:
            path=Path(item['file']).resolve()
            if not path.is_relative_to(root):continue
            record=files.setdefault(path.relative_to(root).as_posix(),{'lines':{},'branches':{},'functions':{}})
            for line in item['lines']:
                n=line['line_number'];record['lines'][n]=record['lines'].get(n,0)+line['count']
                for i,b in enumerate(line.get('branches',[])):
                    key=(n,i);record['branches'][key]=record['branches'].get(key,0)+b['count']
            for f in item.get('functions',[]):record['functions'][f['name']]=record['functions'].get(f['name'],0)+f['execution_count']
    lines=[]
    for path,item in sorted(files.items()):
        lines.append('SF:'+path)
        lines.extend(f'DA:{n},{c}' for n,c in sorted(item['lines'].items()))
        lines.extend(f'BRDA:{n},0,{i},{c}' for (n,i),c in sorted(item['branches'].items()))
        lines.extend(f'FNDA:{c},{name}' for name,c in sorted(item['functions'].items()))
        lines.append('end_of_record')
    (output/'native.lcov').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'fixtures_passed':len(fixtures),'instrumented_files':len(files),'report':str(output/'native.lcov')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();collect(args.root,args.output)
