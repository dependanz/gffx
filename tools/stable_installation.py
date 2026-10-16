"""Check isolated base install, optional-framework refusal, upgrades and uninstall."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',required=True,type=Path)
    parser.add_argument('--previous-wheel',action='append',default=[],type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','GFFX_CUDA_PLUGIN_PATH','GFFX_TEST_SOURCE_TREE']:env.pop(key,None)
    env.update(PYTHONUTF8='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    receipts=[]
    def run(label,command):
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (root/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        receipts.append(dict(label=label,command=command,returncode=result.returncode))
        (root/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
        if result.returncode:raise RuntimeError('Installation check failed: '+label)
        return result.stdout
    cases=[None,*args.previous_wheel]
    for index,previous in enumerate(cases):
        name='base' if previous is None else 'upgrade-'+str(index)
        folder=root/name
        run(name+'-venv',[sys.executable,'-m','venv',str(folder)])
        python=str(folder/'Scripts/python.exe')
        if previous:
            run(name+'-old',[python,'-m','pip','install','--no-index','--no-deps',str(previous.resolve())])
            run(name+'-old-identity',[python,'-I','-c','import importlib.metadata as m; assert m.version("gffx")!="0.2.0";print(m.version("gffx"))'])
        run(name+'-install',[python,'-m','pip','install','--no-index','--no-deps','--upgrade',str(args.wheel.resolve())])
        probe='''import ctypes,importlib.util,json,sys
from pathlib import Path
assert importlib.util.find_spec('torch') is None
assert not ctypes.windll.kernel32.GetModuleHandleW('nvcuda.dll')
assert not ctypes.windll.kernel32.GetModuleHandleW('gffx_cuda12.dll')
import gffx
assert gffx.__version__=='0.2.0'
assert Path(gffx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
cap=gffx.capabilities()
assert cap['native_core']['available'] and cap['gpu_probed'] is False
assert not ctypes.windll.kernel32.GetModuleHandleW('nvcuda.dll')
assert not ctypes.windll.kernel32.GetModuleHandleW('gffx_cuda12.dll')
try:
    import gffx.torch
except ImportError:
    pass
else:
    raise AssertionError('Optional framework absence must be explicit')
print(json.dumps(dict(version=gffx.__version__,native_core=True,framework_present=False,driver_loaded=False,provider_loaded=False)))
'''
        run(name+'-probe',[python,'-I','-c',probe])
        run(name+'-check',[python,'-m','pip','check'])
        run(name+'-uninstall',[python,'-m','pip','uninstall','--yes','gffx'])
        run(name+'-cleanup',[python,'-I','-c','import importlib.util,sys;from pathlib import Path;assert importlib.util.find_spec("gffx") is None;site=Path(sys.prefix)/"Lib/site-packages";assert not (site/"gffx").exists();assert not (site/"gffx.libs").exists();assert not list(site.glob("gffx-*.dist-info"));print("complete package cleanup")'])
    (root/'result.json').write_text(json.dumps(dict(cases=len(cases),previous_wheels=[x.name for x in args.previous_wheel],passed=True,commands=len(receipts)),indent=2)+'\n',encoding='utf-8')
    print('Base/upgrade/uninstall checks passed:',len(cases))


if __name__=='__main__':main()
