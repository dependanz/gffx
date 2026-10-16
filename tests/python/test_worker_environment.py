"""A supervised Windows worker must keep the installed environment for child checks."""
import json
import subprocess
import sys
from pathlib import Path


def test_supervised_worker_child_imports_same_installed_package(tmp_path,monkeypatch):
    import gffx
    from tools import material_loss_cost as cost
    package=Path(gffx.__file__).resolve()
    environment=package.parents[3]
    executable=environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    assert executable.is_file()
    monkeypatch.setattr(sys,'executable',str(executable))
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path/'worker_environment_fixture.py').write_text(
        'import subprocess,sys,json\n'
        'def main():\n'
        ' result=subprocess.run([sys.executable,"-c","import gffx;print(gffx.__file__)"],capture_output=True,text=True,timeout=15)\n'
        ' print(json.dumps(dict(code=result.returncode,out=result.stdout,error=result.stderr)))\n'
        ' return 0\n')
    command=cost.worker_command(tmp_path,tmp_path/'unused.json',worker_module='worker_environment_fixture')
    assert command[0]==getattr(sys,'_base_executable',sys.executable)
    child=subprocess.run(command,capture_output=True,text=True,timeout=25)
    assert child.returncode==0,child.stderr
    result=json.loads(child.stdout)
    assert result['code']==0,result['error']
    assert Path(result['out'].strip()).resolve()==package
