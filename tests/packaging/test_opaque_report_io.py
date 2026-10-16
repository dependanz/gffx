"""Bounded atomic report publication; no numerical retry or budget reset."""
import json,os,threading,time
from pathlib import Path
import pytest
from tools import material_loss_cost as cost


def error(code):
    e=PermissionError('simulated Windows file contention');e.winerror=code;return e


@pytest.mark.parametrize('code',[5,32,33])
def test_transient_windows_lock_retries_same_report_without_recomputing(tmp_path,monkeypatch,code):
    path=tmp_path/'report.json';path.write_text('{"status":"old"}')
    replace=Path.replace;attempts=[];sleeps=[]
    def locked(src,dst):
        attempts.append(src.read_bytes())
        if len(attempts)<3:raise error(code)
        return replace(src,dst)
    monkeypatch.setattr(Path,'replace',locked);monkeypatch.setattr(cost.time,'sleep',sleeps.append)
    cost.write_report(path,{'status':'working','cells':123})
    assert json.loads(path.read_bytes())=={'status':'working','cells':123}
    assert len(attempts)==3 and len(set(attempts))==1
    assert sleeps==[.01,.02]


def test_persistent_lock_is_bounded_and_preserves_previous_report(tmp_path,monkeypatch):
    path=tmp_path/'report.json';path.write_bytes(b'{"status":"old"}');calls=[];sleeps=[]
    def locked(src,dst):calls.append(1);raise error(5)
    monkeypatch.setattr(Path,'replace',locked);monkeypatch.setattr(cost.time,'sleep',sleeps.append)
    with pytest.raises(PermissionError):cost.write_report(path,{'status':'new'})
    assert len(calls)==7 and sum(sleeps)==pytest.approx(.63)
    assert path.read_bytes()==b'{"status":"old"}'
    assert json.loads(path.with_suffix('.json.tmp').read_bytes())=={'status':'new'}


def test_unrelated_io_failure_is_not_retried(tmp_path,monkeypatch):
    sleeps=[];monkeypatch.setattr(cost.time,'sleep',sleeps.append)
    def denied(src,dst):raise PermissionError('ordinary permissions error')
    monkeypatch.setattr(Path,'replace',denied)
    with pytest.raises(PermissionError):cost.write_report(tmp_path/'r.json',{'x':1})
    assert sleeps==[]


@pytest.mark.skipif(os.name!='nt',reason='actual Windows sharing semantics')
def test_actual_windows_reader_lock_release_allows_atomic_publication(tmp_path):
    import ctypes
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE];k.CreateFileW.restype=w.HANDLE
    k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
    p=tmp_path/'r.json';p.write_text('{"old":true}')
    h=k.CreateFileW(str(p),0x80000000,1,None,3,128,None)
    assert h!=ctypes.c_void_p(-1).value
    t=threading.Thread(target=lambda:(time.sleep(.05),k.CloseHandle(h)));t.start()
    try:cost.write_report(p,{'new':True})
    finally:t.join()
    assert json.loads(p.read_bytes())=={'new':True}
