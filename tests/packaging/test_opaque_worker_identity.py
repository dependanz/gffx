"""Installed-mode identity must be established before running its tests."""
import json,sys,types
import pytest

@pytest.mark.parametrize('mode',['original-public-motion-installed','opaque-cuda-original-public-motion-installed'])
@pytest.mark.parametrize('bad',[False,True])
def test_public_worker_checks_wheel_and_thread_limit_before_tests(tmp_path,monkeypatch,mode,bad):
    from tools import opaque_coverage_probe as probe,shading_cuda_closeout as closeout
    calls=[];output=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['probe','--fixtures',str(tmp_path),'--output',str(output),'--mode',mode,'--worker'])
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(set_num_threads=lambda n:calls.append(('threads',n)),set_num_interop_threads=lambda n:calls.append(('interop',n))))
    def identity(*args):
        calls.append(('identity',))
        if bad:raise ValueError('mismatched installed bytes')
        return dict(sha256='a'*64,matched_members=74)
    def execute(*a,**kw):calls.append(('tests',));return 0
    monkeypatch.setattr(closeout,'identity',identity);monkeypatch.setattr(pytest,'main',execute)
    if bad:
        with pytest.raises(ValueError,match='mismatched installed bytes'):probe.main()
        assert ('tests',) not in calls
    else:
        assert probe.main()==0;assert json.loads(output.read_bytes())['wheel']==dict(sha256='a'*64,matched_members=74)
    assert calls[:3]==[('threads',1),('interop',1),('identity',)]
