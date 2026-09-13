"""Pages, execution evidence and live frames have independent lifetimes."""
import json
import pytest
from tools.testing_dashboard.model import Index, test_id as identity


def finished(index, run, names=('a',), *, check=None, code=0):
    index.accept(dict(event='start', run=run, kind='tests'))
    index.accept(dict(event='collect', run=run, nodeids=list(names)))
    if check:
        index.accept(dict(event='check_start', run=run, nodeid=names[0], name=check))
        index.accept(dict(event='check_finish', run=run, nodeid=names[0], name=check, outcome='passed'))
    for name in names:
        for when in ('call', 'teardown'):
            index.accept(dict(event='report', run=run, nodeid=name, when=when,
                              outcome='failed' if code else 'passed'))
    index.accept(dict(event='finish', run=run, exit_code=code))


def test_more_than_one_hundred_executions_do_not_consume_page_capacity(tmp_path):
    from tools.testing_dashboard.catalog import PAGES
    assert len({p['id'] for p in PAGES})==len(PAGES)
    index = Index(tmp_path/'index.sqlite')
    for i in range(105):
        finished(index, 'run'+str(i))
    assert index.history_status()['executions'] == 105
    assert len(index.runs(offset=100)) == 5
    assert index.page('test/'+identity('a'))['evidence']['run'] == 'run104'
    assert len([p for p in index.directory()['items'] if p['id']=='test/'+identity('a')]) == 1


def test_retention_protects_latest_failures_checks_partial_subsets_and_pins(tmp_path):
    index = Index(tmp_path/'index.sqlite', history_keep=0)
    finished(index, 'old', ('a','b'), check='visible')
    finished(index, 'pinned')
    index.pin('pinned', True)
    finished(index, 'latest', code=1)
    finished(index, 'discard')
    finished(index, 'last', code=1)
    index.accept(dict(event='start', run='active', kind='tests'))
    index.accept(dict(event='collect', run='active', nodeids=['a']))
    status = index.history_status()
    assert status['eligible'] == ['latest','discard']
    assert {'old','pinned','last','active'} <= set(status['protected'])
    assert Index(tmp_path/'index.sqlite', history_keep=0).summary('pinned')['pinned']


def test_retention_keeps_newest_unprotected_history_not_a_total_run_cap(tmp_path):
    index = Index(tmp_path/'index.sqlite', history_keep=2)
    for i in range(6): finished(index, 'r'+str(i))
    assert index.history_status()['eligible'] == ['r0','r1','r2']
    assert index.history_status()['history_keep'] == 2


def test_retire_is_transactional_and_preserves_page_identity_and_retired_ids(tmp_path):
    index = Index(tmp_path/'index.sqlite', history_keep=0)
    finished(index, 'old'); finished(index, 'new')
    before=index.page('test/'+identity('a'))['created_on']
    with pytest.raises(ValueError): index.retire(['old','new'])
    assert index.summary('old')
    index.retire(['old'])
    with pytest.raises(KeyError): index.summary('old')
    with pytest.raises(ValueError, match='retired'): index.accept(dict(event='start',run='old'))
    assert index.page('test/'+identity('a'))['created_on'] == before
    assert index.history_status()['retired'] == 1


def test_active_execution_capacity_and_metadata_byte_budget_are_separate(tmp_path):
    index=Index(tmp_path/'index.sqlite', max_active=1, metadata_bytes=1024*1024)
    index.accept(dict(event='start',run='active'))
    with pytest.raises(ValueError, match='active execution'): index.accept(dict(event='start',run='other'))
    index.accept(dict(event='finish',run='active',exit_code=2))
    index.accept(dict(event='start',run='other'))
    assert index.history_status()['metadata_limit_bytes'] == 1024*1024


@pytest.mark.parametrize('kwargs',[{'history_keep':-1},{'history_keep':True},{'max_active':0},
                                 {'metadata_bytes':1},{'metadata_bytes':True}])
def test_invalid_retention_policies_fail_before_database_creation(tmp_path,kwargs):
    with pytest.raises(ValueError): Index(tmp_path/'index.sqlite',**kwargs)
    assert not (tmp_path/'index.sqlite').exists()


def test_latest_coverage_and_demo_evidence_survive_retention(tmp_path):
    from tools.testing_dashboard.model import coverage_report
    index=Index(tmp_path/'index.sqlite',history_keep=0)
    finished(index,'covered')
    coverage=coverage_report({'files':{}})
    coverage['artifact_sha256']='a'*64
    index.accept(dict(event='coverage',run='covered',coverage=coverage))
    finished(index,'no-coverage')
    for run in ('demo-old','demo-new'):
        index.accept(dict(event='start',run=run,kind='demo'))
        index.accept(dict(event='demo',run=run,page='dashboard/panel-demo',artifacts=[]))
        index.accept(dict(event='finish',run=run,exit_code=0))
    assert index.history_status()['eligible'] == ['demo-old']
    # Retention must not make the newest unmeasured coverage page borrow old numbers.
    assert index.latest_coverage()['measured'] is False


def test_history_preview_token_rejects_changed_protection_and_prunes_exact_artifacts(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    server=TestingServer(tmp_path,port=0,allow_loopback_for_tests=True,history_keep=0)
    finished(server.index,'old'); finished(server.index,'new')
    directory=tmp_path/'viewer'/'old';directory.mkdir(parents=True)
    (directory/'log.bin').write_bytes(b'saved evidence')
    preview=server.history_preview()
    server.index.pin('old',True)
    with pytest.raises(ValueError,match='preview'): server.prune_history(preview['token'])
    assert (directory/'log.bin').exists()
    server.index.pin('old',False)
    preview=server.history_preview()
    result=server.prune_history(preview['token'])
    assert result['removed']==['old'] and not directory.exists()
    assert server.index.summary('new')
    with pytest.raises(Exception,match='retired'): server._open_run('old')
    with pytest.raises(Exception,match='retired'): server.replay('old')


def test_retention_never_deletes_through_symlink_or_unknown_files(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    server=TestingServer(tmp_path/'root',port=0,allow_loopback_for_tests=True,history_keep=0)
    finished(server.index,'old');finished(server.index,'new')
    directory=server.root/'old';directory.mkdir(parents=True)
    (directory/'unexpected.txt').write_text('must survive')
    preview=server.history_preview()
    assert 'old' in preview['blocked'] and not preview['candidates']
    server.prune_history(preview['token'])
    assert (directory/'unexpected.txt').read_text()=='must survive'


def test_metadata_exhaustion_is_explicit_and_rolls_back_evidence(tmp_path):
    index=Index(tmp_path/'index.sqlite',metadata_bytes=1024*1024)
    index.accept(dict(event='start',run='large'))
    names=['test_'+str(i)+'_'+'x'*60 for i in range(6000)]
    with pytest.raises(ValueError,match='metadata storage budget'):
        index.accept(dict(event='collect',run='large',nodeids=names))
    assert index.tests('large')['total']==0


def test_live_viewer_leases_block_cleanup(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    server=TestingServer(tmp_path,port=0,allow_loopback_for_tests=True,history_keep=0)
    finished(server.index,'old');finished(server.index,'new')
    loaded=server._open_run('old');lease=object();loaded.pages.add(lease)
    assert 'old' in server.history_preview()['blocked']
    loaded.pages.remove(lease)
    assert server.history_preview()['candidates']==['old']
    server.prune_history(server.history_preview()['token'])


def test_linked_artifact_directory_is_never_followed(tmp_path):
    from tools.testing_dashboard.server import TestingServer
    server=TestingServer(tmp_path/'root',port=0,allow_loopback_for_tests=True,history_keep=0)
    finished(server.index,'old');finished(server.index,'new')
    outside=tmp_path/'outside';outside.mkdir();(outside/'log.bin').write_bytes(b'keep')
    server.root.mkdir(parents=True,exist_ok=True)
    try:(server.root/'old').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('local permission does not allow creating a symlink')
    assert 'old' in server.history_preview()['blocked']
    server.prune_history(server.history_preview()['token'])
    assert (outside/'log.bin').read_bytes()==b'keep'


def test_history_http_and_pin_controls_are_explicit(tmp_path):
    import urllib.request
    import urllib.error
    from tools.testing_dashboard.server import TestingServer
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        finished(server.index,'one')
        url='http://'+server.address+'/testing/api/'
        with urllib.request.urlopen(url+'history') as response:
            assert json.load(response)['executions']==1
        for value in (True,False):
            request=urllib.request.Request(url+'pin',data=json.dumps(dict(run='one',pinned=value)).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:assert json.load(response)['accepted']
            assert server.index.summary('one')['pinned'] is value
        request=urllib.request.Request(url+'pin',data=b'[]',headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(request)
