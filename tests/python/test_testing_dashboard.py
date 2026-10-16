"""Developer application tests; imports tooling without redirecting the selected gffx."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def module(name):
    return importlib.import_module('tools.testing_dashboard.' + name)

def start(index, run='one', kind='tests'):
    index.accept({'event': 'start', 'run': run, 'kind': kind, 'provenance': {}})


def test_classic_demo_assets_are_offline_bounded_and_attributed():
    import math
    D=module('classic_demo')
    for name in ('bunny','teapot'):
        mesh=D.load_asset(name)
        assert 100 < len(mesh['vertices']) < 5000
        assert 100 < len(mesh['faces']) < 10000
        assert mesh['source_url'].startswith('https://') and mesh['license_note']
        assert all(math.isfinite(x) for p in mesh['vertices'] for x in p)
        assert all(len(set(f))==3 and all(0<=i<len(mesh['vertices']) for i in f) for f in mesh['faces'])
    with pytest.raises(ValueError):D.load_asset('../escape')


def test_elastic_ball_freefall_rebound_energy_and_convergence():
    D=module('classic_demo');states=D.simulate();fine=D.simulate(substeps=80)
    assert len(states)==181 and states[0]['t']==0 and states[-1]['t']==3
    # Analytic free fall before contact; first-order integrator tolerance is explicit.
    assert abs(states[24]['height']-(2-9.81*.4**2/2)) < .002
    assert any(s['velocity']>2 and s['height']>.4 for s in states[40:])
    assert min(s['height'] for s in states) > .12
    assert max(s['energy'] for s in states) <= states[0]['energy']*1.01
    assert states[-1]['energy'] < states[0]['energy']*.8
    assert max(abs(a['height']-b['height']) for a,b in zip(states,fine)) < .015
    with pytest.raises(ValueError):D.simulate(substeps=0)


def test_elastic_ball_scene_squashes_without_floor_penetration():
    D=module('classic_demo');states=D.simulate();frames=[D.scene(s) for s in states]
    vertices,faces,colors=frames[0]
    assert len(vertices)<400 and len(faces)<800 and len(colors)==len(vertices)
    assert all(f[1]==faces for f in frames)
    assert all(min(p[1] for p in v)>=-1e-7 for v,_,_ in frames)
    assert vertices!=frames[45][0]


def test_classic_demo_publishes_finite_assets_physics_and_restart_history(tmp_path):
    D=module('classic_demo');Server=module('server').TestingServer
    with Server(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        D.publish('http://'+server.address,'classics')
        page=server.index.page('geometry/classic-meshes-demo')
        assert len(page['artifacts'])==4 and page['state']=='demonstration'
        page=server.index.page('simulation/elastic-ball-demo')
        assert len(page['artifacts'])==4 and page['collected']==0
        first=server.state.get('ball/motion',step=0,run='classics-ball')
        last=server.state.get('ball/motion',step=180,run='classics-ball')
        assert first['vertices'].data!=last['vertices'].data
        with pytest.raises(KeyError):server.state.get('ball/motion',step=181,run='classics-ball')
        assert server.index.summary('classics')['state']=='finished'
        assert server.index.summary('classics-ball')['state']=='finished'
    with Server(tmp_path,port=0,allow_loopback_for_tests=True) as replay:
        replay.replay('classics-ball')
        assert replay.state.get('ball/motion',step=0,run='classics-ball')['vertices'].data==first['vertices'].data
        assert replay.state.get('ball/motion',step=180,run='classics-ball')['vertices'].data==last['vertices'].data


def test_coverage_uses_statement_summary_not_raw_execution_lines():
    raw={'files':{'a.py':{'executed_lines':[1,2,3], 'missing_lines':[4],
         'summary':{'covered_lines':2,'num_statements':3}}}}
    report=module('model').coverage_report(raw)
    assert (report['covered'],report['total'])==(2,3)
    assert report['layer']=='python'


def test_mesh_animation_demo_is_finite_and_replays_frames(tmp_path):
    D=module('mesh_animation_demo');Server=module('server').TestingServer
    root=tmp_path/'server'
    with Server(root,port=0,allow_loopback_for_tests=True) as server:
        D.publish('http://'+server.address,'mesh-animation')
        page=server.index.page('geometry/mesh-animation-demo')
        assert page['state']=='demonstration' and page['collected']==0
        assert len(page['artifacts'])==9
        assert server.index.summary('mesh-animation')['state']=='finished'
        first=server.state.get('gallery/animation-3d',step=0,run='mesh-animation')
        last=server.state.get('gallery/animation-3d',step=23,run='mesh-animation')
        assert first['vertices'].shape==(192,3) and first['faces'].shape==(384,3)
        assert first['vertices'].data!=last['vertices'].data
        assert first['faces'].data==last['faces'].data
        image=server.state.get('gallery/animation-2d',step=0,run='mesh-animation')
        assert image['data'].shape==(180,240,3)
        with pytest.raises(KeyError):server.state.get('gallery/animation-3d',step=24,run='mesh-animation')
    with Server(root,port=0,allow_loopback_for_tests=True) as replay:
        replay.replay('mesh-animation')  # Viewer runs are deliberately opened lazily.
        assert replay.state.get('gallery/animation-3d',step=0,run='mesh-animation')['vertices'].data==first['vertices'].data
        assert replay.state.get('gallery/animation-2d',step=23,run='mesh-animation')['data'].shape==(180,240,3)


def test_mesh_animation_geometry_has_valid_topology_and_changes_over_time():
    import math
    D=module('mesh_animation_demo')
    for planar in (True,False):
        initial,faces,colors=D.geometry(planar,0)
        moved,other_faces,_=D.geometry(planar,7)
        assert initial!=moved and faces==other_faces
        assert len(colors)==len(initial)
        assert all(math.isfinite(x) for row in initial for x in row)
        assert all(len(set(face))==3 and all(0<=i<len(initial) for i in face) for face in faces)
        if planar:assert all(v[2]==0 for v in initial+moved)


def test_camera_fitting_demo_has_bounded_history_and_correct_derivative_curve():
    demo=module('camera_fit_demo');result=demo.solve();curve=demo.derivatives()
    assert len(result['history'])==61 and result['history'][-1][0]==300
    assert result['history'][-1][1]<1e-8
    assert max(abs(a-b) for a,b in zip(result['target'],result['recovered']))<.001
    assert len(curve)==21 and max(row[3] for row in curve)<1e-8


def test_lcov_layers_preserve_independent_reports_and_unknown_gaps(tmp_path):
    M=module('model');C=module('coverage_formats');index=M.Index(tmp_path/'index.sqlite');start(index)
    raw='SF:src/a.js\nDA:1,2\nDA:2,0\nBRDA:1,0,0,1\nBRDA:1,0,1,-\nFNDA:1,main\nend_of_record\n'
    js=C.lcov_report(raw,layer='javascript')
    assert (js['covered'],js['total'])==(1,2)
    assert js['files'][0]['missing_lines']==[2]
    assert js['files'][0]['covered_branches']==1 and js['files'][0]['branches']==2
    repeated=C.lcov_report(raw.replace('FNDA:1,main','FNDA:1,constructor\nFNDA:0,constructor'),layer='javascript')
    assert repeated['files'][0]['functions']==2 and repeated['files'][0]['covered_functions']==1
    index.accept({'event':'coverage','run':'one','coverage':js})
    py=M.coverage_report({'files':{'a.py':{'executed_lines':[1],'missing_lines':[]}}})
    index.accept({'event':'coverage','run':'one','coverage':py})
    assert index.coverage('one',layer='javascript')['covered']==1
    assert index.coverage('one',layer='python')['total']==1
    assert not index.coverage('one',layer='native')['measured']
    start(index,'two');assert not index.latest_coverage(layer='javascript')['measured']
    assert index.latest_coverage(run='one',layer='javascript')['measured']
    with pytest.raises(ValueError):C.lcov_report('SF:a\nDA:-1,0\nend_of_record\n',layer='native')
    with pytest.raises(ValueError):C.lcov_report(raw,layer='made-up')
    duplicate='SF:src\\a.js\nDA:1,1\nend_of_record\nSF:src/a.js\nDA:1,1\nend_of_record\n'
    with pytest.raises(ValueError):C.lcov_report(duplicate,layer='javascript')
    assert M.Index(tmp_path/'index.sqlite').coverage('one',layer='javascript')['covered']==1
    js['files'][0]['covered_functions']=2
    with pytest.raises(ValueError):index.accept({'event':'coverage','run':'one','coverage':js})


def test_coverage_inventory_never_calls_uninstrumented_files_zero_coverage(tmp_path):
    C=module('coverage_formats');(tmp_path/'a.py').write_text('pass\n');(tmp_path/'b.cu').write_text('// kernel\n')
    (tmp_path/'c.js').write_text('// unused\n')
    inventory=C.inventory(tmp_path,['a.py','b.cu','c.js'])
    assert inventory=={'python':['a.py'],'cuda':['b.cu'],'javascript':['c.js']}
    with pytest.raises(ValueError):C.inventory(tmp_path,['../outside.py'])


def test_demo_pages_do_not_borrow_another_pages_artifacts(tmp_path):
    index=module('model').Index(tmp_path/'index.sqlite')
    start(index,'panels','demo')
    index.accept({'event':'demo','run':'panels','artifacts':[{'kind':'text','path':'panels/info'}]})
    start(index,'camera','demo')
    index.accept({'event':'demo','run':'camera','page':'camera/pinhole-demo',
                  'artifacts':[{'kind':'text','path':'camera/info'}]})
    assert index.page('dashboard/panel-demo')['evidence']['run']=='panels'
    assert index.page('camera/pinhole-demo')['evidence']['run']=='camera'
    with pytest.raises(KeyError):index.page('camera/pinhole-demo',run='panels')
    with pytest.raises(ValueError):index.accept({'event':'demo','run':'camera','page':'camera/projection'})
    with pytest.raises(ValueError):index.accept({'event':'demo','run':'camera','page':'dashboard/panel-demo'})


def test_camera_demonstration_is_finite_and_reports_its_own_page(tmp_path):
    Server=module('server').TestingServer
    with Server(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        module('camera_demo').publish('http://'+server.address,'camera-example')
        page=server.index.page('camera/pinhole-demo')
        assert page['evidence']['run']=='camera-example' and page['evidence']['state']=='finished'
        assert page['collected']==0 and len(page['artifacts'])==10
        assert {a['kind'] for a in page['artifacts']}=={'mesh','image','text'}
        assert server.state.get('camera-demo/rig-camera-plan',run='camera-example')['data'].shape==(180,240,3)
        assert server.state.get('camera-demo/landmarks-projected',run='camera-example')['data'].shape==(180,240,3)
        assert server.index.page('dashboard/panel-demo')['state']=='not measured'

def test_every_repository_test_has_an_authored_explanation():
    import ast
    guide=module('explanations')
    missing=[]
    for folder in ('python','pytorch','packaging'):
        for path in sorted((ROOT/'tests'/folder).glob('test_*.py')):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            def visit(body, parents=()):
                for item in body:
                    if isinstance(item,ast.ClassDef): yield from visit(item.body,(*parents,item.name))
                    elif isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name.startswith('test_'):
                        yield '::'.join((path.relative_to(ROOT).as_posix(),*parents,item.name))
            for nodeid in visit(tree.body):
                detail=guide.explain(nodeid)
                if not detail['authored']: missing.append(nodeid)
                else:
                    assert all(len(detail[key])>=25 for key in ('purpose','method','limits')),nodeid
                    assert guide.explain(nodeid+'[variant]')==detail
    assert not missing, 'Add plain-language explanations for: '+str(missing)

def test_explanations_are_read_only_guides_not_historical_test_results(tmp_path):
    M=module('model');index=M.Index(tmp_path/'index.sqlite');start(index)
    name='tests/packaging/test_ci_contract.py::test_action_pins_are_exact_and_immutable'
    index.accept({'event':'collect','run':'one','nodeids':[name,'external.py::test_unknown']})
    page=index.page('test/'+M.test_id(name))
    assert page['explanation']['authored'] is True
    assert 'five' in page['explanation']['method']
    assert 'secure' in page['explanation']['limits']
    assert page['state']=='pending' and page['explanation']['basis']=='current author guide'
    assert index.page('test/'+M.test_id('external.py::test_unknown'))['explanation']['authored'] is False
    assert module('explanations').explain('../../secret.py::test_x')['authored'] is False

def test_named_check_explanations_describe_the_comparison_and_its_limits(tmp_path):
    M=module('model');index=M.Index(tmp_path/'index.sqlite');start(index)
    name='tests/pytorch/test_face_geometry.py::test_ta01_forward_parity'
    index.accept({'event':'collect','run':'one','nodeids':[name]})
    for mesh in ('tetrahedron','degenerate_face'):
        for field in ('normals','areas','validity'):
            check=mesh+' / '+field
            index.accept({'event':'check_start','run':'one','nodeid':name,'name':check})
            page=index.page('check/'+M.test_id(name)+'/'+M.test_id(check))
            assert page['explanation']['authored']
            assert 'bit-for-bit' in page['explanation']['method']
            assert 'C reference' in page['explanation']['limits']
            assert page['state']=='pending'

def test_individual_and_named_check_pages_isolate_artifacts_and_never_fallback(tmp_path):
    M=module('model'); index=M.Index(tmp_path/'index.sqlite')
    name='tests/pytorch/test_face_geometry.py::test_normal[a+b]'; other=name+'other'
    tid=M.test_id(name); cid=M.test_id('normal direction')
    start(index); index.accept({'event':'collect','run':'one','nodeids':[name,other]})
    index.accept({'event':'check_start','run':'one','nodeid':name,'name':'normal direction'})
    index.accept({'event':'artifact','run':'one','nodeid':name,'artifact':{'kind':'text','path':'tests/a/check','check':cid,'label':'comparison'}})
    index.accept({'event':'artifact','run':'one','nodeid':other,'artifact':{'kind':'text','path':'tests/b/other'}})
    index.accept({'event':'check_finish','run':'one','nodeid':name,'name':'normal direction','outcome':'passed','duration':.1})
    page=index.page('test/'+tid)
    assert page['parent']=='mesh/face-geometry' and page['items'][0]['nodeid']==name
    assert [a['path'] for a in page['artifacts']]==['tests/a/check']
    check=index.page('check/'+tid+'/'+cid)
    assert check['state']=='passed' and check['parent']=='test/'+tid
    assert check['artifacts'][0]['label']=='comparison'
    assert index.summary('one')['collected']==2  # subchecks never inflate test counts
    directory=index.directory()
    assert any(p['id']=='check/'+tid+'/'+cid for p in directory['items'])
    assert M.Index(tmp_path/'index.sqlite').page('check/'+tid+'/'+cid)['state']=='passed'
    start(index,'two');index.accept({'event':'collect','run':'two','nodeids':[name]})
    newest=index.page('check/'+tid+'/'+cid)
    assert newest['state']=='not measured' and newest['artifacts']==[] and newest['evidence']['run']=='two'
    assert index.page('check/'+tid+'/'+cid,run='one')['state']=='passed'
    with pytest.raises(KeyError):index.page('test/'+M.test_id('missing'))
    with pytest.raises(KeyError):index.page('check/'+M.test_id(other)+'/'+cid)

def test_named_check_lifecycle_and_quotas_fail_closed(tmp_path):
    M=module('model');index=M.Index(tmp_path/'index.sqlite',max_checks=2)
    start(index);index.accept({'event':'collect','run':'one','nodeids':['a']})
    def send(event,name='x',**extra):
        index.accept(dict(event=event,run='one',nodeid='a',name=name,**extra))
    with pytest.raises(KeyError):send('check_finish',outcome='passed')
    send('check_start');send('check_start','y')
    with pytest.raises(ValueError):send('check_start','z')
    with pytest.raises(ValueError):send('check_start')
    with pytest.raises(ValueError):send('check_finish',outcome='passed',duration=-1)
    send('check_finish',outcome='failed',duration=.01,message='assertion failed')
    with pytest.raises(ValueError):send('check_finish',outcome='passed')
    page=index.page('test/'+M.test_id('a'))
    assert len(page['checks'])==2
    assert index.directory(limit=1)['total']==len(module('catalog').PAGES)+3  # catalog + test + two checks
    assert len(index.directory(limit=1)['items'])==1

def test_named_check_fixture_reports_failure_without_swallowing_assertion(monkeypatch):
    plugin=module('pytest_plugin');events=[]
    class Session:
        def send(self,event,**kw):events.append((event,kw))
        def evidence(self,*args,**kw):events.append(('artifact',kw))
    # Restore before pytest emits this test's call report, not at fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(plugin,'_session',Session())
        evidence=plugin.Evidence('node')
        with evidence.check('shape') as check: check.text('result','right shape')
        assert events[0][0]=='check_start' and events[-1][1]['outcome']=='passed'
        assert events[1][1]['check']==module('model').test_id('shape')
        with pytest.raises(AssertionError):
            with evidence.check('normals'):assert False,'wrong direction'
        assert events[-1][1]['outcome']=='failed'
        with pytest.raises(pytest.skip.Exception):
            with evidence.check('optional'):pytest.skip('no device')
        assert events[-1][1]['outcome']=='skipped'

def test_missing_call_report_cannot_finish_green(tmp_path):
    index=module('model').Index(tmp_path/'index.sqlite');start(index)
    index.accept({'event':'collect','run':'one','nodeids':['a']})
    index.accept({'event':'report','run':'one','nodeid':'a','when':'teardown','outcome':'passed'})
    index.accept({'event':'finish','run':'one','exit_code':0})
    assert index.summary('one')['state']=='incomplete'

def test_progress_counts_tests_not_reports_and_keeps_teardown_failure(tmp_path):
    index = module('model').Index(tmp_path / 'index.sqlite')
    start(index)
    index.accept({'event': 'collect', 'run': 'one', 'nodeids': ['a', 'b']})
    for when, outcome in [('setup', 'passed'), ('call', 'passed'), ('teardown', 'failed')]:
        index.accept({'event': 'report', 'run': 'one', 'nodeid': 'a', 'when': when,
                      'outcome': outcome, 'duration': .1, 'message': 'detail'})
    result = index.summary('one')
    assert result['collected'] == 2
    assert result['completed'] == 1
    assert result['counts'] == {'failed': 1, 'pending': 1}
    assert index.test('one', module('model').test_id('a'))['outcome'] == 'failed'

def test_more_than_128_tests_are_paginated_and_survive_restart(tmp_path):
    M = module('model')
    index = M.Index(tmp_path / 'index.sqlite')
    start(index)
    names = ['test_%d' % i for i in range(305)] + ['x[a+b]', 'x[a b]']
    index.accept({'event': 'collect', 'run': 'one', 'nodeids': names})
    assert len(index.tests('one')['items']) == 50
    assert index.tests('one')['total'] == 307
    assert M.test_id(names[-1]) != M.test_id(names[-2])
    assert M.Index(tmp_path / 'index.sqlite').summary('one')['collected'] == 307
    assert len(index.tests('one', offset=300)['items']) == 7

def test_heartbeat_stale_is_not_completion_and_demo_is_not_a_test(tmp_path):
    index = module('model').Index(tmp_path / 'index.sqlite')
    start(index)
    start(index, 'example', 'demo')
    s = index.summary('one', now=10**12)
    assert s['state'] == 'running' and s['stale'] is True
    assert index.summary('example')['kind'] == 'demo'
    assert index.summary('example')['completed'] == 0
    index.accept({'event': 'finish', 'run': 'one', 'exit_code': 2})
    assert index.summary('one')['state'] == 'interrupted'

def test_coverage_explicit_unknown_and_validated_counts(tmp_path):
    M = module('model')
    index = M.Index(tmp_path / 'index.sqlite'); start(index)
    assert index.summary('one')['coverage'] is None
    result = M.coverage_report({'meta': {'version': 'test'}, 'files': {
        'src/gffx/a.py': {'executed_lines': [1, 3], 'missing_lines': [2],
                         'summary': {'num_branches': 2, 'covered_branches': 1}}}})
    assert result['covered'] == 2 and result['total'] == 3
    index.accept({'event': 'coverage', 'run': 'one', 'coverage': result})
    assert index.summary('one')['coverage']['source'] == 'coverage.py JSON (Python only)'
    with pytest.raises(ValueError):
        M.coverage_report({'files': {'bad': {'executed_lines': [1], 'missing_lines': [1]}}})

def test_quota_rejection_is_transactional_and_unknown_ids_do_not_fallback(tmp_path):
    index = module('model').Index(tmp_path / 'index.sqlite', max_tests=2)
    start(index)
    with pytest.raises(ValueError):
        index.accept({'event': 'collect', 'run': 'one', 'nodeids': ['a', 'b', 'c']})
    assert index.summary('one')['collected'] == 0
    with pytest.raises(KeyError): index.test('one', 'missing')
    with pytest.raises(ValueError): start(index, '../escape')

def test_separate_http_app_reuses_general_assets_and_rejects_cross_origin(tmp_path):
    Server = module('server').TestingServer
    with Server(tmp_path, port=0, allow_loopback_for_tests=True) as server:
        base = 'http://' + server.address
        def get(route): return urllib.request.urlopen(base + route, timeout=3).read()
        assert b'gffx testing' in get('/testing/')
        assert b'gffx dashboard' in get('/')
        assert b'/app.js' in get('/testing/app.js')
        assert json.loads(get('/testing/api/runs')) == {'runs': []}
        payload = json.dumps({'event': 'start', 'run': 'one', 'kind': 'tests'}).encode()
        request = urllib.request.Request(base + '/testing/api/events', data=payload,
            headers={'Content-Type': 'application/json', 'Origin': 'https://evil.invalid'})
        with pytest.raises(urllib.error.HTTPError) as e: urllib.request.urlopen(request)
        assert e.value.code == 403

def test_offline_reporter_never_raises_even_with_warnings_as_errors():
    reporter = module('reporter').Reporter('http://127.0.0.1:1', 'offline')
    assert reporter.send('start', kind='tests') is False
    assert reporter.failed and reporter.send('finish', exit_code=0) is False

def test_real_pytest_session_reports_progress_outcomes_evidence(tmp_path):
    Server = module('server').TestingServer
    sample = tmp_path / 'test_sample.py'
    sample.write_text('''import pytest
@pytest.mark.parametrize("n", range(135))
def test_many(n): assert n >= 0
def test_picture(dashboard):
    with dashboard.check("score range") as check:
        check.scalar("score", 0.5)
        assert 0 <= .5 <= 1
@pytest.mark.xfail(reason="baseline")
def test_expected(): assert False
@pytest.fixture
def broken():
    yield
    assert False, "teardown failed"
def test_teardown(broken): pass
''')
    with Server(tmp_path / 'server', port=0, allow_loopback_for_tests=True) as server:
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path),
               'GFFX_TEST_DASHBOARD': 'http://' + server.address, 'GFFX_TEST_RUN': 'integration',
               'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
        env.pop('GFFX_DASHBOARD', None)
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p',
            'tools.testing_dashboard.pytest_plugin', '-o', 'addopts=', '-q', str(sample)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 1, result.stdout + result.stderr
        summary = server.index.summary('integration')
        assert summary['collected'] == summary['completed'] == 138
        assert summary['counts'] == {'passed': 136, 'xfailed': 1, 'failed': 1}
        assert summary['state'] == 'failed'
        rows = server.index.tests('integration', search='test_picture')['items']
        detail = server.index.test('integration', rows[0]['id'])
        assert detail['artifacts'][0]['kind'] == 'plot'
        assert server.state.get(detail['artifacts'][0]['path'], run='integration')['value'] == .5
        check=server.index.page('check/'+rows[0]['id']+'/'+module('model').test_id('score range'))
        assert check['state']=='passed' and check['artifacts'][0]['check']==module('model').test_id('score range')
        from urllib.parse import urlencode
        url='http://'+server.address  # query is identity-safe, not a file path
        remote=json.loads(urllib.request.urlopen(url+'/testing/api/page?'+urlencode({'page':check['id']})).read())
        assert remote['artifacts']==check['artifacts']

def test_testing_frontend_contracts():
    import shutil
    node = shutil.which('node')
    if not node: pytest.skip('Node is optional development tooling')
    import gffx.dashboard
    result = subprocess.run([node, '--experimental-default-type=module', '--test',
        str(ROOT / 'tests/dashboard/testing.test.mjs')], capture_output=True, text=True, encoding='utf-8', timeout=30,
        env={**os.environ,'GFFX_DASHBOARD_STATIC':str(Path(gffx.dashboard.__file__).parent/'static')})
    assert result.returncode == 0, result.stdout + result.stderr

def test_demo_is_finite_persisted_and_uses_shared_visual_data(tmp_path):
    Server = module('server').TestingServer
    with Server(tmp_path, port=0, allow_loopback_for_tests=True) as server:
        module('demo').publish('http://' + server.address, 'demo-one')
        summary = server.index.summary('demo-one')
        assert summary['kind'] == 'demo' and summary['state'] == 'finished'
        assert summary['collected'] == 0
        assert len(server.index.test('demo-one','demo')['artifacts']) == 4
        assert server.state.get('demo/triangle',run='demo-one')['vertices'].shape == (3,3)

def test_finish_never_fabricates_success_or_overwrites_terminal_result(tmp_path):
    index=module('model').Index(tmp_path/'index.sqlite'); start(index)
    index.accept({'event':'collect','run':'one','nodeids':['a']})
    index.accept({'event':'finish','run':'one','exit_code':0})
    assert index.summary('one')['state']=='incomplete'
    with pytest.raises(ValueError):
        index.accept({'event':'finish','run':'one','exit_code':1})

def test_offline_plugin_preserves_test_outcome_under_warnings_as_errors(tmp_path):
    sample=tmp_path/'test_ok.py';sample.write_text('def test_ok(dashboard): dashboard.scalar("x",1)\n')
    env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path),'GFFX_TEST_DASHBOARD':'http://127.0.0.1:1',
         'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    env.pop('GFFX_DASHBOARD',None)
    result=subprocess.run([sys.executable,'-m','pytest','-p','tools.testing_dashboard.pytest_plugin',
        '-o','addopts=','-W','error','-q',str(sample)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'reporting disabled' in result.stderr

def test_cli_collects_real_python_coverage_when_tool_is_available(tmp_path):
    if importlib.util.find_spec('coverage') is None:
        pytest.skip('coverage.py is optional developer tooling')
    sample=tmp_path/'test_cov.py'
    sample.write_text('import gffx.dashboard\ndef test_cov(): assert gffx.dashboard.Server\n')
    with module('server').TestingServer(tmp_path/'server',port=0,allow_loopback_for_tests=True) as server:
        env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
        env.pop('GFFX_DASHBOARD',None)
        result=subprocess.run([sys.executable,'-m','tools.testing_dashboard','run','--address',
            'http://'+server.address,'--run','coverage-run','--coverage','--output',str(tmp_path/'output'),
            '--','-o','addopts=','-q',str(sample)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        summary=server.index.summary('coverage-run')
        assert summary['state']=='passed'
        assert summary['coverage']['covered']>0
        assert summary['coverage']['total']>=summary['coverage']['covered']
        import hashlib
        report=tmp_path/'output/coverage-run/coverage.json'
        assert summary['coverage']['artifact_sha256']==hashlib.sha256(report.read_bytes()).hexdigest()

def test_coverage_boundary_rejects_malformed_file_rows(tmp_path):
    M=module('model');index=M.Index(tmp_path/'index.sqlite');start(index)
    cov=M.coverage_report({'files':{'a.py':{'executed_lines':[1],'missing_lines':[2]}}})
    cov['files'][0]['missing_lines']=None
    with pytest.raises(ValueError):
        index.accept({'event':'coverage','run':'one','coverage':cov})
    assert index.summary('one')['coverage'] is None

def test_explicit_existing_run_refused_before_executing_tests(tmp_path):
    with module('server').TestingServer(tmp_path/'server',port=0,allow_loopback_for_tests=True) as server:
        start(server.index,run='existing')
        sample=tmp_path/'test_no.py';sample.write_text('raise RuntimeError("MUST NOT COLLECT")\n')
        result=subprocess.run([sys.executable,'-m','tools.testing_dashboard','run','--address',
            'http://'+server.address,'--run','existing','--',str(sample)],cwd=ROOT,
            env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path)},text=True,capture_output=True,timeout=15)
        assert result.returncode==2
        assert 'run already exists' in result.stderr
        assert 'MUST NOT COLLECT' not in result.stdout

def test_catalog_pages_exist_before_execution_and_reject_unknown_pages(tmp_path):
    catalog = module('catalog')
    ids = [page['id'] for page in catalog.PAGES]
    assert len(ids) == len(set(ids)), 'duplicate page IDs would merge unrelated evidence'
    articles = {'learn/index'} | {chapter['id'] for chapter in catalog.CHAPTERS}
    assert all(page.get('learning_article') in articles for page in catalog.PAGES if page.get('learning_article'))
    index = module('model').Index(tmp_path/'index.sqlite')
    pages = index.pages()
    assert any(p['id'] == 'camera/projection' for p in pages)
    assert all(p['evidence'] is None for p in pages)
    page = index.page('camera/projection')
    assert page['state'] == 'not measured' and page['items'] == []
    assert page['evidence'] is None
    with pytest.raises(KeyError): index.page('invented/page')

def test_page_selects_latest_relevant_execution_not_latest_success(tmp_path):
    index = module('model').Index(tmp_path/'index.sqlite')
    name = 'tests/python/test_camera.py::test_cam11_project_is_bit_identical_to_the_manual_chain[f32]'
    start(index, 'old')
    index.accept({'event':'collect','run':'old','nodeids':[name]})
    for phase in ('setup','call','teardown'):
        index.accept({'event':'report','run':'old','nodeid':name,'when':phase,'outcome':'passed'})
    index.accept({'event':'artifact','run':'old','nodeid':name,'artifact':{'kind':'text','path':'old/note'}})
    index.accept({'event':'finish','run':'old','exit_code':0})
    start(index, 'new')
    index.accept({'event':'collect','run':'new','nodeids':[name]})
    start(index, 'unrelated')
    index.accept({'event':'collect','run':'unrelated','nodeids':['tests/packaging/test_ci_contract.py::test_a']})
    page = index.page('camera/projection')
    assert page['evidence']['run'] == 'new' and page['counts'] == {'pending':1}
    assert page['artifacts'] == []  # Do not reuse an older execution's successful visual.
    assert index.page('camera/projection', run='old')['artifacts'][0]['path'] == 'old/note'
    with pytest.raises(KeyError): index.page('camera/projection', run='unrelated')
    with pytest.raises(KeyError): index.page('camera/projection', run='missing')
    assert module('model').Index(tmp_path/'index.sqlite').page('camera/projection')['evidence']['run'] == 'new'

def test_page_pagination_counts_are_local_and_filters_do_not_change_summary(tmp_path):
    index = module('model').Index(tmp_path/'index.sqlite'); start(index)
    names = ['tests/python/test_camera.py::test_cam11_case[%d]' % n for n in range(61)]
    index.accept({'event':'collect','run':'one','nodeids':names + ['tests/packaging/test_x.py::test_other']})
    page = index.page('camera/projection', offset=50)
    assert page['collected'] == 61 and page['completed'] == 0
    assert len(page['items']) == 11 and page['total'] == 61
    filtered = index.page('camera/projection', search='not present')
    assert filtered['total'] == 0 and filtered['collected'] == 61
    with pytest.raises(ValueError): index.page('camera/projection', offset=-1)

def test_demo_page_is_permanent_and_independent_of_test_execution(tmp_path):
    index = module('model').Index(tmp_path/'index.sqlite')
    start(index, 'visual', 'demo')
    index.accept({'event':'demo','run':'visual','title':'Panels','artifacts':[{'kind':'mesh','path':'demo/triangle'}]})
    index.accept({'event':'finish','run':'visual','exit_code':0})
    start(index, 'newer-tests')
    page = index.page('dashboard/panel-demo')
    assert page['kind'] == 'demo' and page['collected'] == 0
    assert page['evidence']['run'] == 'visual' and page['artifacts'][0]['kind'] == 'mesh'
    assert index.page('camera/projection')['state'] == 'not measured'

def test_page_api_requires_no_run_and_keeps_general_dashboard(tmp_path):
    with module('server').TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        base = 'http://' + server.address
        def get(path): return urllib.request.urlopen(base+path,timeout=3).read()
        assert json.loads(get('/testing/api/pages'))['pages']
        assert json.loads(get('/testing/api/page?page=camera/projection'))['evidence'] is None
        assert b'id="runs"' not in get('/testing/')
        assert b'gffx dashboard' in get('/')

def test_catalog_routes_existing_test_families_without_silent_loss():
    catalog = module('catalog')
    assert catalog.page_for('tests/python/test_camera.py::test_cam17_gradcheck_reaches_every_intrinsic') == 'camera/gradients'
    assert catalog.page_for('tests/pytorch/test_face_geometry.py::test_forward') == 'mesh/face-geometry'
    assert catalog.page_for('tests/packaging/test_ci_contract.py::test_pins') == 'infrastructure/packaging'
    assert catalog.page_for('unexpected/path.py::test_new') == 'infrastructure/unassigned'

def test_page_creation_dates_are_explicit_and_independent_of_execution(tmp_path):
    from datetime import date
    catalog=module('catalog')
    index=module('model').Index(tmp_path/'index.sqlite')
    baseline={p['id']:p['created_on'] for p in index.pages()}
    for page in catalog.PAGES:
        assert date.fromisoformat(page['created_on']).isoformat()==page['created_on']
    start(index)
    index.accept({'event':'collect','run':'one','nodeids':['tests/pytorch/test_face_geometry.py::test_forward']})
    assert {p['id']:p['created_on'] for p in index.pages()}==baseline


def test_catalog_cache_invalidates_new_evidence_and_keeps_live_staleness(tmp_path,monkeypatch):
    model=module('model');index=model.Index(tmp_path/'index.sqlite')
    monkeypatch.setattr(model,'PAGES',tuple(p for p in model.PAGES if p['id']=='camera/projection'))
    name='tests/python/test_camera.py::test_cam11_cache_control'
    start(index,'cache_old');index.accept({'event':'collect','run':'cache_old','nodeids':[name]})
    original=index.page;calls=[]
    def counted(*args,**kw):calls.append(args);return original(*args,**kw)
    monkeypatch.setattr(index,'page',counted)
    first=index.pages();second=index.pages();assert first==second and len(calls)==1
    first[0]['evidence']['run']='tampered caller';assert index.pages()[0]['evidence']['run']=='cache_old'
    updated=second[0]['evidence']['updated'];monkeypatch.setattr(model.time,'time',lambda:updated+20)
    assert index.pages()[0]['evidence']['stale'] is True
    start(index,'cache_new');index.accept({'event':'collect','run':'cache_new','nodeids':[name]})
    fresh=index.pages()[0];assert fresh['evidence']['run']=='cache_new' and fresh['counts']=={'pending':1}
    index.accept({'event':'report','run':'cache_new','nodeid':name,'when':'call','outcome':'failed'})
    index.accept({'event':'report','run':'cache_new','nodeid':name,'when':'teardown','outcome':'passed'})
    index.accept({'event':'finish','run':'cache_new','exit_code':1})
    assert index.pages()[0]['state']=='failed'
    index.pin('cache_new',True);assert index.pages()[0]['evidence']['pinned'] is True


def test_live_checkpoint_refreshes_without_server_restart(tmp_path, monkeypatch):
    S=module('server')
    static=tmp_path/'static';static.mkdir()
    checkpoint=static/'checkpoint.json'
    checkpoint.write_text(json.dumps({'status':'running','acceptedForeground':213}),encoding='utf-8')
    monkeypatch.setattr(S,'STATIC',static)
    with S.TestingServer(tmp_path/'server',port=0,allow_loopback_for_tests=True) as server:
        url='http://'+server.address+'/testing/api/pages'
        with urllib.request.urlopen(url) as response:
            assert response.headers['Cache-Control']=='no-store'
            assert json.load(response)['checkpoint']['acceptedForeground']==213
        checkpoint.write_text(json.dumps({'status':'stopped','acceptedForeground':214}),encoding='utf-8')
        with urllib.request.urlopen(url) as response:
            assert json.load(response)['checkpoint']=={'status':'stopped','acceptedForeground':214}
