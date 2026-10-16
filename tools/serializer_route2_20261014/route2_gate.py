import argparse,json,os,sys,time
from pathlib import Path
from route2_common import ROOT as P
H=Path('D:/devspace/gffit/qualification-20261006/boundary-sh28/gradient136-independent-visibility-bruteforce-preparation1')
sys.path.insert(0,'C:/dev/gffx');sys.path.insert(0,str(H));sys.path.insert(0,str(H/'runtime2'));sys.path.insert(0,str(P))
from tools import material_loss_cost as cost
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--worker',action='store_true');pa.add_argument('--output',type=Path,required=True);pa.add_argument('--launch-index',type=int,required=True);a=pa.parse_args();start=time.perf_counter()
    d={'status':'working','phase':'admission','worker_pid':os.getpid(),'launch_index':a.launch_index,'tests':0,'failed':0,'skipped':0,'arrays_opened':False,'native_calls':0,'render_calls':0,'photo_readiness':False}
    from report_journal142 import Scope
    from route2_phases import Recorder
    with Recorder(a.output) as phases, Scope(cost,a.output):
        try:
            cost.write_report(a.output,d)
            from route2_admission import before_arrays,read,sha
            p=read(P/'pipeline-packet1.json');from route2_admission import task_for_index;task=task_for_index(0,p);assert a.worker and a.launch_index==p['base_launch_index'] and a.output==Path(p['stage'])/'controls.json' and read(str(a.output)+'.spec.json')==task['spec']
            sha=phases.wrap(sha,'worker_initial_source_binding_verification')
            for path,h in p['sources'].items():assert sha(path)==h,path
            with phases.measure('worker_prerequisite_verification'):before_arrays(0,check=lambda:cost.check_deadline(start+task['spec']['seconds']))
            d.update(phase='fixture-gate',all_current_COMPLETE_and_stopped_typed_chains_before_arrays=True);cost.write_report(a.output,d)
            binding=p['binding_evidence'];assert sha(binding['path'])==binding['sha256'];b=read(binding['path']);assert b['status']=='passed' and b['failures']==b['errors']==b['skips']==0 and b['tests']==binding['tests']
            for path,h in b['sources'].items():assert sha(path)==h,path
            d.update(binding_tests_passed=True,serializer_adoption={'verified':True,'calls':0,'reused_focused_evidence':binding,'prototype_sha256':'56184c56fed001b9f5e7585da10a5bfbc324ceb7540662b930d7bddad023d391'})
            import pytest,xml.etree.ElementTree as ET
            xml=a.output.with_suffix('.xml')
            with phases.measure('fixture_execution'):rc=pytest.main(['-q','-p','no:cacheprovider','--junitxml='+str(xml),*p['controls']])
            counts=[s.attrib for s in ET.parse(xml).getroot().iter('testsuite')];d.update(tests=sum(int(s.get('tests',0)) for s in counts),failed=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in counts),skipped=sum(int(s.get('skipped',0)) for s in counts))
            assert rc==0 and d['tests']==1639 and d['failed']==d['skipped']==0
            cost.check_deadline(start+task['spec']['seconds']);d.update(status='passed',phase='complete',retained1466_fixture_bodies_unchanged=True,new_field_controls=21,new_attribute_controls=16,retained1487_fixture_bodies_unchanged=True,retained1503_fixture_bodies_unchanged=True,new_whole_reference_controls=27,retained1530_fixture_bodies_unchanged=True,new_scene_support_controls=24,new_visibility_view_controls=14,retained1554_fixture_bodies_unchanged=True,new_stream_controls=14,retained1568_fixture_bodies_unchanged=True,new_composition_controls=8,new_capacity_controls=10,retained1582_fixture_bodies_unchanged=True,retained1600_fixture_bodies_unchanged=True,new_hash_controls=12,retained1612_fixture_bodies_unchanged=True,new_byte_capacity_controls=12,retained1624_fixture_bodies_unchanged=True,new_compact_index_controls=12,new_measurement_controls=3,complete_field_integration=False,independent_derivative_accuracy=False)
        except Exception as ex:d.update(status='stopped',error=repr(ex));raise
        finally:d['phase_metrics']=phases.finish();d['seconds']=time.perf_counter()-start;cost.write_report(a.output,d)
if __name__=='__main__':main()
