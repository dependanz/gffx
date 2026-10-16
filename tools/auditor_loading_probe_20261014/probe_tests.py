"""Small focused measurement tests; never uses the actual saved-pixel491 input."""
import argparse,json,os,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
from probe_core import *
from probe_worker import Recorder,loader_observation
CASES=None
class Tests(unittest.TestCase):
    def setUp(self):self.root=CASES/self._testMethodName;self.root.mkdir()
    def record(self):return {'pixel':491,'payload':[{'rational_hex':['1','3']},None,True,{'text':'é','values':[1,2,3]}]}
    def fixture(self):
        path=self.root/'record.json';path.write_text(json.dumps(self.record()),encoding='utf-8');return path
    def test_exact_production_loader_decoder_and_owner(self):
        load_record,decode,OwnedWork,optional_none=production();owner=OwnedWork(9223372036854775807,9223372036854775807,lambda:None);optional_none(owner,None)
        record=decode(load_record(self.fixture()));owner.hold(record)
        self.assertIs(owner.history[-1],record);self.assertIn(id(record),owner.storage.seen);self.assertEqual(owner.work.count,1)
        result=ownership(owner,deep=True);self.assertGreater(result['seen_count'],10);self.assertGreater(result['seen_total_set_plus_identity_bytes'],result['seen_set_shallow_bytes']);self.assertGreater(result['scientific_accounting_used_bytes'],0)
    def test_loader_observation_restores_original_methods(self):
        load_record,*_=production();path=self.fixture();r=Recorder(self.root,time.perf_counter()+10,GIB);original_read=Path.read_bytes;original_load=json.loads
        with loader_observation(r,path,sha(path)):record=load_record(path)
        self.assertIs(original_read,Path.read_bytes);self.assertIs(original_load,json.loads);self.assertEqual(record['pixel'],491)
        self.assertEqual(set(r.rows),{'bytes_read','JSON_parse'});self.assertEqual(r.consumed_bytes,path.stat().st_size)
    def test_loader_hash_refusal_restores_methods(self):
        load_record,*_=production();path=self.fixture();r=Recorder(self.root,time.perf_counter()+10,GIB);original_read=Path.read_bytes;original_load=json.loads
        with self.assertRaises(ValueError):
            with loader_observation(r,path,'0'*64):load_record(path)
        self.assertIs(original_read,Path.read_bytes);self.assertIs(original_load,json.loads)
    def test_JSON_refusal_restores_methods(self):
        load_record,*_=production();path=self.root/'bad.json';path.write_bytes(b'{');r=Recorder(self.root,time.perf_counter()+10,GIB);original_load=json.loads
        with self.assertRaises(json.JSONDecodeError):
            with loader_observation(r,path,sha(path)):load_record(path)
        self.assertIs(original_load,json.loads)
    def test_phase_exception_identity_preserved(self):
        r=Recorder(self.root,time.perf_counter()+10,GIB);error=ValueError('original')
        with self.assertRaises(ValueError)as caught:
            with r.phase('refusal'):raise error
        self.assertIs(caught.exception,error);self.assertIn('exit',r.rows['refusal'])
    def test_sampler_fault_is_refusal(self):
        r=Recorder(self.root,time.perf_counter()+10,GIB);r.fault=OSError('memory sampler')
        with self.assertRaises(RuntimeError):r.check()
    def test_deadline_is_refusal(self):
        r=Recorder(self.root,time.perf_counter()-1,GIB)
        with self.assertRaises(TimeoutError):r.check()
    def snapshot(self):return {'available_RAM_bytes':25*GIB,'commit_headroom_bytes':25*GIB,'D_free_bytes':2*GIB}
    def test_20GiB_admission_uses_24GiB_floor(self):admit_snapshot(self.snapshot())
    def test_RAM_refusal(self):
        s=self.snapshot();s['available_RAM_bytes']=24*GIB-1
        with self.assertRaises(MemoryError):admit_snapshot(s)
    def test_commit_refusal(self):
        s=self.snapshot();s['commit_headroom_bytes']=24*GIB-1
        with self.assertRaises(MemoryError):admit_snapshot(s)
    def test_disk_reserve_refusal(self):
        s=self.snapshot();s['D_free_bytes']=GIB+64*MIB
        with self.assertRaises(OSError):admit_snapshot(s)
    def test_storage_permanent_loss_refuses(self):
        path=self.root/'report.json';path.write_bytes(b'keep');s=Storage(self.root,limit=MIB);s.bytes_used();path.unlink()
        with self.assertRaises(FileNotFoundError):s.bytes_used()
    def test_owned_fixture_cleanup_and_quota(self):
        path=self.root/'scratch/fixtures/temporary.json';path.parent.mkdir(parents=True);path.write_bytes(b'owned');s=Storage(self.root,limit=MIB);s.bytes_used();path.unlink();s.bytes_used()
        with self.assertRaises(OSError):Storage(self.root,limit=1).bytes_used()if(self.root/'report.json').write_bytes(b'xx')else None
    def test_supervisor_tiny_production_pipeline_and_live_ack(self):
        from probe_controller import supervise
        path=self.fixture();spec={'root':str(self.root),'input':str(path),'input_sha256':sha(path),'pixel':491,'process_bytes':GIB,'seconds':30,'report_limit_bytes':64*MIB,'source_bindings':{str(CODE/'probe_worker.py'):sha(CODE/'probe_worker.py')},'actual_subject':False}
        result=supervise(spec)
        self.assertEqual(result['status'],'passed',result);self.assertTrue(result['actual_worker_identity_matches']);self.assertTrue(result['measurement_acknowledged_with_state_alive'])
        child=result['child'];self.assertEqual(child['mathematical_audit_operations'],0);self.assertEqual(child['serialization_calls'],0);self.assertEqual(child['held_record_identity'],child['retained_record_identity']);self.assertEqual(child['ownership_final']['history_entries'],2)
        for phase in('bytes_read','JSON_parse','exact_decode','OwnedWork.hold'):
            row=child['phases'][phase];self.assertGreater(row['sampled_peak_RSS_bytes'],0);self.assertGreater(row['sampled_peak_private_commit_bytes'],0);self.assertGreaterEqual(row['exit']['OS_lifetime_peak_commit_bytes'],row['entry']['OS_lifetime_peak_commit_bytes'])
    def test_supervisor_input_refusal_is_preserved(self):
        from probe_controller import supervise
        path=self.fixture();spec={'root':str(self.root),'input':str(path),'input_sha256':'0'*64,'pixel':491,'process_bytes':GIB,'seconds':30,'report_limit_bytes':64*MIB,'source_bindings':{},'actual_subject':False}
        result=supervise(spec);self.assertEqual(result['status'],'failed');self.assertTrue((self.root/'worker-result.json').exists());self.assertFalse((self.root/'measurement-complete.json').exists())
    def test_no_checkpoint_or_serializer_called_by_pipeline(self):
        import duffy_integral271 as decoder
        load_record,decode,OwnedWork,optional_none=production();owner=OwnedWork(9223372036854775807,9223372036854775807,lambda:None);optional_none(owner,None)
        with patch.object(decoder,'validate_certificate',side_effect=AssertionError('math audit forbidden')):
            record=decode(load_record(self.fixture()));owner.hold(record)
        self.assertNotIn('disk_exact_json',sys.modules);self.assertEqual(owner.work.count,1)
    def test_external_phase_observer_handles_incomplete_tail(self):
        from probe_controller import PhaseObserver
        from types import SimpleNamespace
        path=self.root/'phase-events.jsonl';path.write_bytes(b'{"sequence":0,"event":"enter","phase":"JSON_parse"}\n{"sequence":1')
        observer=PhaseObserver(self.root);observer.poll();observer.sample(SimpleNamespace(rss=10,private=12,peak_wset=14,peak_pagefile=16))
        self.assertEqual(observer.rows['JSON_parse']['sampled_peak_private_commit_bytes'],12)
        with path.open('ab')as out:out.write(b',"event":"exit","phase":"JSON_parse"}\n')
        observer.poll();self.assertEqual(observer.stack,[])
    def test_external_phase_sequence_gap_refuses(self):
        from probe_controller import PhaseObserver
        (self.root/'phase-events.jsonl').write_bytes(b'{"sequence":1,"event":"enter","phase":"JSON_parse"}\n')
        with self.assertRaises(ValueError):PhaseObserver(self.root).poll()
def main(root):
    global CASES
    root=Path(root);root.mkdir();CASES=root/'cases';CASES.mkdir();started=time.perf_counter()
    with(root/'tests.log').open('x',encoding='utf-8')as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={'status':'passed'if result.wasSuccessful()else'failed','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'worker_pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started,'actual_saved_pixel491_read':False,'no_actual_probe':True}
    seal(root/'test-result.json',report);return 0 if result.wasSuccessful()else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);raise SystemExit(main(parser.parse_args().root))
