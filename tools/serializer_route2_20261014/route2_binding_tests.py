"""Focused route2 bindings only: no original fixture suite or numerical main."""
import ast,copy,hashlib,io,json,os,sys,time,unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
import route2_common as common
import route2_serializer as adapter
import route2_environment as environment
import route2_phases as phases
CASES=None
class BindingTests(unittest.TestCase):
    def setUp(self):
        self.root=CASES/self._testMethodName;self.root.mkdir()
        self.output=self.root/"complete-history.json"
    def graph(self):
        shared={"bounds":(Q(1,10**12),Q(1,10**10)),"state":(True,None,"é")}
        rows=[(i,Q(i+1,7),{"shared":shared})for i in range(20)]
        return {"coordinator_history":tuple(rows),"texture":{"queries":tuple(rows)},"comparison_ordinary_history":((0,True),),"comparison_array_shapes":{"colour":(1,32,32,3)},"comparison_array_sha256":{"colour":"0"*64},"pixel_traces":rows,"audit_history":tuple(rows)}
    def invoke(self,graph=None,limits=None,checkpoint=lambda:None):
        graph=self.graph()if graph is None else graph
        with adapter.context(self.root,"case",checkpoint,limits)as context:
            with self.output.open("x",encoding="utf-8",newline="")as out:stats=adapter.dump_exact(graph,out)
        return graph,stats,context
    def test_complete_history_exact_readback_and_counts(self):
        graph,stats,context=self.invoke()
        row=context.rows[-1]
        self.assertEqual(row["status"],"passed");self.assertEqual(row["readback"]["sha256"],common.sha(self.output))
        self.assertEqual(stats["serialized_fraction_occurrences"],20*12)
        self.assertTrue(all(key in json.loads(self.output.read_text())for key in graph))
        legacy=common.load_file("route2_trusted_legacy",common.FROZEN/"streamed_exact_json312.py");target=io.StringIO();expected=legacy.dump_exact(graph,target)
        self.assertEqual(stats,expected);self.assertEqual(self.output.read_bytes(),target.getvalue().encode())
    def test_other_outputs_count_in_prewrite_budget(self):
        with patch.object(environment.StorageObserver,"bytes_used",return_value=120*common.GIB):
            with self.assertRaises(OSError):self.invoke()
        self.assertEqual(self.output.stat().st_size,0)
    def test_sampler_fault_visible_during_serialization(self):
        with phases.Recorder(self.root/"metrics")as recorder:
            recorder.fault=OSError("observer coverage")
            with self.assertRaises(RuntimeError):self.invoke()
            recorder.fault=None
    def test_native_temp_scope_is_exact_owned_namespace(self):
        observer=environment.StorageObserver(self.root,limit=1024)
        self.assertTrue(observer._temporary(self.root/"scratch/worker-5288/test.tmp"))
        self.assertFalse(observer._temporary(self.root/"scratch/worker-9999/test.tmp"))
        self.assertFalse(observer._temporary(self.root/"report.tmp"))
    def test_original_identity_and_aliases_held(self):
        graph=self.graph();original=graph["coordinator_history"][0]
        self.invoke(graph)
        self.assertIs(graph["pixel_traces"][0],original);self.assertIs(graph["audit_history"][0],original)
    def test_no_context_refuses(self):
        with self.output.open("x",encoding="utf-8")as out:
            with self.assertRaisesRegex(ValueError,"context"):adapter.dump_exact(self.graph(),out)
    def test_cycle_refuses_before_bytes(self):
        graph=[];graph.append(graph)
        with self.assertRaises(ValueError):self.invoke(graph)
        self.assertEqual(self.output.stat().st_size,0)
    def test_type_refuses_before_bytes(self):
        with self.assertRaises(TypeError):self.invoke({"x":1.0})
        self.assertEqual(self.output.stat().st_size,0)
    def test_key_refuses_before_bytes(self):
        with self.assertRaises(TypeError):self.invoke({1:Q(1,3)})
        self.assertEqual(self.output.stat().st_size,0)
    def test_validation_allocation_refusal_preserved(self):
        limits=adapter.prototype.IndexLimits(max_entries=2,max_database_bytes=8192,max_scratch_bytes=16384)
        with self.assertRaises(MemoryError):self.invoke(limits=limits)
        self.assertEqual(self.output.stat().st_size,0)
    def test_functions_restored_after_failure(self):
        validate=adapter.prototype.validate_exact_graph;dump=json.dump;scratch=adapter.prototype.DiskHeldIndex._check_scratch
        with self.assertRaises(TypeError):self.invoke({"bad":object()})
        self.assertIs(validate,adapter.prototype.validate_exact_graph);self.assertIs(dump,json.dump);self.assertIs(scratch,adapter.prototype.DiskHeldIndex._check_scratch)
    def test_functions_restored_after_success(self):
        validate=adapter.prototype.validate_exact_graph;dump=json.dump
        self.invoke();self.assertIs(validate,adapter.prototype.validate_exact_graph);self.assertIs(dump,json.dump)
    def test_same_callback_exception_object(self):
        error=TimeoutError("injected original callback")
        def check():raise error
        with self.assertRaises(TimeoutError)as caught:self.invoke(checkpoint=check)
        self.assertIs(caught.exception,error)
    def test_corrupt_output_readback_refuses(self):
        real=adapter.ComparingWriter.write
        def changed(writer,text):
            if text=="{":text="["
            return real(writer,text)
        with patch.object(adapter.ComparingWriter,"write",changed):
            with self.assertRaisesRegex(ValueError,"bytes"):self.invoke()
    def test_entry_snapshot_uses_fresh_stat(self):
        class Entry:
            path=str(self.output)
            def stat(self,**kw):raise AssertionError("cached stat")
        self.output.write_bytes(b"real")
        self.assertEqual(environment.base._entry_stat(Entry()).st_size,4)
    def test_known_owned_report_temp_cleanup(self):
        report=self.root/"report.json";report.write_bytes(b"keep")
        tmp=Path(str(report)+".tmp");tmp.write_bytes(b"owned")
        observer=environment.StorageObserver(self.root,limit=1024,report_paths=[report])
        observer.bytes_used();tmp.unlink();self.assertEqual(observer.bytes_used(),4)
    def test_missing_permanent_report_still_refuses(self):
        report=self.root/"report.json";report.write_bytes(b"keep")
        observer=environment.StorageObserver(self.root,limit=1024,report_paths=[report]);observer.bytes_used();report.unlink()
        with self.assertRaises(FileNotFoundError):observer.bytes_used()
    def test_unknown_temp_still_permanent(self):
        tmp=self.root/"unknown.tmp";tmp.write_bytes(b"x")
        observer=environment.StorageObserver(self.root,limit=1024);observer.bytes_used();tmp.unlink()
        with self.assertRaises(FileNotFoundError):observer.bytes_used()
    def test_temp_permission_refusal_preserved(self):
        report=self.root/"report.json";report.write_bytes(b"keep");tmp=Path(str(report)+".tmp");tmp.write_bytes(b"x")
        observer=environment.StorageObserver(self.root,limit=1024,report_paths=[report])
        real=environment.base._entry_stat
        def denied(entry):
            if Path(entry.path)==tmp:raise PermissionError("owned temp denied")
            return real(entry)
        with patch.object(environment.base,"_entry_stat",denied):
            with self.assertRaises(PermissionError):observer.bytes_used()
    def test_quota_still_refuses(self):
        self.output.write_bytes(b"xx")
        with self.assertRaises(OSError):environment.StorageObserver(self.root,limit=1).bytes_used()
    def test_focused_fixture_cleanup_is_owned(self):
        from run_route2_tests import FocusedStorageObserver
        import shutil
        fixture=self.root/"case-run/cases/test_fixture/scratch/exact-index-abcdefgh"
        fixture.mkdir(parents=True);(fixture/"held.sqlite-journal").write_bytes(b"owned")
        report=self.root/"focused-results.json";report.write_bytes(b"permanent")
        observer=FocusedStorageObserver(self.root,limit=1024);observer.bytes_used()
        shutil.rmtree(fixture);observer.bytes_used()
        self.assertFalse(observer._temporary(report));self.assertTrue(observer._temporary(fixture))
    def test_focused_permanent_report_loss_refuses(self):
        from run_route2_tests import FocusedStorageObserver
        report=self.root/"focused-results.json";report.write_bytes(b"permanent")
        observer=FocusedStorageObserver(self.root,limit=1024);observer.bytes_used();report.unlink()
        with self.assertRaises(FileNotFoundError):observer.bytes_used()
    def test_focused_owned_fixture_permissions_and_quota_refuse(self):
        from run_route2_tests import FocusedStorageObserver
        fixture=self.root/"case-run/cases/test_fixture";fixture.mkdir(parents=True)
        output=fixture/"held.sqlite-journal";output.write_bytes(b"owned")
        observer=FocusedStorageObserver(self.root,limit=1024);real=environment.base._entry_stat
        def denied(entry):
            if Path(entry.path)==output:raise PermissionError("owned fixture denied")
            return real(entry)
        with patch.object(environment.base,"_entry_stat",denied):
            with self.assertRaises(PermissionError):observer.bytes_used()
        with self.assertRaises(OSError):FocusedStorageObserver(self.root,limit=1).bytes_used()
    def contract(self):
        return {"category":"geometry","output":"chunks/chunk-30/result.json","output_forecast_bytes":64*common.GIB,"pixels":[480,496],"spec":{"module":"route2_producer","module_root":"D:/prepared","args":["--launch-index","5289"],"host_bytes":"FRESH_AVAILABLE_RAM_BYTES_MINUS_4GIB","seconds":18000,"watchdog_seconds":18060}}
    def limits(self):
        return {"process_bytes":40*common.GIB,"RAM_reserve_bytes":4*common.GIB,"commit_reserve_bytes":4*common.GIB,"combined_bytes":128*common.GIB,"disk_reserve_bytes":common.GIB}
    def snapshot(self):
        return {"total_RAM_bytes":64*common.GIB,"available_RAM_bytes":48*common.GIB,"commit_limit_bytes":96*common.GIB,"commit_used_bytes":40*common.GIB,"commit_headroom_bytes":56*common.GIB,"free_disk_bytes":200*common.GIB}
    def test_fresh_physical_commit_resolution_and_ceiling(self):
        contract=self.contract();result=common.resolve_task(contract,self.snapshot(),self.limits())
        self.assertEqual(result["spec"]["host_bytes"],40*common.GIB);self.assertEqual(contract["spec"]["host_bytes"],"FRESH_AVAILABLE_RAM_BYTES_MINUS_4GIB")
    def test_RAM_refusal(self):
        s=self.snapshot();s["available_RAM_bytes"]=43*common.GIB
        with self.assertRaises(MemoryError):common.resolve_task(self.contract(),s,self.limits())
    def test_commit_refusal(self):
        s=self.snapshot();s["commit_used_bytes"]=60*common.GIB;s["commit_headroom_bytes"]=36*common.GIB
        with self.assertRaises(MemoryError):common.resolve_task(self.contract(),s,self.limits())
    def test_commit_arithmetic_refusal(self):
        s=self.snapshot();s["commit_headroom_bytes"]+=1
        with self.assertRaises(ValueError):common.resolve_task(self.contract(),s,self.limits())
    def test_disk_refusal(self):
        s=self.snapshot();s["free_disk_bytes"]=129*common.GIB
        with self.assertRaises(OSError):common.resolve_task(self.contract(),s,self.limits())
    def test_changed_pixel_scope_refusal(self):
        c=self.contract();c["pixels"]=[496,512]
        with self.assertRaises(ValueError):common.resolve_task(c,self.snapshot(),self.limits())
    def test_unbounded_time_refusal(self):
        c=self.contract();c["spec"]["seconds"]=2147483647
        with self.assertRaises(ValueError):common.resolve_task(c,self.snapshot(),self.limits())
    def test_bool_time_refusal(self):
        c=self.contract();c["spec"]["seconds"]=True
        with self.assertRaises(ValueError):common.resolve_task(c,self.snapshot(),self.limits())
    def test_failed_gate_prevents_producer(self):
        with self.assertRaises(ValueError):common.check_prior_status(0,{"status":"stopped"})
    def test_incomplete_producer_prevents_audit(self):
        with self.assertRaises(ValueError):common.check_prior_status(1,{"status":"passed","supervisor":{"failure":None,"exit_code":0},"pixels_complete":15})
    def test_no_fourth_subject(self):
        with self.assertRaises(ValueError):common.check_prior_status(3,{"status":"passed","supervisor":{"failure":None,"exit_code":0}})
    def prepared_packet(self):
        limits=self.limits();limits.update(checkpoint_seconds=43200,terminal_reserve_seconds=3600)
        tasks=[]
        for i,(module,category,output,seconds)in enumerate((("route2_gate","control_repair","controls.json",3600),("route2_producer","geometry","chunks/chunk-30/result.json",18000),("route2_auditor","geometry","audit-30/result.json",10800))):
            task=self.contract();task.update(category=category,output=output)
            if i==0:task.pop("pixels")
            task["spec"].update(module=module,module_root=str(common.ROOT),args=["--launch-index",str(5288+i)],seconds=seconds,watchdog_seconds=seconds+60);tasks.append(task)
        return {"schema":"route2-prepared-checkpoint/1","status":"PREPARED_NOT_AUTHORIZED","base_launch_index":5288,"stage":str(common.ROOT/"results"),"tasks":tasks,"limits":limits}
    def test_exact_prepared_three_subject_packet(self):
        common.validate_packet(self.prepared_packet())
    def test_prepared_extra_subject_refuses(self):
        p=self.prepared_packet();p["tasks"].append(p["tasks"][-1])
        with self.assertRaises(ValueError):common.validate_packet(p)
    def test_prepared_audit_category_preserves_geometry_charge(self):
        p=self.prepared_packet();p["tasks"][-1]["category"]="control_repair"
        with self.assertRaises(ValueError):common.validate_packet(p)
    def test_prepared_changed_task_module_or_arguments_refuses(self):
        for field,value in (("module","route2_producer"),("args",["--launch-index","5291"])):
            p=self.prepared_packet();p["tasks"][-1]["spec"][field]=value
            with self.assertRaises(ValueError):common.validate_packet(p)
    def test_prepared_changed_process_cap_refuses(self):
        p=self.prepared_packet();p["limits"]["process_bytes"]+=1
        with self.assertRaises(ValueError):common.validate_packet(p)
    def test_phase_exception_identity(self):
        error=ValueError("original")
        with phases.Recorder(self.root/"metrics")as recorder:
            with self.assertRaises(ValueError)as caught:
                with recorder.measure("probe"):raise error
        self.assertIs(caught.exception,error)
    def test_phase_observer_fault_refuses(self):
        recorder=phases.Recorder(self.root/"metrics");recorder.fault=OSError("observer")
        with self.assertRaises(RuntimeError):recorder.check()
    def test_source_reversals_preserve_all_numerical_bodies(self):
        items=common.read(Path(__file__).with_name("adoption-transform.json"))
        for item in items:
            text=Path(__file__).with_name(item["candidate"]).read_text()
            for before,after in reversed(item["replacements"]):text=text.replace(after,before)
            self.assertEqual(ast.dump(ast.parse(text)),ast.dump(ast.parse(Path(item["original"]).read_text())))
    def test_both_real_callsites_bind_same_tested_serializer(self):
        for name in("route2_producer.py","route2_auditor.py"):
            tree=ast.parse(Path(__file__).with_name(name).read_text())
            nodes=[node for node in ast.walk(tree)if isinstance(node,ast.ImportFrom)and node.module=="route2_serializer"and any(a.name=="dump_exact"for a in node.names)]
            self.assertEqual(len(nodes),1);space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),name,"exec"),space)
            self.assertIs(space["dump_exact"],adapter.dump_exact)
    def test_original_refusal_after_uninstalled_context(self):
        self.assertFalse(adapter.finish_context()["verified"])
    def test_missing_authority_scope_refuses(self):
        from route2_checkpoint import verify_authority
        common.seal(self.root/"approval.json",{"explicit_launch_authorization":False})
        with self.assertRaises(PermissionError):verify_authority(self.root/"approval.json",{})
    def test_altered_sources_refuse(self):
        source=self.root/"source.py";source.write_text("bound")
        with self.assertRaises(ValueError):common.verify_sources({"sources":{str(source):"0"*64}})
    def test_source_scan_preserves_bounded_deadline_exception(self):
        source=self.root/"source.py";source.write_text("bound");error=TimeoutError("source scan deadline")
        def check():raise error
        with self.assertRaises(TimeoutError)as caught:common.verify_sources({"sources":{str(source):common.sha(source)}},check=check)
        self.assertIs(caught.exception,error)
    def test_initial_resource_refusal_precedes_expensive_prerequisites(self):
        import route2_checkpoint as controller,route2_terminal as terminal
        p=self.prepared_packet();p["stage"]=str(self.root/"results")
        baseline={"count":5288,"previous":"retained"};common.seal(self.root/"pipeline-ledger-before1.json",baseline)
        fresh=self.snapshot();fresh["available_RAM_bytes"]=43*common.GIB
        with patch.object(controller,"ROOT",self.root),patch.object(controller,"packet",return_value=p),patch.object(controller,"verify_authority"),patch.object(controller,"quiet"),patch.object(controller,"psutil_ancestors",return_value=[]),patch.object(controller,"inventory",return_value=baseline),patch.object(controller,"snapshot",return_value=fresh),patch.object(controller,"verify_sources")as sources,patch.object(controller,"before_arrays")as prerequisites,patch.object(controller,"admit")as admit,patch.object(terminal,"reconcile",return_value={"status":"refusal","accepted_pixels":480})as reconcile:
            self.assertEqual(controller.run(self.root/"unit-approval"),1)
            sources.assert_not_called();prerequisites.assert_not_called();admit.assert_not_called();reconcile.assert_called_once()
        result=common.read(self.root/"pipeline-stopped1.json")
        self.assertEqual(result["attempted_launches"],[]);self.assertIn("RAM admission",result["error"])
    def test_terminal_refusal_never_reports_passing_receipt(self):
        import route2_checkpoint as controller,route2_terminal as terminal
        from types import SimpleNamespace
        p=self.prepared_packet();p["stage"]=str(self.root/"results");p["sources"]={}
        common.seal(self.root/"unit-approval",{"memory_record_hashes":{}})
        common.seal(self.root/"pipeline-packet1.json",p)
        baseline={"count":5288,"previous":"retained"};common.seal(self.root/"pipeline-ledger-before1.json",baseline)
        ledger=self.root/"mock-ledger";ledger.mkdir();calls=[]
        def launch(ledger,category,out,packet,qualification,spec,resolution):
            path=self.root/("unit-launch-"+spec["args"][-1]+".json");common.seal(path,{"previous":"unit"});return path
        def run_worker(args,**kw):
            out=Path(args[-1]);index=int(common.read(str(out)+".spec.json")["args"][-1]);calls.append(index)
            data={"status":"passed","seconds":1.,"supervisor":{"elapsed_seconds":1.,"exit_code":0,"failure":None},"serializer_adoption":{"verified":True}}
            if index==5288:data.update(tests=1639,failed=0,skipped=0,binding_tests_passed=True)
            elif index==5289:data.update(pixels_complete=16,pixel_range=[480,496],certificates=[{}for _ in range(16)],all_histories_saved_losslessly=True)
            else:data.update(pixels_audited=16,pixel_range=[480,496],pixel_group_audit_complete=True,native_group_colour_and_mask_agreement=True,whole_image_complete=False)
            common.seal(out,data);return SimpleNamespace(returncode=0)
        def metadata(out,*args):return {"data":common.read(out),"receipt_sha256":"unit-receipt"}
        with patch.object(controller,"ROOT",self.root),patch.object(controller,"LEDGER",ledger),patch.object(controller,"packet",return_value=p),patch.object(controller,"verify_authority"),patch.object(controller,"quiet"),patch.object(controller,"psutil_ancestors",return_value=[]),patch.object(controller,"inventory",return_value=baseline),patch.object(controller,"snapshot",return_value=self.snapshot()),patch.object(controller,"verify_sources"),patch.object(controller.win,"lock_files",return_value=[]),patch.object(controller,"before_arrays"),patch.object(controller,"storage_inventory",return_value={"bytes":0,"files":0}),patch.object(controller,"admit",side_effect=launch),patch.object(controller.subprocess,"run",side_effect=run_worker),patch.object(controller,"metadata",side_effect=metadata),patch.object(terminal,"reconcile",side_effect=TimeoutError("terminal reserve")):
            self.assertEqual(controller.run(self.root/"unit-approval"),1)
        self.assertEqual(calls,[5288,5289,5290]);self.assertEqual(common.read(self.root/"pipeline-complete1.json")["completed_launches"],calls)
        refusal=common.read(self.root/"terminal-reconciliation-refusal.json")
        self.assertTrue(refusal["numerical_acceptance_forbidden"]);self.assertEqual(refusal["accepted_pixels"],480)
        receipt=common.read(self.root/"controller-terminal-engineering-receipt.json")
        self.assertEqual(receipt["status"],"stopped");self.assertIn("terminal reserve",receipt["terminal_error"])
    def test_closed_snapshot_types_refuse(self):
        snapshot=self.snapshot();snapshot["available_RAM_bytes"]=True
        with self.assertRaises(TypeError):common.resolve_task(self.contract(),snapshot,self.limits())
    def test_parent_and_terminal_deadlines_in_source(self):
        text=Path(__file__).with_name("route2_checkpoint.py").read_text()
        self.assertIn('terminal_reserve_seconds',text);self.assertIn('deadline=deadline',text)
        self.assertIn('controller-terminal-engineering-receipt.json',text)
    def mock_metadata(self,stopped=False):
        import route2_metadata as m
        root=self.root;ledger=root/"mock-ledger";ledger.mkdir();output=root/"output.json"
        contract=self.contract();contract["output"]="output.json"
        limits=self.limits();p={"base_launch_index":5288,"tasks":[contract,contract,contract],"limits":limits}
        common.seal(root/"pipeline-packet1.json",p);task=common.resolve_task(contract,self.snapshot(),limits)
        common.seal(root/"evidence.json",{"snapshot":self.snapshot(),"used":{"count":5289},"matching_other_heavy_commands":[],"independent_psutil_quiet_excluding_exact_admitted_controller":True})
        common.seal(root/"job-5289-resolved-task1.json",{"schema":"route2-fresh-resolution/1","packet_sha256":common.sha(root/"pipeline-packet1.json"),"task_contract_sha256":common.digest(contract),"snapshot":self.snapshot(),"task":task,"physical_evidence_path":str(root/"evidence.json"),"physical_evidence_sha256":common.sha(root/"evidence.json")})
        common.seal(str(output)+".spec.json",task["spec"])
        data={"status":"stopped"if stopped else"passed","seconds":1.,"supervisor":{"elapsed_seconds":1.,"exit_code":1 if stopped else 0,"failure":None,"actual_worker_identity_matches":True,"kernel_job_peak_process_commit_bytes":1,"sampled_peak_working_set_bytes":1}}
        if not stopped:
            data.update(pixels_complete=16,pixel_range=[480,496],all_histories_saved_losslessly=True)
            paths=[root/("pixel-"+str(i)+".json")for i in range(16)]+[root/"history.json"]
            rows=[]
            for path in paths:
                path.write_bytes(b"{}");rows.append({"status":"passed","path":str(path),"graph_still_same":True,"readback":{"bytes":2,"sha256":common.sha(path)},"stats":{"validated_unique_objects":1,"validated_container_edges":0,"serialized_fraction_occurrences":0}})
            journal=root/"adoption.jsonl";journal.write_text("".join(json.dumps(row)+"\n"for row in rows))
            data.update(certificates=[{"path":str(path),"sha256":common.sha(path)}for path in paths[:16]],chunk_history={"path":str(paths[-1]),"sha256":common.sha(paths[-1])},serializer_adoption={"verified":True,"calls":17,"journal":str(journal),"journal_sha256":common.sha(journal)})
        from report_journal142 import final_bytes
        output.write_bytes(final_bytes(data))
        Path(str(output)+".live.jsonl").write_text(json.dumps({"sequence":0,"status":data["status"],"seconds":1.})+"\n")
        launch={"index":5289,"category":task["category"],"output":str(output),"packet_sha256":common.sha(root/"pipeline-packet1.json"),"profile_sha256":None,"resolved_task_sha256":common.sha(root/"job-5289-resolved-task1.json"),"reserved_worker_seconds":18000,"reserved_supervised_seconds":18060,"previous":"prior"}
        (root/"profile2.json").write_bytes(b"{}");launch["profile_sha256"]=common.sha(root/"profile2.json")
        common.seal(ledger/"launch-5289.json",launch)
        finish={"launch_sha256":common.sha(ledger/"launch-5289.json"),"report_sha256":common.sha(output),"status":data["status"],"worker_seconds":1.,"supervised_seconds":1.}
        common.seal(ledger/"finish-5289.json",finish);common.seal(str(output)+".receipt.json",finish)
        return m,root,ledger,output,task
    def test_complete_mock_producer_receipt_chain(self):
        m,root,ledger,out,task=self.mock_metadata()
        with patch.object(m,"ROOT",root),patch.object(m,"LEDGER",ledger):
            self.assertTrue(m.metadata(out,task,5289,root/"pipeline-packet1.json","prior")["clean_supervision"])
    def test_changed_finish_hash_refuses(self):
        m,root,ledger,out,task=self.mock_metadata()
        finish=common.read(ledger/"finish-5289.json");finish["report_sha256"]="0"*64
        (ledger/"finish-5289.json").write_text(json.dumps(finish))
        with patch.object(m,"ROOT",root),patch.object(m,"LEDGER",ledger):
            with self.assertRaises(ValueError):m.metadata(out,task,5289,root/"pipeline-packet1.json","prior")
    def test_partial_failed_journal_is_preserved_cost_only(self):
        m,root,ledger,out,task=self.mock_metadata(stopped=True)
        Path(str(out)+".live.jsonl").write_bytes(b'{"torn":')
        with patch.object(m,"ROOT",root),patch.object(m,"LEDGER",ledger):
            with self.assertRaises(ValueError):m.metadata(out,task,5289,root/"pipeline-packet1.json","prior")
            result=m.failure_cost_metadata(out,task,5289,root/"pipeline-packet1.json","prior",ValueError("partial journal"))
            self.assertFalse(result["clean_supervision"]);self.assertTrue(result["numerical_acceptance_forbidden"])

    def test_missing_launch_authority_refuses_before_run(self):
        from route2_checkpoint import verify_authority
        with self.assertRaises(FileNotFoundError):verify_authority(self.root/"absent.json",{})
    def test_imports_do_not_call_numerical_mains(self):
        import route2_producer,route2_auditor,route2_gate,route2_checkpoint,route2_runtime,route2_terminal
        self.assertNotIn("torch",sys.modules);self.assertNotIn("numpy",sys.modules)
def main(root):
    global CASES
    started=time.perf_counter();root=Path(root).resolve();root.relative_to(common.QUAL.resolve());root.mkdir()
    CASES=root/"cases";CASES.mkdir();log=root/"tests.log"
    paths=list(Path(__file__).parent.glob("route2_*.py"))+[Path(__file__).with_name("adoption-transform.json"),adapter.PROTO]
    sources={str(p):common.sha(p)for p in paths}
    with log.open("x",encoding="utf-8")as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingTests))
    report={"status":"passed"if result.wasSuccessful()else"failed","tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),"sources":sources,"worker_pid":os.getpid(),"elapsed_seconds":time.perf_counter()-started,"original_source_reversal_verified":not result.failures and not result.errors,"no_numerical_main_run":True,"no_benchmark_run":True}
    common.seal(root/"result.json",report);print(json.dumps(report));return 0 if result.wasSuccessful()else 1
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    raise SystemExit(main(parser.parse_args().root))
