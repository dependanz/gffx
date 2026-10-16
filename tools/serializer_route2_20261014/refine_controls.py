from pathlib import Path
import json,ast
s=Path("C:/Users/Danze/AppData/Local/Temp/gffx-route2-prepare-20261014-01")
f=s/"route2_serializer.py";v=f.read_text()
v=v.replace("def dump_exact(value,out):","def _dump_exact_bound(value,out):",1)
v=v.replace('try:return call()\n        finally:row["phases"][label]', 'try:\n            from route2_phases import current\n            recorder=current()\n            if recorder is None:return call()\n            name={"validation":"serializer_graph_validation","encoding_write":"serializer_json_encode_write"}.get(label,"serializer_"+label)\n            with recorder.measure(name):return call()\n        finally:row["phases"][label]',1)
v=v.replace('def __init__(self):self.bytes=0;self.hash=hashlib.sha256()','def __init__(self):self.bytes=0;self.other_bytes=0;self.hash=hashlib.sha256()',1)
v=v.replace('c.checkpoint();data=text.encode("utf-8");self.bytes+=len(data)\n            if self.bytes+16*GIB+64*1024**2>128*GIB:raise OSError("output forecast exceeds combined allowance")\n            self.hash.update(data);return out.write(text)','c.checkpoint();data=text.encode("utf-8");next_bytes=self.bytes+len(data)\n            if self.other_bytes+next_bytes+16*GIB+64*1024**2>128*GIB:raise OSError("output forecast exceeds combined allowance")\n            count=out.write(text);self.bytes=next_bytes;self.hash.update(data);return count',1)
v=v.replace('        c.check()\n        prototype.validate_exact_graph','        writer.other_bytes=c.check()\n        prototype.validate_exact_graph',1)
v += '\ndef dump_exact(value,out):\n    from route2_phases import current\n    recorder=current()\n    if recorder is None:return _dump_exact_bound(value,out)\n    with recorder.measure("serializer_total"):return _dump_exact_bound(value,out)\n'
f.write_text(v)
f=s/"route2_runtime.py";v=f.read_text()
v=v.replace('data=read(args.output)if args.output.exists()else{"status":"stopped","phase":"abrupt-worker-stop","error":failure or"worker ended without terminal report"}','''raw=args.output.read_bytes()if args.output.exists()else None
        if raw is not None:
            with Path(str(args.output)+".worker-final.json").open("xb")as out:out.write(raw)
        try:data=json.loads(raw)if raw is not None else None
        except (json.JSONDecodeError,UnicodeDecodeError):data=None
        if data is None:
            failure=failure or"missing/incomplete worker terminal report"
            data={"status":"stopped","phase":"abrupt-worker-stop","error":failure}''')
v=v.replace('''if args.output.exists():
            with Path(str(args.output)+".worker-final.json").open("xb")as out:out.write(args.output.read_bytes())
        elif not Path(str(args.output)+".live.jsonl").exists():''','''if not Path(str(args.output)+".live.jsonl").exists():''')
f.write_text(v)
f=s/"route2_gate.py";v=f.read_text();v=v.replace("b['failures']==b['errors']==b['skips']==0","b['failures']==b['errors']==b['skips']==0 and b['tests']==binding['tests']")
f.write_text(v)
items=json.loads((s/"adoption-transform.json").read_text());items[-1]["replacements"][-1][1]=items[-1]["replacements"][-1][1].replace("b['failures']==b['errors']==b['skips']==0","b['failures']==b['errors']==b['skips']==0 and b['tests']==binding['tests']")
(s/"adoption-transform.json").write_text(json.dumps(items,indent=2))
f=s/"route2_binding_tests.py";v=f.read_text();needle='    def test_original_identity_and_aliases_held(self):'
addition='''    def test_other_outputs_count_in_prewrite_budget(self):
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
'''
assert needle in v;v=v.replace(needle,addition+needle,1);f.write_text(v)
for f in s.glob("route2_*.py"):ast.parse(f.read_text(),str(f))
print("Ordinary control-path and combined-budget corrections compiled.")
