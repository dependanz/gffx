"""Bounded durable test metadata. Viewer arrays remain in gffx.dashboard, not this index."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from .catalog import PAGES, definition, page_for

def test_id(nodeid):
    return hashlib.sha256(nodeid.encode("utf-8")).hexdigest()

def plain(value, maximum=4096):
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError("invalid or oversized text")
    return value

def run_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("run needs a plain identifier")
    return value

def coverage_report(raw):
    """Normalize coverage.py's explicit executable-line sets; never infer native coverage."""
    files = []
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise ValueError("coverage needs files")
    if len(raw["files"]) > 10000:
        raise ValueError("coverage file limit")
    for path, item in sorted(raw["files"].items()):
        plain(path)
        if not isinstance(item, dict): raise ValueError("invalid coverage file")
        executed, missing = item.get("executed_lines"), item.get("missing_lines")
        if not isinstance(executed, list) or not isinstance(missing, list):
            raise ValueError("coverage needs executed_lines and missing_lines")
        if any(type(n) is not int or n < 1 for n in executed + missing):
            raise ValueError("invalid coverage line")
        if len(set(executed)) != len(executed) or len(set(missing)) != len(missing) or set(executed) & set(missing):
            raise ValueError("ambiguous coverage lines")
        summary = item.get("summary", {})
        branches, covered_branches = summary.get("num_branches"), summary.get("covered_branches")
        if branches is not None and (type(branches) is not int or type(covered_branches) is not int or not 0 <= covered_branches <= branches):
            raise ValueError("invalid branch counts")
        covered=summary.get('covered_lines',len(executed));total=summary.get('num_statements',len(executed)+len(missing))
        if type(covered) is not int or type(total) is not int or not 0<=covered<=total or total-covered!=len(missing):
            raise ValueError('invalid statement summary')
        files.append({"path": path, "covered": covered, "total": total,
                      "missing_lines": sorted(missing), "branches": branches, "covered_branches": covered_branches})
    return {"source": "coverage.py JSON (Python only)", "layer":"python", "files": files,
            "covered": sum(f["covered"] for f in files), "total": sum(f["total"] for f in files),
            "artifact_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()}

class Index:
    def __init__(self, path, *, max_tests=10000, max_checks=100, max_pages=100000,
                 history_keep=100, max_active=32, metadata_bytes=64*1024*1024):
        for name,value,minimum in (('history_keep',history_keep,0),('max_active',max_active,1),
                                   ('metadata_bytes',metadata_bytes,1024*1024)):
            if type(value) is not int or value < minimum:
                raise ValueError(name+' has an invalid retention limit')
        self.history_keep, self.max_active, self.metadata_bytes = history_keep,max_active,metadata_bytes
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_tests = max_tests
        self.max_checks, self.max_pages = max_checks, max_pages
        self.lock = threading.RLock()
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, kind TEXT, state TEXT, started REAL, updated REAL,
              exit_code INTEGER, provenance TEXT, coverage TEXT, diagnostic TEXT);
            CREATE TABLE IF NOT EXISTS tests(
              run TEXT, id TEXT, nodeid TEXT, outcome TEXT, completed INTEGER,
              phases TEXT, artifacts TEXT, PRIMARY KEY(run,id));
            CREATE INDEX IF NOT EXISTS tests_run ON tests(run);
            CREATE TABLE IF NOT EXISTS page_bindings(id TEXT PRIMARY KEY, page TEXT);
            CREATE TABLE IF NOT EXISTS demo_pages(run TEXT PRIMARY KEY, page TEXT);
            CREATE INDEX IF NOT EXISTS bindings_page ON page_bindings(page);
            CREATE TABLE IF NOT EXISTS checks(
              run TEXT, test TEXT, id TEXT, name TEXT, outcome TEXT, duration REAL,
              message TEXT, PRIMARY KEY(run,test,id));
            CREATE TABLE IF NOT EXISTS inspection_pages(
              id TEXT PRIMARY KEY, parent TEXT, title TEXT, kind TEXT, group_name TEXT,
              created_on TEXT);
            CREATE INDEX IF NOT EXISTS inspection_parent ON inspection_pages(parent);
            CREATE TABLE IF NOT EXISTS pins(run TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS retired_runs(id TEXT PRIMARY KEY, retired REAL);
            """)
            for page in PAGES:
                db.execute('INSERT OR IGNORE INTO inspection_pages VALUES(?,?,?,?,?,?)',
                           (page['id'],None,page['title'],page['kind'],page['group'],page['created_on']))
                db.execute('UPDATE inspection_pages SET title=?,kind=?,group_name=? WHERE id=?',
                           (page['title'],page['kind'],page['group'],page['id']))
            # Rebuild only the derived bindings on startup so catalog edits also apply to old
            # evidence. Never rewrite test outcomes, artifacts, or producer provenance.
            for row in db.execute("SELECT DISTINCT id,nodeid FROM tests WHERE id!='demo'"):
                db.execute("INSERT OR REPLACE INTO page_bindings VALUES(?,?)", (row['id'],page_for(row['nodeid'])))
                self._test_page(db,row['nodeid'])

    def _register_page(self, db, identifier, parent, title, kind, group):
        # Creation means first registration as an inspection page, not source-code age.
        inserted=db.execute('INSERT OR IGNORE INTO inspection_pages VALUES(?,?,?,?,?,?)',
                            (identifier,parent,title,kind,group,datetime.now(timezone.utc).date().isoformat())).rowcount
        if inserted and db.execute('SELECT count(*) FROM inspection_pages').fetchone()[0] > self.max_pages:
            raise ValueError('inspection page quota reached')

    def _test_page(self, db, nodeid):
        parent=page_for(nodeid)
        self._register_page(db,'test/'+test_id(nodeid),parent,nodeid.split('::')[-1].removeprefix('test_').replace('_',' '),
                            'test',definition(parent)['group'])
        # Catalog membership is a current derived view; creation dates and saved outcomes stay intact.
        db.execute('UPDATE inspection_pages SET parent=?,group_name=? WHERE id=?',
                   (parent,definition(parent)['group'],'test/'+test_id(nodeid)))

    def _db(self):
        # One short-lived connection per operation; serial writer, transactional quota rejection.
        import contextlib
        @contextlib.contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=2)
            db.row_factory = sqlite3.Row
            size=db.execute('PRAGMA page_size').fetchone()[0]
            requested=self.metadata_bytes//size
            actual=db.execute('PRAGMA max_page_count='+str(requested)).fetchone()[0]
            if actual > requested:
                db.close()
                raise ValueError('metadata storage exceeds configured budget; preserve database and review retention')
            try:
                with db: yield db
            except sqlite3.OperationalError as error:
                if 'full' in str(error).lower():
                    raise ValueError('metadata storage budget reached; preview eligible history cleanup; protected evidence is not discarded') from error
                raise
            finally: db.close()
        return connection()

    def accept(self, event):
        if not isinstance(event, dict) or len(json.dumps(event, allow_nan=False).encode()) > 1024 * 1024:
            raise ValueError("metadata event limit is 1 MiB")
        run = run_name(event.get("run")); kind = event.get("event"); now = time.time()
        with self.lock, self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
            if kind == "start":
                if db.execute('SELECT 1 FROM retired_runs WHERE id=?',(run,)).fetchone():
                    raise ValueError('execution identifier was retired; choose a fresh identifier')
                if row: raise ValueError("run already exists; choose a fresh identifier")
                mode = event.get("kind", "tests")
                if mode not in ("tests", "demo"): raise ValueError("unknown run kind")
                if db.execute("SELECT count(*) FROM runs WHERE state='running'").fetchone()[0] >= self.max_active:
                    raise ValueError('active execution limit reached; stale executions are not automatically finished')
                from gffx.dashboard._server import check_provenance
                provenance = check_provenance(event.get("provenance")) or {}
                db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)",
                           (run, mode, "running", now, now, None, json.dumps(provenance), None, None))
                return
            if row is None: raise KeyError("unknown run")
            if kind == "heartbeat":
                if row["state"] != "running": return
            elif kind == "collect":
                if row["state"] != "running" or row["kind"] != "tests": raise ValueError("not collecting tests")
                names = event.get("nodeids")
                if not isinstance(names, list): raise ValueError("nodeids needs a list")
                for name in names:
                    plain(name)
                    self._test_page(db,name)
                    db.execute("INSERT OR REPLACE INTO page_bindings VALUES(?,?)", (test_id(name),page_for(name)))
                    db.execute("INSERT OR IGNORE INTO tests VALUES(?,?,?,?,?,?,?)",
                               (run, test_id(name), name, "pending", 0, "{}", "[]"))
                if db.execute("SELECT count(*) FROM tests WHERE run=?", (run,)).fetchone()[0] > self.max_tests:
                    raise ValueError("test quota reached")
            elif kind in ('check_start','check_finish'):
                if row['state']!='running' or row['kind']!='tests': raise ValueError('run is not running tests')
                nodeid=plain(event.get('nodeid')); tid=test_id(nodeid)
                item=db.execute('SELECT completed FROM tests WHERE run=? AND id=?',(run,tid)).fetchone()
                if item is None: raise KeyError('test not collected')
                if item['completed']: raise ValueError('test has completed')
                name=plain(event.get('name'),200); cid=test_id(name)
                check=db.execute('SELECT * FROM checks WHERE run=? AND test=? AND id=?',(run,tid,cid)).fetchone()
                if kind=='check_start':
                    if check: raise ValueError('duplicate check name in test execution')
                    if db.execute('SELECT count(*) FROM checks WHERE run=? AND test=?',(run,tid)).fetchone()[0]>=self.max_checks:
                        raise ValueError('named check quota reached')
                    self._register_page(db,'check/'+tid+'/'+cid,'test/'+tid,name,'check',definition(page_for(nodeid))['group'])
                    db.execute('INSERT INTO checks VALUES(?,?,?,?,?,?,?)',(run,tid,cid,name,'pending',0,''))
                else:
                    if check is None: raise KeyError('check not started')
                    if check['outcome']!='pending': raise ValueError('check already finished')
                    outcome=event.get('outcome'); duration=event.get('duration',0)
                    if outcome not in ('passed','failed','skipped','interrupted'): raise ValueError('invalid check outcome')
                    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0: raise ValueError('invalid check duration')
                    db.execute('UPDATE checks SET outcome=?,duration=?,message=? WHERE run=? AND test=? AND id=?',
                               (outcome,duration,str(event.get('message') or '')[:4000],run,tid,cid))
            elif kind in ("report", "artifact"):
                nodeid = plain(event.get("nodeid"))
                item = db.execute("SELECT * FROM tests WHERE run=? AND id=?", (run, test_id(nodeid))).fetchone()
                if item is None: raise KeyError("test not collected")
                if kind == "report":
                    if row["state"] != "running": raise ValueError("run has ended")
                    when, outcome = event.get("when"), event.get("outcome")
                    if when not in ("setup", "call", "teardown") or outcome not in ("passed", "failed", "skipped", "xfailed", "xpassed"):
                        raise ValueError("invalid report")
                    duration = event.get("duration", 0)
                    if isinstance(duration, bool) or not isinstance(duration, (int,float)) or not math.isfinite(duration) or duration < 0:
                        raise ValueError("invalid duration")
                    phases = json.loads(item["phases"])
                    phases[when] = {"outcome": outcome, "duration": duration, "message": str(event.get("message") or "")[:4000]}
                    outcomes = [v["outcome"] for v in phases.values()]
                    result = next((o for o in ("failed", "xpassed", "xfailed", "skipped") if o in outcomes),
                                  "passed" if "call" in phases else "pending")
                    db.execute("UPDATE tests SET outcome=?,completed=?,phases=? WHERE run=? AND id=?",
                               (result, int("teardown" in phases), json.dumps(phases), run, item["id"]))
                else:
                    artifact = self._artifact(event.get("artifact"))
                    if artifact.get('check') and not db.execute('SELECT 1 FROM checks WHERE run=? AND test=? AND id=?',(run,item['id'],artifact['check'])).fetchone():
                        raise KeyError('artifact check is not part of this test')
                    artifacts = json.loads(item["artifacts"])
                    artifacts = [a for a in artifacts if a["path"] != artifact["path"]]
                    if len(artifacts) >= 32: raise ValueError("artifact metadata limit")
                    artifacts.append(artifact)
                    db.execute("UPDATE tests SET artifacts=? WHERE run=? AND id=?", (json.dumps(artifacts),run,item["id"]))
            elif kind == "demo":
                if row["kind"] != "demo": raise ValueError("not a demo run")
                page=event.get('page','dashboard/panel-demo')
                if definition(page)['kind']!='demo': raise ValueError('demo requires a demonstration page')
                previous=db.execute('SELECT page FROM demo_pages WHERE run=?',(run,)).fetchone()
                if previous and previous['page']!=page: raise ValueError('demo page cannot change within an execution')
                artifacts = [self._artifact(a) for a in event.get("artifacts", [])]
                if len(artifacts) > 32: raise ValueError("demo artifact limit")
                db.execute('INSERT OR REPLACE INTO demo_pages VALUES(?,?)',(run,page))
                # One explicitly non-test page per execution, never part of test counts.
                db.execute("INSERT OR REPLACE INTO tests VALUES(?,?,?,?,?,?,?)",
                           (run,"demo",str(event.get("title","Demonstration"))[:200],"demo",0,"{}",json.dumps(artifacts)))
            elif kind == "coverage":
                coverage = event.get("coverage")
                from .coverage_formats import LAYERS
                if not isinstance(coverage, dict) or coverage.get("source") not in ("coverage.py JSON (Python only)","LCOV"):
                    raise ValueError("unsupported coverage source")
                layer=coverage.get('layer','python')
                if layer not in LAYERS:raise ValueError('unknown coverage layer')
                if coverage['source'].startswith('coverage.py') and layer!='python':raise ValueError('Python report cannot measure native code')
                inventory=coverage.get('inventory',{})
                if not isinstance(inventory,dict) or set(inventory)-set(LAYERS):raise ValueError('invalid inventory')
                if sum(len(v) for v in inventory.values() if isinstance(v,list))>10000:raise ValueError('inventory limit')
                for paths in inventory.values():
                    if not isinstance(paths,list):raise ValueError('invalid inventory')
                    for path in paths:plain(path)
                files = coverage.get("files")
                if not isinstance(files, list) or len(files) > 10000: raise ValueError("invalid coverage files")
                for f in files:
                    if not isinstance(f, dict): raise ValueError("invalid coverage file")
                    plain(f.get("path"))
                    if type(f.get("covered")) is not int or type(f.get("total")) is not int or not 0 <= f["covered"] <= f["total"]:
                        raise ValueError("invalid coverage counts")
                    missing = f.get("missing_lines")
                    if not isinstance(missing,list) or any(type(n) is not int or n < 1 for n in missing) or len(set(missing)) != len(missing) or len(missing) != f["total"]-f["covered"]:
                        raise ValueError("invalid missing lines")
                    branches, covered_branches = f.get("branches"), f.get("covered_branches")
                    if branches is not None and (type(branches) is not int or type(covered_branches) is not int or not 0 <= covered_branches <= branches):
                        raise ValueError("invalid branch counts")
                    functions,covered_functions=f.get('functions'),f.get('covered_functions')
                    if functions is not None and (type(functions) is not int or type(covered_functions) is not int or not 0<=covered_functions<=functions):
                        raise ValueError('invalid function counts')
                if coverage.get("covered") != sum(f["covered"] for f in files) or coverage.get("total") != sum(f["total"] for f in files):
                    raise ValueError("coverage totals disagree")
                if not re.fullmatch(r"[a-f0-9]{64}",str(coverage.get("artifact_sha256",""))):
                    raise ValueError("coverage needs an artifact hash")
                existing=db.execute('SELECT coverage FROM runs WHERE id=?',(run,)).fetchone()[0]
                old=json.loads(existing) if existing else None
                reports={r.get('layer','python'):r for r in (old.get('reports',[old]) if old else [])}
                reports[layer]={k:v for k,v in coverage.items() if k!='reports'}
                combined={**reports.get('python',reports[layer]),'reports':list(reports.values())}
                db.execute("UPDATE runs SET coverage=? WHERE id=?", (json.dumps(combined),run))
            elif kind == "diagnostic":
                db.execute("UPDATE runs SET diagnostic=? WHERE id=?", (str(event.get("message",""))[:4000],run))
            elif kind == "finish":
                if row["state"] != "running": raise ValueError("run already finished")
                code = event.get("exit_code")
                if type(code) is not int: raise ValueError("invalid exit code")
                state = "finished" if row["kind"] == "demo" else {0:"passed",1:"failed",2:"interrupted"}.get(code,"error")
                pending = db.execute("SELECT count(*) FROM tests WHERE run=? AND (completed=0 OR outcome='pending')", (run,)).fetchone()[0]
                if row["kind"] == "tests" and code == 0 and pending: state = "incomplete"
                failed = db.execute("SELECT count(*) FROM tests WHERE run=? AND outcome='failed'", (run,)).fetchone()[0]
                if row["kind"] == "tests" and code == 0 and failed: state = "inconsistent"
                db.execute("UPDATE runs SET state=?,exit_code=? WHERE id=?", (state,code,run))
            else: raise ValueError("unknown event")
            db.execute("UPDATE runs SET updated=? WHERE id=?", (now,run))

    @staticmethod
    def _artifact(artifact):
        if not isinstance(artifact, dict): raise ValueError("invalid artifact")
        path = plain(artifact.get("path"), 512)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", path): raise ValueError("invalid artifact path")
        if artifact.get("kind") not in ("mesh","image","plot","text","record"): raise ValueError("invalid panel kind")
        result={"path": path, "kind": artifact["kind"]}
        if 'check' in artifact:
            if not isinstance(artifact['check'],str) or not re.fullmatch('[a-f0-9]{64}',artifact['check']): raise ValueError('invalid check identity')
            result['check']=artifact['check']
        if 'label' in artifact: result['label']=plain(artifact['label'],200)
        return result

    def summary(self, run, *, now=None):
        with self.lock, self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
            if row is None: raise KeyError("unknown run")
            result = dict(row)
            result["run"] = result.pop("id")
            result['pinned'] = bool(db.execute('SELECT 1 FROM pins WHERE run=?',(run,)).fetchone())
            result["provenance"] = json.loads(result["provenance"])
            result["coverage"] = json.loads(result["coverage"]) if result["coverage"] else None
            if result["coverage"]:
                result["coverage"].pop("files", None)
                result["coverage"].pop("reports", None)
            counts = db.execute("SELECT outcome,count(*) FROM tests WHERE run=? AND outcome!='demo' GROUP BY outcome", (run,)).fetchall()
            result["counts"] = dict(counts)
            # Fail closed for older stored runs accepted before this report-integrity guard.
            if result['state']=='passed' and result['counts'].get('pending'): result['state']='incomplete'
            result["collected"] = sum(result["counts"].values())
            result["completed"] = db.execute("SELECT count(*) FROM tests WHERE run=? AND completed=1", (run,)).fetchone()[0]
            result["stale"] = result["state"] == "running" and (time.time() if now is None else now) - result["updated"] > 15
            return result

    def runs(self, *, offset=0, limit=100):
        offset=int(offset);limit=int(limit)
        if offset<0 or not 1<=limit<=100:raise ValueError('invalid history pagination')
        with self.lock, self._db() as db:
            names = [r[0] for r in db.execute("SELECT id FROM runs ORDER BY started DESC,rowid DESC LIMIT ? OFFSET ?",(limit,offset))]
        return [self.summary(name) for name in names]

    def history_status(self):
        from .retention import plan
        with self.lock,self._db() as db:
            result=plan(db,self.history_keep)
            size=db.execute('PRAGMA page_size').fetchone()[0]
            pages=db.execute('PRAGMA page_count').fetchone()[0]
            free=db.execute('PRAGMA freelist_count').fetchone()[0]
            return dict(result,metadata_file_bytes=pages*size,metadata_used_bytes=(pages-free)*size,
                        metadata_limit_bytes=self.metadata_bytes,max_active=self.max_active)

    def pin(self, run, value):
        if type(value) is not bool:raise ValueError('pin must be a boolean')
        with self.lock,self._db() as db:
            if not db.execute('SELECT 1 FROM runs WHERE id=?',(run_name(run),)).fetchone():raise KeyError('unknown execution')
            if value:db.execute('INSERT OR IGNORE INTO pins VALUES(?)',(run,))
            else:db.execute('DELETE FROM pins WHERE run=?',(run,))

    def retire(self, names):
        from .retention import retire
        with self.lock,self._db() as db:retire(db,names,self.history_keep)

    def is_retired(self, run):
        with self.lock,self._db() as db:
            return bool(db.execute('SELECT 1 FROM retired_runs WHERE id=?',(run,)).fetchone())

    def tests(self, run, *, offset=0, search="", outcome=""):
        offset = int(offset)
        if offset < 0 or offset > self.max_tests: raise ValueError("invalid offset")
        self.summary(run)
        with self.lock, self._db() as db:
            args = (run, "%" + str(search)[:200] + "%", outcome, outcome)
            where = "run=? AND nodeid LIKE ? AND (?='' OR outcome=?)"
            total = db.execute("SELECT count(*) FROM tests WHERE " + where,args).fetchone()[0]
            rows = db.execute("SELECT id,nodeid,outcome,completed FROM tests WHERE " + where + " ORDER BY nodeid LIMIT 50 OFFSET ?",(*args,offset)).fetchall()
            return {"items":[dict(r) for r in rows],"total":total,"offset":offset}

    def test(self, run, identifier):
        with self.lock, self._db() as db:
            row = db.execute("SELECT * FROM tests WHERE run=? AND id=?", (run,identifier)).fetchone()
            if row is None: raise KeyError("unknown test/page")
            result = dict(row)
            result["phases"] = json.loads(result["phases"])
            result["artifacts"] = json.loads(result["artifacts"])
            return result

    def coverage(self, run, *, offset=0, layer='python'):
        from .coverage_formats import LAYERS
        if layer not in LAYERS:raise ValueError('unknown coverage layer')
        if not 0 <= int(offset) <= 10000: raise ValueError("invalid offset")
        with self.lock, self._db() as db:
            row = db.execute("SELECT coverage FROM runs WHERE id=?", (run,)).fetchone()
            if row is None: raise KeyError("unknown run")
            stored=json.loads(row[0]) if row[0] else None
            reports={r.get('layer','python'):r for r in (stored.get('reports',[stored]) if stored else [])}
            inventory=stored.get('inventory',{}) if stored else {}
            layers=[{'id':key,'label':label,'measured':key in reports,'source_files':len(inventory.get(key,[]))} for key,label in LAYERS.items()]
            if layer not in reports:return {'items':[],'total':0,'measured':False,'layer':layer,'layers':layers,'unmeasured_files':inventory.get(layer,[])}
            result={**reports[layer]};files=result.pop('files');result.pop('reports',None)
            result.pop('inventory',None)
            missing=sorted(set(inventory.get(layer,[]))-{f['path'] for f in files})
            return {**result,'layer':layer,'layers':layers,'unmeasured_files':missing,"items":files[int(offset):int(offset)+50], "total_files":len(files), "measured":True}

    def pages(self):
        """Small permanent catalog with page-local summaries, not run navigation."""
        return [self.page(p['id'], summary_only=True) for p in PAGES]

    def directory(self, *, offset=0, limit=50):
        offset=int(offset);limit=int(limit)
        if not 0<=offset<=self.max_pages or not 1<=limit<=50: raise ValueError('invalid page pagination')
        with self.lock,self._db() as db:
            total=db.execute('SELECT count(*) FROM inspection_pages').fetchone()[0]
            rows=db.execute('SELECT id,parent,title,kind,group_name AS "group",created_on FROM inspection_pages ORDER BY created_on DESC,title COLLATE NOCASE,id LIMIT ? OFFSET ?',(limit,offset))
            return {'items':[dict(r) for r in rows],'total':total}

    def detail_page(self, identifier, *, run=None):
        """A test/check page always follows the newest collection of its parent test."""
        from .explanations import explain
        with self.lock,self._db() as db:
            registered=db.execute('SELECT id,parent,title,kind,group_name AS "group",created_on FROM inspection_pages WHERE id=?',(identifier,)).fetchone()
            if registered is None: raise KeyError('unknown inspection page')
            page=dict(registered);parts=identifier.split('/');tid=parts[1]
            args=(tid,)+( (run_name(run),) if run else ())
            rows=db.execute('SELECT r.id FROM runs r JOIN tests t ON t.run=r.id WHERE t.id=?'+(' AND r.id=?' if run else '')+' ORDER BY r.started DESC,r.rowid DESC LIMIT 10',args).fetchall()
            if not rows: raise KeyError('referenced execution has no evidence for test')
            selected=rows[0]['id'];evidence=self.summary(selected);item=self.test(selected,tid)
            checks=[dict(c) for c in db.execute('SELECT * FROM checks WHERE run=? AND test=? ORDER BY name',(selected,tid))]
            for check in checks: check['page']='check/'+tid+'/'+check['id']
            page.update(description=item['nodeid'],evidence=evidence,history=[self.summary(r['id']) for r in rows],
                        counts={item['outcome']:1},collected=1,completed=item['completed'],
                        state=item['outcome'],items=[item],total=1,artifacts=item['artifacts'],checks=checks,
                        parent_outcome=item['outcome'])
            if page['kind']=='check':
                check=next((c for c in checks if c['id']==parts[2]),None)
                page.update(state=check['outcome'] if check else 'not measured',checks=[check] if check else [],
                            counts={},items=[],total=0,collected=0,completed=0,
                            artifacts=[a for a in item['artifacts'] if a.get('check')==parts[2]])
            if page['state']=='pending' and evidence['state']!='running':page['state']='incomplete'
            page['nodeid']=item['nodeid']
            page['explanation']=explain(item['nodeid'],page['title'] if page['kind']=='check' else None)
            page['description']=page['explanation']['purpose']
            return page

    def page(self, identifier, *, run=None, offset=0, search='', outcome='', summary_only=False):
        if identifier.startswith(('test/','check/')):
            return self.detail_page(identifier,run=run)
        page = definition(identifier)
        if page['kind']=='article':
            if run:raise KeyError('Learning articles have no execution history')
            from pathlib import Path
            from .build_lessons import asset_name
            content=json.loads((Path(__file__).parent/'static/learn'/asset_name(identifier)).read_text(encoding='utf-8'))
            if content['id']!=identifier:raise KeyError('Article content does not match its page')
            return {**page,'state':'article','evidence':None,'counts':{},'collected':0,
                    'completed':0,'items':[],'total':0,'artifacts':[],'history':[],
                    'updated_on':content['updated_on'],'source_sha256':content['source_sha256'],
                    **({} if summary_only else {'article_html':content['html']})}
        offset = int(offset)
        if not 0 <= offset <= self.max_tests: raise ValueError('invalid offset')
        # Selection is by execution start, not last update/coverage import or last success.
        # A newer collected-but-pending subset supersedes old success on the same page.
        with self.lock, self._db() as db:
            condition = ("r.kind='demo' AND EXISTS(SELECT 1 FROM tests t WHERE t.run=r.id AND t.id='demo') AND COALESCE((SELECT page FROM demo_pages d WHERE d.run=r.id),'dashboard/panel-demo')=?"
                         if page['kind'] == 'demo' else
                         "r.kind='tests' AND EXISTS(SELECT 1 FROM tests t JOIN page_bindings b ON b.id=t.id WHERE t.run=r.id AND b.page=?)")
            args = (identifier,)
            if run:
                condition += ' AND r.id=?'; args += (run_name(run),)
            rows = db.execute('SELECT r.id FROM runs r WHERE '+condition+' ORDER BY r.started DESC,r.rowid DESC LIMIT 10',args).fetchall()
            if not rows:
                if run: raise KeyError('Referenced execution has no evidence for this page')
                return {**page,'state':'not measured','evidence':None,'counts':{},'collected':0,
                        'completed':0,'items':[],'total':0,'artifacts':[],'history':[]}
            run = rows[0]['id']
            evidence = self.summary(run)
            history = [] if summary_only else [self.summary(r['id']) for r in rows]
            if page['kind'] == 'demo':
                detail = self.test(run,'demo')
                return {**page,'state':'demonstration','evidence':evidence,'counts':{},'collected':0,
                        'completed':0,'items':[],'total':0,'artifacts':[] if summary_only else detail['artifacts'],'history':history}
            where = 't.run=? AND b.page=?'
            counts = dict(db.execute('SELECT t.outcome,count(*) FROM tests t JOIN page_bindings b ON b.id=t.id WHERE '+where+' GROUP BY t.outcome',(run,identifier)))
            completed = db.execute('SELECT count(*) FROM tests t JOIN page_bindings b ON b.id=t.id WHERE '+where+' AND t.completed=1',(run,identifier)).fetchone()[0]
            collected = sum(counts.values())
            state = ('running' if evidence['state']=='running' else 'incomplete' if completed < collected
                     else 'failed' if counts.get('failed') or counts.get('xpassed') else
                     'expected failures' if counts.get('xfailed') else 'with skips' if counts.get('skipped') else 'passed')
            if evidence['state'] in ('interrupted','error','inconsistent','incomplete'): state = evidence['state']
            result = {**page,'state':state,'evidence':evidence,'counts':counts,'collected':collected,
                      'completed':completed,'items':[],'total':collected,'artifacts':[],'history':history}
            if summary_only: return result
            # Filter only the detail list, never the page's status or evidence selection.
            args = (run,identifier,'%'+str(search)[:200]+'%',outcome,outcome)
            filtered = where + " AND t.nodeid LIKE ? AND (?='' OR t.outcome=?)"
            result['total'] = db.execute('SELECT count(*) FROM tests t JOIN page_bindings b ON b.id=t.id WHERE '+filtered,args).fetchone()[0]
            entries = db.execute('SELECT t.* FROM tests t JOIN page_bindings b ON b.id=t.id WHERE '+filtered+' ORDER BY t.nodeid LIMIT 50 OFFSET ?',(*args,offset))
            for entry in entries:
                item = dict(entry); item['phases']=json.loads(item['phases']); item['artifacts']=json.loads(item['artifacts'])
                result['items'].append(item)
            # Always mount at most 32 panels. Pinned detail links can select a specific test.
            for entry in db.execute('SELECT t.artifacts FROM tests t JOIN page_bindings b ON b.id=t.id WHERE '+where+' ORDER BY t.nodeid',(run,identifier)):
                result['artifacts'].extend(json.loads(entry['artifacts']))
                if len(result['artifacts']) >= 32: break
            result['artifacts'] = result['artifacts'][:32]
            return result

    def latest_coverage(self, *, run=None, offset=0, layer='python'):
        with self.lock, self._db() as db:
            row = db.execute("SELECT id FROM runs WHERE kind='tests'" + (' AND id=?' if run else '') +
                             ' ORDER BY started DESC,rowid DESC LIMIT 1', (run,) if run else ()).fetchone()
            if row is None:
                if run: raise KeyError('Referenced execution not found')
                return {'measured':False,'items':[],'evidence':None}
            return {**self.coverage(row['id'],offset=offset,layer=layer),'evidence':self.summary(row['id'])}
