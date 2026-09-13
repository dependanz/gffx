"""Tester-only evidence retention; never changes general dashboard live buffers."""
import json
import time


def plan(db, keep):
    rows = db.execute('SELECT rowid AS ordinal,* FROM runs ORDER BY started DESC,rowid DESC').fetchall()
    protected = {r['id'] for r in rows if r['state']=='running'}
    protected.update(r[0] for r in db.execute('SELECT run FROM pins'))
    # Preserve latest collected AND latest completed evidence for each individual test.
    for condition in ('', 'WHERE t.completed=1'):
        seen=set()
        for row in db.execute('SELECT t.id,t.run FROM tests t JOIN runs r ON r.id=t.run '+condition+
                              ' ORDER BY r.started DESC,r.rowid DESC'):
            if row['id']=='demo': continue
            if row['id'] not in seen: protected.add(row['run']);seen.add(row['id'])
    # A check can disappear from a newer test. Keep its old evidence accessible explicitly,
    # without making the current page fall back to that old success.
    seen=set()
    for row in db.execute("SELECT c.test,c.id,c.run FROM checks c JOIN runs r ON r.id=c.run "
                          "WHERE c.outcome!='pending' ORDER BY r.started DESC,r.rowid DESC"):
        key=(row['test'],row['id'])
        if key not in seen:protected.add(row['run']);seen.add(key)
    seen=set()
    for row in db.execute("SELECT r.id,COALESCE(d.page,'dashboard/panel-demo') AS page FROM runs r "
                          "JOIN tests t ON t.run=r.id AND t.id='demo' LEFT JOIN demo_pages d ON d.run=r.id "
                          "ORDER BY r.started DESC,r.rowid DESC"):
        for key in ((row['page'],'latest'),):
            if key not in seen:protected.add(row['id']);seen.add(key)
    seen=set()
    for r in rows:
        if r['kind']=='demo' and r['state']=='finished':
            page=db.execute('SELECT page FROM demo_pages WHERE run=?',(r['id'],)).fetchone()
            key=('demo',page[0] if page else 'dashboard/panel-demo')
            if key not in seen:protected.add(r['id']);seen.add(key)
        if r['kind']=='tests' and 'latest-tests' not in seen:
            protected.add(r['id']);seen.add('latest-tests')
        cov=json.loads(r['coverage']) if r['coverage'] else None
        for report in (cov.get('reports',[cov]) if cov else []):
            key=('coverage',report.get('layer','python'))
            if key not in seen:protected.add(r['id']);seen.add(key)
    unprotected=[r['id'] for r in rows if r['id'] not in protected]
    return dict(executions=len(rows), active=sum(r['state']=='running' for r in rows),
                protected=sorted(protected), history_keep=keep,
                eligible=list(reversed(unprotected[keep:])),
                retired=db.execute('SELECT count(*) FROM retired_runs').fetchone()[0])


def retire(db, names, keep):
    eligible=set(plan(db,keep)['eligible'])
    if len(names)!=len(set(names)) or any(n not in eligible for n in names):
        raise ValueError('retention selection contains protected or ineligible evidence')
    for name in names:
        for table in ('tests','checks','demo_pages','pins'):
            db.execute('DELETE FROM '+table+' WHERE run=?',(name,))
        db.execute('DELETE FROM runs WHERE id=?',(name,))
        db.execute('INSERT INTO retired_runs VALUES(?,?)',(name,time.time()))
