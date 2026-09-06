import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const base = new URL('../../tools/testing_dashboard/static/', import.meta.url);
const model = await import(new URL('model.js', base));
test('readable references keep stable IDs in links, not in the display label', () => {
  assert.equal(model.referenceLabel('test/abcdef','Build tools use approved revisions'),'gffx · Build tools use approved revisions');
  assert.equal(model.referenceLabel('test/abcdef',null),'gffx · Test page');
});
test('test guides precede artifacts and render as safe readable text', async () => {
  const html=await readFile(new URL('index.html',base),'utf8');
  const app=await readFile(new URL('app.js',base),'utf8');
  assert.ok(html.indexOf('id="explanation"')<html.indexOf('id="evidence"') && html.includes('id="explanation"'));
  for(const label of ['What this checks','How it checks it','What a pass means']) assert.ok(app.includes(label));
  assert.match(app,/referenceLabel\(/);
  assert.ok(!app.includes('innerHTML'));
});
test('guide rendering preserves open details on refresh and copying uses the loaded title', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  class Element {
    constructor(tag){this.tag=tag;this.children=[];}
    setAttribute(){} append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=children;}focus(){}select(){}
  }
  const elements=new Map(),previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator');let copied;
  globalThis.document={createElement:tag=>new Element(tag),getElementById:id=>{
    if(!elements.has(id))elements.set(id,new Element('div'));return elements.get(id);
  }};
  globalThis.location={href:'https://x/testing/'};
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{writeText:async value=>{copied=value;}}}});
  try {
    const app=Object.create(TestingApp.prototype);
    app.selected={page:'test/abc',run:'',path:null,step:null};
    app.header('Build tools use approved revisions','Purpose','Infrastructure');
    const detail={id:'test/abc',nodeid:'test_file.py::test_pins',explanation:{purpose:'<b>literal text</b>',method:'Compare five entries',limits:'Not a security audit'}};
    app.explanation(detail);
    const box=elements.get('explanation'),technical=box.children.at(-1);technical.open=true;
    assert.equal(box.children[0].children[1].textContent,'<b>literal text</b>');
    app.explanation(detail);assert.equal(box.children.at(-1),technical);assert.ok(technical.open);
    await app.copyReference();assert.match(copied,/^gffx · Build tools use approved revisions\n/);
    assert.equal(new URL(copied.split('\n')[1]).searchParams.get('page'),'test/abc');
    app.selected.page='coverage';app.header('Code coverage','Purpose','Evidence');assert.ok(box.hidden);
  } finally {
    for(const key of ['document','location'])delete globalThis[key];
    if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);else delete globalThis.navigator;
  }
});
test('theme defaults to light, remembers dark, and tolerates unavailable storage', () => {
  const root={dataset:{}}, control={setAttribute(k,v){this[k]=v;}};
  const values=new Map(), storage={getItem:k=>values.get(k),setItem:(k,v)=>values.set(k,v)};
  model.setupTheme(root,control,storage);
  assert.equal(root.dataset.theme,'light');
  control.onclick();assert.equal(root.dataset.theme,'dark');
  assert.equal(control['aria-label'],'Switch to light mode');
  model.setupTheme(root,control,storage);assert.equal(root.dataset.theme,'dark');
  model.setupTheme(root,control,{getItem(){throw Error('denied');},setItem(){throw Error('denied');}});
  assert.equal(root.dataset.theme,'light');control.onclick();assert.equal(root.dataset.theme,'dark');
});
test('check pages report their own state without counting as zero or extra tests', () => {
  assert.equal(model.pageLabel({kind:'check',state:'passed',evidence:{}}),'passed · named check');
  assert.equal(model.pageLabel({kind:'check',state:'not measured',evidence:{}}),'not measured · named check');
  assert.equal(model.pageLabel({kind:'test',state:'pending',evidence:{},completed:0,collected:1}),'pending · 0/1 checked');
});
test('feature and test pages keep check-owned visuals on child pages unless explicitly linked', () => {
  const artifacts=[{path:'parent'},{path:'child',check:'c'}];
  assert.deepEqual(model.pageArtifacts({kind:'test',artifacts},null),[artifacts[0]]);
  assert.deepEqual(model.pageArtifacts({kind:'tests',artifacts},null),[artifacts[0]]);
  assert.deepEqual(model.pageArtifacts({kind:'check',artifacts},null),artifacts);
  assert.deepEqual(model.pageArtifacts({kind:'test',artifacts},'child'),artifacts);
});
test('stable references preserve run, test and pinned artifact without unrelated URL state', () => {
  const ref = model.reference('https://example.test/testing/?secret=bad', {run:'abc',page:'test:hash',path:'tests/hash/mesh',step:0});
  assert.ok(!ref.includes('secret')); assert.equal(new URL(ref).searchParams.get('step'),'0');
  assert.equal(model.selection(new URL(ref)).page, 'test:hash');
  assert.throws(()=>model.selection(new URL('https://x/testing/?step=-1')));
});
test('unknown coverage and stale progress are not green success', () => {
  assert.equal(model.coverageLabel(null), 'Not measured');
  assert.equal(model.coverageLabel({covered:0,total:0}), 'No executable lines measured');
  assert.match(model.progressLabel({state:'running',stale:true,completed:2,collected:3}), /stale/i);
  assert.match(model.progressLabel({state:'running',stale:false,completed:2,collected:3}), /2.*3/);
});
test('dedicated shell uses shared panels, safe text and mobile navigation', async () => {
  const app = await readFile(new URL('app.js',base),'utf8');
  const html = await readFile(new URL('index.html',base),'utf8');
  const css = await readFile(new URL('app.css',base),'utf8');
  assert.match(app,/import.*PANEL_KINDS.*from ['"]\/app.js/);
  assert.ok(!app.includes('innerHTML'));
  assert.ok(!html.includes('id="runs"'));
  assert.ok(html.includes('Project overview') && html.includes('Coverage'));
  assert.ok(html.includes('id="page-groups"'));
  assert.match(css, /760px/); assert.match(css,/44px/);
  assert.match(app,/visibilitychange/); assert.match(app,/clearInterval/);
});

test('page links are readable and execution-independent unless evidence is pinned', () => {
  const first=model.reference('https://example.test/testing/?run=old',{page:'camera/projection'});
  assert.equal(new URL(first).searchParams.get('run'),null);
  assert.equal(model.selection(new URL(first)).page,'camera/projection');
  assert.equal(model.pageLabel({state:'not measured',evidence:null}),'Not measured');
  assert.match(model.pageLabel({kind:'demo',evidence:{}}),/demonstration/i);
  assert.match(model.pageLabel({state:'expected failures',evidence:{stale:false},completed:3,collected:3}),/expected failures/i);
  assert.throws(()=>model.selection(new URL('https://x/testing/?page=camera/projection&step=0')));
});

test('page shell loads catalog evidence instead of selecting runs and hides raw diagnostics', async () => {
  const app=await readFile(new URL('app.js',base),'utf8');
  assert.ok(!app.includes('$("runs")'));
  assert.match(app,/this.get\("pages"\)/);
  assert.match(app,/this.get\("page"/);
  assert.match(app,/Execution details/);
  assert.match(app,/node\("details"/);
});

test('shared-panel reconnect cleans old subscriptions and leaving the page closes the viewer', async () => {
  const {pathToFileURL} = await import('node:url');
  const {join} = await import('node:path');
  const shared = pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href;
  let source = await readFile(new URL('app.js',base),'utf8');
  source = source.replace('from "/app.js"', 'from '+JSON.stringify(shared))
    .replace('from "./model.js"', 'from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp} = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  class Element {
    constructor(){this.children=[];this.style={setProperty(){}};this.dataset={};}
    append(...items){for(const child of items){if(child&&typeof child==='object')child.parent=this;this.children.push(child);}}
    replaceChildren(...items){this.children=[];this.append(...items);}
    setAttribute(){} addEventListener(){}
    remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
  }
  const elements = new Map();
  globalThis.document={createElement:()=>new Element(),getElementById:id=>{
    if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
  }};
  globalThis.location={protocol:'http:',host:'localhost',href:'http://localhost/testing/'};
  globalThis.WebSocket=class {static OPEN=1;constructor(){this.readyState=1;}send(){}close(){this.wasClosed=true;}};
  const app=Object.create(TestingApp.prototype);
  Object.assign(app,{selected:{run:'r',page:'test:a',path:null,step:null},panels:[],renderKey:null});
  app.artifacts('r',[{kind:'text',path:'tests/a/note'}]);
  const ws=app.connection.ws;
  const snapshot={t:'snapshot',paths:{'tests/a/note':{kind:'text',steps:[0],latest:{step:0,value:'safe text'}}}};
  ws.onmessage({data:JSON.stringify(snapshot)});
  assert.equal(app.panels.length,1);assert.equal(app.store.listeners.size,1);
  ws.onmessage({data:JSON.stringify(snapshot)});
  assert.equal(app.panels.length,1);assert.equal(app.store.listeners.size,1);
  app.closePanels();
  assert.equal(app.store.listeners.size,0);assert.equal(ws.wasClosed,true);
  assert.equal(elements.get('evidence').children.length,0);
  ws.onmessage({data:JSON.stringify(snapshot)}); // a late event from the disposed socket
  assert.equal(app.panels.length,0);
  delete globalThis.document;delete globalThis.WebSocket;delete globalThis.location;
});
test('disposed shared connections refuse late reconnects', async () => {
  const {pathToFileURL} = await import('node:url'); const {join}=await import('node:path');
  const {Connection}=await import(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href);
  const connection=new Connection('r',null,()=>{});connection.close();
  connection.connect(); // no WebSocket global: it must return without constructing one
  assert.equal(connection.ws,null);
});

test('mobile page selection clears old evidence, collapses navigation and returns to the top', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const elements=new Map();let scrolled=false,href=null,closed=false,polled=false;
  globalThis.document={getElementById:id=>{if(!elements.has(id))elements.set(id,{setAttribute(){}});return elements.get(id);}};
  globalThis.window={matchMedia:()=>({matches:true}),scrollTo:options=>{scrolled=options.top===0;}};
  globalThis.location={href:'https://x/testing/?run=old&page=mesh/face-geometry&step=0'};
  globalThis.history={pushState:(_s,_t,url)=>{href=url;}};
  try {
    const app=Object.create(TestingApp.prototype);
    Object.assign(app,{selected:{run:'old',page:'mesh/face-geometry',path:'mesh',step:0},generation:0,
      closePanels:()=>{closed=true;},poll:()=>{polled=true;}});
    app.navigate('camera/projection');
    assert.deepEqual(app.selected,{run:'',page:'camera/projection',path:null,step:null});
    assert.equal(new URL(href).searchParams.get('run'),null);
    assert.ok(elements.get('navigation').hidden && closed && polled);
    assert.ok(scrolled,'navigation must not leave the phone scrolled past its menu');
  } finally {for(const key of ['document','window','location','history'])delete globalThis[key];}
});

test('coverage percentages distinguish zero, empty, unknown and invalid measurements', () => {
  assert.equal(model.coveragePercent(2436,3162),2436/3162*100);
  assert.equal(model.coveragePercent(0,87),0);
  assert.equal(model.coveragePercent(7,7),100);
  for(const [covered,total] of [[0,0],[null,10],[11,10],[-1,10],[1,Infinity]])
    assert.equal(model.coveragePercent(covered,total),null);
});

test('coverage file labels shorten installation prefixes without losing full evidence paths', () => {
  assert.deepEqual(model.coverageFile('build\\installed\\gffx\\dashboard\\_server.py'),
    {name:'_server.py',directory:'gffx/dashboard',full:'build\\installed\\gffx\\dashboard\\_server.py'});
  assert.equal(model.coverageFile('plain.py').name,'plain.py');
  assert.equal(model.lineRanges([9,3,4,5,11,12]),'3–5, 9, 11–12');
  assert.equal(model.lineRanges([]),'None');
});

test('coverage meters render accessible progress with exact counts, never a fake unknown bar', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  class Element {constructor(tag){this.tag=tag;this.children=[];this.attrs={};}setAttribute(k,v){this.attrs[k]=String(v);}append(...children){this.children.push(...children);}}
  globalThis.document={createElement:tag=>new Element(tag)};
  try {
    const app=Object.create(TestingApp.prototype);
    const measured=app.coverageMeter('Lines',15,23,'lines');
    const bar=measured.children.find(c=>c.tag==='progress');
    assert.equal(Number(bar.attrs.value),15/23*100);
    assert.equal(bar.attrs.max,'100');assert.match(bar.attrs['aria-valuetext'],/15 of 23 lines/);
    assert.ok(!app.coverageMeter('Branches',0,0,'branches').children.some(c=>c.tag==='progress'));
    const css=await readFile(new URL('app.css',base),'utf8');
    assert.match(css,/\.coverage-files/);assert.match(css,/progress/);
  } finally {delete globalThis.document;}
});

test('creation listing uses page dates, not latest executions, and never invents unknown dates', () => {
  const pages=[{id:'old',title:'Old',created_on:'2026-09-01',evidence:{started:999}},
    {id:'b',title:'Beta',created_on:'2026-09-05'}, {id:'a',title:'Alpha',created_on:'2026-09-05'},
    {id:'unknown',title:'Unknown'}];
  assert.deepEqual(model.newestPages(pages).map(p=>p.id),['a','b','old','unknown']);
  assert.equal(pages[0].id,'old');
  assert.equal(model.creationLabel(pages[3]),'Creation date not recorded');
  assert.match(model.creationLabel(pages[1]),/2026-09-05/);
});

test('slow refresh never collapses visible filters and unchanged results retain their summary', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  class Element {
    constructor(){this.children=[];this.hidden=false;this.value='';this.dataset={};this.textContent='';}
    append(...children){this.children.push(...children);}replaceChildren(...children){this.children=children;}
    setAttribute(){}querySelectorAll(){return [];}querySelector(){return null;}
  }
  const elements=new Map();globalThis.document={hidden:false,querySelectorAll:()=>[],createElement:()=>new Element(),
    getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);}};
  document.getElementById('filters').hidden=false; // already visible before the refresh
  const detail={id:'mesh/face-geometry',title:'Mesh',description:'Check geometry',group:'Geometry',kind:'tests',state:'passed',
    evidence:{run:'r',provenance:{}},history:[],counts:{passed:1},collected:1,completed:1,items:[],artifacts:[],total:0};
  const app=Object.create(TestingApp.prototype);let release,started;
  Object.assign(app,{selected:{page:detail.id,run:'',path:null,step:null},offset:0,generation:0,busy:false,
    navigation(){},execution(){},artifacts(){},closePanels(){},pager(){},
    get:async route=>route==='pages'?{pages:[]}:new Promise(resolve=>{release=resolve;started();})});
  let pending;
  try {
    const entered=new Promise(resolve=>{started=resolve;});pending=app.poll();await entered;
    assert.equal(elements.get('filters').hidden,false,'filters disappeared while awaiting page data');
    release(detail);await pending;
    const summary=elements.get('summary').children[0];
    app.get=async route=>route==='pages'?{pages:[]}:detail;
    await app.poll();
    assert.equal(elements.get('summary').children[0],summary,'unchanged summary was remounted');
    app.get=async()=>{throw new Error('temporary connection loss');};
    await app.poll();
    assert.equal(elements.get('summary').children[0],summary,'transient network loss cleared last received evidence');
    assert.match(elements.get('connection').textContent,/offline/i);
    app.get=async()=>{const error=new Error('Unknown page');error.status=404;throw error;};
    await app.poll();
    assert.equal(elements.get('summary').children.length,0,'unknown references must still fail closed');
  } finally {release?.(detail);await pending;delete globalThis.document;}
});

test('unchanged pagination preserves button identity and focus targets', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const pager={children:[],replaceChildren(...children){this.children=children;}};
  globalThis.document={getElementById:()=>pager,createElement:()=>({setAttribute(){}})};
  try {const app=Object.create(TestingApp.prototype);app.offset=0;app.selected={page:'mesh/face-geometry'};
    app.pager(60);const previous=pager.children[0];app.pager(60);assert.equal(pager.children[0],previous);
  } finally {delete globalThis.document;}
});
