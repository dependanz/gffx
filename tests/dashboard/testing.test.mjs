import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
const base = new URL('../../tools/testing_dashboard/static/', import.meta.url);
const model = await import(new URL('model.js', base));

test('retention UI separates stable pages, historical evidence and destructive maintenance',async()=>{
  const app=await readFile(new URL('app.js',base),'utf8');
  const html=await readFile(new URL('index.html',base),'utf8');
  assert.match(html,/data-page="history"/);
  assert.match(app,/history_keep/);
  assert.match(app,/Pin this evidence/);
  assert.match(app,/evidence\?\.pinned/);
  assert.match(app,/\/testing\/api\/pin/);
  assert.doesNotMatch(app,/\/testing\/api\/history\/prune/);
  assert.equal(model.selection(new URL('https://local/testing/?page=history')).page,'history');
});

test('cinema manifests accept only bounded content-addressed local assets and ordered real steps',async()=>{
  const {validateFilm}=await import(new URL('cinema.js',base));
  const asset={url:'/testing/media/'+'a'.repeat(64)+'.png',sha256:'a'.repeat(64),bytes:80};
  const value={version:1,mode:'fit',width:1280,height:720,target:asset,poster:asset,frames:[{step:0,image:asset,director:asset},{step:10,image:asset,director:asset}]};
  assert.equal(validateFilm(value),value);
  for(const bad of [{...value,width:99999},{...value,frames:[]},{...value,frames:[value.frames[1],value.frames[0]]},
                   {...value,target:{...asset,url:'https://example.com/image.png'}},{...value,target:{...asset,bytes:9*1024*1024}}]){
    assert.throws(()=>validateFilm(bad));
  }
});

test('cinema requests only the current frame pair and releases image resources on exit',async()=>{
  const {attachCinema}=await import(new URL('cinema.js',base));
  const {createHash}=await import('node:crypto');
  const bytes=new Uint8Array(24);bytes.set([137,80,78,71,13,10,26,10]);
  new DataView(bytes.buffer).setUint32(16,320);new DataView(bytes.buffer).setUint32(20,180);
  const sha256=createHash('sha256').update(bytes).digest('hex');
  const asset={sha256,bytes:bytes.length,url:'/testing/media/'+sha256+'.png'};
  const film={version:1,mode:'fit',title:'Recover the shot',width:320,height:180,target:asset,poster:asset,
    objective:'known landmarks',frames:[0,5].map(step=>({step,image:asset,director:asset}))};
  const nodes=[],listeners=new Map(),images=[],requests=[];
  class Element {
    constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.classes=new Set();nodes.push(this);
      this.classList={toggle:key=>{if(this.classes.has(key)){this.classes.delete(key);return false;}this.classes.add(key);return true;}};}
    append(...items){for(const item of items){if(item.parent)item.parent.children=item.parent.children.filter(x=>x!==item);item.parent=this;this.children.push(item);}}
    setAttribute(key,value){this.attributes[key]=value;}
    getContext(){return {drawImage(){},save(){},beginPath(){},rect(){},clip(){},restore(){},fillRect(){}};}
    remove(){if(this.parent)this.parent.children=this.parent.children.filter(x=>x!==this);}
  }
  const saved={document:globalThis.document,fetch:globalThis.fetch,createImageBitmap:globalThis.createImageBitmap};
  globalThis.document={hidden:false,createElement:tag=>new Element(tag),
    addEventListener:(type,fn)=>listeners.set(type,fn),removeEventListener:(type,fn)=>{if(listeners.get(type)===fn)listeners.delete(type);}};
  globalThis.fetch=async(url,options)=>{requests.push({url,...options});return new Response(bytes);};
  // Browser decoding is mocked: this fixture verifies request and bitmap ownership.
  globalThis.createImageBitmap=async()=>{const image={width:320,height:180,closed:false,close(){assert.equal(this.closed,false);this.closed=true;}};images.push(image);return image;};
  const ready=async predicate=>{for(let i=0;i<100&&!predicate();i++)await new Promise(resolve=>setTimeout(resolve,5));assert.ok(predicate());};
  let dispose;
  try {
    const host=new Element('div'),diagnostic=new Element('article');host.append(diagnostic);
    const app={panels:[],store:{paths:new Map([['presentation/film',{latest:{value:film}}]])}};
    dispose=attachCinema(app,host);
    const status=nodes.find(e=>e.tag==='output');
    await ready(()=>status.textContent.includes('optimizer step 0')&&images.length===3);
    assert.equal(requests.length,3,'must not prefetch the whole clip or poster');
    assert.equal(images.filter(x=>!x.closed).length,2,'retain only target and current image');
    assert.equal(diagnostic.parent.tag,'details');
    assert.equal(nodes.some(e=>e.tag==='h2'),false,'the page already has a heading');
    const focus=nodes.find(e=>e.textContent==='Focus view');focus.onclick();
    assert.equal(focus.attributes['aria-pressed'],'true');
    const slider=nodes.find(e=>e.attributes['aria-label']==='Cinematic saved timeline');slider.value=1;
    await slider.oninput();
    assert.equal(requests.length,5);assert.equal(images.filter(x=>!x.closed).length,2);
    assert.match(status.textContent,/optimizer step 5/);
    document.hidden=true;listeners.get('visibilitychange')();
    assert.ok(requests.at(-1).signal.aborted);
    dispose();dispose=null;
    assert.ok(images.every(x=>x.closed));assert.equal(listeners.size,0);assert.equal(host.children.length,0);
  } finally {
    dispose?.();
    for(const [key,value] of Object.entries(saved)){if(value===undefined)delete globalThis[key];else globalThis[key]=value;}
  }
});

test('tiny image swatches receive a visible display-only outline without modifying source pixels',async()=>{
  const {highlightSwatch}=await import(new URL('playback.js',base));
  const previous=globalThis.document,calls=[];
  const context={drawImage(){},strokeRect(...args){calls.push(args);}};
  const canvas={width:2,height:2,getContext:()=>context};
  const source={arrays:{data:new Uint8Array([1,2,3])}};
  const panel={canvas,image:source,overlay:{},show(value){assert.equal(value,source);canvas.width=canvas.height=2;}};
  try{
    globalThis.document={createElement:()=>({getContext:()=>context})};
    highlightSwatch(panel,2);
    assert.equal(canvas.width,128);assert.equal(context.lineWidth,4);
    assert.deepEqual(calls,[[4,68,56,56]]);assert.deepEqual([...source.arrays.data],[1,2,3]);
    highlightSwatch(panel,8);assert.equal(calls.length,1);
    highlightSwatch({canvas},0);
  }finally{globalThis.document=previous;}
});

test('demo selectors group shared timelines without confusing unrelated frame indices',()=>{
  assert.deepEqual(model.demoClipGroup('studio/orbit/color'),{group:'orbit',view:'color'});
  assert.deepEqual(model.demoClipGroup('studio/dolly-zoom/director'),{group:'dolly-zoom',view:'director'});
  assert.equal(model.demoClipGroup('camera-fit/derivative'),null);
  assert.equal(model.demoClipGroup('untrusted/not-a-studio/color'),null);
});

test('synchronized demo selection keeps only common exact saved steps',()=>{
  assert.deepEqual(model.commonClipSteps([[0,1,2,3],[0,2,3],[0,2,4]]),[0,2]);
  assert.deepEqual(model.commonClipSteps([]),[]);
  assert.deepEqual(model.commonClipSteps([[1,2],[]]),[]);
});

test('studio selector cancels old playback, synchronizes both views and disposes visibility hooks',async()=>{
  class Element{
    constructor(tag){this.tag=tag;this.children=[];this.value='';this.classList={add(){},toggle(){}};}
    append(...items){this.children.push(...items);if(this.tag==='select'&&!this.value)this.value=items[0].value;}
    prepend(...items){this.children.unshift(...items);}replaceChildren(){this.children=[];this.value='';}
    setAttribute(k,v){this[k]=v;}remove(){this.removed=true;}closest(){return null;}
  }
  const old=globalThis.document,listeners=new Map(),calls=[];
  globalThis.document={createElement:tag=>new Element(tag),addEventListener:(k,v)=>listeners.set(k,v),removeEventListener:k=>listeners.delete(k)};
  const panel=path=>({spec:{path},el:new Element('panel'),_showRequest:0,visibleSteps:()=>[0,1,2],async showStep(step){this.shown=step;calls.push([path,step]);}});
  const app={panels:[panel('studio/lights/color'),panel('studio/orbit/color'),panel('studio/orbit/director'),panel('studio/light-z')]},host=new Element('host');
  let dispose;
  try{
    dispose=model.attachShowcase(app,host);await new Promise(resolve=>setImmediate(resolve));
    const [choose,view,play,slider]=host.children[0].children;
    choose.value='orbit';await choose.onchange();
    assert.ok(app.panels[0].el.hidden);assert.ok(!app.panels[1].el.hidden&&!app.panels[2].el.hidden);
    assert.ok(app.panels[3].el.hidden);
    const css=await readFile(new URL('app.css',base),'utf8');
    assert.match(css,/\.static-artifact\[hidden\]\s*\{display:none!important;/);
    slider.value='2';await slider.oninput();
    assert.deepEqual(calls.slice(-2),[['studio/orbit/color',2],['studio/orbit/director',2]]);
    await play.onclick();globalThis.document.hidden=true;listeners.get('visibilitychange')();
    dispose();dispose=null;assert.equal(listeners.size,0);
  }finally{dispose?.();globalThis.document=old;}
});

test('article section links require an existing portable local target',()=>{
  const targets=new Set(['energy-intuition','light-recipes','light-sh','future_section_2']);
  for(const href of ['#energy-intuition','#light-recipes','#light-sh','#future_section_2',
                    '/testing/?page=learn/index','https://example.org/paper']){
    assert.equal(model.articleLinkAllowed(href,targets),true,href);
  }
  for(const href of ['#missing','#energy-missing','#','#light-sh extra','#light-sh?x=1',
                    '#light%2Dsh','javascript:alert(1)','data:text/html,x','//example.org',
                    '/other','http://example.org',' https://example.org']){
    assert.equal(model.articleLinkAllowed(href,targets),false,href);
  }
});

test('all generated textbook links pass the reader policy and fragments resolve',async()=>{
  const root=new URL('learn/',base);
  const names=(await readdir(root)).filter(name=>name.endsWith('.json'));
  assert.equal(names.length,20);
  for(const name of names){
    const {html}=JSON.parse(await readFile(new URL(name,root),'utf8'));
    // Extract fixtures from trusted generated source; this is not an HTML sanitizer.
    // Runtime validation continues to inspect the browser-parsed DOM.
    const targets=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]));
    for(const [,href] of html.matchAll(/\bhref="([^"]+)"/g)){
      assert.equal(model.articleLinkAllowed(href,targets),true,`${name}: ${href}`);
    }
  }
});

test('article reader accepts forward section references without bypassing validation',()=>{
  const originalParser=globalThis.DOMParser, originalDocument=globalThis.document;
  const link={localName:'a',attributes:[{name:'href',value:'#light-sh'}]};
  const target={localName:'h2',attributes:[{name:'id',value:'light-sh'}]};
  const content={append(){},querySelector(){return null;}};
  try{
    globalThis.document={createElement:()=>content,importNode:n=>n};
    globalThis.DOMParser=class{parseFromString(){return {body:{querySelectorAll:()=>[link,target],childNodes:[]}};}};
    assert.equal(model.learningArticle('<a href="#light-sh">SH</a><h2 id="light-sh">SH</h2>'),content);
    link.attributes[0].value='#missing';
    assert.throws(()=>model.learningArticle('fixture'),/Unsafe article link/);
    link.attributes[0].value='javascript:alert(1)';
    assert.throws(()=>model.learningArticle('fixture'),/Unsafe article link/);
  }finally{globalThis.DOMParser=originalParser;globalThis.document=originalDocument;}
});

test('learning articles are not unmeasured tests and energy calculator stays bounded',()=>{
  assert.equal(model.pageLabel({kind:'article',evidence:null}),'Learning article');
  for(const e of [0,.3,1])for(const a of [0,.5,1]){
    const budget=model.energyBudget(e,a);
    assert.ok(budget.reflected>=0&&budget.reflected<=1);
    assert.ok(Math.abs(budget.reflected+budget.absorbed-1)<1e-12);
  }
  assert.throws(()=>model.energyBudget(NaN,.5));assert.throws(()=>model.energyBudget(.5,2));
});
test('learning markup rejects active content and unsafe attributes before import',()=>{
  const original=globalThis.DOMParser;
  try{
    for(const element of [
      {localName:'script',attributes:[]},
      {localName:'p',attributes:[{name:'onclick',value:'attack()'}]},
      {localName:'a',attributes:[{name:'href',value:'javascript:attack()'}]},
    ]){
      globalThis.DOMParser=class{parseFromString(){return {body:{querySelectorAll(){return [element];}}};}};
      assert.throws(()=>model.learningArticle('<p>bad</p>'));
    }
  }finally{if(original)globalThis.DOMParser=original;else delete globalThis.DOMParser;}
});
test('learning reader scopes equation overflow and preserves article state on polling',async()=>{
  const app=await readFile(new URL('app.js',base),'utf8'),css=await readFile(new URL('app.css',base),'utf8');
  assert.match(app,/cached\.source_sha256/);assert.match(app,/if\(this\.contentKey!==key\).*learningArticle/);
  assert.match(css,/\.lesson-equation \{overflow-x:auto/);assert.match(css,/\.energy-playground input.*min-height:44px/);
});

test('playback supports 60 fps targets and subtracts display time without catch-up queues',async()=>{
  const {ClipPlayer,PLAYBACK_RATES}=await import(new URL('playback.js',base));
  assert.deepEqual(PLAYBACK_RATES,[4,8,12,24,30,60]);
  let time=0;const pending=[],delays=[];
  const player=new ClipPlayer([0,1,2],async()=>{time+=5;},{now:()=>time,
    schedule:(f,ms)=>{pending.push(f);delays.push(ms);},cancel:()=>{}});
  player.setRate(60);await player.play();
  assert.ok(Math.abs(delays[0]-(1000/60-5))<1e-6);
  player.show=async()=>{time+=40;};await pending.shift()();
  assert.equal(delays[1],0);assert.equal(pending.length,1);
  assert.throws(()=>player.setRate(61));player.dispose();
});

test('finite playback is opt-in, serial and stops at the final frame',async()=>{
  const {ClipPlayer}=await import(new URL('playback.js',base));
  const shown=[],pending=[];
  const player=new ClipPlayer([0,1,2],async step=>shown.push(step),{schedule:f=>{pending.push(f);return f;},cancel:()=>{}});
  assert.deepEqual(shown,[]);assert.equal(player.running,false);
  await player.play();assert.deepEqual(shown,[0]);
  await pending.shift()();assert.deepEqual(shown,[0,1]);
  await pending.shift()();assert.deepEqual(shown,[0,1,2]);
  assert.equal(player.running,false);assert.equal(pending.length,0);
});

test('playback pause/dispose cancel late work and frame seeking is bounded',async()=>{
  const {ClipPlayer}=await import(new URL('playback.js',base));
  let release;const pending=[];
  const player=new ClipPlayer([3,7],()=>new Promise(resolve=>{release=resolve;}),{schedule:f=>pending.push(f),cancel:()=>{}});
  const active=player.play();player.pause();release();await active;
  assert.equal(pending.length,0);assert.equal(player.running,false);
  assert.throws(()=>player.setRate(0),/rate/);
  await assert.rejects(player.seek(2),/frame/);
  player.dispose();await player.play();assert.equal(pending.length,0);
});

test('playback errors stop the clip instead of silently claiming a displayed frame',async()=>{
  const {ClipPlayer}=await import(new URL('playback.js',base));
  let message;
  const player=new ClipPlayer([0,1],async()=>{throw new Error('frame expired');},{onError:e=>{message=e.message;}});
  await player.play();assert.equal(player.running,false);assert.equal(player.index,-1);assert.equal(message,'frame expired');
});

test('browser timers are called without a ClipPlayer receiver',async()=>{
  const {ClipPlayer}=await import(new URL('playback.js',base));
  const original=globalThis.setTimeout,clear=globalThis.clearTimeout;let scheduled=0;
  globalThis.setTimeout=function(){assert.equal(this,undefined);scheduled++;return 7;};
  globalThis.clearTimeout=function(){assert.equal(this,undefined);};
  try {const player=new ClipPlayer([0,1],async()=>{});await player.play();assert.equal(scheduled,1);player.pause();}
  finally {globalThis.setTimeout=original;globalThis.clearTimeout=clear;}
});

test('clip controls retain touch targets and pause on hidden/disposed pages',async()=>{
  const source=await readFile(new URL('playback.js',base),'utf8'),css=await readFile(new URL('app.css',base),'utf8');
  assert.match(source,/visibilitychange/);assert.match(source,/document\.hidden\)pause\(\)/);
  assert.match(source,/panel\.destroy=.*player\.dispose\(\)/);
  assert.match(source,/stopImmediatePropagation/);assert.match(source,/player\.seek\(0\)/);
  assert.match(css,/\.clip-controls button.*min-height:44px/);
  assert.match(css,/\.static-artifact:has\(\.clip-controls\)/);
});
test('tablet navigation responds to breakpoint changes and static panels are explicit',async()=>{
  const app=await readFile(new URL('app.js',base),'utf8'),css=await readFile(new URL('app.css',base),'utf8');
  assert.match(app,/addEventListener\("change",.*navigation/);
  assert.match(css,/@media\s*\(max-width:\s*1000px\)/);
  assert.match(css,/position:fixed/);assert.match(app,/staticEvidence/);
  assert.match(css,/artifact-section/);assert.match(css,/min-width:44px/);
});
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
    .replace("from './portrait-review.js'",'from '+JSON.stringify(new URL('portrait-review.js',base).href))
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
    .replace("from './portrait-review.js'", 'from '+JSON.stringify(new URL('portrait-review.js',base).href))
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
    .replace("from './portrait-review.js'",'from '+JSON.stringify(new URL('portrait-review.js',base).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const elements=new Map();let scrolled=false,href=null,closed=false,polled=false;
  globalThis.document={getElementById:id=>{if(!elements.has(id))elements.set(id,{setAttribute(){}});return elements.get(id);}};
  globalThis.window={matchMedia:()=>({matches:true}),scrollTo:options=>{scrolled=options.top===0;}};
  globalThis.location={href:'https://x/testing/?run=old&page=mesh/face-geometry&step=0'};
  globalThis.history={pushState:(_s,_t,url)=>{href=url;}};
  try {
    const app=Object.create(TestingApp.prototype);
    Object.assign(app,{tablet:{matches:true},selected:{run:'old',page:'mesh/face-geometry',path:'mesh',step:0},generation:0,
      closePanels:()=>{closed=true;},poll:()=>{polled=true;}});
    app.navigate('camera/projection');
    assert.deepEqual(app.selected,{run:'',page:'camera/projection',path:null,step:null});
    assert.equal(new URL(href).searchParams.get('run'),null);
    assert.ok(elements.get('navigation').hidden && closed && polled);
    assert.ok(scrolled,'navigation must not leave the phone scrolled past its menu');
    app.navigationState(true);
    assert.equal(elements.get('nav-backdrop').hidden,false);
    app.tablet.matches=false;app.navigationState(true);
    assert.equal(elements.get('navigation').hidden,false);
    assert.equal(elements.get('nav-backdrop').hidden,true);
    app.tablet.matches=true;app.navigationState(false);
    assert.equal(elements.get('navigation').hidden,true);
  } finally {for(const key of ['document','window','location','history'])delete globalThis[key];}
});

test('coverage layers and scalar-prefix plots have explicit display contracts',async()=>{
  const source=await readFile(new URL('app.js',base),'utf8');
  assert.match(source,/Coverage layers/);assert.match(source,/unmeasured_files/);
  assert.match(source,/cov\.layer,cov\.layers/);
  assert.match(source,/spec\.kind==='plot'.*path\.startsWith/);
});

test('grouped scalar artifacts bind all child curves without matching siblings',async()=>{
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  const {Store,PANEL_KINDS}=await import(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')));
  const store=new Store();
  for(const name of ['fx','fy','cx','cy'])store.paths.set('fit/parameters/'+name,{kind:'scalar',values:[[0,1],[1,2]]});
  store.paths.set('fit/parameters-other',{kind:'scalar',values:[[0,99]]});
  const panel=Object.create(PANEL_KINDS.plot.prototype);
  panel.store=store;panel.spec=model.panelBinding(store,{kind:'plot',path:'fit/parameters'});
  assert.equal(panel.series().length,4);
  assert.equal(model.panelBinding(store,{kind:'image',path:'fit/parameters'}).path,'fit/parameters');
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
    .replace("from './portrait-review.js'",'from '+JSON.stringify(new URL('portrait-review.js',base).href))
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
    .replace("from './portrait-review.js'",'from '+JSON.stringify(new URL('portrait-review.js',base).href))
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
    app.get=async()=>{const error=new Error('signal is aborted without reason');error.name='AbortError';throw error;};
    await app.poll();
    assert.equal(elements.get('summary').children[0],summary);
    assert.match(elements.get('error').textContent,/retrying automatically/i);
    app.get=async()=>{const error=new Error('Unknown page');error.status=404;throw error;};
    await app.poll();
    assert.equal(elements.get('summary').children.length,0,'unknown references must still fail closed');
    app.selected={page:'overview',run:'',path:null,step:null};app.displayedSelection=null;
    app.get=async()=>{const error=new Error('signal is aborted without reason');error.name='AbortError';throw error;};
    await app.poll();
    assert.equal(elements.get('title').textContent,'Where we are in M1');
    assert.match(elements.get('error').textContent,/dated M1 checkpoint/);
    assert.match(elements.get('connection').textContent,/Offline/);
    assert.equal(elements.get('summary').children.length,2);
  } finally {release?.(detail);await pending;delete globalThis.document;}
});

test('learning articles hide execution details and tests link to their lesson', async () => {
  const source=await readFile(new URL('app.js',base),'utf8');
  assert.match(source,/\$\("copy-details"\)\.hidden=this\.selected\.page\.startsWith\('learn\/'\)/);
  assert.match(source,/detail\.learning_article/);
  assert.match(source,/Read the mathematics behind these checks/);
});

test('unchanged pagination preserves button identity and focus targets', async () => {
  const {pathToFileURL}=await import('node:url');const {join}=await import('node:path');
  let source=await readFile(new URL('app.js',base),'utf8');
  source=source.replace('from "/app.js"','from '+JSON.stringify(pathToFileURL(join(process.env.GFFX_DASHBOARD_STATIC,'app.js')).href))
    .replace("from './portrait-review.js'",'from '+JSON.stringify(new URL('portrait-review.js',base).href))
    .replace('from "./model.js"','from '+JSON.stringify(new URL('model.js',base).href));
  const {TestingApp}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const pager={children:[],replaceChildren(...children){this.children=children;}};
  globalThis.document={getElementById:()=>pager,createElement:()=>({setAttribute(){}})};
  try {const app=Object.create(TestingApp.prototype);app.offset=0;app.selected={page:'mesh/face-geometry'};
    app.pager(60);const previous=pager.children[0];app.pager(60);assert.equal(pager.children[0],previous);
  } finally {delete globalThis.document;}
});


test('M1 orientation keeps limited evidence separate from milestone completion',()=>{
  assert.deepEqual(model.M1_CHECKPOINT.groups.map(g=>g.number),[1,2,3,4,5]);
  assert.equal(model.M1_CHECKPOINT.groups.filter(g=>g.state==='Working here').length,1);
  assert.equal(model.M1_CHECKPOINT.groups[1].state,'Qualified in stated scope');
  assert.equal(model.M1_CHECKPOINT.groups[2].state,'Qualified in stated scope');
  assert.equal(model.M1_CHECKPOINT.groups[3].state,'Working here');
  assert.match(model.M1_CHECKPOINT.groups[0].text,/does not qualify a whole frame/);
  const pages=[{id:'image/chroma-demo',kind:'tests'},{id:'camera/fit-demo',kind:'demo'}, {id:'fake',kind:'demo'}];
  assert.deepEqual(model.examplePages(pages),[pages[1]]);
  assert.equal(model.M1_CHECKPOINT.evidence.page,'fitting/gffit-flame-demo');
});


test('UI source retains readable UTF-8 and compact human labels',async()=>{
  for(const name of ['model.js','app.js','index.html','checkpoint.json']){
    const text=await readFile(new URL(name,base),'utf8');
    assert.doesNotMatch(text,/[\u00c3\u00c2]\S|\ufffd/);
  }
  assert.equal(model.pageLabel({kind:'demo',evidence:{}}),'Demonstration · not a test');
  assert.equal(model.lineRanges([1,2,3]),'1–3');
});

test('checkpoint separates scientific campaign state from dashboard connectivity',async()=>{
  const cp=JSON.parse(await readFile(new URL('checkpoint.json',base),'utf8'));
  assert.ok(['stopped','running','qualifying','investigating','checkpoint','forward accuracy passed'].includes(cp.status));
  assert.ok(Number.isInteger(cp.acceptedForeground)&&cp.acceptedForeground>=0&&cp.acceptedForeground<=cp.totalForeground);
  assert.equal(cp.totalForeground,984);
  const hash='a'.repeat(64),stopped={...cp,status:'stopped',referenceCampaign:'saved subject',recoveryProgress:{
    schema:'gffx-recovery-progress/1',campaign:'saved subject',status:'stopped',acceptedForeground:214,totalForeground:984,
    checkedAt:'2026-10-06T17:21:14Z',packetSha256:hash,pipelineStartSha256:hash,frameSha256:{fitted:hash,truth:hash}}};
  assert.equal(model.withRecoveryProgress(stopped).acceptedForeground,214);
  const active={...stopped,status:'qualifying',acceptedForeground:0};
  assert.equal(model.withRecoveryProgress(active).acceptedForeground,0,'old terminal subjects cannot override current qualification');
  const app=await readFile(new URL('app.js',base),'utf8');
  assert.ok(app.includes('const checkpoint=withRecoveryProgress(result.checkpoint||M1_CHECKPOINT)'));
  assert.match(app,/Dashboard online/);
  assert.ok(app.includes('JSON.stringify([page,checkpoint,directory.total'));
});
