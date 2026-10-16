export { demoClipGroup, commonClipSteps, attachShowcase, enhanceDemoPanels } from './playback.js';



export function selection(url) {

  const q = url.searchParams, step = q.get("step");

  if (step !== null && (!/^\d+$/.test(step) || !Number.isSafeInteger(Number(step)))) throw Error("Invalid pinned step");

  if (step !== null && (!q.get("run") || !q.get("path"))) throw Error("Pinned steps require an execution and artifact path");

  return {run:q.get("run") || "",page:q.get("page") || "overview",path:q.get("path"),step:step === null ? null : Number(step)};

}



export function energyBudget(specular,albedo) {

  if(!Number.isFinite(specular)||!Number.isFinite(albedo)||specular<0||specular>1||albedo<0||albedo>1)throw Error('Invalid energy fraction');

  const matte=albedo*(1-specular),reflected=specular+matte;

  return {matte,reflected,absorbed:1-reflected};

}



export function articleLinkAllowed(href, targets) {

  if(href.startsWith('#'))return /^#[A-Za-z][A-Za-z0-9_-]*$/.test(href)&&targets.has(href.slice(1));

  return /^(https:\/\/|\/testing\/\?)/.test(href);

}



export function learningArticle(html) {

  if(typeof html!=='string'||html.length>100000)throw Error('Invalid article size');

  const parsed=new DOMParser().parseFromString(html,'text/html');

  const tags=new Set('article section p strong nav h2 h3 ol ul li a figure figcaption svg rect text details summary pre code label input output math semantics annotation mrow mi mn mo mfrac msqrt mstyle mover munder munderover msub msup msubsup mtable mtr mtd mspace'.split(' '));

  const attrs=new Set('class id href aria-label aria-live role tabindex viewbox x y width height fill type min max step value data-energy xmlns display encoding largeop mathvariant stretchy accent columnalign'.split(' '));

  const elements=[...parsed.body.querySelectorAll('*')];

  // Collect targets before validating links: the contents normally precedes its sections.

  const targets=new Set();

  for(const element of elements)for(const a of element.attributes)if(a.name==='id')targets.add(a.value);

  for(const element of elements){

    if(!tags.has(element.localName))throw Error('Unsupported article element');

    for(const a of element.attributes){

      if(!attrs.has(a.name.toLowerCase()))throw Error('Unsupported article attribute');

      if(a.name==='href'&&!articleLinkAllowed(a.value,targets))throw Error('Unsafe article link');

    }

    if(element.localName==='input'&&element.getAttribute('type')!=='range')throw Error('Unsupported article control');

  }

  const content=document.createElement('div');content.className='learning-content';

  content.append(...[...parsed.body.childNodes].map(n=>document.importNode(n,true)));

  const spec=content.querySelector('[data-energy="specular"]'),albedo=content.querySelector('[data-energy="albedo"]'),result=content.querySelector('[data-energy="result"]');

  if(spec&&albedo&&result){

    const update=()=>{const e=Number(spec.value),a=Number(albedo.value),b=energyBudget(e,a);

      result.textContent=`E=${e.toFixed(2)}; A=${a.toFixed(2)}; glossy ${e.toFixed(2)} + matte ${b.matte.toFixed(2)} = ${b.reflected.toFixed(2)} reflected; ${b.absorbed.toFixed(2)} absorbed.`;};

    spec.addEventListener('input',update);albedo.addEventListener('input',update);update();

  }

  return content;

}

export function reference(base, selected) {

  const url = new URL("/testing/",base);

  for (const key of ["run","page","path","step"]) if (selected[key] !== undefined && selected[key] !== null && selected[key] !== "") url.searchParams.set(key,selected[key]);

  return url.href;

}

export function coverageLabel(value) {

  if (!value) return "Not measured";

  if (!value.total) return "No executable lines measured";

  return (100*value.covered/value.total).toFixed(1) + "% · " + value.covered + "/" + value.total + " executable lines";

}

export function progressLabel(value) {

  return value.state + (value.stale ? " · HEARTBEAT STALE (not completion)" : "") + " · " + value.completed + " / " + value.collected + " tests completed";

}

export function pageLabel(page) {

  if (page.kind === 'article') return 'Learning article';

  if (!page.evidence) return "Not measured";

  if (page.kind === "demo") return "Demonstration · not a test";

  if (page.kind === 'check') return page.state+' · named check'+(page.evidence.stale?' · heartbeat stale':'');

  return page.state + (page.evidence.stale ? " · heartbeat stale" : "") +

    " · " + page.completed + "/" + page.collected + " checked";

}

export function testTitle(nodeid) {

  return nodeid.split("::").at(-1).replace(/^test_/, "").replaceAll("_", " ");

}

export function coveragePercent(covered,total) {

  return Number.isSafeInteger(covered) && Number.isSafeInteger(total) && total>0 && covered>=0 && covered<=total

    ? covered/total*100 : null;

}

export function coverageFile(path) {

  const normalized=path.replaceAll('\\','/'), parts=normalized.split('/');

  const name=parts.pop(), root=parts.lastIndexOf('gffx');

  return {name,directory:(root>=0?parts.slice(root):parts).join('/'),full:path};

}

export function lineRanges(lines) {

  const values=[...new Set(lines)].sort((a,b)=>a-b), ranges=[];

  for(let i=0;i<values.length;i++){

    const start=values[i];let end=start;

    while(values[i+1]===end+1)end=values[++i];

    ranges.push(start===end?String(start):start+'–'+end);

  }

  return ranges.join(', ')||'None';

}

export function newestPages(pages) {

  return [...pages].sort((a,b)=>(b.created_on||'').localeCompare(a.created_on||'') ||

    a.title.localeCompare(b.title) || a.id.localeCompare(b.id));

}

export function creationLabel(page) {

  return page.created_on?'Created '+page.created_on:'Creation date not recorded';

}

export function setupTheme(root,control,storage) {

  let theme='light';

  try {if(storage.getItem('gffx.testing.theme')==='dark')theme='dark';} catch {}

  const apply=()=>{

    root.dataset.theme=theme;

    control.textContent=theme==='light'?'Dark mode':'Light mode';

    control.setAttribute('aria-label','Switch to '+(theme==='light'?'dark':'light')+' mode');

  };

  control.onclick=()=>{theme=theme==='light'?'dark':'light';apply();try{storage.setItem('gffx.testing.theme',theme);}catch{}};

  apply();

}

export function pageArtifacts(detail,pinnedPath) {

  return pinnedPath||detail.kind==='check'||detail.kind==='demo' ? detail.artifacts : detail.artifacts.filter(a=>!a.check);

}

export { attachPlayback } from './playback.js';

export {attachCinema,attachGallery} from './cinema.js';

export function panelBinding(store,spec) {

  // Metadata references an artifact group; the shared viewer spells a prefix with '/'.

  return spec.kind==='plot'&&!store.paths.has(spec.path)

    ? {...spec,path:spec.path.replace(/\/$/,'')+'/'} : spec;

}

export function referenceLabel(page,title) {

  return 'gffx · '+(title|| (page.startsWith('check/')?'Named check page':page.startsWith('test/')?'Test page':page));

}





// Dated presentation checkpoint, not a second plan or live acceptance oracle.

// Authority: devbrain projects/gffx PLAN.md; opaque223 offline Group 3 closeout, retaining earlier Group 1/2 scope.

export const M1_CHECKPOINT = {

  date: '6 October 2026',

  title: 'A graphics library you can fit and compose with',

  purpose: 'M1 qualifies the graphics needed by gffit and Compositron. Concurrent gffit development has its own acceptance requirements.',

  current: "Group 4 paused: one pixel failed the accuracy check.",

  blocker: "214 of 984 foreground pixels passed independent accuracy checks, including the largest. The next pixel failed: its enclosed colour error is 2.15e-10, above the unchanged 1e-10 limit. The worker finished within its resource limits. No campaign is running. Next: investigate this pixel before another trial. Complete CUDA images, full-image gradients and synthetic fitting have not started in this campaign. Your portrait has not been fitted.",

  why: 'After the visible triangle seams, we adopted geometry-aware opaque pixel integration. That changed the image model and required new independent accuracy and gradient checks. M1 has substantial verification left; this is not just final polish.',

  groups: [

    {number:1,state:'Qualified in stated scope',title:'Original fixed-pixel accuracy',text:'All 13 visible pieces of the selected original pixel have independent fixed-colour evidence. This does not qualify a whole frame.'},

    {number:2,state:'Qualified in stated scope',title:'Moving geometry and gradients',text:'Normal, principal-point, world, camera and shear/recomputed-normal checks pass on CPU/CUDA against independent references. Knot/horizon and refusal controls are reconciled. A pixel is not a whole frame.'},

    {number:3,state:'Qualified in stated scope',title:'Complete frames and scale',text:'All 24 legacy scale cases and original images through 1080p pass. Larger original images use the approved offline profile, with exact CPU/CUDA image and albedo-gradient bytes.'},

    {number:4,state:'Working here',title:'gffit readiness',text:'All 203 FLAME model controls pass local derivative checks on both devices. Camera/light fitting and four-group checkpoint replay work. Separate expression, jaw and texture fits improve; shape coefficient recovery remains unresolved even with the other groups known. Broader integration and real observations remain open. Accepted Compositron supply is separate.'},

    {number:5,state:'Still to finish',title:'Final 3D showcase and closeout',text:'Deliver the actual qualified original-mesh demonstrations, finish explanations, and reconcile the acceptance records.'}

  ],

  evidence: {page:'fitting/gffit-flame-demo',run:'pc-gffit-shape-diagnostic2-20261004'},

  staticReport: {"url": "/testing/media/c681fd02d8dd4e6c08347977c0fa890c1241d0d44b8ced344dd0a0c7884224af.png", "alt": "Actual complete 32 by 32 filtered CPU renders of the known FLAME target and saved shape-fit state.", "caption": "SH25, 5 October 2026. Actual complete CPU forward images: known synthetic target on the left, previously saved shape-fit parameters on the right. Displayed in sRGB at nearest-neighbour scale, without smoothing or retouching. No new optimization was run. Complete-image independent accuracy, CUDA, gradients and synthetic recovery remain pending; this is a numerical qualification image, not the final showcase."},

  examples: ['image/chroma-demo','animation/deformation-demo','camera/fit-demo','image/full-recovery-demo'],

};

export function examplePages(pages) {

  return M1_CHECKPOINT.examples.map(id=>pages.find(p=>p.id===id && p.kind==='demo')).filter(Boolean);

}


// Fresh-frame metadata is separate from the frozen controller's legacy counter.
export function withRecoveryProgress(base) {
  const p=base.recoveryProgress,hash=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
  if(!p||p.schema!=='gffx-recovery-progress/1'||p.campaign!==base.referenceCampaign||p.status!==base.status||
     !Number.isInteger(p.acceptedForeground)||p.acceptedForeground<0||p.acceptedForeground>984||p.totalForeground!==984||
     !Number.isFinite(Date.parse(p.checkedAt))||!hash(p.packetSha256)||!hash(p.pipelineStartSha256)||
     !hash(p.frameSha256?.fitted)||!hash(p.frameSha256?.truth))return base;
  return {...base,acceptedForeground:p.acceptedForeground,totalForeground:p.totalForeground,referenceProgressTime:p.checkedAt};
}
