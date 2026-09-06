// The developer shell composes the installed general-purpose viewer objects.
import { Store, Connection, PANEL_KINDS } from "/app.js";
import { selection, reference, coverageLabel, pageLabel, testTitle, coveragePercent, coverageFile, lineRanges, newestPages, creationLabel, setupTheme, pageArtifacts, referenceLabel } from "./model.js";
const $ = id => document.getElementById(id);
const node = (tag,text="",attrs={}) => {
  const element = document.createElement(tag); element.textContent = text;
  for (const [key,value] of Object.entries(attrs)) element.setAttribute(key,value);
  return element;
};
const button = (label, action) => {const b=node("button",label); b.onclick=action; return b;};

export class TestingApp {
  constructor() {
    let storage;try{storage=window.localStorage;}catch{}
    setupTheme(document.documentElement,$("theme"),storage);
    this.selected=selection(new URL(location.href)); this.offset=0; this.panels=[]; this.renderKey=null;
    this.generation=0; this.busy=false; this.provenance={}; this.catalogKey=null; this.contentKey=null;
    $("menu").onclick=()=>{const hidden=$("navigation").hidden=!$("navigation").hidden; $("menu").setAttribute("aria-expanded",String(!hidden));};
    if(window.matchMedia("(max-width:760px)").matches){$("navigation").hidden=true;$("menu").setAttribute("aria-expanded","false");}
    document.querySelectorAll("[data-page]").forEach(b=>b.onclick=()=>this.navigate(b.dataset.page));
    for(const id of ["search","outcome"]) $(id).onchange=()=>{this.offset=0;this.generation++;this.contentKey=null;this.poll();};
    $("copy").onclick=()=>this.copyReference();
    $("copy-details").onclick=()=>this.copyReference(this.selected.path,this.selected.step,true);
    this.visibility=()=>{if(document.hidden){clearInterval(this.timer);this.timer=null;this.closePanels();}
      else {this.renderKey=null;this.poll();this.timer=setInterval(()=>this.poll(),2000);}};
    document.addEventListener("visibilitychange",this.visibility);
    window.addEventListener("pagehide",()=>{clearInterval(this.timer);this.closePanels();});
    window.addEventListener("popstate",()=>{this.selected=selection(new URL(location.href));this.offset=0;this.generation++;this.renderKey=null;this.poll();});
    this.visibility();
  }
  navigate(page, run="") {
    this.selected={run,page,path:null,step:null};this.offset=0;this.generation++;this.renderKey=null;this.contentKey=null;
    $("search").value="";$("outcome").value="";
    if(window.matchMedia("(max-width:760px)").matches){$("navigation").hidden=true;$("menu").setAttribute("aria-expanded","false");}
    history.pushState(null,"",reference(location.href,this.selected));this.closePanels();window.scrollTo({top:0,behavior:"instant"});this.poll();
  }
  async get(route, params={}) {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
    try {
      const response=await fetch("/testing/api/"+route+"?"+new URLSearchParams(params),{cache:"no-store",signal:controller.signal});
      const value=await response.json();
      if(!response.ok){const error=Error(value.error||"Request failed");error.status=response.status;throw error;}
      return value;
    } finally {clearTimeout(timer);}
  }
  async poll() {
    if(document.hidden||this.busy)return;
    this.busy=true;const generation=this.generation;
    try {
      const result=await this.get("pages");
      if(generation!==this.generation)return;
      const pages=result.pages;this.navigation(pages);
      $("connection").textContent="Connected · refreshed "+new Date().toLocaleTimeString();
      document.querySelectorAll("[data-page]").forEach(b=>b.setAttribute("aria-current",b.dataset.page===this.selected.page?"page":"false"));
      const page=this.selected.page;
      if(page==="overview"){
        const directory=await this.get('directory',{limit:5});if(generation!==this.generation)return;
        $("filters").hidden=true;this.closePanels();this.pager(null);
        this.header("Project overview","Inspect a feature, its checks and its visual evidence. Pages stay in place as development progresses.","GFFX / DEVELOPMENT");
        const checks=pages.filter(p=>p.kind==="tests");
        this.replaceSummary(['overview',directory.total,checks.map(p=>[p.state,p.evidence?.stale])],()=>[this.metric("Inspection pages",String(directory.total)),
          this.metric("Awaiting evidence",String(checks.filter(p=>!p.evidence).length)),
          this.metric("Need attention",String(checks.filter(p=>p.evidence && (p.state!=="passed" || p.evidence.stale)).length))]);
        $("evidence-context").textContent="Page results may come from different executions or revisions. This is not a combined suite pass or milestone-completion claim.";
        this.execution(null);
        const key=JSON.stringify(['overview',directory, pages.map(p=>[p.id,p.title,p.created_on,p.state,p.counts,p.evidence?.run,p.evidence?.stale])]);
        if(this.contentKey!==key){
          const cards=pages.map(p=>{const b=button("",()=>this.navigate(p.id));b.className="page-card";
            b.append(node("span",p.group,{class:"eyebrow"}),node("h2",p.title),node("p",p.description),node("span",pageLabel(p),{class:"badge",'data-state':p.state}));return b;});
          const grid=node("div","",{class:"page-cards"});grid.append(...cards);
          $("details").replaceChildren(node("h2","Recently created pages"),this.creationList(directory.items,5),
            button("All pages by creation date",()=>this.navigate("newest")),node("h2","Explore features"),grid);this.contentKey=key;
        }
      } else if(page==="newest"){
        const directory=await this.get('directory',{offset:this.offset});if(generation!==this.generation)return;
        $("filters").hidden=true;this.closePanels();this.execution(null);
        this.header("Newest pages","Test and demonstration pages, newest creation date first. This is when the page was created—not when its tests last ran.","GFFX / PAGE DIRECTORY");
        this.replaceSummary(['newest',directory.total],()=>[this.metric("Pages",String(directory.total))]);
        $("evidence-context").textContent="Same-day pages are alphabetical; unknown dates appear last. Original creation times were not recorded for the initial pages.";
        const key=JSON.stringify(['newest',directory,this.offset]);
        if(this.contentKey!==key){$("details").replaceChildren(this.creationList(directory.items));this.contentKey=key;}
        this.pager(directory.total);
      } else if(page==="coverage"){
        $("filters").hidden=true;
        this.closePanels();
        const cov=await this.get("latest-coverage",{...(this.selected.run?{run:this.selected.run}:{}),offset:this.offset});
        if(generation!==this.generation)return;
        this.header("Code coverage","How much Python code did the tests exercise? Coverage measures execution, not correctness.","GFFX / EVIDENCE");
        $("evidence-context").textContent="Python only · Native C/CUDA and per-test attribution are not measured. Showing the selected execution, never an older fallback.";
        this.execution(cov.evidence);
        this.coverageView(cov);
        if(cov.measured)this.pager(cov.total_files);else this.pager(null);
      } else {
        const detail=await this.get("page",{page,...(this.selected.run?{run:this.selected.run}:{}),offset:this.offset,search:$("search").value,outcome:$("outcome").value});
        if(generation!==this.generation)return;
        this.header(detail.title,detail.description,"GFFX / "+detail.group.toUpperCase());
        this.explanation(detail);
        this.replaceSummary([page,pageLabel(detail),detail.counts],()=>[
          node("div",pageLabel(detail),{class:"badge",'data-state':detail.state}),
          ...(detail.kind==="tests"?Object.entries(detail.counts).map(([label,count])=>this.metric(label,String(count))):[])]);
        $("evidence-context").textContent=detail.kind==='check' ? "Named check in the selected test execution. Parent test: "+detail.parent_outcome+". A passing check does not make the whole test pass." : detail.kind==="demo" ? "DEMONSTRATION · illustrative visuals, never test acceptance." :
          "Evidence covers the collected checks for this page in one execution, not the entire feature contract. Visuals appear below when published; no fabricated figures.";
        this.execution(detail.evidence,detail.history);
        if(detail.evidence){
          const visible=pageArtifacts(detail,this.selected.path);this.artifacts(detail.evidence.run,visible);
          const hint='Check-owned visuals are available on the named check pages below each test.';
          if(!visible.length&&detail.artifacts.length&&$("evidence").textContent!==hint)$("evidence").textContent=hint;
        }else this.closePanels();
        $("filters").hidden=detail.kind!=="tests" || !detail.evidence;
        const key=JSON.stringify([page,detail.evidence?.run,detail.items,detail.checks,detail.state,this.offset]);
        if(this.contentKey!==key){
          const open=new Set([...$("details").querySelectorAll("details[open]")].map(d=>d.dataset.test));
          const entries=detail.items.map(t=>{const d=node("details","",{class:"check"});d.dataset.test=t.id;d.open=open.has(t.id);
            const summary=node("summary");summary.append(node("span",testTitle(t.nodeid)),node("span",t.outcome+(t.completed?"":" · pending completion"),{class:"badge",'data-state':t.outcome}));
            d.append(summary,node("code",t.nodeid));
            if(detail.kind==='tests')d.append(button('Open test page',()=>this.navigate('test/'+t.id,this.selected.run)));
            for(const [phase,v] of Object.entries(t.phases)){d.append(node("p",phase+" · "+v.outcome+" · "+v.duration.toFixed(4)+" s"));if(v.message)d.append(node("pre",v.message));}
            return d;});
          if(detail.kind==='check'||detail.kind==='test'){
            const back=button(detail.kind==='check'?'← Parent test':'← Feature overview',()=>this.navigate(detail.parent,this.selected.run));
            const children=(detail.checks||[]).map(c=>{
              const card=node('article','',{class:'check-card'});
              card.append(node('h3',c.name),node('span',c.outcome,{class:'badge','data-state':c.outcome}),node('p',c.duration.toFixed(4)+' s'));
              if(c.message)card.append(node('pre',c.message));
              if(detail.kind==='test')card.append(button('Open check page',()=>this.navigate(c.page,this.selected.run)));
              return card;
            });
            $("details").replaceChildren(back,...entries,node('h2',detail.kind==='test'?'Named checks':'Check result'),
              ...(children.length?children:[node('p','No named checks reported in this execution. Ordinary assertions are not inferred or marked passed.') ]));
          } else $("details").replaceChildren(...(entries.length?[node("h2","Tests"),...entries]:[node("p",detail.evidence?(detail.kind==="demo"?"Use the panels to inspect, scrub and copy an evidence reference.":"No tests match this filter."):"No evidence published yet. This page is ready for its fixtures and visual artifacts.")]));
          this.contentKey=key;
        }
        if(detail.kind==="tests"&&detail.evidence)this.pager(detail.total);else this.pager(null);
      }
      this.displayedSelection=JSON.stringify(this.selected);this.lastSuccess=new Date().toLocaleTimeString();
      $("error").hidden=true;
    } catch(error){
      if(generation!==this.generation)return;
      if(this.displayedSelection===JSON.stringify(this.selected) && !(error.status>=400 && error.status<500)){
        $("error").hidden=false;$("error").textContent="Refresh unavailable; showing last received evidence from "+this.lastSuccess+". "+error.message;
        $("connection").textContent="Offline · showing last received evidence";return;
      }
      this.closePanels();this.contentKey=null;this.summaryKey=null;this.pagerKey=null;this.execution(null);
      for(const id of ["details","summary","pager","evidence-context","explanation"])$(id).replaceChildren();
      this.guideKey=null;this.pageTitle=null;this.titlePage=null;
      $("error").hidden=false;$("error").textContent=error.message;$("connection").textContent="Unavailable";
    }
    finally{this.busy=false;if(generation!==this.generation&&!document.hidden)queueMicrotask(()=>this.poll());}
  }
  replaceSummary(value,build) {
    const key=JSON.stringify(value);if(this.summaryKey===key)return;
    this.summaryKey=key;$("summary").replaceChildren(...build());
  }
  creationList(pages,limit=pages.length) {
    const list=node("ol","",{class:"creation-list"});
    for(const page of newestPages(pages).slice(0,limit)){
      const item=node("li"),b=button("",()=>this.navigate(page.id));
      b.append(node("span",creationLabel(page),{class:"creation-date"}),node("strong",page.title),
        node("span",page.group+" · "+({demo:'Demonstration',test:'Test',check:'Named check',tests:'Feature'}[page.kind]||page.kind),{class:"creation-kind"}));
      item.append(b);list.append(item);
    }
    return list;
  }
  header(title,description,eyebrow) {
    this.pageTitle=title;this.titlePage=this.selected.page;
    $("reference-label").textContent=referenceLabel(this.selected.page,title);
    if(!this.selected.page.startsWith('test/')&&!this.selected.page.startsWith('check/')){
      $("explanation").hidden=true;this.guideKey=null;
    }
    for(const [id,value] of [["title",title],["description",description],["eyebrow",eyebrow]])if($(id).textContent!==value)$(id).textContent=value;
  }
  explanation(detail) {
    const box=$("explanation"),guide=detail.explanation;
    box.hidden=!guide;if(!guide)return;
    const key=JSON.stringify([detail.id,guide,detail.nodeid]);if(this.guideKey===key)return;
    this.guideKey=key;
    const sections=[['What this checks',guide.purpose],['How it checks it',guide.method],
      ['What a pass means — and does not mean',guide.limits]];
    box.replaceChildren(...sections.map(([label,text])=>{
      const section=node('section');section.append(node('h2',label),node('p',text));return section;
    }));
    box.append(node('p','Current author guide · explains today’s test source, not a description captured during the selected historical execution.',{class:'guide-note'}));
    box.append(node('p','Read the result separately: passed = assertions succeeded; failed = at least one failed; xfailed = expected failure, not acceptance; skipped = not exercised; pending/incomplete = no complete result.',{class:'guide-note'}));
    const technical=node('details');technical.append(node('summary','Technical identifiers'),
      node('p','Test source'),node('code',detail.nodeid||''),node('p','Stable page ID'),node('code',detail.id));box.append(technical);
  }
  coverageMeter(label,covered,total,unit) {
    const percent=coveragePercent(covered,total), meter=node("div","",{class:"coverage-meter"});
    const heading=node("div","",{class:"coverage-meter-heading"});
    heading.append(node("span",label),node("strong",percent===null?"—":percent.toFixed(1)+"%"));meter.append(heading);
    if(percent!==null){
      meter.append(node("progress","",{max:100,value:percent,'aria-label':label,
        'aria-valuetext':covered.toLocaleString('en-US')+" of "+total.toLocaleString('en-US')+" "+unit+" covered ("+percent.toFixed(1)+"%)"}));
      meter.append(node("p",covered.toLocaleString('en-US')+" / "+total.toLocaleString('en-US')+" "+unit+" covered",{class:"coverage-count"}));
    } else meter.append(node("p",total===0?(unit==="branches"?"No branches":"No executable lines"):"Not measured",{class:"coverage-count"}));
    return meter;
  }
  coverageView(cov) {
    const key=JSON.stringify(['coverage',cov.evidence?.run,cov.artifact_sha256,cov.items,this.offset,cov.measured]);
    if(this.contentKey===key)return;
    this.contentKey=key;
    const opened=new Set([...$("details").querySelectorAll("details[open]")].map(d=>d.dataset.file));
    const overview=node("section","",{class:"coverage-overview"});
    overview.append(this.coverageMeter("Overall line coverage",cov.measured?cov.covered:null,cov.measured?cov.total:null,"lines"));
    if(cov.measured)overview.append(node("p",(cov.total-cov.covered).toLocaleString('en-US')+" uncovered lines · "+cov.total_files+" Python files",{class:"coverage-count"}));
    this.summaryKey=null;$("summary").replaceChildren(overview);
    if(!cov.measured){$("details").replaceChildren(node("p","No coverage report for this execution yet."));return;}
    const files=node("div","",{class:"coverage-files"});
    for(const f of cov.items){
      const label=coverageFile(f.path), card=node("article","",{class:"coverage-file"});
      card.append(node("p",label.directory||"Python module",{class:"coverage-directory"}),node("h3",label.name));
      card.append(this.coverageMeter("Lines",f.covered,f.total,"lines"),this.coverageMeter("Branches",f.covered_branches,f.branches,"branches"));
      const details=node("details","",{class:"coverage-diagnostics"});details.dataset.file=f.path;details.open=opened.has(f.path);
      details.append(node("summary","Source & missing lines"),node("p","Full measured path"),node("code",label.full),
        node("p","Missing lines (ranges)"),node("code",lineRanges(f.missing_lines)));
      card.append(details);files.append(card);
    }
    $("details").replaceChildren(node("h2","Coverage by file"),files);
  }
  metric(label,value) {const card=node("div","",{class:"card"});card.append(node("span",label),node("strong",value));return card;}
  navigation(pages) {
    const key=JSON.stringify(pages.map(p=>[p.id,p.group,p.title]));
    if(this.catalogKey!==key){
      $("page-groups").replaceChildren();const groups=new Map();
      for(const p of pages){if(!groups.has(p.group)){const section=node("details","",{class:"nav-group"});section.open=true;section.append(node("summary",p.group));groups.set(p.group,section);$("page-groups").append(section);}
        const b=button(p.title,()=>this.navigate(p.id));b.dataset.page=p.id;b.setAttribute("aria-label",p.title);b.append(node("span","",{'data-page-status':p.id,class:"nav-status"}));groups.get(p.group).append(b);}
      this.catalogKey=key;
    }
    for(const p of pages){const badge=[...document.querySelectorAll("[data-page-status]")].find(b=>b.dataset.pageStatus===p.id);badge.textContent=p.evidence?(p.evidence.stale?"stale":p.state):"not measured";badge.setAttribute("data-state",p.state);}
  }
  execution(evidence, historyItems=[]) {
    this.evidenceRun=evidence?.run||"";this.provenance=evidence?.provenance||{};
    const key=JSON.stringify([this.selected.page,this.selected.run,evidence?.run,evidence?.state,evidence?.stale,evidence?.coverage,evidence?.diagnostic,evidence?.provenance,historyItems.map(r=>r.run)]);
    if(this.executionKey===key)return;this.executionKey=key;
    const wasOpen=$("provenance").querySelector("details")?.open;
    $("provenance").replaceChildren();
    if(!evidence)return;
    const details=node("details","",{class:"execution"});details.open=Boolean(wasOpen);
    details.append(node("summary","Execution details · "+new Date(evidence.started*1000).toLocaleString()),
      node("p","Source: "+evidence.run+" · "+evidence.state+(evidence.stale?" · heartbeat stale":"")),
      node("p","Producer revision: "+(evidence.provenance.revision||"not reported")+" · dirty: "+String(evidence.provenance.dirty??"unknown")),
      node("p","Execution-scoped Python coverage: "+coverageLabel(evidence.coverage)+". Not per-page coverage."));
    if(evidence.diagnostic)details.append(node("p","Reporting warning: "+evidence.diagnostic));
    if(this.selected.run)details.append(button("Return to latest evidence",()=>this.navigate(this.selected.page)));
    for(const item of historyItems.slice(1))details.append(button("Inspect evidence from "+new Date(item.started*1000).toLocaleString()+" · "+item.run,()=>this.navigate(this.selected.page,item.run)));
    const raw=node("details");raw.append(node("summary","Raw producer metadata"),node("pre",JSON.stringify(evidence,null,2)));details.append(raw);$("provenance").append(details);
  }
  table(headers,rows) {
    const table=node("table"),head=node("tr");headers.forEach(h=>head.append(node("th",h)));table.append(head);
    for(const cells of rows){const row=node("tr");for(const value of cells){const cell=node("td");cell.append(value instanceof Node?value:document.createTextNode(String(value)));row.append(cell);}table.append(row);}
    $("details").replaceChildren(table);
  }
  pager(total) {
    const key=JSON.stringify([this.selected.page,this.offset,total]);if(this.pagerKey===key)return;
    this.pagerKey=key;
    if(total===null){$("pager").replaceChildren();return;}
    const prev=button("Previous",()=>{this.offset=Math.max(0,this.offset-50);this.generation++;this.poll();});
    const next=button("Next",()=>{this.offset+=50;this.generation++;this.poll();});
    prev.disabled=this.offset===0;next.disabled=this.offset+50>=total;
    $("pager").replaceChildren(prev,node("span",total? (this.offset+1)+"–"+Math.min(total,this.offset+50)+" of "+total:"No entries"),next);
  }
  closePanels() {
    for(const panel of this.panels)panel.destroy();this.panels=[];
    this.connection?.close();this.connection=null;this.renderKey=null;
    $("evidence").replaceChildren();
  }
  artifacts(run, artifacts) {
    const key=JSON.stringify([run,this.selected.page,artifacts,this.selected.path,this.selected.step]);
    if(key===this.renderKey)return;
    this.closePanels();this.renderKey=key;
    if(!artifacts.length){$("evidence").append(node("p",this.selected.path?"Referenced artifact is unavailable; no substitute selected.":"No visual artifacts published for this page’s selected evidence.",{class:"empty-evidence"}));return;}
    $("evidence").replaceChildren();this.store=new Store();
    this.reference={path:this.selected.path,step:this.selected.step};
    this.layout={panels:artifacts.map((a,i)=>({...a,col:0,row:i*6,w:12,h:6}))};
    this.connection=new Connection(run,this.store,(text,error)=>{
      if(this.connection!==connection)return;
      if(error){$("error").hidden=false;$("error").textContent="Artifact viewer: "+text;}
    });
    const connection=this.connection;
    this.connection.viewOptions={fps:2,bytes_per_second:131072,mesh_deltas:true,paused:false};
    this.connection.onSnapshot=()=>{
      if(this.connection!==connection)return;
      for(const panel of this.panels)panel.destroy();
      this.panels=[];$("evidence").replaceChildren();
      for(const spec of this.layout.panels){
        const panel=new PANEL_KINDS[spec.kind](this,spec);this.panels.push(panel);$("evidence").append(panel.el);
        if(spec.label){const label=panel.head.querySelector('.path');if(label)label.textContent=spec.label;}
        if(!this.store.paths.has(spec.path))panel.body.replaceChildren(node("p","Artifact unavailable: not retained or not delivered."));
        if(this.selected.path===spec.path && this.selected.step!==null){
          if(typeof panel.showStep==="function"){panel.following=false;panel.showStep(this.selected.step);}
          else{panel.unsubscribe();panel.body.replaceChildren(node("p","Pinned steps unsupported for this panel; no latest-value substitute."));}
        }
      }
      if(this.selected.path&&!artifacts.some(a=>a.path===this.selected.path)){this.closePanels();$("evidence").textContent="Referenced artifact is not on this page.";}
    };
    this.connection.connect();
  }
  async copyReference(path=this.selected.path,step=this.selected.step,detailed=false) {
    const url=reference(location.href,{...this.selected,run:path?this.evidenceRun:this.selected.run,path,step});
    const text=referenceLabel(this.selected.page,this.titlePage===this.selected.page?this.pageTitle:null)+"\n"+url+((detailed||path||this.selected.run)?"\n"+(path?"Pinned artifact execution: "+this.evidenceRun:this.selected.run?"Selected historical execution: "+this.selected.run:"Live page; evidence may update")+"\nProducer provenance: "+JSON.stringify(this.provenance)+"\nArtifacts may expire; demos are not test acceptance.":"");
    try{await navigator.clipboard.writeText(text);}
    catch{$("reference-fallback").hidden=false;$("reference-fallback").value=text;$("reference-fallback").focus();$("reference-fallback").select();}
  }
}
if (typeof document !== "undefined" && document.body?.dataset.gffxApp === "testing") {
  try {new TestingApp();}
  catch(error){$("error").hidden=false;$("error").textContent=error.message;$("connection").textContent="Invalid page reference";}
}
