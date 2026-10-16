import { portraitReview } from './portrait-review.js';
// The developer shell composes the installed general-purpose viewer objects.
import { Store, Connection, PANEL_KINDS } from "/app.js";
import { M1_CHECKPOINT, withRecoveryProgress, examplePages, selection, reference, coverageLabel, pageLabel, testTitle, coveragePercent, coverageFile, lineRanges, newestPages, creationLabel, setupTheme, pageArtifacts, referenceLabel, panelBinding, attachPlayback, learningArticle, attachShowcase, demoClipGroup, enhanceDemoPanels, attachCinema, attachGallery } from "./model.js";
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
    this.tablet=window.matchMedia("(max-width:1000px)");
    this.navigationState(!this.tablet.matches);
    this.tablet.addEventListener("change",()=>this.navigationState(!this.tablet.matches));
    $("menu").onclick=()=>this.navigationState($("navigation").hidden);
    $("nav-backdrop").onclick=()=>this.navigationState(false);
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&this.tablet.matches&&!$("navigation").hidden){this.navigationState(false);$("menu").focus();}});
    this.headerObserver=new ResizeObserver(()=>document.documentElement.style.setProperty('--toolbar-height',document.querySelector('.bar').getBoundingClientRect().height+'px'));
    this.headerObserver.observe(document.querySelector('.bar'));
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
    if(this.tablet.matches)this.navigationState(false);
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
  navigationState(open) {
    $("navigation").hidden=!open;$("menu").setAttribute('aria-expanded',String(open));
    $("nav-backdrop").hidden=!(open&&this.tablet.matches);
  }
  async poll() {
    if(document.hidden||this.busy)return;
    this.busy=true;const generation=this.generation;
    try {
      let result,offlineOverview=false;
      try{result=await this.get("pages");}catch(error){
        if(this.selected.page!=='overview'||this.displayedSelection===JSON.stringify(this.selected)||(error.status>=400&&error.status<500))throw error;
        // A dated explanation remains useful on first load without the index.
        // No test result, live status or substitute evidence is invented.
        result={pages:[]};offlineOverview=true;
      }
      if(generation!==this.generation)return;
      const checkpoint=withRecoveryProgress(result.checkpoint||M1_CHECKPOINT);
      const progressUpdated=Date.parse(checkpoint.referenceProgressTime||"");
      const progressStale=checkpoint.status==='running' && (!Number.isFinite(progressUpdated)||Date.now()-progressUpdated>60000);
      const pages=result.pages;this.navigation(pages);
      $("connection").textContent="Dashboard online · checked "+new Date().toLocaleTimeString();
      document.querySelectorAll("[data-page]").forEach(b=>b.setAttribute("aria-current",b.dataset.page===this.selected.page?"page":"false"));
      const page=this.selected.page;
      if(page==="overview" || page==="visuals" || page==="features"){
        const directory={total:pages.length};if(generation!==this.generation)return;
        $("filters").hidden=true;$("explanation").hidden=true;this.closePanels();this.pager(null);
        this.execution(null);
        const isOverview=page==='overview',isVisuals=page==='visuals';
        this.header(isOverview?'Where we are in M1':isVisuals?'See gffx in action':'Features & evidence',
          isOverview?checkpoint.purpose:isVisuals?'Saved graphics and fitting examples. Each page explains its scope and where the output came from.':'The detailed lab notebook. A page result covers one selected execution, not the whole feature.',
          isOverview?'GFFX / MILESTONE 1':'GFFX / EXPLORE');
        this.replaceSummary([page,checkpoint.groups],()=>isOverview?[this.metric('Milestone','M1 unfinished'),this.metric('Current work',`Group ${checkpoint.groups.find(g=>g.state==='Working here').number} of ${checkpoint.groups.length}`)]:[]);
        $('evidence-context').textContent=isOverview?'Dated checkpoint · '+checkpoint.date+' · progress is not a percentage of test pages.':
          'Historical failures and older demonstrations remain available. Demonstrations are not milestone acceptance.';
        const key=JSON.stringify([page,checkpoint,directory.total,progressStale,pages.map(p=>[p.id,p.state,p.evidence?.run])]);
        if(this.contentKey!==key){
          const card=p=>{const b=button('',()=>this.navigate(p.id));b.className='page-card';
            b.append(node('span',p.group,{class:'eyebrow'}),node('h2',p.title),node('p',p.description),
              node('span',p.kind==='tests'?'Selected checks: '+pageLabel(p):pageLabel(p),{class:'badge'}));return b;};
          const grid=list=>{const g=node('div','',{class:'page-cards'});g.append(...list.map(card));return g;};
          if(isOverview){
            const focus=node('section','',{class:'m1-focus','data-status':checkpoint.status||'checkpoint'});
            focus.append(node('p',checkpoint.status==='stopped'?'CAMPAIGN STOPPED':progressStale?'PROGRESS UPDATE DELAYED':checkpoint.status==='running'?(checkpoint.progressStage==='images'?'FRESH IMAGE RENDERS RUNNING':'INDEPENDENT CHECKS RUNNING'):checkpoint.status==='forward accuracy passed'?'FORWARD ACCURACY VERIFIED':'RECORDED CHECKPOINT',{class:'eyebrow'}),node('h2',checkpoint.current),node('p',checkpoint.blocker),
              button('See FLAME working in gffit',()=>this.navigate('fitting/gffit-flame-demo')),
              button('Review portrait landmarks and skin mask',()=>this.navigate('portrait-review')),
              button(checkpoint.staticReport?'Open the earlier fitting checkpoint':'Read the latest engineering checkpoint',()=>this.navigate(checkpoint.evidence.page,checkpoint.evidence.run)));
            const qualification=checkpoint.qualificationProgress;
            if(qualification && Number.isInteger(qualification.passed) && Number.isInteger(qualification.total) &&
               qualification.total>0 && qualification.passed>=0 && qualification.passed<=qualification.total){
              const section=node('section','',{class:'m1-progress','aria-label':'Current qualification stage'});
              const heading=node('div','',{class:'m1-progress-heading'});
              heading.append(node('label',qualification.title||'Current qualification',{for:'m1-qualification-progress'}),
                node('strong',`${qualification.passed.toLocaleString()} / ${qualification.total.toLocaleString()}`));
              section.append(heading,node('progress','',{id:'m1-qualification-progress',max:qualification.total,value:qualification.passed}),
                node('p',qualification.scope||'Passing reports in the current stage; other acceptance gates remain separate.',{class:'m1-progress-note'}));
              focus.insertBefore(section,focus.children[3]);
            }
            const segments=checkpoint.replacementSegmentsPassed,segmentTotal=checkpoint.replacementSegmentsTotal;
            if(Number.isInteger(segments)&&Number.isInteger(segmentTotal)&&segmentTotal>0&&segments>=0&&segments<=segmentTotal){
              const section=node('section','',{class:'m1-progress','aria-label':'Fresh CPU and CUDA rendering progress'});
              const heading=node('div','',{class:'m1-progress-heading'});
              heading.append(node('label',checkpoint.renderProgressLabel||'Fresh image segments rendered and matched on CPU/CUDA',{for:'m1-render-progress'}),node('strong',`${segments.toLocaleString()} / ${segmentTotal.toLocaleString()}`));
              section.append(heading,node('progress','',{id:'m1-render-progress',max:segmentTotal,value:segments}),node('p','Four pixels per segment. Rendering progress is separate from independent accuracy and gradient acceptance.',{class:'m1-progress-note'}));
              focus.append(section);
            }
            const accepted=checkpoint.acceptedForeground,total=checkpoint.totalForeground;
            if(Number.isInteger(accepted)&&Number.isInteger(total)&&total>0&&accepted>=0&&accepted<=total){
              const progress=node('section','',{class:'m1-progress','aria-label':'Independent image comparison progress'});
              const heading=node('div','',{class:'m1-progress-heading'});
              heading.append(node('label',checkpoint.accuracyProgressLabel||'Foreground pixels independently checked',{for:'m1-accuracy-progress'}),
                node('strong',`${accepted.toLocaleString()} / ${total.toLocaleString()} · ${(100*accepted/total).toFixed(1)}%`));
              const bar=node('progress','',{id:'m1-accuracy-progress',max:total,value:accepted,'aria-describedby':'m1-progress-scope'});
              const scope=node('p','Pixel checks only — not overall M1 completion or time remaining. Transparent background and final whole-image checks are separate.',{id:'m1-progress-scope',class:'m1-progress-note'});
              const timing=Number.isFinite(progressUpdated)?`Results checked at ${new Date(progressUpdated).toLocaleTimeString()}. `:'';
              progress.append(heading,bar,scope,
                node('p',timing+(progressStale?'Updates are delayed; the last verified count is shown.':checkpoint.status==='running'?'Results refresh about every 20 seconds; this page polls every 2 seconds while visible.':'This campaign is not reporting live progress.'),{class:'m1-progress-note'}),
                node('p','The count advances when a complete pixel passes both colour and coverage checks. Revalidating saved certificates or calculating several parts of one pixel can leave it unchanged.',{class:'m1-progress-note'}));
              focus.append(progress);
            }
            if(checkpoint.staticReport){
              const report=checkpoint.staticReport,figure=node('figure','',{class:'m1-static-report'});
              const link=node('a','',{href:report.url,target:'_blank',rel:'noopener','aria-label':'Open the saved validation image at full size'});
              link.append(node('img','',{src:report.url,alt:report.alt,width:1600,height:960,loading:'lazy',style:'display:block;width:100%;height:auto;border-radius:8px'}));
              figure.append(link,node('figcaption',report.caption));focus.append(figure);
            }
            const roadmap=node('ol','',{class:'m1-roadmap','aria-label':'M1 exit checklist'});
            for(const g of checkpoint.groups){const row=node('li','',{'data-current':String(g.state==='Next'||g.state==='Working here')});
              row.append(node('span',g.state,{class:'eyebrow'}),node('h3',g.title),node('p',g.text));roadmap.append(row);}
            const why=node('details','',{class:'m1-why'});why.append(node('summary','Why has M1 taken so long?'),node('p',checkpoint.why));
            const library=node('details','',{class:'m1-why'});library.append(node('summary','What is already built?'),
              node('p','Geometry and cameras, scene assembly, animation and deformation, image layers, textures, materials, lighting, and CPU/CUDA execution all have implementation and test evidence. The remaining work qualifies the expanded opaque-rendering and integration requirements.'),
              button('Explore features and their checks',()=>this.navigate('features')));
            $('details').replaceChildren(focus,node('h2','The M1 closeout checklist'),roadmap,why,library,
              node('h2','Start with a visual example'),node('p','These are existing saved demonstrations, not the final M1 3D showcase.'),grid(examplePages(pages)),
              button('Browse all demonstrations',()=>this.navigate('visuals')),
              node('p','Need the details? Newest pages is the chronological lab notebook; green checks there do not mean M1 is complete.',{class:'m1-footnote'}));
          }else $('details').replaceChildren(grid(pages.filter(p=>isVisuals?p.kind==='demo':true)));
          this.contentKey=key;
        }
      } else if(page==='portrait-review'){
        const review=await this.get('portrait-review');if(generation!==this.generation)return;
        $('filters').hidden=true;$('explanation').hidden=true;this.closePanels();this.execution(null);this.pager(null);
        this.header('Review the portrait inputs','Toggle the overlays and inspect the observations before fitting.','GFFX / GROUP 4 / INPUT REVIEW');
        this.replaceSummary([page,review.revision_sha256],()=>[this.metric('Status','Draft annotations'),this.metric('Landmarks','68')]);
        $('evidence-context').textContent='Not a fitted result or a passing test. The original photo is unchanged.';
        const key=JSON.stringify([page,review.revision_sha256]);
        if(this.contentKey!==key){$('details').replaceChildren(portraitReview(review));this.contentKey=key;}
      } else if(page==="newest"){
        const directory=await this.get('directory',{offset:this.offset});if(generation!==this.generation)return;
        $("filters").hidden=true;this.closePanels();this.execution(null);
        this.header("Newest pages","Learning, test and demonstration pages, newest creation date first. This is when the page was created—not when its tests last ran.","GFFX / PAGE DIRECTORY");
        this.replaceSummary(['newest',directory.total],()=>[this.metric("Pages",String(directory.total))]);
        $("evidence-context").textContent="Same-day pages are alphabetical; unknown dates appear last. Original creation times were not recorded for the initial pages.";
        const key=JSON.stringify(['newest',directory,this.offset]);
        if(this.contentKey!==key){$("details").replaceChildren(this.creationList(directory.items));this.contentKey=key;}
        this.pager(directory.total);
      } else if(page==='history'){
        const info=await this.get('history');if(generation!==this.generation)return;
        this.closePanels();this.execution(null);this.pager(null);$('filters').hidden=true;
        this.header('History & storage','Pages are permanent identities. Execution history and live-frame buffers have separate limits.','GFFX / RETENTION');
        this.replaceSummary(['history',info.executions,info.protected.length,info.candidates.length],()=>[
          this.metric('Saved executions',String(info.executions)),this.metric('Protected executions',String(info.protected.length)),
          this.metric('Eligible for cleanup',String(info.candidates.length))]);
        $('evidence-context').textContent='Cleanup is explicit, never triggered by viewing this page. Latest evidence, active executions and pins are protected. Live window and snapshot settings are unchanged.';
        const key=JSON.stringify(['history',info]);if(this.contentKey!==key){
          const raw=node('details','',{class:'m1-why'});raw.append(node('summary','Technical retention inventory'),node('pre',JSON.stringify({eligible:info.candidates,blocked:info.blocked},null,2)));
          $('details').replaceChildren(node('p','Retain '+info.history_keep+' extra older executions, plus all protected evidence. This is not a lifetime execution cap.'),
            node('p','Active executions: '+info.active+' / '+info.max_active+'. Stale sessions still need explicit completion.'),
            node('p','Metadata: '+(info.metadata_used_bytes/1048576).toFixed(2)+' MiB used / '+(info.metadata_limit_bytes/1048576).toFixed(0)+' MiB budget. Freed SQLite pages are reused.'),
            node('p','Saved-log budget: '+(info.log_limit_bytes/1048576).toFixed(0)+' MiB. Shared presentation assets have a separate preserve-and-stop budget.'),
            node('p','Pin important evidence under its Execution details. Ask the developer to preview and apply eligible cleanup; this page cannot delete history.'),
            raw);this.contentKey=key;
        }
      } else if(page==="coverage"){
        $("filters").hidden=true;
        this.closePanels();
        const cov=await this.get("latest-coverage",{...(this.selected.run?{run:this.selected.run}:{}),offset:this.offset,layer:this.coverageLayer||'python'});
        if(generation!==this.generation)return;
        this.header("Code coverage","What code did the instrumented tests exercise? Select a layer; there is no misleading whole-repository percentage.","GFFX / EVIDENCE");
        $("evidence-context").textContent="Coverage remeasurement is deferred until after M1. Coverage measures execution, not correctness. A Python or host test does not measure GPU device code. Uninstrumented files and missing layers are not zero-percent coverage. Reports stay attached to their execution.";
        this.execution(cov.evidence);
        this.coverageView(cov);
        if(cov.measured)this.pager(cov.total_files);else this.pager(null);
      } else {
        const cached=this.articleDetail;
        const detail=cached?.id===page && cached.source_sha256===pages.find(p=>p.id===page)?.source_sha256 && !this.selected.run ? cached :
          await this.get("page",{page,...(this.selected.run?{run:this.selected.run}:{}),offset:this.offset,search:$("search").value,outcome:$("outcome").value});
        if(generation!==this.generation)return;
        this.header(detail.title,detail.description,"GFFX / "+detail.group.toUpperCase());
        if(detail.kind==='article'){
          if(this.selected.path)throw Error('Learning articles do not have execution artifacts');
          this.articleDetail=detail;this.closePanels();this.execution(null);this.pager(null);$("filters").hidden=true;
          this.replaceSummary([page,detail.updated_on],()=>[node('span','Learning article · updated '+detail.updated_on,{class:'badge'})]);
          $("evidence-context").textContent='Authored explanation, not test acceptance. Measurements are dated; linked test pages may show newer evidence.';
          const key=JSON.stringify([page,detail.source_sha256]);
          if(this.contentKey!==key){$("details").replaceChildren(learningArticle(detail.article_html));this.contentKey=key;}
        }else{
        this.explanation(detail);
        this.replaceSummary([page,pageLabel(detail),detail.counts],()=>[
          node("div",(detail.kind==='tests'?'Selected execution · ':'')+pageLabel(detail),{class:"badge",'data-state':detail.state}),
          ...(detail.kind==="tests"?Object.entries(detail.counts).map(([label,count])=>this.metric(label,String(count))):[])]);
        $("evidence-context").textContent=detail.kind==='check' ? "Named check in the selected test execution. Parent test: "+detail.parent_outcome+". A passing check does not make the whole test pass." : detail.kind==="demo" ? "DEMONSTRATION · illustrative visuals, never test acceptance." :
          "Evidence covers the collected checks for this page in one execution, not the entire feature contract. Visuals appear below when published; no fabricated figures.";
        this.execution(detail.evidence,detail.history);
        if(detail.evidence){
          this.staticEvidence=detail.kind==='demo';
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
      }
      this.displayedSelection=offlineOverview?null:JSON.stringify(this.selected);if(!offlineOverview)this.lastSuccess=new Date().toLocaleTimeString();
      $("error").hidden=!offlineOverview;
      if(offlineOverview){$('error').textContent='The evidence index is temporarily unavailable. Showing the dated M1 checkpoint; live results and demonstration links will return when the connection recovers.';$('connection').textContent='Offline · dated checkpoint';}
    } catch(error){
      if(generation!==this.generation)return;
      if(this.displayedSelection===JSON.stringify(this.selected) && !(error.status>=400 && error.status<500)){
        $("error").hidden=false;$("error").textContent="Refresh unavailable; showing last received evidence from "+this.lastSuccess+". "+(error.name==='AbortError'?'The server took too long to respond; retrying automatically.':error.message);
        $("connection").textContent="Offline · showing last received evidence";return;
      }
      this.closePanels();this.contentKey=null;this.summaryKey=null;this.pagerKey=null;this.execution(null);
      for(const id of ["details","summary","pager","evidence-context","explanation"])$(id).replaceChildren();
      this.guideKey=null;this.pageTitle=null;this.titlePage=null;
      $("error").hidden=false;$('error').textContent=error.name==='AbortError'?'The evidence index is taking too long to respond. Retrying automatically.':error.message;$('connection').textContent='Unavailable';
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
        node("span",page.group+" · "+({article:'Learning article',demo:'Demonstration',test:'Test',check:'Named check',tests:'Feature'}[page.kind]||page.kind),{class:"creation-kind"}));
      item.append(b);list.append(item);
    }
    return list;
  }
  header(title,description,eyebrow) {
    const referenceBar=$("reference-bar");if(referenceBar)referenceBar.hidden=this.selected.page==='overview';
    $("description").hidden=false;
    $("copy-details").hidden=this.selected.page.startsWith('learn/');
    this.pageTitle=title;this.titlePage=this.selected.page;
    $("reference-label").textContent=referenceLabel(this.selected.page,title);
    if(!this.selected.page.startsWith('test/')&&!this.selected.page.startsWith('check/')){
      $("explanation").hidden=true;this.guideKey=null;
    }
    for(const [id,value] of [["title",title],["description",description],["eyebrow",eyebrow]])if($(id).textContent!==value)$(id).textContent=value;
  }
  explanation(detail) {
    const box=$("explanation"),guide=detail.explanation;
    if(!guide&&detail.learning_article){
      box.hidden=false;
      const key='lesson:'+detail.learning_article;
      if(this.guideKey!==key){box.replaceChildren(node('a','Read the mathematics behind these checks',
        {href:reference(location.href,{page:detail.learning_article})}));this.guideKey=key;}
      return;
    }
    box.hidden=!guide;if(!guide)return;
    $("description").hidden=true;
    const key=JSON.stringify([detail.id,guide,detail.nodeid]);if(this.guideKey===key)return;
    this.guideKey=key;
    const sections=[['What this checks',guide.purpose],['How it checks it',guide.method],
      ['What a pass means — and does not mean',guide.limits]];
    box.replaceChildren(...sections.map(([label,text],i)=>{
      const section=node(i?'details':'section');section.append(node(i?'summary':'h2',label),node('p',text));return section;
    }));
    const technical=node('details');technical.append(node('summary','Technical identifiers'),
      node('p','Current author guide, separate from historical execution evidence. A skipped or expected-failure result is not acceptance.'),node('p','Test source'),node('code',detail.nodeid||''),node('p','Stable page ID'),node('code',detail.id));box.append(technical);
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
    const key=JSON.stringify(['coverage',cov.evidence?.run,cov.artifact_sha256,cov.items,this.offset,cov.measured,cov.layer,cov.layers]);
    if(this.contentKey===key)return;
    this.contentKey=key;
    const opened=new Set([...$("details").querySelectorAll("details[open]")].map(d=>d.dataset.file));
    const overview=node("section","",{class:"coverage-overview"});
    const layers=node('div','',{class:'coverage-layers','aria-label':'Coverage layers'});
    for(const layer of cov.layers||[]){const b=button(layer.label+(layer.measured?'':' · not measured')+(layer.source_files?' · '+layer.source_files+' source files':''),()=>{this.coverageLayer=layer.id;this.offset=0;this.generation++;this.contentKey=null;this.poll();});b.setAttribute('aria-pressed',String((cov.layer||'python')===layer.id));layers.append(b);}
    overview.append(layers,this.coverageMeter("Reported line coverage",cov.measured?cov.covered:null,cov.measured?cov.total:null,"lines"));
    if(cov.unmeasured_files?.length){const gaps=node('details');gaps.append(node('summary',cov.unmeasured_files.length+' source files not represented in this report'),node('p','This includes tests, headers or files outside the instrumented build; executable-line counts are unknown, not zero.'),node('pre',cov.unmeasured_files.join('\n')));overview.append(gaps);}
    if(cov.measured)overview.append(node("p",(cov.total-cov.covered).toLocaleString('en-US')+" uncovered lines · "+cov.total_files+" instrumented files · "+cov.source,{class:"coverage-count"}));
    this.summaryKey=null;$("summary").replaceChildren(overview);
    if(!cov.measured){$("details").replaceChildren(node("p","No coverage report for this execution yet."));return;}
    const files=node("div","",{class:"coverage-files"});
    for(const f of cov.items){
      const label=coverageFile(f.path), card=node("article","",{class:"coverage-file"});
      card.append(node("p",label.directory||"Source file",{class:"coverage-directory"}),node("h3",label.name));
      card.append(this.coverageMeter("Lines",f.covered,f.total,"lines"),this.coverageMeter("Branches",f.covered_branches,f.branches,"branches"));
      if(f.functions!=null)card.append(this.coverageMeter('Functions',f.covered_functions,f.functions,'functions'));
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
      for(const p of pages){if(!groups.has(p.group)){const section=node("details","",{class:"nav-group"});section.open=false;section.append(node("summary",p.group));groups.set(p.group,section);$("page-groups").append(section);}
        const b=button(p.title,()=>this.navigate(p.id));b.dataset.page=p.id;b.setAttribute("aria-label",p.title);b.append(node("span","",{'data-page-status':p.id,class:"nav-status"}));groups.get(p.group).append(b);}
      this.catalogKey=key;
    }
    for(const p of pages){const badge=[...document.querySelectorAll("[data-page-status]")].find(b=>b.dataset.pageStatus===p.id);badge.textContent=p.kind==='article'?'article':p.evidence?(p.evidence.stale?"stale":p.kind==='tests'?"checks: "+p.state:p.state):"not measured";badge.setAttribute("data-state",p.state);}
  }
  execution(evidence, historyItems=[]) {
    this.evidenceRun=evidence?.run||"";this.provenance=evidence?.provenance||{};
    const key=JSON.stringify([this.selected.page,this.selected.run,evidence?.run,evidence?.pinned,evidence?.state,evidence?.stale,evidence?.coverage,evidence?.diagnostic,evidence?.provenance,historyItems.map(r=>r.run)]);
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
    details.append(button(evidence.pinned?'Unpin this evidence':'Pin this evidence',async()=>{
      try{
        const response=await fetch('/testing/api/pin',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:evidence.run,pinned:!evidence.pinned})});
        const result=await response.json();if(!response.ok)throw Error(result.error||'Pin update failed');
        this.executionKey=null;this.poll();
      }catch(error){$('error').hidden=false;$('error').textContent=error.message;}
    }));
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
    this.disposeCinema?.();this.disposeCinema=null;
    this.disposeShowcase?.();this.disposeShowcase=null;
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
      this.disposeCinema?.();this.disposeCinema=null;
      this.disposeShowcase?.();this.disposeShowcase=null;
      for(const panel of this.panels)panel.destroy();
      this.panels=[];$("evidence").replaceChildren();
      let section=null,grid=null;
      for(const spec of this.layout.panels){
        if(this.staticEvidence && (spec.kind==='text'||!section)){
          section=node('section','',{class:'artifact-section'});grid=node('div','',{class:'artifact-pair'});
          const title=(spec.label||spec.path.split('/').at(-1)).replace(/-explained$/,'').replace(/-/g,' ');
          section.append(node('h2',title),grid);$("evidence").append(section);
        }
        const panel=new PANEL_KINDS[spec.kind](this,panelBinding(this.store,spec));this.panels.push(panel);
        panel.el.dataset.kind=spec.kind;
        if(this.staticEvidence){
          panel.el.classList.add('static-artifact');
          if(spec.kind==='text'){
            const explanation=node('details','',{class:'demo-explanation'});
            explanation.open=!this.layout.panels.some(p=>['image','mesh','plot'].includes(p.kind));
            explanation.append(node('summary','Explanation, mathematics & limits'),panel.el);section.append(explanation);
          }else if(spec.kind==='record'){const technical=node('details','',{class:'demo-explanation'});technical.append(node('summary','Technical record · '+(spec.label||spec.path)),panel.el);grid.append(technical);}else grid.append(panel.el);
        }else $("evidence").append(panel.el);
        if(spec.label){const label=panel.head.querySelector('.path');if(label)label.textContent=spec.label;}
        const available=this.store.paths.has(spec.path)||(spec.kind==='plot'&&[...this.store.paths.keys()].some(path=>path.startsWith(spec.path+'/')));
        if(!available)panel.body.replaceChildren(node("p","Artifact unavailable: not retained or not delivered."));
        if(available&&this.staticEvidence&&this.selected.step===null&&typeof panel.visibleSteps==='function'&&!demoClipGroup(spec.path))attachPlayback(panel);
        if(this.selected.path===spec.path && this.selected.step!==null){
          if(typeof panel.showStep==="function"){panel.following=false;panel.showStep(this.selected.step);}
          else{panel.unsubscribe();panel.body.replaceChildren(node("p","Pinned steps unsupported for this panel; no latest-value substitute."));}
        }
      }
      if(this.staticEvidence&&this.selected.step===null){enhanceDemoPanels(this);this.disposeShowcase=attachShowcase(this,$("evidence"));}
      if(this.staticEvidence&&!this.selected.path&&this.selected.step===null){try{this.disposeCinema=attachCinema(this,$("evidence"))||attachGallery(this,$("evidence"));}catch(error){$("error").hidden=false;$("error").textContent=error.message;}}
      if(this.selected.path&&!artifacts.some(a=>a.path===this.selected.path)){this.closePanels();$("evidence").textContent="Referenced artifact is not on this page.";}
    };
    this.connection.connect();
  }
  async copyReference(path=this.selected.path,step=this.selected.step,detailed=false) {
    path=path?.replace(/\/$/,'')??path;
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
