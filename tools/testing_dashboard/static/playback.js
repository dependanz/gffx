// Bounded finite-clip playback. No producer commands, autoplay or persistent writes.
export const PLAYBACK_RATES=Object.freeze([4,8,12,24,30,60]);

export function demoClipGroup(path) {
  const match=/^studio\/([a-z-]+)\/(color|normals|depth_image|director)$/.exec(path);
  return match?{group:match[1],view:match[2]}:null;
}
export function commonClipSteps(lists) {
  if(!lists.length)return [];
  return [...new Set(lists[0])].filter(step=>lists.every(list=>list.includes(step))).sort((a,b)=>a-b);
}

export function highlightSwatch(panel,index) {
  if(!panel?.canvas||!panel.image||!Number.isInteger(index)||index<0||index>2)return;
  panel.show(panel.image);
  const copy=document.createElement('canvas');copy.width=panel.canvas.width;copy.height=panel.canvas.height;
  copy.getContext('2d').drawImage(panel.canvas,0,0);
  // Display-only upscale: a fractional-pixel outline on the 2x2 source is invisible.
  panel.canvas.width=panel.canvas.height=128;
  const context=panel.canvas.getContext('2d');context.imageSmoothingEnabled=false;
  context.drawImage(copy,0,0,128,128);context.strokeStyle='#ff00ff';context.lineWidth=4;
  context.strokeRect(index%2*64+4,Math.floor(index/2)*64+4,56,56);
  panel.overlay.textContent=`Triangle vertex ${index} · matching RGB swatch`;
}

export function syncPlots(app,step,path) {
  const prefix=path.startsWith('studio/')?'studio/':path.startsWith('camera-fit/')?'camera-fit/':path.split('/')[0]+'/';
  for(const plot of app.panels.filter(p=>typeof p.series==='function'&&p.spec.path.startsWith(prefix)&&!p.spec.path.includes('derivative'))){
    if(!plot._demoDraw){
      plot._demoDraw=plot.draw.bind(plot);
      plot.draw=()=>{
        plot._demoDraw();const m=plot.map;if(!m||!Number.isFinite(plot.demoStep))return;
        const ctx=plot.canvas.getContext('2d');ctx.strokeStyle='#168c80';ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(m.px(plot.demoStep),m.pad.t);ctx.lineTo(m.px(plot.demoStep),m.h-m.pad.b);ctx.stroke();
        plot.overlay.textContent=plot.series().map(([name,e])=>{
          const exact=e.values.find(pair=>pair[0]===plot.demoStep);
          return `${name.split('/').at(-1)}: ${exact?Number(exact[1]).toPrecision(4):'no saved scalar at this step'} @ ${plot.demoStep}`;
        }).join('\n');
      };
    }
    plot.demoStep=step;plot.draw();
  }
}

export function enhanceDemoPanels(app) {
  for(const panel of app.panels){
    if(panel.spec.kind==='mesh')panel.el.classList.add('demo-mesh-stage');
    const info=app.store.paths.get(panel.spec.path+'-inspection')?.latest?.value;
    if(info&&panel.view){
      let normals=false;const originalPick=panel.view.onPick;
      const drawOverlay=panel.view._drawOverlay.bind(panel.view);
      panel.view._drawOverlay=mvp=>{
        drawOverlay(mvp);const face=panel.pick?.face;if(!mvp||!Number.isInteger(face)||!panel.mesh)return;
        const ids=Array.from(panel.mesh.faces.slice(face*3,face*3+3));
        const center=[0,1,2].map(axis=>ids.reduce((sum,i)=>sum+panel.mesh.vertices[i*3+axis],0)/3);
        const end=center.map((x,i)=>x+info.face_normals[face][i]*panel.view.orbit.radius*.18);
        const project=p=>{const c=[0,1,2,3].map(row=>mvp[row]*p[0]+mvp[4+row]*p[1]+mvp[8+row]*p[2]+mvp[12+row]);return c[3]>0?[(c[0]/c[3]*.5+.5)*panel.canvas.clientWidth,(.5-c[1]/c[3]*.5)*panel.canvas.clientHeight]:null;};
        const a=project(center),b=project(end);if(!a||!b)return;
        const ctx=panel.marker.getContext('2d'),angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.strokeStyle='#ffb454';ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);for(const sign of [-1,1]){ctx.moveTo(...b);ctx.lineTo(b[0]-8*Math.cos(angle+sign*.5),b[1]-8*Math.sin(angle+sign*.5));}ctx.stroke();
      };
      panel.tool('normal colours','Show gffx vertex normals as RGB',()=>{
        normals=!normals;if(!panel.mesh)return;
        panel.view.setMesh({...panel.mesh,colors:normals?new Float32Array(info.normal_colors.flat()):panel.mesh.colors});
      });
      panel.view.onPick=pick=>{
        originalPick?.(pick);
        if(pick&&Number.isInteger(pick.face)){
          panel.overlay.textContent+=`\narea ${Number(info.areas[pick.face]).toPrecision(5)} · native gffx\nnormal (${info.face_normals[pick.face].map(x=>Number(x).toFixed(3)).join(', ')})`;
        }
      };
    }
    if(panel.spec.path==='demo/triangle'&&panel.view){
      const originalPick=panel.view.onPick;
      panel.view.onPick=pick=>{
        originalPick?.(pick);if(!pick)return;
        const image=app.panels.find(p=>p.spec.path==='demo/image');
        highlightSwatch(image,pick.index);
      };
    }
  }
}

// One serial timeline owns all selected views. Switching clips cancels old work.
export function attachShowcase(app,host) {
  const clips=app.panels.filter(p=>demoClipGroup(p.spec.path));
  if(!clips.length)return null;
  const bar=document.createElement('div');bar.className='showcase-controls';
  const create=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;bar.append(e);return e;};
  const choose=create('select','');choose.setAttribute('aria-label','Scene demonstration');
  const groups=[...new Set(clips.map(p=>demoClipGroup(p.spec.path).group))];
  for(const group of groups){const o=document.createElement('option');o.value=group;o.textContent=group.replaceAll('-',' ');choose.append(o);}
  const view=create('select','');view.setAttribute('aria-label','Render view');
  const play=create('button','Play');const slider=create('input','');slider.type='range';slider.min=0;slider.step=1;slider.setAttribute('aria-label','Shared scene timeline');
  const speed=create('select','');speed.setAttribute('aria-label','Scene playback speed');
  for(const fps of PLAYBACK_RATES){const o=document.createElement('option');o.value=fps;o.textContent=fps+' fps max';speed.append(o);}speed.value='8';
  const label=create('output','');let player=null,active=[],closed=false;
  function invalidate(){player?.dispose();for(const p of clips)p._showRequest++;}
  async function select(){
    invalidate();if(closed)return;
    const group=choose.value;
    const options=clips.filter(p=>demoClipGroup(p.spec.path).group===group&&demoClipGroup(p.spec.path).view!=='director');
    const chosen=options.find(p=>demoClipGroup(p.spec.path).view===view.value)||options[0];
    view.replaceChildren();for(const p of options){const o=document.createElement('option');o.value=demoClipGroup(p.spec.path).view;o.textContent=o.value.replaceAll('_',' ');view.append(o);}
    view.value=demoClipGroup(chosen.spec.path).view;view.hidden=options.length<2;
    active=clips.filter(p=>p===chosen||(demoClipGroup(p.spec.path).group===group&&demoClipGroup(p.spec.path).view==='director'));
    for(const p of clips){p.el.hidden=!active.includes(p);p.el.classList.add('showcase-panel');p.el.classList.toggle('single-view',active.length===1);}
    const heading=chosen.el.closest('section')?.querySelector('h2');if(heading)heading.textContent=group.replaceAll('-',' ')+' · '+view.value.replaceAll('_',' ');
    for(const p of app.panels)if(p.spec.path==='studio/light-z')p.el.hidden=group!=='lights';
    const steps=commonClipSteps(active.map(p=>p.visibleSteps()));slider.max=Math.max(0,steps.length-1);
    const selected=[...active];
    player=new ClipPlayer(steps,async(step,index)=>{
      await Promise.all(selected.map(async p=>{p.following=false;await p.showStep(step);if(p.shown!==step)throw Error('Selected scene frame unavailable');}));
    },{onState:p=>{play.textContent=p.running?'Pause':'Play';slider.value=Math.max(0,p.index);label.textContent=`${p.running?'Playing':'Paused'} · ${group} · frame ${Math.max(1,p.index+1)} / ${steps.length} · saved clip`;
      syncPlots(app,steps[Math.max(0,p.index)],chosen.spec.path);
    },onError:error=>label.textContent=error.message});
    player.setRate(Number(speed.value));if(steps.length)await player.seek(0);
  }
  choose.onchange=select;view.onchange=select;play.onclick=()=>player.running?player.pause():player.play();
  slider.oninput=()=>player.seek(Number(slider.value));speed.onchange=()=>player.setRate(Number(speed.value));
  const hidden=()=>{if(document.hidden){for(const p of active)p._showRequest++;player?.pause();}};
  document.addEventListener('visibilitychange',hidden);host.prepend(bar);select();
  return ()=>{closed=true;invalidate();document.removeEventListener('visibilitychange',hidden);bar.remove();};
}
export class ClipPlayer {
  constructor(steps,show,{schedule=(fn,ms)=>setTimeout(fn,ms),cancel=id=>clearTimeout(id),now=()=>performance.now(),onState=()=>{},onError=()=>{}}={}) {
    this.steps=[...steps];this.show=show;this.schedule=schedule;this.cancel=cancel;
    this.onState=onState;this.onError=onError;this.now=now;this.index=-1;this.rate=8;
    this.running=false;this.disposed=false;this.generation=0;this.timer=null;
  }
  setRate(rate) {if(!PLAYBACK_RATES.includes(rate))throw new Error('Unsupported playback rate');this.rate=rate;}
  pause() {this.running=false;this.generation++;if(this.timer!==null)this.cancel(this.timer);this.timer=null;this.onState(this);}
  async frame(index,generation) {
    try {
      await this.show(this.steps[index],index);
      if(this.disposed||generation!==this.generation)return false;
      this.index=index;this.onState(this);return true;
    } catch(error) {if(generation===this.generation){this.pause();this.onError(error);}return false;}
  }
  async seek(index) {
    if(!Number.isInteger(index)||index<0||index>=this.steps.length)throw new Error('Invalid clip frame');
    if(this.disposed)return;this.pause();await this.frame(index,this.generation);
  }
  async play() {
    if(this.disposed||this.running||!this.steps.length)return;
    if(this.index===this.steps.length-1)this.index=-1;
    this.running=true;const generation=++this.generation;this.onState(this);
    const tick=async()=>{
      if(!this.running||this.disposed||generation!==this.generation)return;
      const started=this.now();
      if(!await this.frame(this.index+1,generation))return;
      if(this.index===this.steps.length-1){this.pause();return;}
      // Schedule only after display finishes: network latency cannot build a frame backlog.
      if(this.running&&generation===this.generation)this.timer=this.schedule(tick,Math.max(0,1000/this.rate-(this.now()-started)));
    };
    await tick();
  }
  dispose() {this.pause();this.disposed=true;}
}

export function attachPlayback(panel) {
  const steps=[...panel.visibleSteps()];
  if(steps.length<2)return;
  const bar=document.createElement('div');bar.className='clip-controls';
  const make=(text,action)=>{const b=document.createElement('button');b.textContent=text;b.onclick=action;bar.append(b);return b;};
  const play=make('Play',()=>player.running?pause():player.play());
  const previous=make('Previous frame',()=>player.seek(Math.max(0,player.index-1)));
  const next=make('Next frame',()=>player.seek(Math.min(steps.length-1,player.index+1)));
  const speed=document.createElement('select');speed.setAttribute('aria-label','Playback speed');
  for(const rate of PLAYBACK_RATES){const option=document.createElement('option');option.value=rate;option.textContent=rate+' fps max';speed.append(option);}
  speed.value='8';speed.onchange=()=>player.setRate(Number(speed.value));bar.append(speed);
  const status=document.createElement('span');status.className='clip-status';bar.append(status);
  const energy=[];
  if(panel.spec.path==='ball/motion'){
    make('Impact close-up',async()=>{
      const values=panel.store.paths.get('ball/parts/spring')?.values||[];
      const maximum=values.reduce((a,b)=>b[1]>a[1]?b:a,[0,-Infinity]);
      const index=steps.indexOf(maximum[0]);if(index<0)return;await player.seek(index);
      if(panel.view){panel.view.orbit.target=[0,.45,0];panel.view.orbit.distance=2.3;panel.view.orbit.pitch=.2;panel.view.requestRender();}
    });
    make('Full stage',()=>{if(panel.view&&panel.mesh)panel.view.setMesh(panel.mesh,{keepCamera:false});});
    const parts=document.createElement('div');parts.className='energy-parts';
    for(const [key,title] of [['potential','Height'],['kinetic','Motion'],['spring','Compression'],['energy_difference','Energy difference from start']]){
      const row=document.createElement('label'),meter=document.createElement('meter'),value=document.createElement('output');
      meter.min=0;meter.max=19.62;meter.setAttribute('aria-label',title+' energy in joules');row.append(title,meter,value);parts.append(row);energy.push({key,meter,value});
    }
    panel.el.append(parts);
  }
  const player=new ClipPlayer(steps,async(step,index)=>{
    panel.following=false;panel.liveButton.classList.remove('on');panel.range.value=index;
    await panel.showStep(step);
    if(panel.shown!==step)throw new Error('Frame unavailable or could not be displayed');
    syncPlots(panel.app,step,panel.spec.path);
  },{onState:p=>{
    play.textContent=p.running?'Pause':'Play';previous.disabled=p.index<=0;next.disabled=p.index>=steps.length-1;
    status.textContent=(p.running?'Playing':'Paused')+' · frame '+Math.max(1,p.index+1)+' / '+steps.length;
    for(const part of energy){
      const value=panel.store.paths.get('ball/parts/'+part.key)?.values?.find(([s])=>s===steps[Math.max(0,p.index)])?.[1];
      part.meter.hidden=!Number.isFinite(value);part.meter.value=value??0;part.value.textContent=Number.isFinite(value)?Number(value).toFixed(2)+' J':'not saved';
    }
  },onError:error=>{status.textContent=error.message;}});
  panel.el.insertBefore(bar,panel.body);
  // Capture prevents the shared scrub handler from issuing a duplicate frame request.
  const onInput=event=>{event.stopImmediatePropagation();player.seek(Number(panel.range.value));};
  panel.range.addEventListener('input',onInput,true);
  const pause=()=>{panel._showRequest++;player.pause();},hidden=()=>{if(document.hidden)pause();};
  panel.range.addEventListener('pointerdown',pause);panel.range.addEventListener('keydown',pause);
  document.addEventListener('visibilitychange',hidden);
  panel.liveButton.hidden=true;panel.checkpointButton.hidden=true;panel.olderButton.hidden=true;
  panel.policyLabel.textContent='finite saved clip';
  const destroy=panel.destroy.bind(panel);
  panel.destroy=()=>{panel._showRequest++;player.dispose();document.removeEventListener('visibilitychange',hidden);panel.range.removeEventListener('input',onInput,true);panel.range.removeEventListener('pointerdown',pause);panel.range.removeEventListener('keydown',pause);destroy();};
  player.seek(0);return player;
}
