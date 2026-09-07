// Finite, on-demand PNG presentation. No autoplay, producer commands or history writes.
import {ClipPlayer,syncPlots} from './playback.js';
const MAX_BYTES=8*1024*1024;
export function validateFilm(film){
  const asset=a=>{if(!a||!Number.isInteger(a.bytes)||a.bytes<1||a.bytes>MAX_BYTES||
    !/^[a-f0-9]{64}$/.test(a.sha256)||a.url!=='/testing/media/'+a.sha256+'.png')throw Error('Invalid presentation asset');};
  if(!film||film.version!==1||!['fit','lights'].includes(film.mode)||!Number.isInteger(film.width)||!Number.isInteger(film.height)||
    film.width<1||film.width>1920||film.height<1||film.height>1080||!Array.isArray(film.frames)||film.frames.length<2||film.frames.length>96)throw Error('Invalid presentation manifest');
  asset(film.target);asset(film.poster);let previous=-1;
  for(const f of film.frames){if(!Number.isInteger(f.step)||f.step<=previous)throw Error('Invalid presentation steps');previous=f.step;asset(f.image);asset(f.director);}
  return film;
}

async function bitmap(asset,signal){
  const response=await fetch(asset.url,{signal,cache:'force-cache'});if(!response.ok)throw Error('Saved presentation image unavailable');
  const reader=response.body.getReader();let size=0,chunks=[];
  try{for(;;){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>MAX_BYTES||size>asset.bytes)throw Error('Presentation byte budget exceeded');chunks.push(value);}}
  finally{await reader.cancel();}
  if(size!==asset.bytes)throw Error('Incomplete presentation image');
  const bytes=new Uint8Array(size);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length;}
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==asset.sha256)throw Error('Presentation digest mismatch');
  const header=new DataView(bytes.buffer);if(size<24||header.getUint32(16)>1920||header.getUint32(20)>1080)throw Error('Presentation pixel budget exceeded');
  const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
  if(signal.aborted){image.close();throw Error('Presentation cancelled');}return image;
}

export function attachCinema(app,host){
  const raw=app.store.paths.get('presentation/film')?.latest?.value;if(!raw)return null;
  const film=validateFilm(raw),root=document.createElement('section');root.className='cinema';
  const el=(tag,text='',cls='')=>{const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;};
  // The page already supplies a title and explanation; keep the presentation visual-first.
  root.setAttribute('aria-label',film.title);
  const grid=el('div','','cinema-grid'),hero=el('div','','cinema-hero'),side=el('div','','cinema-side');
  const canvas=el('canvas'),targetCanvas=el('canvas'),director=el('canvas');canvas.width=targetCanvas.width=film.width;canvas.height=targetCanvas.height=film.height;
  canvas.setAttribute('aria-label','Rendered camera estimate and target comparison');
  const badge=el('span','Estimated view','cinema-badge');hero.append(canvas,badge);
  const targetTitle=el('h3',film.mode==='fit'?'The shot to recover':'Lighting reference'),directorTitle=el('h3','The camera in the scene');
  side.append(targetTitle,targetCanvas,directorTitle,director);grid.append(hero,side);
  const controls=el('div','','cinema-controls'),play=el('button','Play'),restart=el('button','Restart'),expand=el('button','Focus view'),timeline=el('input'),state=el('output','Loading saved images…');
  timeline.type='range';timeline.min=0;timeline.max=film.frames.length-1;timeline.value=0;timeline.setAttribute('aria-label','Cinematic saved timeline');
  const rate=el('select');rate.setAttribute('aria-label','Cinematic playback speed');
  for(const value of [4,8,12,24,30,60]){const option=el('option',value+' fps max');option.value=value;rate.append(option);}rate.value='4';
  controls.append(play,restart,expand,timeline,rate,state);
  expand.setAttribute('aria-pressed','false');expand.onclick=()=>{const on=root.classList.toggle('cinema-focus');expand.setAttribute('aria-pressed',String(on));expand.textContent=on?'Show reference views':'Focus view';};
  const wipeLabel=el('label','Compare with target'),wipe=el('input');wipe.type='range';wipe.min=0;wipe.max=100;wipe.value=0;wipe.setAttribute('aria-label','Target comparison wipe');wipeLabel.append(wipe);wipeLabel.hidden=film.mode!=='fit';
  const poster=el('a','Open the 1080p still');poster.href=film.poster.url;poster.target='_blank';poster.rel='noopener';
  const note=el('p',`${film.width} × ${film.height} · finite saved frames · ${film.objective}.`,'cinema-note');
  root.append(grid,controls,wipeLabel,poster,note);
  const diagnostics=el('details','','cinema-diagnostics');diagnostics.append(el('summary','Measurements, method & rendering limits'));
  for(const child of [...host.children])diagnostics.append(child);host.append(root,diagnostics);
  let target=null,current=null,closed=false,request=null,ticket=0;
  const draw=()=>{if(!current)return;const ctx=canvas.getContext('2d');ctx.drawImage(current,0,0,canvas.width,canvas.height);
    const width=Number(wipe.value)/100*canvas.width;
    if(target&&width>0){ctx.save();ctx.beginPath();ctx.rect(0,0,width,canvas.height);ctx.clip();ctx.drawImage(target,0,0,canvas.width,canvas.height);ctx.restore();ctx.fillStyle='#75e6d0';ctx.fillRect(width-1,0,2,canvas.height);}
  };
  const player=new ClipPlayer(film.frames.map(f=>f.step),async(step,index)=>{
    const id=++ticket;request?.abort();request=new AbortController();const signal=request.signal;
    if(!target){target=await bitmap(film.target,signal);targetCanvas.getContext('2d').drawImage(target,0,0);}
    const f=film.frames[index];const results=await Promise.allSettled([bitmap(f.image,signal),bitmap(f.director,signal)]);
    if(closed||id!==ticket||results.some(r=>r.status==='rejected')){for(const r of results)if(r.status==='fulfilled')r.value.close();throw Error('Presentation unavailable or cancelled');}
    current?.close();current=results[0].value;const outside=results[1].value;director.width=outside.width;director.height=outside.height;director.getContext('2d').drawImage(outside,0,0);outside.close();draw();
    syncPlots(app,step,'presentation/');
  },{onState:p=>{play.textContent=p.running?'Pause':'Play';timeline.value=Math.max(0,p.index);const f=film.frames[Math.max(0,p.index)];
    state.textContent=`${p.running?'Playing':'Paused'} · frame ${Math.max(1,p.index+1)}/${film.frames.length} · ${film.mode==='fit'?'optimizer step ':'frame '}${f.step}`;
    badge.textContent=film.mode==='fit'?(p.index===film.frames.length-1?'Recovered camera':'Estimated camera'):'Moving light study';
  },onError:e=>{state.textContent=e.message;}});
  player.setRate(4);play.onclick=()=>player.running?player.pause():player.play();restart.onclick=()=>player.seek(0);timeline.oninput=()=>player.seek(Number(timeline.value));rate.onchange=()=>player.setRate(Number(rate.value));wipe.oninput=draw;
  const hidden=()=>{if(document.hidden){player.pause();request?.abort();}};document.addEventListener('visibilitychange',hidden);player.seek(0);
  return ()=>{closed=true;ticket++;player.dispose();request?.abort();target?.close();current?.close();document.removeEventListener('visibilitychange',hidden);root.remove();diagnostics.remove();};
}
