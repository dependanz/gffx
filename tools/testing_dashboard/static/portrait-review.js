const ns='http://www.w3.org/2000/svg';
const el=(tag,text='',attrs={})=>{const n=document.createElement(tag);n.textContent=text;for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;};
const svgEl=(tag,attrs={})=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;};
export function portraitReview(data){
  const root=el('section','',{class:'portrait-review'});
  root.append(el('p','Your original photograph with draft observations. This is not a render or a fitted face.',{class:'portrait-notice'}));
  const controls=el('div','',{class:'portrait-controls'}),viewport=el('div','',{class:'portrait-viewport'});
  const svg=svgEl('svg',{role:'img','aria-label':'Original portrait with draft facial landmarks and skin mask'});
  svg.append(svgEl('image',{href:data.image_url,x:-.5,y:-.5,width:data.width,height:data.height}));
  const mask=svgEl('g'),points=svgEl('g'),numbers=svgEl('g');
  for(const poly of data.skin_polygons)mask.append(svgEl('polygon',{points:poly.map(p=>p.join(',')).join(' '),fill:'#04d7a3','fill-opacity':.24,stroke:'#00f4c4','stroke-width':1}));
  const selected=el('p','Tap a yellow point to identify it. Use its number when suggesting a correction.',{'aria-live':'polite',class:'portrait-selection'});
  data.landmarks.forEach(([x,y],i)=>{
    const circle=svgEl('circle',{cx:x,cy:y,r:2.4,fill:'#ffe55b',stroke:'#17202a','stroke-width':.65});
    const hit=svgEl('circle',{cx:x,cy:y,r:7,fill:'transparent',role:'button',tabindex:0,'aria-label':'Landmark '+i});
    const choose=()=>{selected.textContent=`Point ${i}: source pixel (${x}, ${y}). Tell me in chat if it should move.`;};
    hit.addEventListener('click',choose);hit.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});points.append(circle,hit);
    const label=svgEl('text',{x:x+4,y:y-4,fill:'#fff08a',stroke:'#13212a','stroke-width':.2,'font-size':7,'font-family':'system-ui','pointer-events':'none'});label.textContent=String(i);numbers.append(label);
  });
  svg.append(mask,points,numbers);numbers.style.display='none';viewport.append(svg);
  const toggle=(title,group,checked)=>{const label=el('label'),input=el('input','',{type:'checkbox'});input.checked=checked;input.addEventListener('change',()=>{group.style.display=input.checked?'':'none';});label.append(input,document.createTextNode(title));controls.append(label);};
  toggle('Landmarks',points,true);toggle('Point numbers',numbers,false);toggle('Skin mask',mask,true);
  let cropped=true;const crop=el('button','Show full photo');const setView=()=>{const [x,y,r,b]=cropped?data.crop_xyxy:[0,0,data.width,data.height];svg.setAttribute('viewBox',`${x-.5} ${y-.5} ${r-x} ${b-y}`);};setView();
  crop.onclick=()=>{cropped=!cropped;crop.textContent=cropped?'Show full photo':'Show face crop';setView();};controls.append(crop);
  const zoomLabel=el('label','Zoom '),zoom=el('input','',{type:'range',min:1,max:3,step:.25,value:1,'aria-label':'Portrait zoom'});zoom.oninput=()=>{svg.style.width=Number(zoom.value)*100+'%';};zoomLabel.append(zoom);controls.append(zoomLabel);
  root.append(controls,viewport,selected);
  const legend=el('div','',{class:'portrait-guide'});
  legend.append(el('h2','What to look for'),el('p','Yellow landmarks should follow the jaw, eyebrows, eyelids, nose and lips. The inner lip points are uncertain because the mouth is closed. The points were estimated visually; they are not detector measurements.'),el('p','Green regions are deliberately conservative skin samples. They should avoid hair, eyebrows, eyes, nostrils, lips, facial hair and background. Uncoloured skin is excluded from the photometric loss; it is not missing geometry.'),el('p','Reply in chat with any corrections, using point numbers or a region name. Viewing this page does not approve the annotations or start a fit.'));
  const provenance=el('details');provenance.append(el('summary','Input identity and coordinates'),el('p',`Unchanged ${data.width} × ${data.height} photo. Coordinates use source pixel centres; x right, y down. Crop: ${data.crop_xyxy.join(', ')}.`),el('code','Photo SHA256: '+data.image_sha256),el('br'),el('code','Annotation SHA256: '+data.annotation_sha256));
  root.append(legend,provenance);return root;
}
