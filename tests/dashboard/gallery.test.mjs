import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import {validateGallery,attachGallery} from '../../tools/testing_dashboard/static/cinema.js';
const asset={url:'/testing/media/'+'a'.repeat(64)+'.png',sha256:'a'.repeat(64),bytes:80};
const gallery={version:1,title:'Saved images',images:[{label:'Actual output',caption:'Measured fixture',asset}]};
test('gallery refuses external, excessive and unidentified saved images',()=>{
 assert.equal(validateGallery(gallery),gallery);
 for(const bad of [{...gallery,images:[]},{...gallery,images:Array(17).fill(gallery.images[0])},{...gallery,images:[{...gallery.images[0],asset:{...asset,url:'https://example.com/x'}}]},{...gallery,images:[{...gallery.images[0],asset:{...asset,bytes:9*1024*1024}}]}])assert.throws(()=>validateGallery(bad));
});
test('gallery loads only the selected image and closes decoded resources',async()=>{
 const bytes=new Uint8Array(24);bytes.set([137,80,78,71,13,10,26,10]);new DataView(bytes.buffer).setUint32(16,320);new DataView(bytes.buffer).setUint32(20,180);
 const sha256=createHash('sha256').update(bytes).digest('hex');const a={sha256,bytes:24,url:'/testing/media/'+sha256+'.png'};
 const nodes=[],images=[],requests=[];
 class Element{constructor(tag){this.tag=tag;this.children=[];this.style={};nodes.push(this);}append(...xs){this.children.push(...xs);}prepend(x){this.children.unshift(x);}setAttribute(){}getContext(){return {drawImage(){}};}remove(){this.removed=true;}}
 const old={document:globalThis.document,fetch:globalThis.fetch,createImageBitmap:globalThis.createImageBitmap};
 globalThis.document={createElement:tag=>new Element(tag)};
 globalThis.fetch=async url=>{requests.push(url);return new Response(bytes);};
 globalThis.createImageBitmap=async()=>{const image={width:320,height:180,closed:false,close(){this.closed=true;}};images.push(image);return image;};
 try{
  const value={...gallery,images:[{label:'Before',caption:'Initial state',asset:a},{label:'After',caption:'Saved result',asset:a}]};
  const dispose=attachGallery({store:{paths:new Map([['presentation/gallery',{latest:{value}}]])}},new Element('host'));
  await new Promise(r=>setTimeout(r,30));assert.equal(requests.length,1);
  nodes.find(n=>n.tag==='button'&&n.textContent==='After').onclick();await new Promise(r=>setTimeout(r,30));
  assert.equal(requests.length,2);assert.equal(images[0].closed,true);dispose();assert.ok(images.every(i=>i.closed));
 }finally{Object.assign(globalThis,old);}
});
