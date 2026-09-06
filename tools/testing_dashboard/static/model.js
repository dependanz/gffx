export function selection(url) {
  const q = url.searchParams, step = q.get("step");
  if (step !== null && (!/^\d+$/.test(step) || !Number.isSafeInteger(Number(step)))) throw Error("Invalid pinned step");
  if (step !== null && (!q.get("run") || !q.get("path"))) throw Error("Pinned steps require an execution and artifact path");
  return {run:q.get("run") || "",page:q.get("page") || "overview",path:q.get("path"),step:step === null ? null : Number(step)};
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
export function referenceLabel(page,title) {
  return 'gffx · '+(title|| (page.startsWith('check/')?'Named check page':page.startsWith('test/')?'Test page':page));
}
