"""Self-contained HTML report: one file, no external assets, works offline."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Dict, Sequence, Union

from . import __version__
from . import text as T
from .analyze import TraceAnalysis


def _payload(a: TraceAnalysis) -> Dict:
    d = a.to_dict()
    for s in d["sentences"]:
        s["segments"] = T.segments(s["text"], drop_filler=True, mask=True)
    for c, chunk in zip(d["chunks"], a.trace.chunks):
        c["segments"] = T.segments(chunk.text)
    d["query_keys"] = T.query_keys(a.trace.query)
    return d


def build_report(analyses: Sequence[TraceAnalysis], *, title: str = "ragscope report") -> str:
    data = {
        "title": title,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "version": __version__,
        "traces": [_payload(a) for a in analyses],
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    return (_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob))


def write_report(analyses: Sequence[TraceAnalysis], path: Union[str, Path], **kwargs) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_report(analyses, **kwargs), encoding="utf-8")
    return p


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;
  --ok:#1f7a4d;--ok-bg:#e3f3ea;--warn:#9a6200;--warn-bg:#fdf1d8;--bad:#b42318;--bad-bg:#fde7e4;
  --info:#2457a6;--info-bg:#e6eefb;--hl:#ffe9a8;--hl-strong:#c9e7d5;--accent:#3b5bdb;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9c9b94;--line:#33332f;
  --ok:#5ccf91;--ok-bg:#183326;--warn:#f0b54a;--warn-bg:#3a2c10;--bad:#ff7b6b;--bad-bg:#3d1a16;
  --info:#8ab4ff;--info-bg:#1a2740;--hl:#5a4a12;--hl-strong:#1f4a33;--accent:#8ea6ff;}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;align-items:center;gap:24px;padding:14px 24px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:-.01em}
header h1 span{color:var(--accent)}
.stats{display:flex;gap:20px;color:var(--muted);font-size:13px;margin-left:auto}
.stats b{color:var(--ink);font-weight:600}
.layout{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 53px)}
nav{border-right:1px solid var(--line);background:var(--panel);overflow:auto;height:calc(100vh - 53px);position:sticky;top:53px}
nav input{width:calc(100% - 24px);margin:12px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit;font-size:13px}
.item{padding:10px 14px;border-left:3px solid transparent;cursor:pointer;border-bottom:1px solid var(--line)}
.item:hover{background:var(--bg)}
.item.active{border-left-color:var(--accent);background:var(--bg)}
.item .q{font-size:13.5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.item .meta{font-size:12px;color:var(--muted);margin-top:3px;display:flex;gap:8px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
main{padding:24px 32px 64px;max-width:1200px}
.query{font-size:21px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.chip{padding:5px 11px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-size:13px}
.chip b{font-weight:650}
.verdict{font-weight:650;text-transform:uppercase;letter-spacing:.04em;font-size:12px;border:0}
.v-grounded{background:var(--ok-bg);color:var(--ok)}
.v-partially{background:var(--warn-bg);color:var(--warn)}
.v-ungrounded{background:var(--bad-bg);color:var(--bad)}
.v-empty{background:var(--line);color:var(--muted)}
.issues{display:flex;flex-direction:column;gap:6px;margin-bottom:22px}
.issue{padding:8px 12px;border-radius:8px;font-size:13.5px;display:flex;gap:10px}
.issue .sev{font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding-top:2px;min-width:54px}
.issue.error{background:var(--bad-bg)}.issue.error .sev{color:var(--bad)}
.issue.warning{background:var(--warn-bg)}.issue.warning .sev{color:var(--warn)}
.issue.info{background:var(--info-bg)}.issue.info .sev{color:var(--info)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;align-items:start}
@media (max-width:1100px){.grid{grid-template-columns:1fr}.layout{grid-template-columns:240px 1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px;font-weight:650;display:flex;justify-content:space-between}
.panel h2 .hint{text-transform:none;letter-spacing:0;font-weight:400}
.answer{font-size:15.5px;line-height:1.75}
.s{border-radius:4px;padding:2px 1px;cursor:pointer;transition:outline .1s;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.s.supported{background:var(--ok-bg)}
.s.partial{background:var(--warn-bg)}
.s.unsupported{background:var(--bad-bg);text-decoration:underline wavy var(--bad);text-decoration-thickness:1px;text-underline-offset:4px}
.s.neutral{color:var(--muted)}
.s.sel{outline:2px solid var(--accent);outline-offset:1px}
.s .miss{color:var(--bad);font-weight:650}
.sdetail{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line);font-size:13.5px;color:var(--muted);min-height:44px}
.sdetail b{color:var(--ink)}
.bar{height:6px;border-radius:3px;background:var(--line);overflow:hidden;width:120px;display:inline-block;vertical-align:middle;margin:0 6px}
.bar i{display:block;height:100%}
.chunk{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px;transition:opacity .15s,border-color .15s}
.chunk.unused{opacity:.62}
.chunk.src{border-color:var(--ok);opacity:1;box-shadow:0 0 0 1px var(--ok)}
.chunk.dim{opacity:.35}
.chead{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin-bottom:6px}
.rank{font-family:var(--mono);font-weight:700}
.cid{font-family:var(--mono);color:var(--accent);word-break:break-all}
.tag{padding:1px 7px;border-radius:999px;font-size:11.5px;background:var(--line);color:var(--muted)}
.tag.ok{background:var(--ok-bg);color:var(--ok)}
.tag.warn{background:var(--warn-bg);color:var(--warn)}
.ctext{font-size:13.5px;line-height:1.6;white-space:pre-wrap}
mark{background:transparent;color:inherit;border-radius:3px}
mark.q{background:var(--hl)}
mark.m{background:var(--hl-strong);font-weight:600}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);margin-top:10px;flex-wrap:wrap}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px;background:var(--c)}
details{margin-top:18px}
summary{cursor:pointer;color:var(--muted);font-size:13px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font:12px/1.5 var(--mono)}
footer{color:var(--muted);font-size:12px;margin-top:40px}
.empty{color:var(--muted);font-style:italic}
</style>
</head>
<body>
<header><h1><span>rag</span>scope</h1><div class="stats" id="stats"></div></header>
<div class="layout">
  <nav><input id="filter" placeholder="Filter traces…" aria-label="Filter traces"><div id="list"></div></nav>
  <main id="main"></main>
</div>
<script id="ragscope-data" type="application/json">__DATA__</script>
<script>
(function(){
const DATA = JSON.parse(document.getElementById('ragscope-data').textContent);
const T = DATA.traces;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => x == null ? 'n/a' : Math.round(x*100) + '%';
const vclass = v => 'v-' + (v||'empty').split(' ')[0];
const vcolor = v => ({grounded:'var(--ok)','partially grounded':'var(--warn)',ungrounded:'var(--bad)'}[v]||'var(--muted)');
const lcolor = l => ({supported:'var(--ok)',partial:'var(--warn)',unsupported:'var(--bad)'}[l]||'var(--muted)');
let current = 0, selected = null;

function stats(){
  const g = T.map(t=>t.groundedness).filter(x=>x!=null);
  const avg = g.length ? g.reduce((a,b)=>a+b,0)/g.length : null;
  const flagged = T.filter(t=>t.sentences.some(s=>s.label==='unsupported')).length;
  document.getElementById('stats').innerHTML =
    `<span><b>${T.length}</b> traces</span><span>avg groundedness <b>${pct(avg)}</b></span>`+
    `<span><b style="color:${flagged?'var(--bad)':'var(--ok)'}">${flagged}</b> with unsupported claims</span>`;
}

function list(){
  const f = document.getElementById('filter').value.toLowerCase();
  document.getElementById('list').innerHTML = T.map((t,i)=>{
    const q = t.trace.query;
    if (f && !(q.toLowerCase().includes(f) || t.verdict.includes(f) || (t.trace.id||'').toLowerCase().includes(f))) return '';
    const errs = t.issues.filter(x=>x.severity==='error').length;
    return `<div class="item ${i===current?'active':''}" data-i="${i}">
      <div class="q">${esc(q)}</div>
      <div class="meta"><span class="dot" style="background:${vcolor(t.verdict)}"></span>${esc(t.verdict)}
      · ${pct(t.groundedness)}${errs?` · <span style="color:var(--bad)">${errs} error${errs>1?'s':''}</span>`:''}</div></div>`;
  }).join('') || '<div class="item empty">No matching traces</div>';
}

function segHTML(segs, mark){
  return segs.map(([txt,key]) => {
    const cls = key ? mark(key) : null;
    return cls ? `<mark class="${cls}">${esc(txt)}</mark>` : esc(txt);
  }).join('');
}

function render(){
  const t = T[current]; if(!t){document.getElementById('main').innerHTML='<p class="empty">No traces.</p>';return;}
  const tr = t.trace;
  const sel = selected!=null ? t.sentences[selected] : null;
  const qset = new Set(t.query_keys);
  const sset = sel ? new Set(sel.keys) : null;
  const allChunkKeys = new Set(t.chunks.flatMap(c=>c.segments.filter(s=>s[1]).map(s=>s[1])));
  const used = t.chunks.filter(c=>c.used).length;

  const answer = t.sentences.length ? t.sentences.map(s=>{
    const inner = s.label==='neutral' ? esc(s.text) :
      segHTML(s.segments, k => allChunkKeys.has(k) ? null : 'miss');
    return `<span class="s ${s.label} ${selected===s.index?'sel':''}" data-s="${s.index}" title="${esc(s.label)} · support ${s.support.toFixed(2)}">${inner}</span>`;
  }).join(' ') : '<span class="empty">(empty answer)</span>';

  let detail = '<span>Click a sentence to see which chunks support it.</span>';
  if (sel){
    const srcs = sel.sources.map(j=>`#${j+1} <span class="bar"><i style="width:${Math.round(sel.chunk_scores[j]*100)}%;background:var(--ok)"></i></span>${sel.chunk_scores[j].toFixed(2)}`).join('<br>');
    detail = `<b style="color:${lcolor(sel.label)}">${esc(sel.label.toUpperCase())}</b> · support ${sel.support.toFixed(2)} — ${esc(sel.reason)}`+
      (sel.missing_terms.length && sel.label!=='supported' ? `<br>Not found in any chunk: <b>${sel.missing_terms.map(esc).join(', ')}</b>`:'')+
      (srcs ? `<div style="margin-top:6px">${srcs}</div>` : (sel.label!=='neutral'?'<br>No chunk supports this sentence.':''));
  }

  const chunks = t.chunks.length ? t.chunks.map((c,j)=>{
    const isSrc = sel && sel.sources.includes(j);
    const cls = ['chunk', c.used?'':'unused', isSrc?'src':'', sel && !isSrc && sel.label!=='neutral' ? 'dim':''].join(' ');
    const mark = sset ? (k => sset.has(k) ? 'm' : null) : (k => qset.has(k) ? 'q' : null);
    const tags = [
      c.score!=null ? `<span class="tag">score ${c.score.toFixed(2)}</span>`:'',
      c.used ? `<span class="tag ok">used by ${c.cited_by.map(i=>'['+(i+1)+']').join(' ')}</span>` : '<span class="tag">unused</span>',
      c.duplicate_of!=null ? `<span class="tag warn">duplicate of #${c.duplicate_of+1}</span>`:'',
      `<span class="tag">query match ${pct(c.query_coverage)}</span>`,
      sel && sel.chunk_scores.length ? `<span class="tag">match ${sel.chunk_scores[j].toFixed(2)}</span>`:''
    ].join('');
    return `<div class="${cls}"><div class="chead"><span class="rank">#${c.rank}</span><span class="cid">${esc(c.id)}</span>${tags}</div>
      <div class="ctext">${segHTML(c.segments, mark)}</div></div>`;
  }).join('') : '<p class="empty">No chunks were retrieved.</p>';

  const issues = t.issues.map(x=>`<div class="issue ${x.severity}"><span class="sev">${x.severity}</span><span>${esc(x.message)}</span></div>`).join('');
  const timings = Object.entries(tr.timings||{}).map(([k,v])=>`${esc(k)} ${v} ms`).join(' · ');
  const meta = [tr.id ? 'id '+esc(tr.id) : '', tr.timestamp ? new Date(tr.timestamp*1000).toLocaleString() : '', timings, tr.metadata && tr.metadata.scenario ? esc(tr.metadata.scenario) : ''].filter(Boolean).join(' · ');

  document.getElementById('main').innerHTML = `
    <h2 class="query">${esc(tr.query)}</h2>
    <div class="sub">${meta}</div>
    <div class="chips">
      <span class="chip verdict ${vclass(t.verdict)}">${esc(t.verdict)}</span>
      <span class="chip">groundedness <b>${pct(t.groundedness)}</b></span>
      <span class="chip">context used <b>${used}/${t.chunks.length}</b></span>
      <span class="chip">query coverage <b>${pct(t.query_coverage)}</b></span>
    </div>
    ${issues ? `<div class="issues">${issues}</div>` : ''}
    <div class="grid">
      <section class="panel"><h2>Answer <span class="hint">click a sentence</span></h2>
        <div class="answer">${answer}</div>
        <div class="legend"><span style="--c:var(--ok-bg)">supported</span><span style="--c:var(--warn-bg)">partial</span><span style="--c:var(--bad-bg)">unsupported</span><span style="--c:transparent;color:var(--bad);font-weight:650">word not in any chunk</span></div>
        <div class="sdetail">${detail}</div>
      </section>
      <section class="panel"><h2>Retrieved chunks <span class="hint">${sel ? 'green = words shared with the selected sentence' : 'yellow = query terms'}</span></h2>${chunks}</section>
    </div>
    <details><summary>Raw trace JSON</summary><pre>${esc(JSON.stringify(tr, null, 2))}</pre></details>
    <footer>Generated by ragscope ${esc(DATA.version)} · ${esc(DATA.generated)}</footer>`;
}

document.getElementById('list').addEventListener('click', e=>{
  const it = e.target.closest('.item[data-i]'); if(!it) return;
  current = +it.dataset.i; selected = null; list(); render(); window.scrollTo(0,0);
});
document.getElementById('main').addEventListener('click', e=>{
  const s = e.target.closest('.s[data-s]'); if(!s) return;
  const i = +s.dataset.s; selected = selected===i ? null : i; render();
});
document.getElementById('filter').addEventListener('input', list);
document.addEventListener('keydown', e=>{
  if (e.target.tagName==='INPUT') return;
  if (e.key==='j'||e.key==='ArrowDown'){current=Math.min(T.length-1,current+1);selected=null;list();render();e.preventDefault();}
  if (e.key==='k'||e.key==='ArrowUp'){current=Math.max(0,current-1);selected=null;list();render();e.preventDefault();}
  if (e.key==='Escape'){selected=null;render();}
});
stats(); list(); render();
})();
</script>
</body>
</html>
"""
