"""
Generate the interactive AUC/AP/Accuracy comparison HTML for all at-20-cm
manual-KNN experiments.

Reads the summary JSONs written by the training scripts and emits a
self-contained HTML file with a bar chart, table, and metric/model filters.

Inputs (auto-discovered under --results_dir)
--------------------------------------------
  rf/at20cm_rf_summary.json                            (train_rf.py)
  mlp/at20cm_mlp_summary.json                          (train_mlp.py)
  multimodal/at20cm_multimodal_summary.json            (train_multimodal.py)
  multimodal_histoscore/at20cm_multimodal_histo_summary.json
                                                       (train_multimodal_histoscore.py)

Output
------
  <results_dir>/results_at20cm_manual_knn_comparison.html

Usage
-----
  python plot_all_at20cm_manual_knn.py \
      [--results_dir  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/at20cm_visit_manual_knn/results] \
      [--output_html  <results_dir>/results_at20cm_manual_knn_comparison.html]

Adding a new experiment
-----------------------
  1. Drop its summary.json under <results_dir>/<subfolder>/<name>_summary.json
  2. Add one entry to STRATEGY_META below mapping every strategy key you want
     rendered to a human label + group.
"""

import argparse
import json
import os

DEFAULT_RESULTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    'at20cm_visit_manual_knn/results'
)

# ── Metadata registry ────────────────────────────────────────────────────────
# One entry per strategy key we render. Any strategy key not present here is
# skipped with a warning — that keeps the plot legend-consistent as new arms
# are added.
GROUPS = [
    'Imaging', 'RNA', 'Clinical', 'Img + RNA', 'Img + BulkFormer', '+ Clinical',
]

STRATEGY_META = {
    # 10-arm RF and MLP (train_rf.py / train_mlp.py) — the model flag comes
    # from the file the strategy was loaded from.
    'img_base_visit':          ('Imaging (prism2 base)',    'Imaging'),
    'img_histoscore_visit':    ('Histological scores',      'Imaging'),
    'bulkformer_visit':        ('BulkFormer',               'RNA'),
    'rna_visit':               ('RNA log2(TPM+1)',          'RNA'),
    'pca640_rna_visit':        ('RNA PCA-640',              'RNA'),
    'concat_raw_visit':        ('Imaging + RNA',            'Img + RNA'),
    'concat_histoscore_visit': ('Histoscores + RNA',        'Img + RNA'),
    'concat_bf_base_visit':    ('Imaging + BulkFormer',     'Img + BulkFormer'),
    'concat_bf_histo_visit':   ('Histoscores + BulkFormer', 'Img + BulkFormer'),
    'concat_pca640_base':      ('Imaging + RNA PCA-640',    'Img + RNA'),
    # Multimodal (train_multimodal.py) — suffixed _rf / _mlp per strategy.
    'clinical_visit':          ('Clinical only',            'Clinical'),
    'img_clinical_visit':      ('Imaging + Clinical',       '+ Clinical'),
    'rna_clinical_visit':      ('RNA + Clinical',           '+ Clinical'),
    'trimodal_visit':          ('Trimodal (Img+RNA+Clin)',  '+ Clinical'),
    # Multimodal histoscore variant (train_multimodal_histoscore.py).
    'histo_clinical_visit':    ('Histoscores + Clinical',   '+ Clinical'),
    'trimodal_histo_visit':    ('Trimodal (Histo+RNA+Clin)','+ Clinical'),
}

# Source registry: file paths (relative to results_dir) and how to interpret
# the strategy field. `model_from_suffix=True` means the strategy string ends
# in `_rf` or `_mlp` (multimodal case). Otherwise, an explicit `model` label
# is applied to every row.
SOURCES = [
    dict(path='rf/at20cm_rf_summary.json',
         model='RF',  model_from_suffix=False),
    dict(path='mlp/at20cm_mlp_summary.json',
         model='MLP', model_from_suffix=False),
    dict(path='multimodal/at20cm_multimodal_summary.json',
         model=None,  model_from_suffix=True),
    dict(path='multimodal_histoscore/at20cm_multimodal_histo_summary.json',
         model=None,  model_from_suffix=True),
]


def collect_rows(results_dir):
    rows, skipped = [], []
    for src in SOURCES:
        full = os.path.join(results_dir, src['path'])
        if not os.path.exists(full):
            print(f'  MISS  {src["path"]}  (skipping)')
            continue
        with open(full) as f:
            entries = json.load(f)
        for e in entries:
            strat = e['strategy']
            if src['model_from_suffix']:
                if strat.endswith('_rf'):    model, base = 'RF',  strat[:-3]
                elif strat.endswith('_mlp'): model, base = 'MLP', strat[:-4]
                else:                        skipped.append(strat); continue
            else:
                model, base = src['model'], strat

            if base not in STRATEGY_META:
                skipped.append(base); continue
            label, group = STRATEGY_META[base]
            rows.append(dict(
                strategy=base, label=label, group=group, model=model,
                auc=e['mean_auc'], std_auc=e['std_auc'],
                ap=e['mean_ap'],  std_ap=e['std_ap'],
                acc=e['mean_acc'], std_acc=e['std_acc'],
            ))
        print(f'  OK    {src["path"]}  ({len(entries)} strategies)')

    if skipped:
        uniq = sorted(set(skipped))
        print(f'\n  Warning: {len(uniq)} unknown strategy key(s) skipped: {uniq}')
        print('  Add them to STRATEGY_META in this script to include them.')
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CD vs UC — at-20-cm manual-KNN comparison</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --text-muted:#898781; --gridline:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100;
  --s4:#eb6834; --s5:#4a3aa7; --s6:#008300;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 13px; background: var(--surface-1); color: var(--text-primary);
  padding: 28px 32px; max-width: 920px; margin: 0 auto;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --text-muted:#898781; --gridline:#2c2c2a; --axis:#383835;
    --s1:#3987e5; --s2:#199e70; --s3:#c98500;
    --s4:#d95926; --s5:#9085e9; --s6:#008300;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --text-muted:#898781; --gridline:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#199e70; --s3:#c98500;
  --s4:#d95926; --s5:#9085e9; --s6:#008300;
}
h2 { font-size: 16px; font-weight: 600; margin: 0 0 3px; }
.subtitle { font-size: 12px; color: var(--text-secondary); margin: 0 0 6px; }
.chart-wrap { position: relative; }
.tooltip {
  position: absolute; background: var(--surface-1);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 12px;
  font-size: 12px; pointer-events: none; opacity: 0; transition: opacity .1s;
  box-shadow: 0 2px 8px rgba(0,0,0,.14);
  color: var(--text-primary); line-height: 1.7; white-space: nowrap; z-index: 10;
}
.tooltip.visible { opacity: 1; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.controls { display: flex; align-items: center; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
.ctrl-group { display: flex; align-items: center; gap: 6px; }
.ctrl-label { font-size: 11px; color: var(--text-muted); }
.btn { font-size: 11px; color: var(--text-muted); cursor: pointer; border: 1px solid var(--axis); background: transparent; padding: 3px 10px; border-radius: 4px; font-family: inherit; }
.btn.active { background: var(--axis); color: var(--text-primary); }
.spacer { flex: 1; }
.provenance { font-size: 11px; color: var(--text-muted); margin-top: 20px; border-top: 1px solid var(--gridline); padding-top: 10px; }
.provenance code { background: var(--gridline); padding: 1px 4px; border-radius: 2px; font-size: 10px; }
details summary { font-size: 12px; color: var(--text-secondary); cursor: pointer; margin-top: 16px; user-select: none; }
table { margin-top: 8px; font-size: 11px; border-collapse: collapse; width: 100%; }
th { text-align: left; padding: 4px 10px; color: var(--text-secondary); border-bottom: 1px solid var(--gridline); font-weight: 600; }
td { padding: 4px 10px; color: var(--text-primary); border-bottom: 1px solid var(--gridline); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="viz-root" id="root">
  <div class="controls">
    <div class="ctrl-group">
      <span class="ctrl-label">Metric:</span>
      <button class="btn active" id="btn-auc" onclick="setMetric('auc')">AUC</button>
      <button class="btn" id="btn-ap"  onclick="setMetric('ap')">AP</button>
      <button class="btn" id="btn-acc" onclick="setMetric('acc')">Accuracy</button>
    </div>
    <div class="ctrl-group" style="margin-left:12px">
      <span class="ctrl-label">Model:</span>
      <button class="btn active" id="filter-all" onclick="setFilter('all')">All</button>
      <button class="btn" id="filter-RF"  onclick="setFilter('RF')">RF</button>
      <button class="btn" id="filter-MLP" onclick="setFilter('MLP')">MLP</button>
    </div>
    <div class="spacer"></div>
    <button class="btn" id="btn-theme" onclick="toggleTheme()">Dark</button>
  </div>

  <h2>CD vs UC — at-20-cm manual-KNN pipeline</h2>
  <p class="subtitle">945 visits / 817 patients · 5-fold patient-level CV · sorted by chosen metric</p>

  <div class="chart-wrap">
    <svg id="chart"></svg>
    <div class="tooltip" id="tooltip"></div>
  </div>
  <div class="legend" id="legend"></div>

  <details>
    <summary>Table view</summary>
    <table id="data-table"></table>
  </details>

  <div class="provenance">
    Generated from: <code>__SOURCES__</code><br>
    Regenerate with:
    <code>python /home/jovyan/ibdplexus/code/training/cd_vs_uc/plot/plot_all_at20cm_manual_knn.py</code>
  </div>
</div>

<script>
const ALL_DATA = __DATA__;
const GROUPS   = __GROUPS__;
const SLOT_KEYS = ['--s1','--s2','--s3','--s4','--s5','--s6'];
const METRIC_LABEL = { auc:'ROC-AUC', ap:'Average Precision', acc:'Accuracy' };
let currentMetric = 'auc', currentFilter = 'all';

function css(prop){ return getComputedStyle(document.getElementById('root')).getPropertyValue(prop).trim(); }
function groupColor(g){ return css(SLOT_KEYS[GROUPS.indexOf(g)] || '--s1'); }

function setMetric(m){
  currentMetric = m;
  ['auc','ap','acc'].forEach(k => document.getElementById('btn-'+k).classList.toggle('active', k===m));
  render();
}
function setFilter(f){
  currentFilter = f;
  ['all','RF','MLP'].forEach(k => document.getElementById('filter-'+k).classList.toggle('active', k===f));
  render();
}

function render(){
  let data = [...ALL_DATA];
  if (currentFilter !== 'all') data = data.filter(d => d.model === currentFilter);
  data.sort((a,b) => b[currentMetric] - a[currentMetric]);

  const metric = currentMetric, sk = 'std_'+metric;
  const W=840, LPAD=238, RPAD=72, TPAD=10, BPAD=38;
  const BAR_H=19, GAP=7, N=data.length;
  const plotW=W-LPAD-RPAD, H=TPAD+N*(BAR_H+GAP)+BPAD;
  const vals=data.map(d=>d[metric]);
  const lo=Math.min(...vals), hi=Math.max(...vals), span=hi-lo||0.2;
  const xMin=Math.max(0, Math.floor((lo-span*.15)*20)/20);
  const xMax=Math.min(1.0,Math.ceil ((hi+span*.10)*20)/20);
  const xs=v=>LPAD+((v-xMin)/(xMax-xMin))*plotW;

  const svg=document.getElementById('chart');
  svg.setAttribute('width',W); svg.setAttribute('height',H);
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML='';

  const gc=css('--gridline'),ac=css('--axis'),mc=css('--text-muted'),sc=css('--text-secondary'),pc=css('--text-primary');

  const step=span>0.25?0.1:0.05;
  for(let t=Math.ceil(xMin/step)*step; t<=xMax+.001; t+=step){
    const x=Math.round(xs(t));
    el('line',{x1:x,y1:TPAD,x2:x,y2:H-BPAD,stroke:gc,'stroke-width':1},svg);
    el('text',{x,y:H-BPAD+14,'text-anchor':'middle',fill:mc,'font-size':11},svg)
      .textContent=t.toFixed(2).replace(/\\.?0+$/,'')||'0';
  }
  el('line',{x1:xs(xMin),y1:H-BPAD,x2:xs(xMax),y2:H-BPAD,stroke:ac,'stroke-width':1},svg);
  el('text',{x:LPAD+plotW/2,y:H-4,'text-anchor':'middle',fill:sc,'font-size':11},svg)
    .textContent=METRIC_LABEL[metric]+' (mean +/- 1 SD, 5-fold)';

  data.forEach((d,i)=>{
    const y=TPAD+i*(BAR_H+GAP), midY=y+BAR_H/2, barY=y+2, barH=BAR_H-4;
    const color=groupColor(d.group), val=d[metric], std=d[sk];
    el('rect',{x:xs(xMin),y:barY,width:Math.max(0,xs(val)-xs(xMin)),height:barH,fill:color,rx:3,ry:3},svg);
    const eLo=xs(Math.max(xMin,val-std)), eHi=Math.min(xs(xMax),xs(val+std));
    el('line',{x1:eLo,y1:midY,x2:eHi,y2:midY,stroke:pc,'stroke-width':1.5,opacity:.4},svg);
    el('line',{x1:eLo,y1:midY-3,x2:eLo,y2:midY+3,stroke:pc,'stroke-width':1.5,opacity:.4},svg);
    el('line',{x1:eHi,y1:midY-3,x2:eHi,y2:midY+3,stroke:pc,'stroke-width':1.5,opacity:.4},svg);
    el('text',{x:Math.min(eHi+5,W-4),y:midY+4,'text-anchor':'start',fill:sc,'font-size':10,'font-variant-numeric':'tabular-nums'},svg)
      .textContent=val.toFixed(3);
    el('text',{x:LPAD-44,y:midY+4,'text-anchor':'middle',fill:mc,'font-size':10},svg).textContent=d.model;
    el('text',{x:LPAD-54,y:midY+4,'text-anchor':'end',fill:pc,'font-size':11.5},svg).textContent=d.label;
    const hz=el('rect',{x:0,y,width:W,height:BAR_H+GAP-1,fill:'transparent',style:'cursor:default'},svg);
    hz.addEventListener('mousemove',e=>showTip(e,d));
    hz.addEventListener('mouseleave',hideTip);
  });
}

function el(tag,attrs,parent){
  const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);
  if(parent)parent.appendChild(e); return e;
}
function showTip(e,d){
  const tt=document.getElementById('tooltip');
  tt.innerHTML=`<strong>${d.label}</strong> <span style="font-size:10px;color:var(--text-muted)">${d.model} - ${d.group}</span><br>`+
    `AUC  ${d.auc.toFixed(4)} +/- ${d.std_auc.toFixed(4)}<br>`+
    `AP   ${d.ap.toFixed(4)} +/- ${d.std_ap.toFixed(4)}<br>`+
    `Acc  ${d.acc.toFixed(4)} +/- ${d.std_acc.toFixed(4)}`;
  const root=document.getElementById('root'),rb=root.getBoundingClientRect();
  let tx=e.clientX-rb.left+14, ty=e.clientY-rb.top-10;
  if(tx+230>root.offsetWidth)tx=e.clientX-rb.left-235;
  tt.style.left=tx+'px'; tt.style.top=ty+'px'; tt.classList.add('visible');
}
function hideTip(){document.getElementById('tooltip').classList.remove('visible');}

function renderLegend(){
  const lg=document.getElementById('legend'); lg.innerHTML='';
  GROUPS.forEach(name=>{
    const item=document.createElement('div'); item.className='legend-item';
    item.innerHTML=`<div class="legend-swatch" style="background:${groupColor(name)}"></div><span>${name}</span>`;
    lg.appendChild(item);
  });
}

function renderTable(){
  const data=[...ALL_DATA].sort((a,b)=>b.auc-a.auc);
  document.getElementById('data-table').innerHTML=
    `<tr><th>Strategy</th><th>Model</th><th>Group</th><th>AUC</th><th>+/-SD</th><th>AP</th><th>+/-SD</th><th>Acc</th><th>+/-SD</th></tr>`+
    data.map(d=>`<tr><td>${d.label}</td><td class="num">${d.model}</td><td style="color:var(--text-secondary)">${d.group}</td>`+
      `<td class="num">${d.auc.toFixed(4)}</td><td class="num" style="color:var(--text-muted)">${d.std_auc.toFixed(4)}</td>`+
      `<td class="num">${d.ap.toFixed(4)}</td><td class="num" style="color:var(--text-muted)">${d.std_ap.toFixed(4)}</td>`+
      `<td class="num">${d.acc.toFixed(4)}</td><td class="num" style="color:var(--text-muted)">${d.std_acc.toFixed(4)}</td></tr>`).join('');
}

let dark=window.matchMedia('(prefers-color-scheme: dark)').matches;
function toggleTheme(){
  dark=!dark;
  document.documentElement.setAttribute('data-theme',dark?'dark':'light');
  document.getElementById('btn-theme').textContent=dark?'Light':'Dark';
  setTimeout(()=>{render();renderLegend();},10);
}
if(dark){document.documentElement.setAttribute('data-theme','dark');document.getElementById('btn-theme').textContent='Light';}
render(); renderLegend(); renderTable();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{render();renderLegend();});
</script>
</body>
</html>
"""


def build_html(rows, source_paths):
    sources_str = ', '.join(source_paths) if source_paths else '(none found)'
    return (HTML_TEMPLATE
            .replace('__DATA__',   json.dumps(rows, indent=2))
            .replace('__GROUPS__', json.dumps(GROUPS))
            .replace('__SOURCES__', sources_str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--output_html', default=None,
                   help='Default: <results_dir>/results_at20cm_manual_knn_comparison.html')
    a = p.parse_args()

    out = a.output_html or os.path.join(
        a.results_dir, 'results_at20cm_manual_knn_comparison.html')

    print(f'Results dir: {a.results_dir}')
    rows = collect_rows(a.results_dir)
    if not rows:
        print('No summary files found — nothing to plot.')
        return

    print(f'\nCollected {len(rows)} strategy-model rows.')
    found = [s['path'] for s in SOURCES
             if os.path.exists(os.path.join(a.results_dir, s['path']))]
    html = build_html(rows, found)

    with open(out, 'w') as f:
        f.write(html)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
