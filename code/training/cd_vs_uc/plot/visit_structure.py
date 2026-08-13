"""
4-panel visit structure plot for at-20-cm biopsies (CD + UC).

Panels
------
A  H&E slides per visit
B  RNA samples per visit
C  Visits per patient (H&E vs RNA, grouped bars)
D  Visit-level data availability (H&E only / both / RNA only)

Usage
-----
    python visit_structure.py

Output
------
    <REPORTS_DIR>/visit_structure.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

AT20 = {'at 20 cm', 'At 20 cm'}
CDUC_RAW   = ["Crohn's disease", 'Ulcerative colitis']
CDUC_OMICS = ["Crohn's Disease","Crohn's disease",'Ulcerative colitis','Ulcerative Colitis']

# ── Load ──────────────────────────────────────────────────────────────────────
wsi = pd.read_csv('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv')
wsi = wsi[wsi['BIOSAMPLE_LOCATION'].isin(AT20) & wsi['diagnosis'].isin(CDUC_RAW)].copy()
wsi['date'] = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')
wsi['visit_key'] = list(zip(wsi['deidentified_master_patient_id'], wsi['date'].dt.date))

MAPPING     = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/ibd_21183_omics_patient_mapping_genestack.csv'
SAMPLE_META = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/GSF1478941_sample_combined_from1stRun.tsv__metadata.csv'
mapping = pd.read_csv(MAPPING)
omics = pd.read_csv('/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv')
pid_dx = omics[['deidentified_master_patient_id','diagnosis']].drop_duplicates()
rna = mapping[mapping['characteristics_bio_material'].isin(AT20)].copy()
rna = rna.merge(pid_dx, on='deidentified_master_patient_id', how='left')
rna = rna[rna['diagnosis'].isin(CDUC_OMICS)].copy()
rna['date'] = pd.to_datetime(rna['sample_collected_date'], dayfirst=True, errors='coerce')
rna['visit_key'] = list(zip(rna['deidentified_master_patient_id'], rna['date'].dt.date))

# ── Aggregate stats ───────────────────────────────────────────────────────────
he_per_visit      = wsi.groupby('visit_key').size()
rna_per_enc       = rna.groupby('visit_encounter_id').size()
he_visits_per_pat = wsi.groupby('deidentified_master_patient_id')['visit_key'].nunique()
rna_visits_per_pat= rna.groupby('deidentified_master_patient_id')['visit_encounter_id'].nunique()

he_only  = set(wsi['visit_key']) - set(rna['visit_key'])
rna_only = set(rna['visit_key']) - set(wsi['visit_key'])
both     = set(wsi['visit_key']) & set(rna['visit_key'])

# ── Style ─────────────────────────────────────────────────────────────────────
HE_C   = '#2a78d6'
RNA_C  = '#1baf7a'
INK    = '#000000'
MUTED  = '#555555'
GRID   = '#dddddd'

fig = plt.figure(figsize=(10, 7.5))
fig.patch.set_alpha(0)

# layout: 2 rows × 2 cols, last row is wider single panel
gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38,
                      left=0.09, right=0.97, top=0.91, bottom=0.09)
ax_he   = fig.add_subplot(gs[0, 0])
ax_rna  = fig.add_subplot(gs[0, 1])
ax_vis  = fig.add_subplot(gs[1, 0])
ax_ovlp = fig.add_subplot(gs[1, 1])

def style_ax(ax):
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.4)
    ax.tick_params(colors=INK, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

# ── Panel A: H&E slides per visit ─────────────────────────────────────────────
counts = he_per_visit.value_counts().sort_index()
ax_he.bar(counts.index.astype(str), counts.values, color=HE_C,
          width=0.55, zorder=2, linewidth=0)
for x, v in zip(counts.index, counts.values):
    ax_he.text(str(x), v + 18, f'{v:,}', ha='center', va='bottom',
               fontsize=7, color=INK)
style_ax(ax_he)
ax_he.set_xlabel('H&E slides per visit', fontsize=8, color=INK)
ax_he.set_ylabel('Number of visits', fontsize=8, color=INK)
ax_he.set_title('A  H&E slides per visit', fontsize=8.5, fontweight='bold',
                color=INK, loc='left', pad=5)
ax_he.set_ylim(0, counts.max() * 1.18)

# ── Panel B: RNA samples per encounter ────────────────────────────────────────
counts_r = rna_per_enc.value_counts().sort_index()
ax_rna.bar(counts_r.index.astype(str), counts_r.values, color=RNA_C,
           width=0.55, zorder=2, linewidth=0)
for x, v in zip(counts_r.index, counts_r.values):
    ax_rna.text(str(x), v + 18, f'{v:,}', ha='center', va='bottom',
                fontsize=7, color=INK)
style_ax(ax_rna)
ax_rna.set_xlabel('RNA samples per visit', fontsize=8, color=INK)
ax_rna.set_ylabel('Number of visits', fontsize=8, color=INK)
ax_rna.set_title('B  RNA samples per visit', fontsize=8.5, fontweight='bold',
                 color=INK, loc='left', pad=5)
ax_rna.set_ylim(0, counts_r.max() * 1.18)

# ── Panel C: visits per patient (grouped H&E + RNA) ───────────────────────────
max_v = max(he_visits_per_pat.max(), rna_visits_per_pat.max())
bins = list(range(1, min(max_v + 1, 8))) + (['7+'] if max_v >= 7 else [])
def bin_count(series, bins):
    out = []
    for b in bins:
        if isinstance(b, str):
            out.append((series >= 7).sum())
        else:
            out.append((series == b).sum())
    return np.array(out)

he_vc  = bin_count(he_visits_per_pat, bins)
rna_vc = bin_count(rna_visits_per_pat, bins)

xs = np.arange(len(bins))
w = 0.38
ax_vis.bar(xs - w/2, he_vc,  width=w, color=HE_C,  zorder=2, linewidth=0, label='H&E')
ax_vis.bar(xs + w/2, rna_vc, width=w, color=RNA_C, zorder=2, linewidth=0, label='RNA-seq')
for xi, (h, r) in zip(xs, zip(he_vc, rna_vc)):
    if h > 0: ax_vis.text(xi - w/2, h + 8, f'{h}', ha='center', fontsize=6.5, color=INK)
    if r > 0: ax_vis.text(xi + w/2, r + 8, f'{r}', ha='center', fontsize=6.5, color=INK)
style_ax(ax_vis)
ax_vis.set_xticks(xs)
ax_vis.set_xticklabels([str(b) for b in bins], fontsize=7)
ax_vis.set_xlabel('Number of visits per patient', fontsize=8, color=INK)
ax_vis.set_ylabel('Number of patients', fontsize=8, color=INK)
ax_vis.set_title('C  Visits per patient', fontsize=8.5, fontweight='bold',
                 color=INK, loc='left', pad=5)
ax_vis.set_ylim(0, max(he_vc.max(), rna_vc.max()) * 1.20)
ax_vis.legend(fontsize=7.5, frameon=False, labelcolor=INK,
              handles=[mpatches.Patch(color=HE_C, label='H&E'),
                       mpatches.Patch(color=RNA_C, label='RNA-seq')])

# ── Panel D: visit-level co-occurrence ────────────────────────────────────────
cats   = ['H&E only', 'Both', 'RNA only']
vals   = [len(he_only), len(both), len(rna_only)]
colors = [HE_C, '#8a6fb5', RNA_C]
ax_ovlp.bar(cats, vals, color=colors, width=0.55, zorder=2, linewidth=0)
for i, v in enumerate(vals):
    ax_ovlp.text(i, v + 15, f'{v:,}', ha='center', va='bottom', fontsize=7, color=INK)
style_ax(ax_ovlp)
ax_ovlp.set_ylabel('Number of visits', fontsize=8, color=INK)
ax_ovlp.set_title('D  Visit-level data availability', fontsize=8.5, fontweight='bold',
                  color=INK, loc='left', pad=5)
ax_ovlp.set_ylim(0, max(vals) * 1.18)
ax_ovlp.tick_params(axis='x', labelsize=8)

# ── Figure title & caption ────────────────────────────────────────────────────
fig.suptitle('At-20-cm visit structure — CD + UC patients',
             fontsize=10, fontweight='bold', color=INK, y=0.97)

REPORTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_09_at20cm_site_controlled/reports')
import os
os.makedirs(REPORTS_DIR, exist_ok=True)
out = os.path.join(REPORTS_DIR, 'visit_structure.png')
plt.savefig(out, dpi=180, bbox_inches='tight', transparent=True)
print(f'Saved: {out}')
