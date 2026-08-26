"""
SHAP plots for the prism2 histological score arms — concept_learning series.

Figures
-------
  Fig 1: histoscore_shap_bar.pdf     — SHAP bar for 11 histological features,
                                       colored by direction (CD/UC), ±1 SD bands
  Fig 2: histoscore_swarm.pdf        — score distribution CD vs UC per feature,
                                       ordered by mean|SHAP|
  Fig 3: histoscore_fusion_bar.pdf   — top 20 RNA genes in the fusion arm +
                                       histoscore modality split
  Fig 4: histoscore_panel.pdf        — 3-panel combined

Requires 10c to have completed first.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

SHAP_DIR  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/data')
PLOTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/plots')
VST_GCT   = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
             'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
             'alltissues_all3releases_header.gct')
MAPPING_CSV = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
HISTOSCORE_CSV = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
WSI_META    = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
               'IBD_meta_data_latest/wsi_metadata_raw.csv')
AT20        = {'at 20 cm', 'At 20 cm'}

UC_COLOR = '#c94040'
CD_COLOR = '#2a78d6'
SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
BASE  = '#c3c2b7'

# ── readable labels for 11 histological features ──────────────────────────────
HISTO_LABELS = {
    'inflammation_involvement':           'Inflammation involvement',
    'crypt_architectural_distortion':     'Crypt architectural distortion',
    'neutrophil_granulocytic_infiltration': 'Neutrophilic infiltration',
    'crypt_abscesses':                    'Crypt abscesses',
    'lymphoid_aggregates':                'Lymphoid aggregates',
    'histiocytic_granulomas':             'Histiocytic granulomas',
    'mucin_depletion':                    'Mucin depletion',
    'pyloric_gland_metaplasia':           'Pyloric gland metaplasia',
    'paneth_cell_metaplasia':             'Paneth cell metaplasia',
    'neuronal_hyperplasia':               'Neuronal hyperplasia',
    'muscular_hypertrophy':               'Muscular hypertrophy',
}

GENE_ANNO = {
    'COL12A1':  ('ECM / Fibrosis',          'Type XII collagen'),
    'POSTN':    ('ECM / Fibrosis',          'Periostin; TGF-β'),
    'COL5A2':   ('ECM / Fibrosis',          'Type V collagen'),
    'COL3A1':   ('ECM / Fibrosis',          'Type III collagen'),
    'SPARC':    ('ECM / Fibrosis',          'ECM remodeling'),
    'CPXM1':    ('ECM / Fibrosis',          'ECM processing'),
    'ITGB8':    ('ECM / Fibrosis',          'Integrin β8'),
    'ADGRV1':   ('ECM / Fibrosis',          'Cell-matrix adhesion'),
    'CARD6':    ('Immunity / Inflammation', 'Inflammasome scaffold'),
    'CCL11':    ('Immunity / Inflammation', 'Eosinophil recruitment'),
    'DAPP1':    ('Immunity / Inflammation', 'B/mast cell signaling'),
    'HYAL1':    ('Immunity / Inflammation', 'ECM-immune crosstalk'),
    'FOXP2':    ('Transcription Factor',    'Mucosal regulation'),
    'TBX3':     ('Transcription Factor',    'Epithelial lineage'),
    'PITX1':    ('Transcription Factor',    'Hindgut TF'),
    'ZNF492':   ('Transcription Factor',    'Transcriptional regulation'),
    'BRINP3':   ('Neural / Receptor',       'Neuronal/ECM'),
    'NPSR1':    ('Neural / Receptor',       'Neuropeptide S receptor'),
    'MYEOV':    ('Neural / Receptor',       'ENS-associated'),
    'DPP10':    ('Neural / Receptor',       'Kv channel modulator'),
    'ACAT1':    ('Metabolism / Transport',  'Ketone metabolism'),
    'SELENBP1': ('Metabolism / Transport',  'Mucosal oxidative stress'),
    'CYP2C18':  ('Metabolism / Transport',  'Xenobiotic metabolism'),
    'ABCA13':   ('Metabolism / Transport',  'Lipid export'),
    'HOXB2':    ('HOX / Positional',        'AP-axis identity'),
    'HOXA13':   ('HOX / Positional',        'Posterior gut'),
    'HOXB13':   ('HOX / Positional',        'Posterior colon'),
    'GCG':      ('Gut Hormone / Peptide',   'GLP-1/GLP-2'),
    'PYY':      ('Gut Hormone / Peptide',   'L-cell, satiety'),
}

PAL_CAT = {
    'ECM / Fibrosis':          '#2a78d6',
    'Immunity / Inflammation': '#1baf7a',
    'Transcription Factor':    '#eda100',
    'Neural / Receptor':       '#e87ba4',
    'Metabolism / Transport':  '#4a3aa7',
    'HOX / Positional':        '#eb6834',
    'Gut Hormone / Peptide':   '#008300',
    'Other / Unknown':         '#898781',
}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': BASE,   'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True,        'grid.color': GRID,
        'grid.linewidth': 0.4,    'axes.axisbelow': True,
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'xtick.color': MUTED,     'ytick.color': INK2,
        'xtick.labelsize': 8,     'ytick.labelsize': 9,
        'text.color': INK,
    })


def load_gene_symbols():
    gmap = pd.read_csv(VST_GCT, sep='\t', skiprows=2,
                       usecols=['Name', 'Description'])
    return dict(zip(gmap['Name'], gmap['Description']))


def gene_display(feat, sym_map):
    if feat.startswith('ENSG'):
        return sym_map.get(feat, feat)
    if feat.startswith('histo_'):
        raw = feat[6:]
        return HISTO_LABELS.get(raw, raw.replace('_', ' '))
    return feat


def gene_cat_color(feat, sym_map):
    sym = gene_display(feat, sym_map)
    cat = GENE_ANNO.get(sym, ('Other / Unknown', ''))[0]
    return PAL_CAT.get(cat, PAL_CAT['Other / Unknown'])


def gene_func(feat, sym_map):
    sym = gene_display(feat, sym_map)
    return GENE_ANNO.get(sym, ('', ''))[1]


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: Histological score SHAP bar (all 11, signed colors)
# ══════════════════════════════════════════════════════════════════════════════

def histoscore_shap_bar(df, auc, n_vis, n_pat):
    setup_style()
    # order: highest mean_abs_shap at top
    df = df.sort_values('mean_abs_shap', ascending=True).reset_index(drop=True)

    labels   = [HISTO_LABELS.get(f, f) for f in df['feature']]
    vals_abs = df['mean_abs_shap'].tolist()
    dirs     = df['direction'].tolist()
    colors   = [UC_COLOR if d == 'UC' else CD_COLOR for d in dirs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    y = np.arange(len(labels))

    ax.barh(y, vals_abs, height=0.6, color=colors, edgecolor=SURF, linewidth=0.8)

    max_v = max(vals_abs)
    for i, (val, d) in enumerate(zip(vals_abs, dirs)):
        dir_label = '↑ UC' if d == 'UC' else '↑ CD'
        dir_color = UC_COLOR if d == 'UC' else CD_COLOR
        ax.text(val + max_v * 0.01, i,
                f'{val:.4f}  {dir_label}',
                va='center', ha='left', fontsize=8,
                color=dir_color, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel('Mean |SHAP value|', fontsize=9, color=INK2)
    ax.set_xlim(0, max_v * 1.55)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)
    ax.set_title(
        f'Histological score SHAP — img_histoscore_visit\n'
        f'At-20-cm · {n_vis} visits · {n_pat} patients · AUC={auc:.3f}',
        fontsize=10, fontweight='bold', color=INK, loc='left', pad=8)

    handles = [
        mpatches.Patch(color=UC_COLOR, label='Higher in UC'),
        mpatches.Patch(color=CD_COLOR, label='Higher in CD'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True,
              framealpha=0.9, edgecolor=GRID, loc='lower right')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Score distribution CD vs UC per feature (violin + strip)
# ══════════════════════════════════════════════════════════════════════════════

def load_histoscore_cohort():
    """Load at-20-cm matched histoscore visits with labels."""
    cv_pids = set(pd.read_csv(CV_PATIENTS)['patient_id'])
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
              wsi['diagnosis'].isin(["Crohn's disease", 'Ulcerative colitis']) &
              wsi['deidentified_master_patient_id'].isin(cv_pids)].copy()
    wsi['date']     = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)

    scores = pd.read_csv(HISTOSCORE_CSV).set_index('slide')
    histo_cols = list(scores.columns)

    rows = []
    for _, r in wsi.iterrows():
        sid = r['slide_id']
        if sid not in scores.index:
            continue
        row = scores.loc[sid]
        rows.append({
            'patient_id': r['deidentified_master_patient_id'],
            'label': 'UC' if r['diagnosis'] == 'Ulcerative colitis' else 'CD',
            **{c: float(row[c]) for c in histo_cols},
        })
    return pd.DataFrame(rows), histo_cols


def score_distribution(df_cohort, histo_cols, shap_order, auc):
    """Box+strip plot of each score by CD vs UC, ordered by SHAP rank."""
    setup_style()
    n = len(histo_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.55 + 1, 5), sharey=False)
    fig.patch.set_facecolor(SURF)

    for ax_i, feat in enumerate(shap_order):
        ax = axes[ax_i]
        ax.set_facecolor(SURF)
        cd_vals = df_cohort[df_cohort['label'] == 'CD'][feat].dropna()
        uc_vals = df_cohort[df_cohort['label'] == 'UC'][feat].dropna()

        # box
        bp = ax.boxplot([cd_vals, uc_vals], positions=[0, 1], widths=0.45,
                        patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color='white', linewidth=1.5))
        bp['boxes'][0].set_facecolor(CD_COLOR); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(UC_COLOR); bp['boxes'][1].set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set_color(MUTED); whisker.set_linewidth(0.8)
        for cap in bp['caps']:
            cap.set_color(MUTED); cap.set_linewidth(0.8)

        # strip
        np.random.seed(42)
        jitter = 0.12
        for vals, pos, col in [(cd_vals, 0, CD_COLOR), (uc_vals, 1, UC_COLOR)]:
            jx = np.random.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(pos + jx, vals, s=2.5, color=col, alpha=0.35, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CD', 'UC'], fontsize=8, fontweight='bold')
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='x', colors=INK, length=0)
        ax.tick_params(axis='y', labelsize=7, colors=MUTED)
        if ax_i == 0:
            ax.set_ylabel('Score', fontsize=8, color=INK2)
        else:
            ax.set_yticklabels([])

        label = HISTO_LABELS.get(feat, feat)
        # wrap long labels
        words = label.split()
        mid   = len(words) // 2
        wrapped = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:]) if len(words) > 2 else label
        ax.set_title(wrapped, fontsize=7, color=INK, pad=4, fontweight='bold')

        ax.spines['left'].set_color(BASE)
        ax.spines['bottom'].set_color(BASE)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.4)

    fig.suptitle(
        f'Histological score distributions — At-20-cm (CD vs UC)\n'
        f'Ordered by SHAP importance  ·  img_histoscore_visit AUC={auc:.3f}',
        fontsize=10, fontweight='bold', color=INK, y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Fusion bar — RNA genes + histoscore modality split
# ══════════════════════════════════════════════════════════════════════════════

def fusion_bar(df_fus, histo_frac, rna_frac, auc, sym_map, n_vis, n_pat):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                             gridspec_kw={'width_ratios': [1.4, 2.8]})
    fig.patch.set_facecolor(SURF)

    # Left: modality split
    ax = axes[0]; ax.set_facecolor(SURF)
    cats   = ['RNA\n(17,963 genes)', 'Histo scores\n(11 features)']
    fracs  = [rna_frac * 100, histo_frac * 100]
    colors = ['#2a78d6', '#e87ba4']
    brs = ax.barh(cats, fracs, height=0.45, color=colors,
                  edgecolor=SURF, linewidth=0.8)
    for bar, val in zip(brs, fracs):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                fontsize=10, fontweight='bold', color=INK)
    ax.set_xlim(0, 115)
    ax.set_xlabel('% of total mean |SHAP|', fontsize=8.5, color=INK2)
    ax.set_title('Fusion SHAP\nmodality split', fontsize=10,
                 fontweight='bold', color=INK, loc='left')
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
    ax.text(0, -0.68,
            f'concat_histoscore_visit  ·  AUC={auc:.3f}\n'
            f'{n_vis} visits  ·  {n_pat} patients  ·  RF  ·  5-fold CV',
            transform=ax.transAxes, fontsize=7, color=MUTED)

    # Right: top RNA genes in fusion
    ax2 = axes[1]; ax2.set_facecolor(SURF)
    rna_only = df_fus[df_fus['feature'].str.startswith('ENSG')].head(20)
    genes  = rna_only['feature'].tolist()[::-1]
    labels = [gene_display(g, sym_map) for g in genes]
    values = rna_only['mean_abs_shap'].tolist()[::-1]
    dirs   = rna_only['direction'].tolist()[::-1]
    colors2 = [gene_cat_color(g, sym_map) for g in genes]
    y = np.arange(len(genes))
    ax2.barh(y, values, height=0.55, color=colors2, edgecolor=SURF, linewidth=0.8)

    max_v = max(values)
    for i, (val, g, d) in enumerate(zip(values, genes, dirs)):
        dir_label = '↑UC' if d == 'UC' else '↑CD'
        dir_col   = UC_COLOR if d == 'UC' else CD_COLOR
        ax2.text(val + max_v * 0.01, i, f'{val:.4f}',
                 va='center', ha='left', fontsize=7, color=INK2)
        ax2.text(val + max_v * 0.01, i - 0.32, dir_label,
                 va='center', ha='left', fontsize=6, color=dir_col,
                 fontweight='bold')
        func = gene_func(g, sym_map)
        ax2.text(max_v * 1.1, i, func, va='center', ha='left',
                 fontsize=6.5, color=MUTED, style='italic')

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax2.set_xlabel('Mean |SHAP value|', fontsize=8.5, color=INK2)
    ax2.set_xlim(0, max_v * 1.6)
    ax2.set_title('Top RNA genes driving fusion predictions',
                  fontsize=10, fontweight='bold', color=INK, loc='left')
    ax2.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax2.yaxis.grid(False)
    ax2.spines['left'].set_color(BASE); ax2.spines['bottom'].set_color(BASE)

    cats_present = {GENE_ANNO.get(gene_display(g, sym_map), ('Other / Unknown', ''))[0]
                    for g in genes}
    handles = [mpatches.Patch(color=PAL_CAT[c], label=c)
               for c in PAL_CAT if c in cats_present]
    fig.legend(handles=handles, title='Biological category',
               fontsize=7, title_fontsize=7.5, frameon=True,
               framealpha=0.9, edgecolor=GRID, loc='lower center',
               bbox_to_anchor=(0.65, -0.03), ncol=min(len(handles), 4))
    fig.suptitle('Multimodal fusion — histological scores + RNA-seq  (concat_histoscore_visit)',
                 fontsize=11, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: 3-panel combined
# ══════════════════════════════════════════════════════════════════════════════

def combined_panel(df_img, df_rna, df_fus, summary, sym_map, n_vis, n_pat):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    fig.patch.set_facecolor(SURF)
    fig.suptitle(
        f'SHAP feature importance — At-20-cm visit-level cohort'
        f'  ({n_vis} visits, {n_pat} patients)',
        fontsize=12, fontweight='bold', color=INK, y=1.01)

    # Panel 1: histological scores
    ax = axes[0]; ax.set_facecolor(SURF)
    df_s = df_img.sort_values('mean_abs_shap', ascending=True)
    labels = [HISTO_LABELS.get(f, f) for f in df_s['feature']]
    vals   = df_s['mean_abs_shap'].tolist()
    dirs   = df_s['direction'].tolist()
    colors = [UC_COLOR if d == 'UC' else CD_COLOR for d in dirs]
    y      = np.arange(len(labels))
    ax.barh(y, vals, height=0.6, color=colors, edgecolor=SURF, linewidth=0.8)
    max_v  = max(vals)
    for i, (val, d) in enumerate(zip(vals, dirs)):
        ax.text(val + max_v * 0.01, i, f'{val:.4f}  {"↑UC" if d=="UC" else "↑CD"}',
                va='center', ha='left', fontsize=6.5,
                color=UC_COLOR if d == 'UC' else CD_COLOR, fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlabel('Mean |SHAP|', fontsize=8, color=INK2)
    ax.set_xlim(0, max_v * 1.6)
    auc_img = summary.get('img_histoscore_visit', {}).get('auc', '?')
    ax.set_title(f'Histological scores  (11 features)\nAUC {auc_img}',
                 fontsize=9.5, fontweight='bold', color=INK, loc='left', pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4); ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
    h = [mpatches.Patch(color=UC_COLOR, label='↑ UC'),
         mpatches.Patch(color=CD_COLOR, label='↑ CD')]
    ax.legend(handles=h, fontsize=7, frameon=True, framealpha=0.9, edgecolor=GRID)

    # Panel 2: RNA only
    def _rna_bar(ax, df, title):
        df = df.head(20)
        genes  = df['feature'].tolist()[::-1]
        labels = [gene_display(g, sym_map) for g in genes]
        values = df['mean_abs_shap'].tolist()[::-1]
        dirs   = df['direction'].tolist()[::-1]
        colors = [gene_cat_color(g, sym_map) for g in genes]
        y      = np.arange(len(genes))
        ax.barh(y, values, height=0.6, color=colors, edgecolor=SURF, linewidth=0.8)
        max_v  = max(values)
        for i, (val, g, d) in enumerate(zip(values, genes, dirs)):
            ax.text(val + max_v * 0.01, i, f'{val:.4f}',
                    va='center', ha='left', fontsize=6.5, color=INK2)
            ax.text(val + max_v * 0.01, i - 0.32,
                    '↑UC' if d == 'UC' else '↑CD',
                    va='center', ha='left', fontsize=5.5,
                    color=UC_COLOR if d == 'UC' else CD_COLOR, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_xlabel('Mean |SHAP|', fontsize=8, color=INK2)
        ax.set_xlim(0, max_v * 1.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold', color=INK, loc='left', pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.4); ax.yaxis.grid(False)
        ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)

    auc_rna = summary.get('rna_visit', {}).get('auc', '?')
    _rna_bar(axes[1], df_rna, f'RNA-seq only  (17,963 genes)\nAUC {auc_rna}')

    auc_cat = summary.get('concat_histoscore_visit', {}).get('auc', '?')
    rna_in_fus = df_fus[df_fus['feature'].str.startswith('ENSG')].reset_index(drop=True)
    hf = summary.get('concat_histoscore_visit', {}).get('histoscore_shap_fraction', 0)
    rf = summary.get('concat_histoscore_visit', {}).get('rna_shap_fraction', 0)
    _rna_bar(axes[2], rna_in_fus,
             f'RNA + Histo scores  (concat)\nAUC {auc_cat}'
             f'  ·  RNA={rf*100:.0f}%  Histo={hf*100:.0f}%')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print('Loading gene symbol map ...')
    sym_map = load_gene_symbols()

    df_img = pd.read_csv(f'{SHAP_DIR}/shap_img_histoscore_visit.csv')
    df_rna = pd.read_csv(f'{SHAP_DIR}/shap_rna_histoscore_visit_top500.csv')
    df_fus = pd.read_csv(f'{SHAP_DIR}/shap_concat_histoscore_visit_top500.csv')
    with open(f'{SHAP_DIR}/shap_histoscore_summary.json') as f:
        summary = json.load(f)

    n_vis   = 945
    n_pat   = 817
    auc_img = summary.get('img_histoscore_visit', {}).get('auc', 0)

    histo_shap_order = df_img.sort_values('mean_abs_shap', ascending=False)['feature'].tolist()
    histo_frac = summary.get('concat_histoscore_visit', {}).get('histoscore_shap_fraction', 0)
    rna_frac   = summary.get('concat_histoscore_visit', {}).get('rna_shap_fraction', 0)
    auc_cat    = summary.get('concat_histoscore_visit', {}).get('auc', 0)

    print('Loading cohort scores for distribution plot ...')
    df_cohort, histo_cols = load_histoscore_cohort()
    print(f'  {len(df_cohort)} at-20-cm slide-level records')

    print('  Fig 1: histoscore SHAP bar')
    fig1 = histoscore_shap_bar(df_img, auc_img, n_vis, n_pat)
    p1 = f'{PLOTS_DIR}/histoscore_shap_bar.pdf'
    fig1.savefig(p1, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig1); print(f'    saved {p1}')

    print('  Fig 2: score distribution')
    fig2 = score_distribution(df_cohort, histo_cols, histo_shap_order, auc_img)
    p2 = f'{PLOTS_DIR}/histoscore_swarm.pdf'
    fig2.savefig(p2, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig2); print(f'    saved {p2}')

    print('  Fig 3: fusion bar')
    fig3 = fusion_bar(df_fus, histo_frac, rna_frac, auc_cat, sym_map, n_vis, n_pat)
    p3 = f'{PLOTS_DIR}/histoscore_fusion_bar.pdf'
    fig3.savefig(p3, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig3); print(f'    saved {p3}')

    print('  Fig 4: combined panel')
    fig4 = combined_panel(df_img, df_rna, df_fus, summary, sym_map, n_vis, n_pat)
    p4 = f'{PLOTS_DIR}/histoscore_panel.pdf'
    fig4.savefig(p4, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig4); print(f'    saved {p4}')

    print(f'\nAll plots saved to {PLOTS_DIR}/')


if __name__ == '__main__':
    main()
