"""
SHAP plots for CD vs UC classifiers — visit-level cohort.

Mirrors 11_shap_plots.py using outputs from 10b_shap_analysis_visit.py.

Figures
-------
  Fig 1: shap_rna_visit_bar.pdf        — top 20 genes, RNA visit-level
  Fig 2: shap_img_visit_bar.pdf        — top 20 imaging dims, img visit-level
  Fig 3: shap_fusion_visit_split.pdf   — fusion modality breakdown
  Fig 4: shap_visit_panel.pdf          — 3-panel combined figure

Run after 10b_shap_analysis_visit.py has completed.
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

DATA_DIR    = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/data_visit'
PLOTS_DIR   = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/plots_visit'
VST_GCT     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct')
MAPPING_CSV = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
AT20        = {'at 20 cm', 'At 20 cm'}

# ── Validated categorical palette ─────────────────────────────────────────────
PAL = {
    'ECM / Fibrosis':          '#2a78d6',
    'HOX / Positional':        '#eb6834',
    'Immunity / Inflammation': '#1baf7a',
    'Transcription Factor':    '#eda100',
    'Neural / Receptor':       '#e87ba4',
    'Gut Hormone / Peptide':   '#008300',
    'Metabolism / Transport':  '#4a3aa7',
    'Other / Unknown':         '#898781',
}

SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
BASE  = '#c3c2b7'

GENE_ANNO = {
    'COL12A1':  ('ECM / Fibrosis',          'Type XII collagen; CD fibrosis/stricture'),
    'POSTN':    ('ECM / Fibrosis',          'Periostin; TGF-β fibrosis marker'),
    'COL5A2':   ('ECM / Fibrosis',          'Type V collagen; fibril assembly'),
    'COL3A1':   ('ECM / Fibrosis',          'Type III collagen; submucosal ECM'),
    'SPARC':    ('ECM / Fibrosis',          'Secreted acidic cysteine-rich; ECM remodeling'),
    'CPXM1':    ('ECM / Fibrosis',          'Carboxypeptidase X; ECM processing'),
    'ITGB8':    ('ECM / Fibrosis',          'Integrin β8; TGF-β activation via ECM'),
    'ADGRV1':   ('ECM / Fibrosis',          'Adhesion GPCR V1; cell-matrix adhesion'),
    'CARD6':    ('Immunity / Inflammation', 'Caspase recruit domain; inflammasome scaffold'),
    'CCL11':    ('Immunity / Inflammation', 'Eotaxin-1; eosinophil recruitment'),
    'DAPP1':    ('Immunity / Inflammation', 'Dual adaptor; B cell / mast cell signaling'),
    'HYAL1':    ('Immunity / Inflammation', 'Hyaluronidase 1; ECM-immune crosstalk'),
    'FOXP2':    ('Transcription Factor',    'Forkhead box P2; mucosal regulation'),
    'TBX3':     ('Transcription Factor',    'T-box 3; epithelial lineage'),
    'PITX1':    ('Transcription Factor',    'Paired-like homeobox; hindgut TF'),
    'ZNF492':   ('Transcription Factor',    'Zinc finger 492; transcriptional regulation'),
    'BRINP3':   ('Neural / Receptor',       'BMP/RA-inducible neural; neuronal/ECM'),
    'NPSR1':    ('Neural / Receptor',       'Neuropeptide S receptor; enteric motility'),
    'MYEOV':    ('Neural / Receptor',       'Myeloma overexpressed; ENS-associated'),
    'DPP10':    ('Neural / Receptor',       'Dipeptidyl peptidase 10; Kv channel modulator'),
    'ACAT1':    ('Metabolism / Transport',  'Acetyl-CoA acetyltransferase; ketone metabolism'),
    'SELENBP1': ('Metabolism / Transport',  'Selenium binding; mucosal oxidative stress'),
    'CYP2C18':  ('Metabolism / Transport',  'Cytochrome P450 2C18; xenobiotic metabolism'),
    'ABCA13':   ('Metabolism / Transport',  'ABC transporter A13; lipid export'),
    'HOXB2':    ('HOX / Positional',        'Homeobox B2; AP-axis identity'),
    'HOXA13':   ('HOX / Positional',        'Homeobox A13; posterior gut identity'),
    'HOXB13':   ('HOX / Positional',        'Homeobox B13; posterior colon position'),
    'GCG':      ('Gut Hormone / Peptide',   'Proglucagon → GLP-1/GLP-2'),
    'PYY':      ('Gut Hormone / Peptide',   'Peptide YY; L-cell, satiety'),
}


_ENSG_TO_SYMBOL = {}

def load_gene_symbols():
    """Populate _ENSG_TO_SYMBOL from the GCT Description column (run once)."""
    global _ENSG_TO_SYMBOL
    if _ENSG_TO_SYMBOL:
        return
    gmap = pd.read_csv(VST_GCT, sep='\t', skiprows=2,
                       usecols=['Name', 'Description'])
    _ENSG_TO_SYMBOL = dict(zip(gmap['Name'], gmap['Description']))


def to_symbol(feature):
    """Map ENSG ID to gene symbol; pass through anything that isn't ENSG-like."""
    if feature.startswith('ENSG'):
        return _ENSG_TO_SYMBOL.get(feature, feature)
    return feature


def get_anno(gene):
    sym = to_symbol(gene)
    if sym in GENE_ANNO:
        return GENE_ANNO[sym]
    if gene.startswith('img_') or gene.startswith('f'):
        return ('Other / Unknown', 'Virchow2 embedding dim')
    return ('Other / Unknown', '')


def display_name(feature):
    """Return gene symbol for labels; keep img_f* as-is."""
    if feature.startswith('ENSG'):
        return _ENSG_TO_SYMBOL.get(feature, feature)
    return feature


# ── Expression-based direction ─────────────────────────────────────────────────
# direction = 'UC' if mean expression is higher in UC, 'CD' otherwise.
# For top RF SHAP features, expression direction ≈ SHAP sign.

_GENE_DIRECTION = {}   # ensg_id -> 'UC' or 'CD'


def _norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return 'CD'
        if 'Ulcerative' in d: return 'UC'
    return None


def compute_gene_directions(ensg_ids):
    """Populate _GENE_DIRECTION for the requested ENSG IDs (skips already cached)."""
    global _GENE_DIRECTION
    needed = {e for e in ensg_ids if e.startswith('ENSG') and e not in _GENE_DIRECTION}
    if not needed:
        return

    print(f'  Computing expression directions for {len(needed)} genes ...')

    # ── get sample IDs → label for at-20-cm cohort ────────────────────────────
    cv_pids = set(pd.read_csv(CV_PATIENTS)['patient_id'])
    mapping = pd.read_csv(MAPPING_CSV)
    smeta   = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(smeta[['SampleID', 'diagnosis', 'Sample QC']],
                        on='SampleID', how='left')
    rna['_dx'] = rna['diagnosis'].map(_norm_dx)
    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['_dx'].isin(['CD', 'UC']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].drop_duplicates(subset='visit_encounter_id', keep='first')
    sample_label = dict(zip(rna['SampleID'], rna['_dx']))  # sid -> 'CD' or 'UC'

    # ── read GCT header once to build column index list ───────────────────────
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header = f.readline().rstrip('\n').split('\t')
    col_info = [(i, sample_label[sid])
                for i, sid in enumerate(header)
                if sid in sample_label]

    # ── scan GCT for target genes only (avoids loading full 17K × 3289 matrix) ─
    uc_sum = {g: 0.0 for g in needed}
    cd_sum = {g: 0.0 for g in needed}
    uc_cnt = {g: 0   for g in needed}
    cd_cnt = {g: 0   for g in needed}

    with open(VST_GCT) as f:
        f.readline(); f.readline(); f.readline()   # skip 3 header lines
        found = 0
        for line in f:
            tab = line.index('\t')
            gene = line[:tab]
            if gene not in needed:
                continue
            parts = line.rstrip('\n').split('\t')
            for col_idx, dx in col_info:
                v = float(parts[col_idx])
                if dx == 'UC':
                    uc_sum[gene] += v; uc_cnt[gene] += 1
                else:
                    cd_sum[gene] += v; cd_cnt[gene] += 1
            found += 1
            if found == len(needed):
                break   # all target genes found — stop early

    for gene in needed:
        if uc_cnt[gene] > 0 and cd_cnt[gene] > 0:
            mean_uc = uc_sum[gene] / uc_cnt[gene]
            mean_cd = cd_sum[gene] / cd_cnt[gene]
            _GENE_DIRECTION[gene] = 'UC' if mean_uc > mean_cd else 'CD'

    print(f'    done ({len([g for g in needed if g in _GENE_DIRECTION])} resolved)')


def direction_tag(feature):
    """Return a short direction tag: '↑UC', '↑CD', or '' for imaging features."""
    if not feature.startswith('ENSG'):
        return ''
    d = _GENE_DIRECTION.get(feature)
    return f'↑{d}' if d else ''


DIR_COLOR = {'UC': '#c94040', 'CD': '#2a78d6'}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': BASE,   'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True,        'grid.color': GRID,
        'grid.linewidth': 0.4,    'grid.alpha': 1.0,
        'axes.axisbelow': True,
        'xtick.color': MUTED,     'ytick.color': INK2,
        'xtick.labelsize': 8,     'ytick.labelsize': 9,
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'text.color': INK,
    })


def cat_color(gene):
    cat, _ = get_anno(gene)
    return PAL.get(cat, PAL['Other / Unknown'])


def legend_handles(categories):
    return [mpatches.Patch(color=PAL[c], label=c)
            for c in PAL if c in categories]


# ── Horizontal bar ─────────────────────────────────────────────────────────────

def bar_figure(df_top20, title, subtitle, figsize=(10, 7)):
    setup_style()
    genes  = df_top20['feature'].tolist()[::-1]
    labels = [display_name(g) for g in genes]
    values = df_top20['mean_abs_shap'].tolist()[::-1]
    colors = [cat_color(g) for g in genes]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURF)
    y = np.arange(len(genes))
    ax.barh(y, values, height=0.6, color=colors, edgecolor=SURF, linewidth=0.8)

    max_v = max(values)
    for i, (val, gene) in enumerate(zip(values, genes)):
        tag = direction_tag(gene)
        tag_color = DIR_COLOR.get(_GENE_DIRECTION.get(gene, ''), INK2)
        ax.text(val + max_v * 0.005, i, f'{val:.4f}',
                va='center', ha='left', fontsize=7.5, color=INK2)
        if tag:
            ax.text(val + max_v * 0.005, i - 0.32, tag,
                    va='center', ha='left', fontsize=6, color=tag_color,
                    fontweight='bold')
    for i, gene in enumerate(genes):
        _, func = get_anno(gene)
        if func:
            ax.text(max_v * 1.08, i, func, va='center', ha='left',
                    fontsize=7, color=MUTED, style='italic')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel('Mean |SHAP value|', fontsize=9, color=INK2)
    ax.set_xlim(0, max_v * 1.55)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold',
                 color=INK, pad=10, loc='left')
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)

    cats_present = {get_anno(g)[0] for g in genes}
    handles = legend_handles(cats_present)
    fig.legend(handles=handles, title='Biological category',
               fontsize=7.5, title_fontsize=8, frameon=True,
               framealpha=0.9, edgecolor=GRID, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(handles), 4),
               borderpad=0.6)
    fig.tight_layout()
    return fig


# ── Fusion split ───────────────────────────────────────────────────────────────

def fusion_figure(df_fus, img_frac, rna_frac, auc, n_visits, n_patients,
                  figsize=(10, 6)):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.4, 2.5]})
    fig.patch.set_facecolor(SURF)

    ax = axes[0]
    ax.set_facecolor(SURF)
    cats   = ['RNA\n(17,963 genes)', 'Imaging\n(2,560 dims)']
    fracs  = [rna_frac * 100, img_frac * 100]
    colors = [PAL['ECM / Fibrosis'], '#6baed6']
    brs = ax.barh(cats, fracs, height=0.45, color=colors,
                  edgecolor=SURF, linewidth=0.8)
    for bar, val in zip(brs, fracs):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                fontsize=10, fontweight='bold', color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel('% of total mean |SHAP|', fontsize=8.5, color=INK2)
    ax.set_title('Fusion SHAP\nmodality split', fontsize=10,
                 fontweight='bold', color=INK, loc='left')
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
    ax.text(0, -0.62,
            f'concat_raw_visit (At-20-cm)\n'
            f'{n_visits} visits · {n_patients} patients · RF · 5-fold CV',
            transform=ax.transAxes, fontsize=7, color=MUTED)

    ax2 = axes[1]
    ax2.set_facecolor(SURF)
    rna_only = df_fus[~df_fus['feature'].str.startswith('img_')].head(20)
    genes  = rna_only['feature'].tolist()[::-1]
    labels = [display_name(g) for g in genes]
    values = rna_only['mean_abs_shap'].tolist()[::-1]
    colors2 = [cat_color(g) for g in genes]
    y = np.arange(len(genes))
    ax2.barh(y, values, height=0.55, color=colors2,
             edgecolor=SURF, linewidth=0.8)
    max_v = max(values)
    for i, (val, g) in enumerate(zip(values, genes)):
        tag = direction_tag(g)
        tag_color = DIR_COLOR.get(_GENE_DIRECTION.get(g, ''), INK2)
        ax2.text(val + max_v * 0.01, i, f'{val:.4f}',
                 va='center', ha='left', fontsize=7, color=INK2)
        if tag:
            ax2.text(val + max_v * 0.01, i - 0.32, tag,
                     va='center', ha='left', fontsize=6, color=tag_color,
                     fontweight='bold')
        _, func = get_anno(g)
        ax2.text(max_v * 1.08, i, func, va='center', ha='left',
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

    cats_present = {get_anno(g)[0] for g in genes}
    handles = legend_handles(cats_present)
    fig.legend(handles=handles, title='Biological category', fontsize=7,
               title_fontsize=7.5, frameon=True, framealpha=0.9,
               edgecolor=GRID, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(handles), 4), borderpad=0.6)
    fig.suptitle(
        f'Multimodal Fusion (concat_raw_visit) — At-20-cm visit-level cohort'
        f'  ·  AUC={auc:.3f}',
        fontsize=11, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout()
    return fig


# ── 3-panel combined ───────────────────────────────────────────────────────────

def combined_panel(rna_df, img_df, fus_df, summary, n_visits, n_patients,
                   figsize=(18, 7)):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor(SURF)
    fig.suptitle(
        f'SHAP feature importance — At-20-cm visit-level cohort'
        f'  ({n_visits} visits, {n_patients} patients)',
        fontsize=12, fontweight='bold', color=INK, y=1.01)

    def _bar(ax, df, title, subtitle, max_genes=20):
        genes  = df['feature'].tolist()[:max_genes][::-1]
        labels = [display_name(g) for g in genes]
        values = df['mean_abs_shap'].tolist()[:max_genes][::-1]
        colors = [cat_color(g) for g in genes]
        y = np.arange(len(genes))
        ax.barh(y, values, height=0.6, color=colors,
                edgecolor=SURF, linewidth=0.8)
        max_v = max(values)
        for i, (val, gene) in enumerate(zip(values, genes)):
            tag = direction_tag(gene)
            tag_color = DIR_COLOR.get(_GENE_DIRECTION.get(gene, ''), INK2)
            ax.text(val + max_v * 0.008, i, f'{val:.4f}',
                    va='center', ha='left', fontsize=6.5, color=INK2)
            if tag:
                ax.text(val + max_v * 0.008, i - 0.32, tag,
                        va='center', ha='left', fontsize=5.5, color=tag_color,
                        fontweight='bold')
            _, func = get_anno(gene)
            ax.text(max_v * 1.12, i, func, va='center', ha='left',
                    fontsize=6, color=MUTED, style='italic')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_xlabel('Mean |SHAP value|', fontsize=8, color=INK2)
        ax.set_xlim(0, max_v * 1.65)
        ax.set_title(f'{title}\n{subtitle}', fontsize=9.5,
                     fontweight='bold', color=INK, loc='left', pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.4)
        ax.yaxis.grid(False)
        ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)

    auc_img = summary.get('img_base_visit', {}).get('auc', '?')
    auc_rna = summary.get('rna_visit', {}).get('auc', '?')
    auc_cat = summary.get('concat_raw_visit', {}).get('auc', '?')
    img_frac = summary.get('concat_raw_visit', {}).get('imaging_shap_fraction', 0)
    rna_frac = summary.get('concat_raw_visit', {}).get('rna_shap_fraction', 0)

    _bar(axes[0], img_df,
         'Imaging only  (prism2_base)',
         f'AUC {auc_img}  ·  2,560 dims')
    _bar(axes[1], rna_df,
         'RNA-seq only  (VST)',
         f'AUC {auc_rna}  ·  17,963 genes')

    # Panel 3: fusion RNA genes + modality split annotation
    rna_in_fus = fus_df[~fus_df['feature'].str.startswith('img_')].reset_index(drop=True)
    _bar(axes[2], rna_in_fus,
         'RNA + Imaging  (concat raw)',
         f'AUC {auc_cat}  ·  RNA={rna_frac*100:.1f}%  Img={img_frac*100:.1f}% of SHAP')

    all_cats = {get_anno(g)[0]
                for df in [rna_df.head(20), img_df.head(20), fus_df.head(20)]
                for g in df['feature']}
    handles = legend_handles(all_cats)
    fig.legend(handles=handles, title='Biological category',
               fontsize=7.5, title_fontsize=8, frameon=True,
               framealpha=0.92, edgecolor=GRID, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=len(handles), borderpad=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print('Loading gene symbol map ...')
    load_gene_symbols()

    rna_df = pd.read_csv(f'{DATA_DIR}/shap_rna_visit_top500.csv')
    img_df = pd.read_csv(f'{DATA_DIR}/shap_img_base_visit_top500.csv')
    fus_df = pd.read_csv(f'{DATA_DIR}/shap_concat_raw_visit_top500.csv')
    with open(f'{DATA_DIR}/shap_visit_summary.json') as f:
        summary = json.load(f)

    n_visits   = 945
    n_patients = 817

    # Pre-compute expression-based directions for all RNA features in top 20 per arm
    all_ensg = set(rna_df.head(20)['feature']) | set(fus_df.head(40)['feature'])
    compute_gene_directions(all_ensg)

    img_frac = summary.get('concat_raw_visit', {}).get('imaging_shap_fraction', 0)
    rna_frac = summary.get('concat_raw_visit', {}).get('rna_shap_fraction', 0)
    auc_cat  = summary.get('concat_raw_visit', {}).get('auc', 0)

    # ── Fig 1: RNA visit bar ───────────────────────────────────────────────────
    print('  Fig 1: rna_visit bar')
    auc_rna = summary.get('rna_visit', {}).get('auc', '?')
    fig1 = bar_figure(
        rna_df.head(20),
        title='Top 20 genes — rna_visit (At-20-cm, visit-level)',
        subtitle=f'Site-controlled · {n_visits} visits · {n_patients} patients'
                 f' · AUC {auc_rna}',
    )
    path1 = f'{PLOTS_DIR}/shap_rna_visit_bar.pdf'
    fig1.savefig(path1, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig1)
    print(f'    saved {path1}')

    # ── Fig 2: imaging visit bar ───────────────────────────────────────────────
    print('  Fig 2: img_base_visit bar')
    auc_img = summary.get('img_base_visit', {}).get('auc', '?')
    fig2 = bar_figure(
        img_df.head(20),
        title='Top 20 dims — img_base_visit (At-20-cm, visit-level)',
        subtitle=f'prism2_base (Virchow2) · {n_visits} visits · AUC {auc_img}',
    )
    path2 = f'{PLOTS_DIR}/shap_img_visit_bar.pdf'
    fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig2)
    print(f'    saved {path2}')

    # ── Fig 3: fusion split ────────────────────────────────────────────────────
    print('  Fig 3: fusion split')
    fig3 = fusion_figure(fus_df, img_frac, rna_frac, auc_cat,
                         n_visits, n_patients)
    path3 = f'{PLOTS_DIR}/shap_fusion_visit_split.pdf'
    fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig3)
    print(f'    saved {path3}')

    # ── Fig 4: 3-panel combined ────────────────────────────────────────────────
    print('  Fig 4: combined panel')
    fig4 = combined_panel(rna_df, img_df, fus_df, summary, n_visits, n_patients)
    path4 = f'{PLOTS_DIR}/shap_visit_panel.pdf'
    fig4.savefig(path4, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig4)
    print(f'    saved {path4}')

    print(f'\nAll visit-level SHAP plots saved to {PLOTS_DIR}/')


if __name__ == '__main__':
    main()
