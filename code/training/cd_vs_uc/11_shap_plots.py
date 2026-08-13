"""
SHAP plots for CD vs UC classifiers — At-20-cm and all-sites arms.

Figures
-------
  Fig 1: shap_rna_20cm_bar.pdf          — top 20 genes, At-20-cm RNA
  Fig 2: shap_rna_allsites_bar.pdf      — top 20 genes, all-sites RNA
  Fig 3: shap_rank_comparison.pdf       — dumbbell rank-shift (20cm vs all-sites)
  Fig 4: shap_fusion_split.pdf          — fusion modality breakdown
  Fig 5: shap_panel.pdf                 — 4-panel combined figure

Design follows the dataviz skill (palette.md):
  Categorical slots (light mode, adjacent-validated):
    #2a78d6  blue     ECM / Fibrosis
    #eb6834  orange   HOX / Positional identity (site confound)
    #1baf7a  aqua     Immunity / Inflammation
    #eda100  yellow   Transcription Factor
    #e87ba4  magenta  Neural / Receptor
    #008300  green    Gut Hormone / Peptide
    #4a3aa7  violet   Metabolism / Transport
    #898781  gray     Other / Unknown
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DATA_DIR  = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/data'
PLOTS_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/plots'

# ── Validated categorical palette (slots 1-8 + gray, palette.md) ──────────────
PAL = {
    'ECM / Fibrosis':          '#2a78d6',   # blue   slot-1
    'HOX / Positional':        '#eb6834',   # orange slot-2  ← site confound
    'Immunity / Inflammation': '#1baf7a',   # aqua   slot-3
    'Transcription Factor':    '#eda100',   # yellow slot-4
    'Neural / Receptor':       '#e87ba4',   # magenta slot-5
    'Gut Hormone / Peptide':   '#008300',   # green  slot-6
    'Metabolism / Transport':  '#4a3aa7',   # violet slot-7
    'Other / Unknown':         '#898781',   # gray   muted
}

SURF  = '#fcfcfb'   # chart surface
INK   = '#0b0b0b'   # primary ink
INK2  = '#52514e'   # secondary ink
MUTED = '#898781'   # axis/muted
GRID  = '#e1e0d9'   # hairline gridline
BASE  = '#c3c2b7'   # axis baseline

# ── Gene annotations: (category, short_function) ──────────────────────────────
GENE_ANNO = {
    # ── rna_20cm top genes (disease biology) ──────────────────────────────────
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
    'CABP4':    ('Neural / Receptor',       'Calcium binding protein 4; retinal/ENS'),
    'SYT10':    ('Neural / Receptor',       'Synaptotagmin 10; calcium sensor / ENS'),
    'ACAT1':    ('Metabolism / Transport',  'Acetyl-CoA acetyltransferase; ketone metabolism'),
    'SELENBP1': ('Metabolism / Transport',  'Selenium binding; mucosal oxidative stress'),
    'CYP2C18':  ('Metabolism / Transport',  'Cytochrome P450 2C18; xenobiotic metabolism'),
    'ABCA13':   ('Metabolism / Transport',  'ABC transporter A13; lipid export'),
    'LRRIQ4':   ('Metabolism / Transport',  'LRR-IQ motif 4; ciliary/metabolic function'),
    'SHC3':     ('Metabolism / Transport',  'SHC adaptor 3; RTK/growth factor signaling'),
    'RUSC2':    ('Other / Unknown',         'RUN/FYVE domain; endosomal trafficking'),
    'STPG4':    ('Other / Unknown',         'Sperm-tail PG-rich repeat; poorly characterized'),
    'XKR9':     ('Other / Unknown',         'XK related 9; phospholipid scramblase'),
    'ABCA13':   ('Metabolism / Transport',  'ABC transporter A13; lipid export'),
    # ── rna_patmean extra genes (site confound signal) ────────────────────────
    'HOXB2':    ('HOX / Positional',        'Homeobox B2; AP-axis identity → cecum signal'),
    'HOXA13':   ('HOX / Positional',        'Homeobox A13; posterior gut identity → cecum'),
    'HOXB13':   ('HOX / Positional',        'Homeobox B13; posterior colon position → cecum'),
    'PITX2':    ('HOX / Positional',        'Paired-like 2; L-R asymmetry, proximal gut'),
    'FOXA2':    ('Transcription Factor',    'FoxA2; endoderm axis patterning'),
    'THRB':     ('Transcription Factor',    'Thyroid hormone receptor β; metabolic TF'),
    'GCG':      ('Gut Hormone / Peptide',   'Proglucagon → GLP-1/GLP-2; proximal gut'),
    'PYY':      ('Gut Hormone / Peptide',   'Peptide YY; L-cell, satiety, proximal gut'),
    'DRD5':     ('Neural / Receptor',       'Dopamine receptor D5; enteric nervous system'),
    'CLDN8':    ('ECM / Fibrosis',          'Claudin 8; tight junction, cecum-enriched'),
    'SLC14A2':  ('Metabolism / Transport',  'Urea transporter UT-A2; position-dependent'),
    'ST3GAL4':  ('Metabolism / Transport',  'Sialyltransferase; mucin glycosylation'),
    'B3GALT5':  ('Metabolism / Transport',  'Galactosyltransferase; Lewis glycan'),
    'B3GNT7':   ('Metabolism / Transport',  'GlcNAc transferase; glycan biosynthesis'),
    'TRPM6':    ('Metabolism / Transport',  'TRPM6; Mg²⁺ absorption, cecum-enriched'),
    'CAPN13':   ('Immunity / Inflammation', 'Calpain 13; calcium-dependent protease'),
    'GPC3':     ('Other / Unknown',         'Glypican 3; Wnt signaling co-receptor'),
    'POPDC3':   ('Other / Unknown',         'Popeye cAMP-binding; smooth muscle'),
    'RIMKLA':   ('Other / Unknown',         'Ribosomal modification; poorly characterized'),
    'CPA6':     ('Other / Unknown',         'Carboxypeptidase A6; extracellular proteolysis'),
}

def get_anno(gene):
    if gene in GENE_ANNO:
        return GENE_ANNO[gene]
    if gene.startswith('img_') or gene.startswith('f'):
        return ('Other / Unknown', 'Virchow2 embedding dim')
    return ('Other / Unknown', '')


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': BASE, 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
        'grid.alpha': 1.0, 'axes.axisbelow': True,
        'xtick.color': MUTED, 'ytick.color': INK2,
        'xtick.labelsize': 8, 'ytick.labelsize': 9,
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'text.color': INK,
    })


def cat_color(gene):
    cat, _ = get_anno(gene)
    return PAL.get(cat, PAL['Other / Unknown'])


def legend_handles(categories):
    return [mpatches.Patch(color=PAL[c], label=c)
            for c in PAL if c in categories]


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 & 2: Horizontal bar — top 20 genes per RNA arm
# ══════════════════════════════════════════════════════════════════════════════

def bar_figure(df_top20, title, subtitle, auc_str, symbol_col='symbol',
               shap_col='mean_abs_shap', right_label_col=None,
               comparison_rank=None, figsize=(10, 7)):
    """
    Horizontal bar chart: genes on y-axis, mean|SHAP| on x.
    Color by biological category.
    Annotate with brief function on the right.
    """
    setup_style()
    genes  = df_top20[symbol_col].tolist()[::-1]   # reverse so rank-1 at top
    values = df_top20[shap_col].tolist()[::-1]
    colors = [cat_color(g) for g in genes]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURF)

    y = np.arange(len(genes))
    bars = ax.barh(y, values, height=0.6, color=colors,
                   edgecolor=SURF, linewidth=0.8)

    # direct value labels inside/beside bars
    max_v = max(values)
    for i, (val, bar) in enumerate(zip(values, bars)):
        ax.text(val + max_v * 0.005, i, f'{val:.4f}',
                va='center', ha='left', fontsize=7.5, color=INK2)

    # right-side function annotation
    for i, gene in enumerate(genes):
        _, func = get_anno(gene)
        if func:
            ax.text(max_v * 1.08, i, func,
                    va='center', ha='left', fontsize=7, color=MUTED,
                    style='italic')

    # rank markers from comparison arm
    if comparison_rank is not None:
        cr = comparison_rank  # dict: gene → rank
        for i, gene in enumerate(genes):
            if gene in cr:
                r = cr[gene]
                label = f'#{r} all-sites' if title.endswith('20cm') else f'#{r} @20cm'
                ax.text(-max_v * 0.01, i, label,
                        va='center', ha='right', fontsize=6.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=9, color=INK)
    ax.set_xlabel('Mean |SHAP value|', fontsize=9, color=INK2)
    ax.set_xlim(0, max_v * 1.55)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)

    # title block
    ax.set_title(f'{title}\n{subtitle}  ·  AUC = {auc_str}',
                 fontsize=11, fontweight='bold', color=INK, pad=10, loc='left')

    # legend
    cats_present = {get_anno(g)[0] for g in genes}
    handles = legend_handles(cats_present)
    leg = ax.legend(handles=handles, title='Biological category',
                    fontsize=7.5, title_fontsize=8,
                    frameon=True, framealpha=0.9, edgecolor=GRID,
                    loc='lower right', bbox_to_anchor=(0.98, 0.01))

    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Dumbbell rank-shift
# ══════════════════════════════════════════════════════════════════════════════

def dumbbell_figure(cmp_df, n=30, cap=350, figsize=(11, 9)):
    """
    For the top-n genes in rna_20cm show their rank in rna_20cm (circle, x-left)
    and rna_patmean (diamond, x-right). X-axis capped at `cap`; genes beyond cap
    get an arrowhead at the edge and a "→ #rank" label.
    """
    setup_style()
    df = cmp_df.head(n).copy()
    df['rank_allsites'] = df['rank_allsites'].fillna(9999).astype(int)
    df = df.iloc[::-1].reset_index(drop=True)   # rank-1 at top

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURF)
    y = np.arange(len(df))
    DOT = 55   # scatter s=

    for i, row in df.iterrows():
        r20  = int(row['rank_20cm'])
        rall = int(row['rank_allsites'])
        gene = row['symbol'] if 'symbol' in df.columns else row['feature']
        cat, func = get_anno(gene)
        col = PAL.get(cat, PAL['Other / Unknown'])

        rall_capped = min(rall, cap)
        rank_diff   = rall - r20

        # line color: red = big demotion (disease-biology gene hidden by site signal)
        if rank_diff > 100:
            line_col, line_a = '#c0392b', 0.65
        elif rank_diff < -20:
            line_col, line_a = '#27ae60', 0.55
        else:
            line_col, line_a = GRID,      0.9

        # connecting line (clipped to cap)
        ax.plot([r20, rall_capped], [i, i], color=line_col,
                linewidth=1.3, alpha=line_a, zorder=1, solid_capstyle='round')

        # at-20-cm dot (circle)
        ax.scatter([r20], [i], color=col, s=DOT, zorder=4,
                   edgecolors=SURF, linewidths=0.8)

        # all-sites dot (diamond); arrowhead if beyond cap
        if rall <= cap:
            ax.scatter([rall], [i], color=col, s=DOT, zorder=4,
                       edgecolors=SURF, linewidths=0.8, marker='D')
        else:
            # arrowhead at the cap edge
            ax.annotate('', xy=(cap, i), xytext=(cap - 18, i),
                        arrowprops=dict(arrowstyle='->', color=line_col,
                                        lw=1.3), zorder=5)

        # gene label (left)
        ax.text(-6, i, gene, va='center', ha='right', fontsize=8.5, color=INK)

        # out-of-cap rank label (right)
        if rall > cap:
            ax.text(cap + 4, i, f'→ #{rall:,}',
                    va='center', ha='left', fontsize=7, color='#c0392b')
        elif rall > r20 + 20:
            # small rank label for visible but shifted genes
            ax.text(rall + 4, i, f'#{rall}',
                    va='center', ha='left', fontsize=6.5, color=INK2)

        # function annotation (far right, muted)
        ax.text(cap + (70 if rall > cap else 50), i, func,
                va='center', ha='left', fontsize=6.5, color=MUTED, style='italic')

    ax.set_yticks([])
    ax.set_xlabel('Gene importance rank (rank 1 = highest SHAP)', fontsize=9, color=INK2)
    ax.set_xlim(-90, cap + 10)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300, cap])
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(BASE)

    # cap boundary marker
    ax.axvline(cap, color=MUTED, linewidth=0.6, linestyle='--', alpha=0.5)
    ax.text(cap + 2, -1.2, f'cap={cap}', fontsize=7, color=MUTED, va='top')

    ax.set_title(
        'Gene rank shift: At-20-cm (●) vs all-sites (◆)\n'
        'Red line = gene important @20cm but ranked much lower in all-sites\n'
        '→ = rank beyond cap (masked by site-identity signal in all-sites analysis)',
        fontsize=9.5, fontweight='bold', color=INK, pad=8, loc='left')

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=INK2,
               markersize=7, label='Rank in rna_20cm (At-20-cm)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=INK2,
               markersize=7, label='Rank in rna_patmean (all-sites)'),
        Line2D([0],[0], color='#c0392b', linewidth=2, alpha=0.7,
               label='Gene demoted in all-sites  (disease biology, not site)'),
        Line2D([0],[0], color='#27ae60', linewidth=2, alpha=0.7,
               label='Gene promoted in all-sites (site-correlated)'),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor=GRID, loc='lower right', bbox_to_anchor=(0.48, 0.0))

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: Fusion modality split + top fusion genes
# ══════════════════════════════════════════════════════════════════════════════

def fusion_figure(df_fus, img_frac, rna_frac, figsize=(10, 6)):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.4, 2.5]})
    fig.patch.set_facecolor(SURF)

    # ── left: modality split bar ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(SURF)
    cats   = ['RNA\n(17,963 genes)', 'Imaging\n(2,560 dims)']
    fracs  = [rna_frac * 100, img_frac * 100]
    colors = [PAL['ECM / Fibrosis'], '#6baed6']   # blue tones: deep=RNA, mid=img
    brs = ax.barh(cats, fracs, height=0.45, color=colors,
                  edgecolor=SURF, linewidth=0.8)
    for bar, val in zip(brs, fracs):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left',
                fontsize=10, fontweight='bold', color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel('% of total mean |SHAP|', fontsize=8.5, color=INK2)
    ax.set_title('Fusion SHAP\nmodality split', fontsize=10,
                 fontweight='bold', color=INK, loc='left')
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
    # note
    ax.text(0, -0.62, 'concat_raw_20cm (At-20-cm)\n828 patients · RF · 5-fold CV',
            transform=ax.transAxes, fontsize=7, color=MUTED)

    # ── right: top 20 RNA genes in fusion ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(SURF)
    rna_only = df_fus[~df_fus['feature'].str.startswith('img_')].head(20)
    genes  = (rna_only['symbol'] if 'symbol' in rna_only.columns
              else rna_only['feature']).tolist()[::-1]
    values = rna_only['mean_abs_shap'].tolist()[::-1]
    colors2 = [cat_color(g) for g in genes]
    y = np.arange(len(genes))
    ax2.barh(y, values, height=0.55, color=colors2,
             edgecolor=SURF, linewidth=0.8)
    max_v = max(values)
    for i, (val, g) in enumerate(zip(values, genes)):
        ax2.text(val + max_v * 0.01, i, f'{val:.4f}',
                 va='center', ha='left', fontsize=7, color=INK2)
        _, func = get_anno(g)
        ax2.text(max_v * 1.08, i, func,
                 va='center', ha='left', fontsize=6.5, color=MUTED, style='italic')
    ax2.set_yticks(y)
    ax2.set_yticklabels(genes, fontsize=8.5, color=INK)
    ax2.set_xlabel('Mean |SHAP value|', fontsize=8.5, color=INK2)
    ax2.set_xlim(0, max_v * 1.6)
    ax2.set_title('Top RNA genes driving fusion predictions',
                  fontsize=10, fontweight='bold', color=INK, loc='left')
    ax2.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax2.yaxis.grid(False)
    ax2.spines['left'].set_color(BASE); ax2.spines['bottom'].set_color(BASE)

    cats_present = {get_anno(g)[0] for g in genes}
    handles = legend_handles(cats_present)
    ax2.legend(handles=handles, title='Category', fontsize=7,
               title_fontsize=7.5, frameon=True, framealpha=0.9,
               edgecolor=GRID, loc='lower right')

    fig.suptitle('Multimodal Fusion (concat_raw_20cm) — At-20-cm Cohort',
                 fontsize=11, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: Side-by-side top-20 panel (20cm vs all-sites)
# ══════════════════════════════════════════════════════════════════════════════

def comparison_panel(rna_20, rna_all, figsize=(16, 8)):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(SURF)
    fig.suptitle(
        'Top 20 genes by SHAP importance: site-controlled (At-20-cm) vs all-sites RNA',
        fontsize=12, fontweight='bold', color=INK, y=1.01)

    def _bar(ax, df, title, subtitle, highlight_hox=False):
        genes  = df['symbol'].tolist()[::-1]
        values = df['mean_abs_shap'].tolist()[::-1]
        colors = [cat_color(g) for g in genes]
        y = np.arange(len(genes))
        ax.barh(y, values, height=0.6, color=colors,
                edgecolor=SURF, linewidth=0.8)
        max_v = max(values)
        for i, (val, gene) in enumerate(zip(values, genes)):
            ax.text(val + max_v * 0.008, i, f'{val:.4f}',
                    va='center', ha='left', fontsize=7, color=INK2)
            _, func = get_anno(gene)
            cat, _ = get_anno(gene)
            fn_color = '#c0392b' if cat == 'HOX / Positional' else MUTED
            ax.text(max_v * 1.12, i, func,
                    va='center', ha='left', fontsize=6.5,
                    color=fn_color, style='italic',
                    fontweight='bold' if cat == 'HOX / Positional' else 'normal')
        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=8.5, color=INK)
        ax.set_xlabel('Mean |SHAP value|', fontsize=8.5, color=INK2)
        ax.set_xlim(0, max_v * 1.65)
        ax.set_title(f'{title}\n{subtitle}', fontsize=10,
                     fontweight='bold', color=INK, loc='left', pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.4)
        ax.yaxis.grid(False)
        ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)

        # no per-axes legend; shared figure legend added below

    _bar(axes[0], rna_20.head(20),
         'rna_20cm  (At-20-cm only)',
         'AUC 0.824 ± 0.019  ·  828 patients  ·  disease biology signal')
    _bar(axes[1], rna_all.head(20),
         'rna_patmean  (all colon sites)',
         'AUC 0.920 ± 0.007  ·  997 patients  ·  ⚠ includes site-identity signal',
         highlight_hox=True)

    # single figure-level legend below both panels
    all_cats = {get_anno(g)[0]
                for df in [rna_20.head(20), rna_all.head(20)]
                for g in df['symbol']}
    handles = legend_handles(all_cats)
    fig.legend(handles=handles, title='Biological category',
               fontsize=7.5, title_fontsize=8,
               frameon=True, framealpha=0.92, edgecolor=GRID,
               loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=len(handles), borderpad=0.6)

    # annotation bracket for HOX genes in right panel
    hox_ranks = [i for i, g in enumerate(rna_all['symbol'].tolist()[::-1])
                 if get_anno(g)[0] == 'HOX / Positional']
    if hox_ranks:
        y_min, y_max = min(hox_ranks) - 0.3, max(hox_ranks) + 0.3
        xref = rna_all['mean_abs_shap'].max() * 0.72
        axes[1].annotate(
            'HOX positional genes\n(site-identity shortcut)',
            xy=(xref, (y_min + y_max) / 2),
            xytext=(xref + rna_all['mean_abs_shap'].max() * 0.15,
                    (y_min + y_max) / 2),
            fontsize=7.5, color='#c0392b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.1),
            va='center',
        )

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rna_20  = pd.read_csv(f'{DATA_DIR}/shap_rna_20cm_top500.csv')
    rna_all = pd.read_csv(f'{DATA_DIR}/shap_rna_patmean_top500.csv')
    df_fus  = pd.read_csv(f'{DATA_DIR}/shap_concat_raw_20cm_top500.csv')
    cmp     = pd.read_csv(f'{DATA_DIR}/shap_rna_20cm_vs_patmean.csv')

    # ensure symbol column exists (added in previous script)
    for df in [rna_20, rna_all, df_fus, cmp]:
        if 'symbol' not in df.columns:
            df['symbol'] = df['feature']

    rank_20_to_all  = dict(zip(cmp['feature'], cmp['rank_allsites'].fillna(9999)))
    rank_all_to_20  = dict(zip(
        rna_all['feature'],
        rna_all.reset_index()['index'] + 1
    ))

    # ── Figure 1: rna_20cm bar ─────────────────────────────────────────────────
    print('  Fig 1: rna_20cm bar')
    fig1 = bar_figure(
        rna_20.head(20),
        title='Top 20 genes — rna_20cm (At-20-cm only)',
        subtitle='Site-controlled · disease biology signal · 828 patients',
        auc_str='0.824 ± 0.019',
    )
    path1 = f'{PLOTS_DIR}/shap_rna_20cm_bar.pdf'
    fig1.savefig(path1, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig1)
    print(f'    saved {path1}')

    # ── Figure 2: rna_patmean bar ──────────────────────────────────────────────
    print('  Fig 2: rna_patmean bar')
    fig2 = bar_figure(
        rna_all.head(20),
        title='Top 20 genes — rna_patmean (all colon sites)',
        subtitle='⚠ Includes site-identity confound (HOX genes) · 997 patients',
        auc_str='0.920 ± 0.007',
    )
    path2 = f'{PLOTS_DIR}/shap_rna_allsites_bar.pdf'
    fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig2)
    print(f'    saved {path2}')

    # ── Figure 3: dumbbell ─────────────────────────────────────────────────────
    print('  Fig 3: rank-shift dumbbell')
    fig3 = dumbbell_figure(cmp, n=30)
    path3 = f'{PLOTS_DIR}/shap_rank_comparison.pdf'
    fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig3)
    print(f'    saved {path3}')

    # ── Figure 4: fusion split ─────────────────────────────────────────────────
    print('  Fig 4: fusion split')
    # compute modality fractions from saved file
    img_shap = df_fus[df_fus['feature'].str.startswith('img_')]['mean_abs_shap'].sum()
    rna_shap = df_fus[~df_fus['feature'].str.startswith('img_')]['mean_abs_shap'].sum()
    total = img_shap + rna_shap
    fig4 = fusion_figure(df_fus, img_shap/total, rna_shap/total)
    path4 = f'{PLOTS_DIR}/shap_fusion_split.pdf'
    fig4.savefig(path4, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig4)
    print(f'    saved {path4}')

    # ── Figure 5: comparison panel ─────────────────────────────────────────────
    print('  Fig 5: comparison panel')
    fig5 = comparison_panel(rna_20, rna_all)
    path5 = f'{PLOTS_DIR}/shap_panel.pdf'
    fig5.savefig(path5, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig5)
    print(f'    saved {path5}')

    print(f'\nAll SHAP plots saved to {PLOTS_DIR}/')


if __name__ == '__main__':
    main()
