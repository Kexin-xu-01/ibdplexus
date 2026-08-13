"""
Pathway enrichment analysis for CD-up vs UC-up SHAP genes — visit-level cohort.

For each arm (rna_visit, concat_raw_visit) we split the top-ranked SHAP genes
by expression direction:
  UC-up  : mean expression higher in UC than CD in the at-20-cm cohort
  CD-up  : mean expression higher in CD than UC

Then run over-representation analysis (ORA) against GO:BP, KEGG, Reactome,
and WikiPathways using g:Profiler (custom background = all VST genes).

A second pass uses gseapy / Enrichr as a cross-check.

Outputs (under OUT_DIR)
-----------------------
  data/  enrichment_{arm}_{direction}_{tool}.csv
  plots/ enrichment_dotplot_{arm}.pdf   — combined UC vs CD dot-plot per arm
         enrichment_barplot_{arm}_{direction}.pdf
  enrichment_summary.json
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
SHAP_DIR    = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '10_11_shap_analysis/data_visit')
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '12_pathway_enrichment')
VST_GCT     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct')
MAPPING_CSV = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
AT20        = {'at 20 cm', 'At 20 cm'}

TOP_N = 200   # top SHAP-ranked genes to analyse per arm

# ── style ─────────────────────────────────────────────────────────────────────
UC_COLOR = '#c94040'   # red
CD_COLOR = '#2a78d6'   # blue
SURF     = '#fcfcfb'
INK      = '#0b0b0b'
INK2     = '#52514e'
MUTED    = '#898781'
GRID     = '#e1e0d9'
BASE     = '#c3c2b7'

DB_COLOR = {
    'GO:BP':  '#4e79a7',
    'KEGG':   '#f28e2b',
    'REAC':   '#76b7b2',
    'WP':     '#59a14f',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Build gene symbol maps
# ══════════════════════════════════════════════════════════════════════════════

def load_gct_symbol_map():
    """Return (ensg_to_symbol dict, all_symbols list) from GCT header."""
    gmap = pd.read_csv(VST_GCT, sep='\t', skiprows=2,
                       usecols=['Name', 'Description'])
    ensg_to_sym = dict(zip(gmap['Name'], gmap['Description']))
    # drop duplicated symbols (keep first by ENSG order)
    seen = set(); all_syms = []
    for sym in gmap['Description']:
        if sym not in seen:
            seen.add(sym); all_syms.append(sym)
    return ensg_to_sym, all_syms


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Expression-direction scan
# ══════════════════════════════════════════════════════════════════════════════

def _norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return 'CD'
        if 'Ulcerative' in d: return 'UC'
    return None


def compute_directions(ensg_ids):
    """Return dict {ensg_id: 'UC' or 'CD'} using mean expression in the at-20-cm cohort."""
    print(f'  Computing expression directions for {len(ensg_ids)} genes ...')

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
    sample_label = dict(zip(rna['SampleID'], rna['_dx']))

    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header = f.readline().rstrip('\n').split('\t')
    col_info = [(i, sample_label[sid])
                for i, sid in enumerate(header)
                if sid in sample_label]

    needed = set(ensg_ids)
    uc_s = {g: 0.0 for g in needed}; uc_n = {g: 0 for g in needed}
    cd_s = {g: 0.0 for g in needed}; cd_n = {g: 0 for g in needed}

    with open(VST_GCT) as f:
        f.readline(); f.readline(); f.readline()
        found = 0
        for line in f:
            gene = line[:line.index('\t')]
            if gene not in needed:
                continue
            parts = line.rstrip('\n').split('\t')
            for ci, dx in col_info:
                v = float(parts[ci])
                if dx == 'UC': uc_s[gene] += v; uc_n[gene] += 1
                else:          cd_s[gene] += v; cd_n[gene] += 1
            found += 1
            if found == len(needed):
                break

    directions = {}
    for g in needed:
        if uc_n[g] > 0 and cd_n[g] > 0:
            directions[g] = 'UC' if (uc_s[g]/uc_n[g]) > (cd_s[g]/cd_n[g]) else 'CD'
    print(f'    {len(directions)}/{len(needed)} resolved')
    return directions


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Load SHAP data and build gene lists
# ══════════════════════════════════════════════════════════════════════════════

def build_gene_lists(arm_csv, arm_name, top_n, ensg_to_sym, directions):
    """
    Returns dict with keys 'UC' and 'CD', each a list of gene symbols,
    and a DataFrame with ensg_id, symbol, rank, shap, direction.
    """
    df = pd.read_csv(arm_csv)
    # filter to RNA features only (concat arm also has img_ features)
    df = df[~df['feature'].str.startswith('img_')].head(top_n).copy()
    df['symbol']    = df['feature'].map(lambda x: ensg_to_sym.get(x, x))
    df['direction'] = df['feature'].map(lambda x: directions.get(x, None))
    df = df.dropna(subset=['direction'])

    gene_lists = {
        'UC': df[df['direction'] == 'UC']['symbol'].tolist(),
        'CD': df[df['direction'] == 'CD']['symbol'].tolist(),
    }
    print(f'  {arm_name}: UC-up={len(gene_lists["UC"])}  CD-up={len(gene_lists["CD"])}')
    return gene_lists, df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Enrichment runners
# ══════════════════════════════════════════════════════════════════════════════

def run_gprofiler(gene_symbols, background, sources=None):
    """Run g:Profiler ORA. Returns DataFrame or None on failure."""
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
        if sources is None:
            sources = ['GO:BP', 'KEGG', 'REAC', 'WP']
        res = gp.profile(
            organism='hsapiens',
            query=gene_symbols,
            background=background,
            sources=sources,
            significance_threshold_method='fdr',
            user_threshold=0.05,
            no_evidences=False,
        )
        return res
    except Exception as e:
        print(f'    gprofiler error: {e}')
        return None


def run_enrichr(gene_symbols, label, outdir):
    """Run Enrichr via gseapy. Returns combined DataFrame or None on failure."""
    try:
        import gseapy as gp
        dbs = ['GO_Biological_Process_2023', 'KEGG_2021_Human',
               'Reactome_2022', 'WikiPathway_2023_Human']
        enr = gp.enrichr(
            gene_list=gene_symbols,
            gene_sets=dbs,
            organism='human',
            outdir=os.path.join(outdir, label),
            verbose=False,
        )
        return enr.results
    except Exception as e:
        print(f'    enrichr error: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': BASE,   'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True,        'grid.color': GRID,
        'grid.linewidth': 0.4,    'axes.axisbelow': True,
        'xtick.color': MUTED,     'ytick.color': INK2,
        'xtick.labelsize': 8,     'ytick.labelsize': 9,
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'text.color': INK,
    })


def _gprofiler_bar(ax, df, direction, n_top=15, db_filter=None):
    """Draw a horizontal bar of top enriched terms on ax."""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No significant terms', ha='center', va='center',
                transform=ax.transAxes, color=MUTED)
        return

    src_map = {'GO:BP': 'GO:BP', 'KEGG': 'KEGG', 'REAC': 'REAC', 'WP': 'WP'}
    if db_filter:
        df = df[df['source'].isin(db_filter)]

    df = df.copy()
    df['-log10_fdr'] = -np.log10(df['p_value'].clip(1e-300))
    df = df.nsmallest(n_top, 'p_value')
    df = df[::-1]   # bottom = most significant

    labels = df['name'].str[:65].tolist()
    values = df['-log10_fdr'].tolist()
    colors = [DB_COLOR.get(s, '#aaaaaa') for s in df['source']]

    y = np.arange(len(labels))
    ax.barh(y, values, height=0.55, color=colors, edgecolor=SURF, linewidth=0.5)

    vmax = max(values) if values else 1
    for i, (val, row) in enumerate(zip(values, df.itertuples())):
        n_genes = getattr(row, 'intersection_size', '')
        ax.text(val + vmax * 0.01, i, f'p={row.p_value:.1e}  n={n_genes}',
                va='center', ha='left', fontsize=6.5, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel('−log₁₀(FDR p-value)', fontsize=8, color=INK2)
    col = UC_COLOR if direction == 'UC' else CD_COLOR
    dx_label = 'UC-upregulated' if direction == 'UC' else 'CD-upregulated'
    ax.set_title(dx_label, fontsize=10, fontweight='bold', color=col, loc='left')
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)

    handles = [mpatches.Patch(color=DB_COLOR[s], label=s)
               for s in DB_COLOR if s in df['source'].values]
    ax.legend(handles=handles, fontsize=7, frameon=True, framealpha=0.9,
              edgecolor=GRID, loc='lower right')


def dotplot(df_uc, df_cd, arm_name, out_path, top_n=15):
    """Combined dotplot: UC pathways on top, CD on bottom."""
    setup_style()

    def prep(df, direction, n):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d['-log10_fdr'] = -np.log10(d['p_value'].clip(1e-300))
        d = d.nsmallest(n, 'p_value')
        d['direction'] = direction
        return d

    top_uc = prep(df_uc, 'UC', top_n)
    top_cd = prep(df_cd, 'CD', top_n)
    combined = pd.concat([top_uc, top_cd], ignore_index=True)
    if combined.empty:
        print(f'  dotplot: no data for {arm_name}')
        return

    combined = combined.sort_values(['-log10_fdr'], ascending=True).reset_index(drop=True)
    n_rows = len(combined)
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.32 + 1.5)))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    y        = np.arange(n_rows)
    lfc      = combined['-log10_fdr'].tolist()
    colors   = [UC_COLOR if d == 'UC' else CD_COLOR for d in combined['direction']]
    sizes    = np.clip(combined['-log10_fdr'] * 15, 20, 180)
    ax.scatter(lfc, y, s=sizes, c=colors,
               alpha=0.85, zorder=3, edgecolors='white', linewidths=0.5)

    for i, (_, row) in enumerate(combined.iterrows()):
        n_g = row.get('intersection_size', '')
        ax.text(lfc[i] + 0.12, i, f'n={n_g}  {row["source"]}',
                va='center', ha='left', fontsize=6.5, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels(combined['name'].str[:70], fontsize=7.5, color=INK)
    ax.set_xlabel('−log₁₀(FDR p-value)', fontsize=9, color=INK2)
    ax.axvline(1.3, color=MUTED, lw=0.5, ls='--')
    ax.text(1.3, n_rows - 0.5, 'FDR 0.05', fontsize=6.5, color=MUTED, ha='left')
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)

    handles = [
        mpatches.Patch(color=UC_COLOR, label='UC-upregulated'),
        mpatches.Patch(color=CD_COLOR, label='CD-upregulated'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9,
              edgecolor=GRID, loc='lower right')
    ax.set_title(f'Pathway enrichment — {arm_name}\n(top {top_n} per direction, g:Profiler)',
                 fontsize=10, fontweight='bold', color=INK, loc='left', pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(f'  saved {out_path}')


def barplot_pair(df_uc, df_cd, arm_name, out_path, top_n=20):
    """Side-by-side horizontal bars: left = CD-up, right = UC-up."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)

    _gprofiler_bar(axes[0], df_cd, 'CD', n_top=top_n)
    _gprofiler_bar(axes[1], df_uc, 'UC', n_top=top_n)

    fig.suptitle(f'Top enriched pathways — {arm_name}  (g:Profiler, FDR < 0.05)',
                 fontsize=12, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(f'  saved {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    data_dir  = os.path.join(OUT_DIR, 'data')
    plots_dir = os.path.join(OUT_DIR, 'plots')
    os.makedirs(data_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print('Loading gene symbol map ...')
    ensg_to_sym, all_background_syms = load_gct_symbol_map()
    print(f'  background: {len(all_background_syms)} unique gene symbols')

    # ── arms to analyse ───────────────────────────────────────────────────────
    arms = [
        ('rna_visit',        os.path.join(SHAP_DIR, 'shap_rna_visit_top500.csv')),
        ('concat_raw_visit', os.path.join(SHAP_DIR, 'shap_concat_raw_visit_top500.csv')),
    ]

    summary = {}

    for arm_name, arm_csv in arms:
        print(f'\n{"="*60}')
        print(f'ARM: {arm_name}')
        print(f'{"="*60}')

        # ── 1. compute expression directions for top TOP_N genes ──────────────
        df_arm = pd.read_csv(arm_csv)
        df_arm = df_arm[~df_arm['feature'].str.startswith('img_')].head(TOP_N)
        directions = compute_directions(df_arm['feature'].tolist())

        # ── 2. build gene lists ───────────────────────────────────────────────
        gene_lists, df_ann = build_gene_lists(
            arm_csv, arm_name, TOP_N, ensg_to_sym, directions)

        df_ann.to_csv(os.path.join(data_dir, f'{arm_name}_genes_annotated.csv'),
                      index=False, float_format='%.6f')

        # ── 3. g:Profiler ─────────────────────────────────────────────────────
        gp_results = {}
        for dx in ['UC', 'CD']:
            genes = gene_lists[dx]
            cached_csv = os.path.join(data_dir, f'{arm_name}_{dx}_gprofiler.csv')
            if os.path.exists(cached_csv):
                print(f'  g:Profiler {dx}-up: loading cached {cached_csv}')
                gp_results[dx] = pd.read_csv(cached_csv)
                continue
            if len(genes) < 5:
                print(f'  {dx}: too few genes ({len(genes)}), skipping')
                gp_results[dx] = None
                continue
            print(f'\n  g:Profiler ORA — {arm_name} {dx}-up ({len(genes)} genes) ...')
            res = run_gprofiler(genes, all_background_syms)
            if res is not None and not res.empty:
                res['arm']       = arm_name
                res['direction'] = dx
                res.to_csv(cached_csv, index=False)
                sig = res[res['p_value'] < 0.05]
                print(f'    {len(sig)} significant terms (FDR<0.05)')
                for src in ['GO:BP', 'KEGG', 'REAC', 'WP']:
                    n = len(sig[sig['source'] == src])
                    if n:
                        print(f'      {src}: {n}')
                gp_results[dx] = res
            else:
                print('    no results returned')
                gp_results[dx] = None
            time.sleep(1)   # be polite to the API

        # ── 4. gseapy / Enrichr ───────────────────────────────────────────────
        for dx in ['UC', 'CD']:
            genes = gene_lists[dx]
            if len(genes) < 5:
                continue
            print(f'\n  Enrichr — {arm_name} {dx}-up ...')
            enr_out = os.path.join(data_dir, f'enrichr_{arm_name}_{dx}')
            res_enr = run_enrichr(genes, f'{arm_name}_{dx}', data_dir)
            if res_enr is not None and not res_enr.empty:
                res_enr['arm'] = arm_name; res_enr['direction'] = dx
                res_enr.to_csv(os.path.join(data_dir,
                                f'{arm_name}_{dx}_enrichr.csv'), index=False)
                sig = res_enr[res_enr['Adjusted P-value'] < 0.05]
                print(f'    {len(sig)} significant terms (adj.p<0.05)')
            time.sleep(1)

        # ── 5. plots ──────────────────────────────────────────────────────────
        barplot_pair(
            gp_results.get('UC'), gp_results.get('CD'),
            arm_name,
            os.path.join(plots_dir, f'{arm_name}_barplot.pdf'))
        dotplot(
            gp_results.get('UC'), gp_results.get('CD'),
            arm_name,
            os.path.join(plots_dir, f'{arm_name}_dotplot.pdf'))

        # ── 6. summary ────────────────────────────────────────────────────────
        summary[arm_name] = {}
        for dx in ['UC', 'CD']:
            r = gp_results.get(dx)
            if r is not None and not r.empty:
                sig = r[r['p_value'] < 0.05]
                top5 = sig.nsmallest(5, 'p_value')[
                    ['source', 'name', 'p_value', 'intersection_size']
                ].to_dict('records')
            else:
                top5 = []
            summary[arm_name][dx] = {
                'n_genes':    len(gene_lists[dx]),
                'n_sig_terms': len(sig) if r is not None and not r.empty else 0,
                'top5':        top5,
            }

    with open(os.path.join(OUT_DIR, 'enrichment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n\nAll outputs saved to {OUT_DIR}/')
    print('\n=== SUMMARY ===')
    for arm, arm_data in summary.items():
        print(f'\n── {arm} ──')
        for dx, d in arm_data.items():
            print(f'  {dx}-up  ({d["n_genes"]} genes)  →  {d["n_sig_terms"]} sig terms')
            for t in d['top5']:
                print(f'    [{t["source"]}] {t["name"][:70]}  p={t["p_value"]:.2e}  n={t["intersection_size"]}')


if __name__ == '__main__':
    main()
