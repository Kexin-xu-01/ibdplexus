"""
UMAP of the RNA-seq TPM matrix — coloured by the same metadata as umap_rna.py.

Leakage-free counterpart to umap_rna.py (which used the CombatSeq + VST matrix).
TPM is per-sample: no cross-sample statistics, no batch correction applied.

Steps
-----
  1. Load full TPM GCT (62,703 genes × 3,304 samples)
  2. log2(TPM + 1) per sample
  3. Filter to expressed genes: mean log2(TPM+1) > 0.5 across all samples
     (no train/test split here — UMAP is visualisation only)
  4. PCA → 50 components
  5. UMAP on PCA coordinates
  6. Join metadata: diagnosis, biopsy_location, macroscopic_appearance,
                    disease_activity (MAYO6_CATEGORY), disease_location,
                    sequencing_batch (first / later),
                    collection_year (2017–2022)
  7. Save per-variable static PNGs + one interactive HTML

Outputs (under OUT_DIR)
-----------------------
  umap_rna_tpm_coords.npz
  umap_rna_tpm_<var>.png
  umap_rna_tpm.html
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
TPM_GCT     = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF2048892_combined_TPM_matrix_with_header.gct')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
OUT_DIR     = '/home/jovyan/kgbk271-ibd-volume/results/rna/tpm'
COORDS_NPZ  = os.path.join(OUT_DIR, 'umap_rna_tpm_coords.npz')

N_PCA          = 50
UMAP_NEIGHBORS = 30
UMAP_DIST      = 0.25
RANDOM_STATE   = 42
MIN_MEAN_LOG2TPM = 0.5  # gene expression filter (all samples — visualisation only)

# ── palettes (matching umap_rna.py / prism2 umap_reports.py) ──────────────────
PALETTES = {
    'diagnosis': {
        "Crohn's Disease":    '#4C72B0',
        'Ulcerative Colitis': '#DD8452',
        'IBD Unclassified':   '#55A868',
        'Unknown':            '#BBBBBB',
    },
    'biopsy_location': {
        'At 20 cm':          '#4C72B0',
        'Ileum':             '#DD8452',
        'Cecum':             '#55A868',
        'Rectum':            '#9467BD',
        'Ascending Colon':   '#8C564B',
        'Descending Colon':  '#E377C2',
        'Transverse Colon':  '#F7B731',
        'Sigmoid Colon':     '#17BECF',
        'Other':             '#7F7F7F',
        'Unknown':           '#BBBBBB',
    },
    'macroscopic_appearance': {
        'Normal':                '#3A9E64',
        'Possible inflammation': '#F0C040',
        'Erosions or Ulcers':    '#C0392B',
        'Unknown':               '#BBBBBB',
    },
    'disease_activity': {
        'Remission': '#3A9E64',
        'Mild':      '#F0C040',
        'Moderate':  '#E07B2A',
        'Severe':    '#C0392B',
        'Unknown':   '#BBBBBB',
    },
    'disease_location': {
        'Ileum (L1)':           '#4C72B0',
        'Colon (L2)':           '#DD8452',
        'Ileocolon (L3)':       '#55A868',
        'Upper GI (L4)':        '#9467BD',
        'Proctitis (E1)':       '#8C564B',
        'Left-sided UC (E2)':   '#E377C2',
        'Pancolitis (E3)':      '#F7B731',
        'Unknown':              '#BBBBBB',
    },
    'sequencing_batch': {
        'first':   '#4C72B0',
        'later':   '#DD8452',
        'Unknown': '#BBBBBB',
    },
    'collection_year': {
        '2017':    '#fee5d9',
        '2018':    '#fcbba1',
        '2019':    '#fc9272',
        '2020':    '#fb6a4a',
        '2021':    '#de2d26',
        '2022':    '#a50f15',
        'Unknown': '#BBBBBB',
    },
}

TITLES = {
    'diagnosis':              'Diagnosis',
    'biopsy_location':        'Biopsy Location',
    'macroscopic_appearance': 'Macroscopic Appearance',
    'disease_activity':       'Disease Activity (Mayo 6)',
    'disease_location':       'Disease Location / Extent',
    'sequencing_batch':       'Sequencing Batch',
    'collection_year':        'Sample Collection Year',
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_tpm(path):
    """Load full TPM GCT, apply log2(TPM+1), filter expressed genes."""
    print('  Reading GCT header ...')
    with open(path) as f:
        f.readline()
        dims = f.readline().strip().split('\t')
        n_genes, n_samples = int(dims[0]), int(dims[1])
        header_cols = f.readline().strip().split('\t')

    sample_ids = header_cols[2:]
    print(f'  {n_genes} genes × {n_samples} samples')

    print('  Loading expression matrix ...')
    gct = pd.read_csv(
        path, sep='\t', skiprows=2, header=0, index_col=0,
        dtype={c: (str if c in ('Name', 'Description') else np.float32)
               for c in header_cols})
    gct = gct.drop(columns=['Description'], errors='ignore')

    print('  Applying log2(TPM+1) ...')
    X = np.log2(gct.values.T.astype(np.float32) + 1.0)  # (n_samples, n_genes)

    print('  Filtering expressed genes ...')
    means = X.mean(axis=0)
    mask  = means > MIN_MEAN_LOG2TPM
    X_filt = X[:, mask]
    print(f'  Genes retained: {mask.sum()} / {n_genes}  '
          f'(mean log2(TPM+1) > {MIN_MEAN_LOG2TPM})')

    return sample_ids, X_filt


def load_metadata(sample_ids):
    mapping = pd.read_csv(MAPPING_CSV)
    smeta   = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    merged  = mapping.merge(
        smeta[['SampleID', 'diagnosis', 'macroscopic_appearance',
               'MAYO6_CATEGORY', 'disease_location', 'batch', 'Sample QC']].rename(
            columns={'macroscopic_appearance': 'macro_smeta'}),
        on='SampleID', how='left')
    merged['macroscopic_appearance'] = merged['macro_smeta'].where(
        merged['macro_smeta'].notna(),
        merged.get('macroscopic_appearance', pd.NA))

    # extract collection year from sample_collected_date (format: 08-JUN-2021)
    merged['_year'] = pd.to_datetime(
        merged['sample_collected_date'], format='%d-%b-%Y', errors='coerce'
    ).dt.year

    def norm_dx(d):
        if isinstance(d, str):
            if 'Crohn' in d:           return "Crohn's Disease"
            if 'Ulcerative' in d:      return 'Ulcerative Colitis'
            if 'Unclassified' in d:    return 'IBD Unclassified'
        return 'Unknown'

    def norm_macro(v):
        known = {'Normal', 'Possible inflammation', 'Erosions or Ulcers'}
        return v.strip() if isinstance(v, str) and v.strip() in known else 'Unknown'

    def norm_mayo(v):
        known = {'Remission', 'Mild', 'Moderate', 'Severe'}
        return v.strip() if isinstance(v, str) and v.strip() in known else 'Unknown'

    def norm_loc(v):
        if not isinstance(v, str): return 'Unknown'
        for k, lab in [('L1','Ileum (L1)'), ('L2','Colon (L2)'), ('L3','Ileocolon (L3)'),
                       ('L4','Upper GI (L4)'), ('E1','Proctitis (E1)'),
                       ('E2','Left-sided UC (E2)'), ('E3','Pancolitis (E3)')]:
            if k in v: return lab
        return 'Unknown'

    def norm_bio(v):
        known = {'At 20 cm', 'Ileum', 'Cecum', 'Rectum',
                 'Ascending Colon', 'Descending Colon',
                 'Transverse Colon', 'Sigmoid Colon', 'Other'}
        return v.strip() if isinstance(v, str) and v.strip() in known else 'Other'

    def norm_batch(v):
        return v.strip() if isinstance(v, str) and v.strip() in ('first', 'later') else 'Unknown'

    merged['diagnosis_norm']     = merged['diagnosis'].map(norm_dx)
    merged['macro_norm']         = merged['macroscopic_appearance'].map(norm_macro)
    merged['disease_act_norm']   = merged['MAYO6_CATEGORY'].map(norm_mayo)
    merged['disease_loc_norm']   = merged['disease_location'].map(norm_loc)
    merged['biopsy_norm']        = merged['characteristics_bio_material'].map(norm_bio)
    merged['batch_norm']         = merged['batch'].map(norm_batch)
    merged['year_norm']          = merged['_year'].apply(
        lambda y: str(int(y)) if pd.notna(y) else 'Unknown')

    lookup = merged.drop_duplicates('SampleID').set_index('SampleID')
    rows = []
    for sid in sample_ids:
        if sid in lookup.index:
            r = lookup.loc[sid]
            rows.append({
                'sample_id':              sid,
                'diagnosis':              r['diagnosis_norm'],
                'biopsy_location':        r['biopsy_norm'],
                'macroscopic_appearance': r['macro_norm'],
                'disease_activity':       r['disease_act_norm'],
                'disease_location':       r['disease_loc_norm'],
                'sequencing_batch':       r['batch_norm'],
                'collection_year':        r['year_norm'],
            })
        else:
            rows.append({'sample_id': sid, 'diagnosis': 'Unknown',
                         'biopsy_location': 'Unknown',
                         'macroscopic_appearance': 'Unknown',
                         'disease_activity': 'Unknown',
                         'disease_location': 'Unknown',
                         'sequencing_batch': 'Unknown',
                         'collection_year': 'Unknown'})

    df = pd.DataFrame(rows).set_index('sample_id')
    for col in PALETTES:
        df[col] = df[col].fillna('Unknown')
    return df


# ── UMAP computation ───────────────────────────────────────────────────────────

def run_umap(X, sample_ids):
    if os.path.exists(COORDS_NPZ):
        print('  Loading cached UMAP coordinates ...')
        d = np.load(COORDS_NPZ, allow_pickle=True)
        if list(d['sample_ids']) == sample_ids:
            return d['xy']
        print('  Cache mismatch — recomputing')

    print(f'  PCA → {N_PCA} components ...')
    pca  = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    Xpca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.cumsum()[N_PCA - 1]
    print(f'    variance explained: {var_exp*100:.1f}%')

    print(f'  UMAP (n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_DIST}) ...')
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_DIST,
        metric='euclidean',
        random_state=RANDOM_STATE,
        low_memory=False,
        verbose=True,
    )
    xy = reducer.fit_transform(Xpca)
    np.savez_compressed(COORDS_NPZ,
                        xy=xy.astype(np.float32),
                        sample_ids=np.array(sample_ids))
    print(f'  Coords saved → {COORDS_NPZ}')
    return xy


# ── plotting ───────────────────────────────────────────────────────────────────

def _scatter_ax(ax, df, xy, col, palette, title, n_total):
    ax.set_facecolor('#FAFAFA')
    order = [k for k in palette if k != 'Unknown'] + ['Unknown']
    for cat in order:
        mask = (df[col] == cat).values
        if not mask.any(): continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=palette[cat], s=4, alpha=0.65, linewidths=0,
                   rasterized=True, label=cat)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    present = [c for c in order if (df[col] == c).any()]
    handles = [mpatches.Patch(color=palette[c], label=c) for c in present]
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              framealpha=0.85, markerscale=1.5, handlelength=1.0)
    ax.text(1, -0.06, f'n = {n_total}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color='#888888')


def save_static_plots(df, xy, out_dir, dpi=200):
    for col, palette in PALETTES.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('white')
        _scatter_ax(ax, df, xy, col, palette, TITLES[col], len(df))
        stem = f'umap_rna_tpm_{col}'
        fig.savefig(os.path.join(out_dir, f'{stem}.png'), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'    {stem}.png')


def make_html(df, xy, out_path, n_genes_retained):
    all_traces, group_spans = [], []
    for col, palette in PALETTES.items():
        order  = [k for k in palette if k != 'Unknown'] + ['Unknown']
        traces = []
        for cat in order:
            mask = (df[col] == cat).values
            if not mask.any(): continue
            sub_xy = xy[mask]; sub_df = df[mask]
            traces.append(go.Scattergl(
                x=sub_xy[:, 0], y=sub_xy[:, 1],
                mode='markers', name=cat,
                marker=dict(color=palette[cat], size=4, opacity=0.70,
                            line=dict(width=0.3, color='rgba(255,255,255,0.2)')),
                text=sub_df.apply(lambda r: (
                    f'<b>{r.name}</b><br>'
                    f'Diagnosis: {r["diagnosis"]}<br>'
                    f'Biopsy: {r["biopsy_location"]}<br>'
                    f'Macro: {r["macroscopic_appearance"]}<br>'
                    f'Activity: {r["disease_activity"]}<br>'
                    f'Location: {r["disease_location"]}<br>'
                    f'Batch: {r["sequencing_batch"]}<br>'
                    f'Year: {r["collection_year"]}'
                ), axis=1),
                hovertemplate='%{text}<extra></extra>',
                legendgroup=col, visible=False,
            ))
        group_spans.append((col, len(traces)))
        all_traces.extend(traces)

    fig = go.Figure()
    for t in all_traces:
        fig.add_trace(t)
    for i in range(group_spans[0][1]):
        fig.data[i].visible = True

    flat_len = len(all_traces)
    buttons, cum = [], 0
    for col, n_tr in group_spans:
        vis = [False] * flat_len
        for j in range(cum, cum + n_tr): vis[j] = True
        buttons.append(dict(
            label=TITLES[col], method='update',
            args=[{'visible': vis},
                  {'title': f'RNA-seq TPM UMAP — {TITLES[col]}'}]))
        cum += n_tr

    fig.update_layout(
        title=dict(text=f'RNA-seq TPM UMAP — {TITLES["diagnosis"]}',
                   font=dict(size=16)),
        updatemenus=[dict(buttons=buttons, direction='down',
                          x=0.01, xanchor='left', y=1.13, yanchor='top',
                          showactive=True, bgcolor='#F0F0F0', bordercolor='#CCCCCC')],
        annotations=[dict(text='Colour by:', x=0.01, xref='paper',
                          y=1.17, yref='paper', showarrow=False,
                          font=dict(size=12))],
        xaxis=dict(title='UMAP 1', showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title='UMAP 2', showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(itemsizing='constant', font=dict(size=11)),
        plot_bgcolor='#FAFAFA', paper_bgcolor='white',
        width=1000, height=700, margin=dict(t=110, r=20, b=40, l=60),
    )
    fig.add_annotation(
        text=(f'{len(df)} RNA-seq samples · log2(TPM+1), no batch correction · '
              f'{n_genes_retained} genes · PCA({N_PCA}) → UMAP'),
        x=1, xref='paper', y=-0.05, yref='paper',
        showarrow=False, font=dict(size=10, color='#888888'), xanchor='right')
    fig.write_html(str(out_path), include_plotlyjs='cdn')
    print(f'  HTML → {out_path}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=== RNA-seq TPM UMAP ===')
    print('Loading TPM GCT ...')
    sample_ids, X = load_tpm(TPM_GCT)
    n_genes_retained = X.shape[1]
    print(f'  Matrix after filter: {X.shape}')

    print('Loading metadata ...')
    meta_df = load_metadata(sample_ids)
    for col in PALETTES:
        n_known = (meta_df[col] != 'Unknown').sum()
        print(f'  {col}: {n_known}/{len(meta_df)} labelled')

    print('Computing UMAP ...')
    xy = run_umap(X, sample_ids)
    print(f'  UMAP shape: {xy.shape}')

    print('Saving static plots ...')
    save_static_plots(meta_df, xy, OUT_DIR)

    print('Building interactive HTML ...')
    make_html(meta_df, xy,
              os.path.join(OUT_DIR, 'umap_rna_tpm.html'),
              n_genes_retained)

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
