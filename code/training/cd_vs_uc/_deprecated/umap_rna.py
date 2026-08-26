"""
UMAP of the RNA-seq VST matrix — coloured by the same metadata as the PRISM2 UMAP.

Steps
-----
  1. Load full VST GCT (3,289 samples × 17,963 genes)
  2. PCA → 50 components
  3. UMAP on PCA coordinates
  4. Join metadata: diagnosis, biopsy_location, macroscopic_appearance,
                    disease_activity (MAYO6_CATEGORY), disease_location
  5. Save per-variable static PNGs + one interactive HTML

Outputs (under OUT_DIR)
-----------------------
  umap_rna_coords.npz
  umap_rna_<var>.png
  umap_rna.html

Usage
-----
    python umap_rna.py
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
VST_GCT     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
               'alltissues_all3releases_header.gct')
MAPPING_CSV = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/'
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
OUT_DIR     = '/home/jovyan/kgbk271-ibd-volume/results/rna/umap'
COORDS_NPZ  = os.path.join(OUT_DIR, 'umap_rna_coords.npz')

N_PCA      = 50
UMAP_NEIGHBORS = 30
UMAP_DIST      = 0.25
RANDOM_STATE   = 42

# ── palettes (mirroring prism2 umap_reports.py) ────────────────────────────────
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
}

TITLES = {
    'diagnosis':             'Diagnosis',
    'biopsy_location':       'Biopsy Location',
    'macroscopic_appearance': 'Macroscopic Appearance',
    'disease_activity':      'Disease Activity (Mayo 6)',
    'disease_location':      'Disease Location / Extent',
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_gct(path):
    """Load the full VST GCT matrix. Returns (sample_ids, X: n_samples × n_genes)."""
    print('  Reading GCT header ...')
    with open(path) as f:
        f.readline()
        dims = f.readline().strip().split('\t')
        n_genes, n_samples = int(dims[0]), int(dims[1])
        header_cols = f.readline().strip().split('\t')

    sample_cols = header_cols[2:]   # skip Name, Description
    print(f'  {n_genes} genes × {n_samples} samples')

    print('  Loading expression matrix (may take ~30 s) ...')
    gct = pd.read_csv(path, sep='\t', skiprows=2, header=0,
                      index_col=0,
                      dtype={c: (str if c in ('Name', 'Description') else np.float32)
                             for c in header_cols})
    gct = gct.drop(columns=['Description'], errors='ignore')
    X = gct.T.values.astype(np.float32)   # (n_samples, n_genes)
    return sample_cols, X


def load_metadata(sample_ids):
    """Build a per-sample metadata DataFrame aligned to sample_ids."""
    mapping = pd.read_csv(MAPPING_CSV)
    smeta   = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})

    merged = mapping.merge(
        smeta[['SampleID', 'diagnosis', 'macroscopic_appearance',
               'MAYO6_CATEGORY', 'disease_location', 'Sample QC']].rename(
            columns={'macroscopic_appearance': 'macro_smeta'}),
        on='SampleID', how='left')
    # prefer SAMPLE_META macroscopic_appearance; fall back to MAPPING_CSV version
    merged['macroscopic_appearance'] = merged['macro_smeta'].where(
        merged['macro_smeta'].notna(), merged.get('macroscopic_appearance', pd.NA))

    # normalise diagnosis
    def norm_dx(d):
        if isinstance(d, str):
            if 'Crohn' in d:      return "Crohn's Disease"
            if 'Ulcerative' in d: return 'Ulcerative Colitis'
            if 'Unclassified' in d or 'IBDU' in d: return 'IBD Unclassified'
        return 'Unknown'

    merged['diagnosis_norm'] = merged['diagnosis'].map(norm_dx)

    # normalise macroscopic appearance
    macro_map = {
        'Normal':                'Normal',
        'Possible inflammation': 'Possible inflammation',
        'Erosions or Ulcers':    'Erosions or Ulcers',
    }
    def norm_macro(v):
        if isinstance(v, str):
            return macro_map.get(v.strip(), 'Unknown')
        return 'Unknown'
    merged['macro_norm'] = merged['macroscopic_appearance'].map(norm_macro)

    # disease activity from MAYO6
    def norm_mayo(v):
        if isinstance(v, str):
            if v.strip() == 'Remission': return 'Remission'
            if v.strip() == 'Mild':      return 'Mild'
            if v.strip() == 'Moderate':  return 'Moderate'
            if v.strip() == 'Severe':    return 'Severe'
        return 'Unknown'
    merged['disease_activity_norm'] = merged['MAYO6_CATEGORY'].map(norm_mayo)

    # disease location
    def norm_loc(v):
        if not isinstance(v, str):
            return 'Unknown'
        v = v.strip()
        mapping_loc = {
            'L1':  'Ileum (L1)',
            'L2':  'Colon (L2)',
            'L3':  'Ileocolon (L3)',
            'L4':  'Upper GI (L4)',
            'E1':  'Proctitis (E1)',
            'E2':  'Left-sided UC (E2)',
            'E3':  'Pancolitis (E3)',
        }
        for k, lab in mapping_loc.items():
            if k in v:
                return lab
        return 'Unknown'
    merged['disease_loc_norm'] = merged['disease_location'].map(norm_loc)

    # biopsy location (characteristics_bio_material)
    def norm_bio(v):
        if not isinstance(v, str):
            return 'Unknown'
        v = v.strip()
        known = {'At 20 cm', 'Ileum', 'Cecum', 'Rectum',
                 'Ascending Colon', 'Descending Colon',
                 'Transverse Colon', 'Sigmoid Colon', 'Other'}
        return v if v in known else 'Other'
    merged['biopsy_norm'] = merged['characteristics_bio_material'].map(norm_bio)

    lookup = merged.drop_duplicates('SampleID').set_index('SampleID')

    rows = []
    for sid in sample_ids:
        if sid in lookup.index:
            r = lookup.loc[sid]
            rows.append({
                'sample_id':             sid,
                'diagnosis':             r['diagnosis_norm'],
                'biopsy_location':       r['biopsy_norm'],
                'macroscopic_appearance': r['macro_norm'],
                'disease_activity':      r['disease_activity_norm'],
                'disease_location':      r['disease_loc_norm'],
                'qc':                    str(r.get('Sample QC', '')),
            })
        else:
            rows.append({
                'sample_id':             sid,
                'diagnosis':             'Unknown',
                'biopsy_location':       'Unknown',
                'macroscopic_appearance': 'Unknown',
                'disease_activity':      'Unknown',
                'disease_location':      'Unknown',
                'qc':                    'unknown',
            })
    df = pd.DataFrame(rows).set_index('sample_id')
    for col in ['diagnosis', 'biopsy_location', 'macroscopic_appearance',
                'disease_activity', 'disease_location']:
        df[col] = df[col].fillna('Unknown')
    return df


# ── UMAP computation ───────────────────────────────────────────────────────────

def run_umap(X, sample_ids):
    if os.path.exists(COORDS_NPZ):
        print('  Loading cached UMAP coordinates ...')
        d = np.load(COORDS_NPZ, allow_pickle=True)
        cached_ids = list(d['sample_ids'])
        if cached_ids == sample_ids:
            return d['xy']
        print('  Cache mismatch — recomputing')

    print(f'  PCA → {N_PCA} components ...')
    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    Xpca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.cumsum()[N_PCA - 1]
    print(f'    variance explained: {var_explained*100:.1f}%')

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
    print(f'  Saved coords → {COORDS_NPZ}')
    return xy


# ── plotting ───────────────────────────────────────────────────────────────────

def _scatter_ax(ax, df, xy, col, palette, title, n_total):
    ax.set_facecolor('#FAFAFA')
    order = [k for k in palette if k != 'Unknown'] + ['Unknown']
    for cat in order:
        mask = (df[col] == cat).values
        if not mask.any():
            continue
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
        stem = f'umap_rna_{col}'
        for ext in ('png', 'pdf'):
            p = os.path.join(out_dir, f'{stem}.{ext}')
            fig.savefig(p, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'    {stem}.png/.pdf')


def make_html(df, xy, out_path):
    """Interactive Plotly HTML with dropdown to switch colour variable."""
    all_traces = []
    group_spans = []

    for col, palette in PALETTES.items():
        order = [k for k in palette if k != 'Unknown'] + ['Unknown']
        traces = []
        for cat in order:
            mask = (df[col] == cat).values
            if not mask.any():
                continue
            sub_xy  = xy[mask]
            sub_df  = df[mask]
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
                    f'Location: {r["disease_location"]}'
                ), axis=1),
                hovertemplate='%{text}<extra></extra>',
                legendgroup=col,
                visible=False,
            ))
        group_spans.append((col, len(traces)))
        all_traces.extend(traces)

    fig = go.Figure()
    for t in all_traces:
        fig.add_trace(t)

    # make first group visible by default
    for i in range(group_spans[0][1]):
        fig.data[i].visible = True

    flat_len = len(all_traces)
    buttons = []
    cum = 0
    for col, n_traces in group_spans:
        vis = [False] * flat_len
        for j in range(cum, cum + n_traces):
            vis[j] = True
        buttons.append(dict(
            label=TITLES[col], method='update',
            args=[{'visible': vis},
                  {'title': f'RNA-seq UMAP — {TITLES[col]}'}],
        ))
        cum += n_traces

    fig.update_layout(
        title=dict(text=f'RNA-seq VST UMAP — {TITLES["diagnosis"]}',
                   font=dict(size=16)),
        updatemenus=[dict(
            buttons=buttons, direction='down',
            x=0.01, xanchor='left', y=1.13, yanchor='top',
            showactive=True, bgcolor='#F0F0F0', bordercolor='#CCCCCC',
        )],
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
        text=f'{len(df)} RNA-seq samples · VST batch-corrected · PCA({N_PCA}) → UMAP',
        x=1, xref='paper', y=-0.05, yref='paper',
        showarrow=False, font=dict(size=10, color='#888888'), xanchor='right')

    fig.write_html(str(out_path), include_plotlyjs='cdn')
    print(f'  HTML → {out_path}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=== RNA-seq UMAP ===')
    print('Loading GCT ...')
    sample_ids, X = load_gct(VST_GCT)
    print(f'  Matrix shape: {X.shape}')

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
    make_html(meta_df, xy, os.path.join(OUT_DIR, 'umap_rna.html'))

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
