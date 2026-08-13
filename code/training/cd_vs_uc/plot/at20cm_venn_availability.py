"""
At-20-cm Venn: H&E (H5-filtered) vs RNA-seq with both patient counts and biopsy counts.

Usage
-----
    python at20cm_venn_availability.py

Output
------
    <REPORTS_DIR>/at20cm_venn_availability.pdf
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

WSI_META = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv'
OMICS    = '/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv'
H5_DIR   = (
    '/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
    '20x_224px_0px_overlap/prism2_base'
)
REPORTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/reports'
)

AT20       = {'at 20 cm', 'At 20 cm'}
CDUC_RAW   = ["Crohn's disease", 'Ulcerative colitis']
CDUC_OMICS = ["Crohn's Disease", "Crohn's disease", 'Ulcerative colitis', 'Ulcerative Colitis']

SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
HE_C  = '#2a78d6'
RNA_C = '#1baf7a'


def plot(wsi_meta: str = WSI_META, omics_csv: str = OMICS,
         h5_dir: str = H5_DIR, out_dir: str = REPORTS_DIR) -> str:
    h5_ids = {f[:-3] for f in os.listdir(h5_dir) if f.endswith('.h5')}

    wsi = pd.read_csv(wsi_meta)
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi_at20 = wsi[
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['slide_id'].isin(h5_ids)
    ].copy()

    omics = pd.read_csv(omics_csv)
    rna_at20 = omics[
        (omics['omics_type'] == 'RNASeq') &
        (omics['characteristics_bio_material'].isin(AT20)) &
        (omics['diagnosis'].isin(CDUC_OMICS))
    ].copy()

    he_pids  = set(wsi_at20['deidentified_master_patient_id'].dropna())
    rna_pids = set(rna_at20['deidentified_master_patient_id'].dropna())

    he_only  = he_pids - rna_pids
    rna_only = rna_pids - he_pids
    both     = he_pids & rna_pids
    total    = he_pids | rna_pids

    he_slides_only = len(wsi_at20[wsi_at20['deidentified_master_patient_id'].isin(he_only)])
    he_slides_both = len(wsi_at20[wsi_at20['deidentified_master_patient_id'].isin(both)])
    rna_samp_only  = len(rna_at20[rna_at20['deidentified_master_patient_id'].isin(rna_only)])
    rna_samp_both  = len(rna_at20[rna_at20['deidentified_master_patient_id'].isin(both)])

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURF)
    ax.set_facecolor(SURF)

    cx, cy, dx = 0.0, 0.0, 0.30
    rx, ry = 0.52, 0.40

    for fc, x in [(HE_C, cx - dx), (RNA_C, cx + dx)]:
        ax.add_patch(Ellipse((x, cy), 2*rx, 2*ry, facecolor=fc, alpha=0.28, linewidth=0))
        ax.add_patch(Ellipse((x, cy), 2*rx, 2*ry, facecolor='none', edgecolor=fc, linewidth=2.0))

    ax.text(cx - dx - 0.10, cy + ry + 0.06, 'H&E',
            ha='center', fontsize=12, fontweight='bold', color=HE_C)
    ax.text(cx + dx + 0.10, cy + ry + 0.06, 'RNA-seq',
            ha='center', fontsize=12, fontweight='bold', color=RNA_C)

    def section(x, y, pts, biop, biop_lbl, col):
        ax.text(x, y + 0.13, f'{pts:,}', ha='center', va='center',
                fontsize=15, fontweight='bold', color=col)
        ax.text(x, y + 0.00, 'patients', ha='center', va='center',
                fontsize=8, color=MUTED)
        ax.text(x, y - 0.13, f'{biop:,}', ha='center', va='center',
                fontsize=12, color=col, style='italic')
        ax.text(x, y - 0.24, biop_lbl, ha='center', va='center',
                fontsize=7.5, color=MUTED)

    section(cx - dx - 0.24, cy, len(he_only),  he_slides_only, 'H&E slides', HE_C)
    section(cx + dx + 0.24, cy, len(rna_only), rna_samp_only,  'RNA samples', RNA_C)

    ax.text(cx, cy + 0.22, f'{len(both):,}', ha='center', va='center',
            fontsize=15, fontweight='bold', color=INK2)
    ax.text(cx, cy + 0.09, 'patients', ha='center', va='center', fontsize=8, color=MUTED)
    ax.text(cx, cy - 0.05, f'{he_slides_both:,}', ha='center', va='center',
            fontsize=11, color=HE_C, style='italic')
    ax.text(cx, cy - 0.15, 'H&E slides', ha='center', va='center', fontsize=7.5, color=MUTED)
    ax.text(cx, cy - 0.27, f'{rna_samp_both:,}', ha='center', va='center',
            fontsize=11, color=RNA_C, style='italic')
    ax.text(cx, cy - 0.37, 'RNA samples', ha='center', va='center', fontsize=7.5, color=MUTED)

    ax.set_title('At-20-cm biopsy — H&E slides (with embeddings) vs RNA-seq  (CD + UC)',
                 fontsize=11, fontweight='bold', color=INK, pad=14)
    ax.text(0, -0.63, f'Total unique patients: {len(total):,}',
            ha='center', fontsize=9, color=MUTED)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.75, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'at20cm_venn_availability.pdf')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=SURF)
    plt.close()
    return out


if __name__ == '__main__':
    path = plot()
    print(f'Saved: {path}')
