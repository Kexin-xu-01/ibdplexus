"""
Side-by-side Venn: data availability for all colon sites vs at-20-cm only.
Shows patient counts per region (H&E only / both / RNA only).

Usage
-----
    python venn_all_vs_at20cm.py

Output
------
    <REPORTS_DIR>/venn_all_vs_at20cm.pdf
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

WSI_META = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv'
OMICS    = '/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv'
REPORTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/reports'
)

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}
AT20  = {'at 20 cm', 'At 20 cm'}
CDUC_RAW  = ["Crohn's disease", 'Ulcerative colitis']
CDUC_OMICS = ["Crohn's Disease", "Crohn's disease", 'Ulcerative colitis', 'Ulcerative Colitis']

SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
HE_C  = '#2a78d6'
RNA_C = '#1baf7a'


def plot(wsi_meta: str = WSI_META, omics_csv: str = OMICS,
         out_dir: str = REPORTS_DIR) -> str:
    wsi = pd.read_csv(wsi_meta)
    wsi = wsi[wsi['diagnosis'].isin(CDUC_RAW)].copy()

    he_all = set(wsi[wsi['BIOSAMPLE_LOCATION'].isin(COLON_LOCS)]['deidentified_master_patient_id'].dropna())
    he_20  = set(wsi[wsi['BIOSAMPLE_LOCATION'].isin(AT20)]['deidentified_master_patient_id'].dropna())

    omics = pd.read_csv(omics_csv)
    rna_all = set(omics[
        (omics['omics_type'] == 'RNASeq') &
        (omics['characteristics_bio_material'].isin(COLON_LOCS)) &
        (omics['diagnosis'].isin(CDUC_OMICS))
    ]['deidentified_master_patient_id'].dropna())
    rna_20 = set(omics[
        (omics['omics_type'] == 'RNASeq') &
        (omics['characteristics_bio_material'].isin(AT20)) &
        (omics['diagnosis'].isin(CDUC_OMICS))
    ]['deidentified_master_patient_id'].dropna())

    panels = [
        ('All colon biopsy sites', he_all, rna_all),
        ('At-20-cm biopsy only',   he_20,  rna_20),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURF)
    fig.suptitle('Data availability — H&E slides vs RNA-seq  (CD + UC patients)',
                 fontsize=12, fontweight='bold', color=INK, y=1.01)

    cx, cy, dx = 0.0, 0.0, 0.28
    rx, ry = 0.50, 0.38

    for ax, (title, he, rna_set) in zip(axes, panels):
        ax.set_facecolor(SURF)
        he_only  = he - rna_set
        rna_only = rna_set - he
        both     = he & rna_set
        total    = he | rna_set

        for fc, x in [(HE_C, cx - dx), (RNA_C, cx + dx)]:
            ax.add_patch(Ellipse((x, cy), 2*rx, 2*ry, facecolor=fc, alpha=0.30, linewidth=0))
            ax.add_patch(Ellipse((x, cy), 2*rx, 2*ry, facecolor='none', edgecolor=fc, linewidth=1.8))

        def lbl(x, y, line1, n, c1):
            ax.text(x, y + 0.10, line1, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=c1)
            ax.text(x, y - 0.08, f'{n:,}', ha='center', va='center',
                    fontsize=13, color=INK)
            ax.text(x, y - 0.24, 'patients', ha='center', va='center',
                    fontsize=8, color=MUTED)

        lbl(cx - dx - 0.22, cy, 'H&E only',  len(he_only),  HE_C)
        lbl(cx + dx + 0.22, cy, 'RNA only',  len(rna_only), RNA_C)
        lbl(cx,             cy, 'Both',      len(both),     INK2)

        ax.text(cx, -0.56, f'Total unique patients: {len(total):,}',
                ha='center', fontsize=8.5, color=MUTED)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', color=INK, pad=10)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'venn_all_vs_at20cm.pdf')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=SURF)
    plt.close()
    return out


if __name__ == '__main__':
    path = plot()
    print(f'Saved: {path}')
