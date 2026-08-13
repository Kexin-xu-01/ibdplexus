import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

AT20 = {'at 20 cm', 'At 20 cm'}
CDUC_RAW   = ["Crohn's disease", 'Ulcerative colitis']
CDUC_OMICS = ["Crohn's Disease","Crohn's disease",'Ulcerative colitis','Ulcerative Colitis']

wsi = pd.read_csv('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv')
wsi = wsi[wsi['BIOSAMPLE_LOCATION'].isin(AT20) & wsi['diagnosis'].isin(CDUC_RAW)].copy()
wsi['date'] = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')

MAPPING = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/ibd_21183_omics_patient_mapping_genestack.csv'
mapping = pd.read_csv(MAPPING)
omics   = pd.read_csv('/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv')
pid_dx  = omics[['deidentified_master_patient_id','diagnosis']].drop_duplicates()
rna     = mapping[mapping['characteristics_bio_material'].isin(AT20)].copy()
rna     = rna.merge(pid_dx, on='deidentified_master_patient_id', how='left')
rna     = rna[rna['diagnosis'].isin(CDUC_OMICS)].copy()
rna['date'] = pd.to_datetime(rna['sample_collected_date'], dayfirst=True, errors='coerce')
rna_enc = rna.drop_duplicates(subset='visit_encounter_id')

def gap_days(series_dates):
    dates = sorted(series_dates.dropna().unique())
    return [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

he_gaps  = np.array([g for _, grp in wsi.groupby('deidentified_master_patient_id')
                     for g in gap_days(grp['date'])])
rna_gaps = np.array([g for _, grp in rna_enc.groupby('deidentified_master_patient_id')
                     for g in gap_days(grp['date'])])

# convert to months for readability
he_mo  = he_gaps  / 30.44
rna_mo = rna_gaps / 30.44

HE_C  = '#2a78d6'
RNA_C = '#1baf7a'
INK   = '#000000'
MUTED = '#555555'
GRID  = '#dddddd'

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=False)
fig.patch.set_alpha(0)

bin_edges = np.arange(0, max(he_mo.max(), rna_mo.max()) + 3, 3)  # 3-month bins

for ax, gaps_mo, color, label, n_gaps in [
    (axes[0], he_mo,  HE_C,  'H&E',     len(he_gaps)),
    (axes[1], rna_mo, RNA_C, 'RNA-seq',  len(rna_gaps)),
]:
    counts, edges = np.histogram(gaps_mo, bins=bin_edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges)*0.88,
           align='edge', color=color, zorder=2, linewidth=0)

    med = np.median(gaps_mo)
    ax.axvline(med, color=INK, lw=1.2, ls='--', zorder=3)
    ax.text(med + 0.8, counts.max() * 0.95, f'median\n{med:.0f} mo',
            fontsize=7, color=INK, va='top')

    # shade >= 12 months
    xlim_max = bin_edges[-1]
    ax.axvspan(12, xlim_max, color='#f0e8d0', alpha=0.45, zorder=0, lw=0)
    ax.text(xlim_max * 0.72, counts.max() * 0.55, '≥ 1 yr', fontsize=7,
            color='#a07030', va='center', style='italic')

    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=7.5)
    ax.set_xlabel('Inter-visit gap (months)', fontsize=8.5, color=INK)
    ax.set_ylabel('Number of gaps', fontsize=8.5, color=INK)
    ax.set_xlim(0, xlim_max)

    pct_over1yr = 100 * (gaps_mo >= 12).mean()
    ax.set_title(f'{label}  (n={n_gaps} gaps, {pct_over1yr:.0f}% ≥ 1 yr)',
                 fontsize=9, fontweight='bold', color=INK, loc='left', pad=5)

fig.suptitle('Time between visits — at-20-cm multi-visit patients',
             fontsize=10, fontweight='bold', color=INK, y=1.02)
plt.tight_layout(pad=0.5)

out = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/08_09_at20cm_site_controlled/reports/visit_gaps.png'
plt.savefig(out, dpi=180, bbox_inches='tight', transparent=True)
print(f'Saved: {out}')
