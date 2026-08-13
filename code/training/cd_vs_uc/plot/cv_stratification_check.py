"""
3-panel stacked bar chart: % composition per CV fold for diagnosis, sex,
and age-at-diagnosis. Left bar = full 1,250-patient cohort; right bar = matched
828 patients. Dashed lines mark the overall proportion (stratification target).

Usage
-----
    python cv_stratification_check.py

Output
------
    <REPORTS_DIR>/cv_stratification_check.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
AT20_PREDS  = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/results/at20cm_patient_predictions.csv'
)
REPORTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/reports'
)

SURF   = '#fcfcfb'
INK    = '#0b0b0b'
INK2   = '#52514e'
MUTED  = '#898781'
GRID   = '#e1e0d9'
BLUE   = '#2a78d6'
ORANGE = '#eb6834'
AQUA   = '#1baf7a'
YELLOW = '#eda100'

PANELS = [
    {
        'title': 'Diagnosis',
        'col':   'diagnosis',
        'cats':  ["Crohn's disease", 'Ulcerative colitis'],
        'labels': ['CD', 'UC'],
        'colors': [BLUE, ORANGE],
    },
    {
        'title': 'Sex',
        'col':   'gender',
        'cats':  ['Female', 'Male'],
        'labels': ['Female', 'Male'],
        'colors': [AQUA, ORANGE],
    },
    {
        'title': 'Age at diagnosis',
        'col':   'age_group',
        'cats':  ['<20 yrs', '20–35 yrs', '>35 yrs'],
        'labels': ['< 20', '20–35', '> 35'],
        'colors': [BLUE, AQUA, YELLOW],
    },
]


def _age_bin(a):
    if pd.isna(a): return 'Unknown'
    if a < 20:     return '<20 yrs'
    if a <= 35:    return '20–35 yrs'
    return '>35 yrs'


def plot(cv_patients: str = CV_PATIENTS, at20_preds: str = AT20_PREDS,
         out_dir: str = REPORTS_DIR) -> str:
    cv = pd.read_csv(cv_patients)
    matched_pids = set(
        pd.read_csv(at20_preds)
        .query("strategy == 'concat_raw_20cm'")['patient_id']
    )
    matched = cv[cv['patient_id'].isin(matched_pids)].copy()

    for df in [cv, matched]:
        df['age_group'] = df['age_at_diagnosis'].apply(_age_bin)

    folds = [0, 1, 2, 3, 4]
    BAR_W = 0.38
    GAP   = 0.06

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=SURF)
    fig.suptitle('CV split stratification — % composition per fold',
                 fontsize=12, fontweight='bold', color=INK, y=1.01)

    for ax, p in zip(axes, PANELS):
        ax.set_facecolor(SURF)
        col, cats, colors = p['col'], p['cats'], p['colors']

        overall = {c: (cv[col] == c).sum() / len(cv) for c in cats}

        xs_full    = np.array(folds) - BAR_W / 2 - GAP / 2
        xs_matched = np.array(folds) + BAR_W / 2 + GAP / 2

        for dataset, xs, alpha, lw in [
            (cv,      xs_full,    0.85, 0),
            (matched, xs_matched, 0.50, 1.2),
        ]:
            bottoms = np.zeros(5)
            for cat, color in zip(cats, colors):
                pcts = [
                    (dataset[dataset['fold'] == f][col] == cat).sum() /
                    len(dataset[dataset['fold'] == f]) * 100
                    for f in folds
                ]
                ax.bar(xs, pcts, bottom=bottoms, width=BAR_W,
                       color=color, alpha=alpha, linewidth=lw,
                       edgecolor=SURF if lw == 0 else INK2, zorder=2)
                bottoms += np.array(pcts)

        cumsum = 0
        for cat, color in zip(cats, colors):
            pct = overall[cat] * 100
            ax.axhline(cumsum + pct, color=color, linewidth=1.0,
                       linestyle='--', alpha=0.7, zorder=3)
            cumsum += pct

        ax.set_xticks(folds)
        ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=8.5, color=INK2)
        ax.set_ylim(0, 105)
        ax.set_ylabel('% of fold', fontsize=9, color=INK2)
        ax.set_title(p['title'], fontsize=10.5, fontweight='bold', color=INK, pad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(colors=INK2, length=3)

        handles = [mpatches.Patch(facecolor=c, label=l)
                   for c, l in zip(colors, p['labels'])]
        ax.legend(handles=handles, fontsize=8, loc='upper right',
                  frameon=False, labelcolor=INK2)

    full_patch = mpatches.Patch(
        facecolor='#888888', alpha=0.85,
        label=f'Full cohort (n={len(cv):,}) — left bar')
    matched_patch = mpatches.Patch(
        facecolor='#888888', alpha=0.50, edgecolor=INK2, linewidth=1.2,
        label=f'Matched {len(matched):,} — right bar')
    dash_patch = plt.Line2D(
        [0], [0], color=MUTED, linestyle='--', linewidth=1.0,
        label='Overall proportion (target)')
    fig.legend(
        handles=[full_patch, matched_patch, dash_patch],
        loc='lower center', ncol=3, fontsize=8.5, frameon=False,
        labelcolor=INK2, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'cv_stratification_check.pdf')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=SURF)
    plt.close()
    return out


if __name__ == '__main__':
    path = plot()
    print(f'Saved: {path}')
