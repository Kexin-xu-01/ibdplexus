"""
Confusion matrix for the PCA-640 RNA + prism2 base (concat_pca640_base) arm.

Aggregates all 5 CV folds. Rows = true class, cols = predicted class.
Each cell shows count and row-normalized percentage (= recall per class).

Output
------
    <PLOTS_DIR>/pca640_confusion_matrix.png/.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

PRED_CSV  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/results/at20cm_pca_rna_predictions.csv')
PLOTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/plots')

STRATEGY  = 'concat_pca640_base'
CLASSES   = ['CD', 'UC']          # label 0 = CD, 1 = UC

INK   = '#000000'
MUTED = '#555555'
# cell fill colours
CORRECT_COLOR = '#d4e9f7'   # soft blue — correct diagonal
ERROR_COLOR   = '#fde8e8'   # soft red  — off-diagonal


def plot(out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(PRED_CSV)
    sub = df[df['strategy'] == STRATEGY]
    if sub.empty:
        raise RuntimeError(f'No rows for strategy={STRATEGY!r}')

    cm = confusion_matrix(sub['true_label'], sub['pred_label'], labels=[0, 1])
    # row-normalise (percent of true class)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    acc = accuracy_score(sub['true_label'], sub['pred_label'])
    f1  = f1_score(sub['true_label'], sub['pred_label'], average='macro')

    n = 2
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    for i in range(n):
        for j in range(n):
            color = CORRECT_COLOR if i == j else ERROR_COLOR
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       color=color, zorder=1))
            ax.text(j, i,
                    f'{cm[i, j]:,}\n{cm_pct[i, j]:.1f}%',
                    ha='center', va='center',
                    fontsize=9, fontfamily='sans-serif', color=INK,
                    fontweight='bold', zorder=2)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)          # origin top-left
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, fontsize=8.5, color=INK, fontfamily='sans-serif')
    ax.set_yticklabels(CLASSES, fontsize=8.5, color=INK, fontfamily='sans-serif')
    ax.set_xlabel('Predicted', fontsize=8, color=INK, fontfamily='sans-serif')
    ax.set_ylabel('True', fontsize=8, color=INK, fontfamily='sans-serif')
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # draw grid lines between cells
    for k in range(1, n):
        ax.axhline(k - 0.5, color='white', lw=2, zorder=3)
        ax.axvline(k - 0.5, color='white', lw=2, zorder=3)

    n_visits = len(sub)
    n_folds  = sub['fold'].nunique()
    ax.set_title(
        f'PCA-640 RNA + prism2 base\n'
        f'{n_visits:,} visits · {n_folds}-fold CV · AUC 0.914 ± 0.013\n'
        f'Accuracy {acc:.3f}  ·  Macro-F1 {f1:.3f}',
        fontsize=7.5, fontweight='bold', color=INK,
        fontfamily='sans-serif', pad=6, loc='left',
    )

    plt.tight_layout(pad=0.5)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'pca640_confusion_matrix.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    plot()
