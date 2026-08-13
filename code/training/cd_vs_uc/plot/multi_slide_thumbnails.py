"""
Thumbnail grid showing same-biopsy vs multi-visit slides for three example patients.

Panels:
  - Patient A: serial sections from the same block (HE1 + HE101)
  - Patient B: two tissue sections same visit (HE1 + HE2)
  - Patient C: four slides across multiple visits (UC, at-20-cm)

Usage
-----
    python multi_slide_thumbnails.py

Output
------
    <REPORTS_DIR>/multi_slide_thumbnails.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

THUMB_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/data/processed/'
    'trident_processed/thumbnails'
)
REPORTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/reports'
)

SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'

# (slide_id, subplot_title, ax_index)
LAYOUT = [
    ('10601326HE101', 'Same block A\n2018-07-21',  0),
    ('10799378HE1',   'Same block B\n2020-02-29',  1),
    ('10799378HE2',   'Same block B\n2020-02-29',  2),
    ('10829795HE1',   'Visit 1 (2020-03-01)',       4),
    ('10866466HE1',   'Visit 2 (2020-11-20)',       5),
    ('10940278HE1',   'Visit 3 (2021-07-18)',       6),
    ('11003737HE1',   'Visit 4 (2021-10-28)',       7),
]


def plot(thumb_dir: str = THUMB_DIR, out_dir: str = REPORTS_DIR) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor=SURF)
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor(SURF)

    for sid, lbl, ax_i in LAYOUT:
        path = os.path.join(thumb_dir, f'{sid}.jpg')
        ax = axes[ax_i]
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path))
        ax.set_title(lbl, fontsize=8, color=INK2, pad=4)
        ax.set_xlabel(sid, fontsize=7, color=MUTED)
        ax.tick_params(left=False, bottom=False)

    fig.text(0.18, 0.97, 'Same visit — serial sections (patient A)',
             ha='center', fontsize=9, fontweight='bold', color=INK)
    fig.text(0.50, 0.97, 'Same visit — HE1 + HE2 sections (patient B)',
             ha='center', fontsize=9, fontweight='bold', color=INK)
    fig.text(0.78, 0.97, 'Multiple visits — same patient (patient C, UC, at-20-cm)',
             ha='center', fontsize=9, fontweight='bold', color=INK)

    for x in [0.37, 0.60]:
        fig.add_artist(plt.Line2D(
            [x, x], [0.05, 0.93], color=MUTED,
            linewidth=0.7, linestyle='--', transform=fig.transFigure))

    fig.suptitle('At-20-cm multi-slide patients — thumbnail comparison',
                 fontsize=11, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    tmp_png = '/tmp/multi_slide_thumbnails.png'
    fig.savefig(tmp_png, dpi=150, bbox_inches='tight', facecolor=SURF)
    plt.close()

    out = os.path.join(out_dir, 'multi_slide_thumbnails.pdf')
    Image.open(tmp_png).convert('RGB').save(out, 'PDF', resolution=200)
    return out


if __name__ == '__main__':
    path = plot()
    print(f'Saved: {path}')
