import os
from pathlib import Path

INPUT_DIR = Path("/alan-data/jinc/share/data/ibd_clean_xavier/sparc-image-ffp/all_wsi_tiff")

accessible, missing, denied = [], [], []
for _, row in df_slides.iterrows():
    sid = row['slide_id']
    for ext in ('.tiff', '.tif'):
        p = INPUT_DIR / f"{sid}{ext}"
        if p.exists():
            if os.access(p, os.R_OK):
                accessible.append(sid)
            else:
                denied.append(sid)
            break
    else:
        missing.append(sid)

print(f"Accessible: {len(accessible)}")
print(f"Missing:    {len(missing)}")
print(f"Denied:     {len(denied)}")
