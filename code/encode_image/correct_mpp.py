import tifffile
import pandas as pd
from pathlib import Path
import shutil

TSV_PATH   = "/home/jovyan/shared-data/users/kexin/vsi_metadata.tsv"
INPUT_DIR  = Path("/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/all_wsi_tiff")
OUTPUT_DIR = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/raw/tiff_mpp_corrected")

df = pd.read_csv(TSV_PATH, sep='\t')
df_slides = df[(df['is_overview'] == 0) & (df['num_scenes'] == 1)].reset_index(drop=True)

# Skip slides already corrected
already_done = set(p.stem for p in OUTPUT_DIR.glob('*.tif*'))
df_slides = df_slides[~df_slides['slide_id'].isin(already_done)].reset_index(drop=True)
print(f"Already done: {len(already_done)} | Remaining: {len(df_slides)}")

# Audit accessibility — os.access() lies on NFS+ACL, so actually try to open
accessible, missing, denied = [], [], []
for _, row in df_slides.iterrows():
    sid = row['slide_id']
    for ext in ('.tiff', '.tif'):
        p = INPUT_DIR / f"{sid}{ext}"
        if p.exists():
            try:
                p.open('rb').close()
                accessible.append(sid)
            except PermissionError:
                denied.append(sid)
            break
    else:
        missing.append(sid)

print(f"Accessible: {len(accessible)}")
print(f"Missing:    {len(missing)}")
print(f"Denied:     {len(denied)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def to_rational(val, denom=10000):
    return (int(round(val * denom)), denom)

def inspect_and_fix(row, dry_run=False):
    sid   = row['slide_id']
    mpp_x = row['mpp_x_um']
    mpp_y = row['mpp_y_um']
    xres  = 10000.0 / mpp_x   # pixels per cm
    yres  = 10000.0 / mpp_y

    for ext in ('.tiff', '.tif'):
        src = INPUT_DIR / f"{sid}{ext}"
        if src.exists():
            break
    else:
        print(f"  [SKIP] {sid} — file not found")
        return

    dst = OUTPUT_DIR / src.name

    try:
        with tifffile.TiffFile(str(src)) as tif:
            if not tif.pages:
                print(f"  [SKIP] {sid} — empty/corrupt TIFF (no pages)")
                return
            p = tif.pages[0]
            h, w = p.shape[:2]
            tsv_w, tsv_h = int(row['width_px']), int(row['height_px'])
            dim_ok = (w == tsv_w and h == tsv_h)

            cur_xres = p.tags.get(282)
            cur_yres = p.tags.get(283)
            cur_unit = p.tags.get(296)

            print(f"\n=== {sid} ===")
            print(f"  file:  {src.name}  ({src.stat().st_size/1e9:.2f} GB)")
            print(f"  shape: {w}x{h}  TSV: {tsv_w}x{tsv_h}  {'OK' if dim_ok else '*** MISMATCH ***'}")
            print(f"  xres:  {cur_xres.value if cur_xres else 'n/a'}  "
                  f"yres: {cur_yres.value if cur_yres else 'n/a'}  "
                  f"unit: {cur_unit.value if cur_unit else 'n/a'}")
            print(f"  ->     xres={xres:.4f}  yres={yres:.4f}  unit=3 (cm)")

            if not dim_ok:
                print("  [SKIP] dimension mismatch")
                return

    except PermissionError:
        print(f"  [SKIP] {sid} — permission denied")
        return

    if dry_run:
        print("  [DRY RUN] no file written")
        return

    # Copy byte-for-byte, then patch only the 3 resolution tags in-place.
    # No pixel decompression — preserves JPEG compression quality exactly.
    shutil.copyfile(str(src), str(dst))
    try:
        with open(str(dst), 'r+b') as f:
            with tifffile.TiffFile(f) as tif:
                p = tif.pages[0]
                if 282 in p.tags: p.tags[282].overwrite(to_rational(xres))
                if 283 in p.tags: p.tags[283].overwrite(to_rational(yres))
                if 296 in p.tags: p.tags[296].overwrite(3)
    except Exception as e:
        dst.unlink(missing_ok=True)
        print(f"  [FAIL] {sid} — {e}")
        return


    print(f"  [DONE] -> {dst}")


from tqdm import tqdm

all_rows = df_slides[df_slides['slide_id'].isin(accessible)]
print(f"\nProcessing {len(all_rows)} slides...")
for _, row in tqdm(all_rows.iterrows(), total=len(all_rows)):
    inspect_and_fix(row, dry_run=False)
