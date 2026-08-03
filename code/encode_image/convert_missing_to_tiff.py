"""
Convert 431 slides missing from tiff_mpp_corrected by reading their source VSIs.
Uses the same format as existing TIFFs: LZW, 256x256 tiles, flat multi-page
pyramid (subfiletype=1), ResolutionUnit=centimeter.
Writes atomically via .converting temp file; skips slides already present.
"""

import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import slideio
import tifffile
from tqdm import tqdm

OUTPUT_DIR = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/raw/tiff_mpp_corrected")
VSI_ROOT   = Path("/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe")
ALL_WSI    = Path("/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/all_wsi_tiff")
TILE_SIZE  = 256


def build_vsi_index():
    index = {}
    for vsi in VSI_ROOT.rglob("*.vsi"):
        stem = vsi.stem
        if "_Overview" in stem:
            continue
        if stem.startswith("Image_Overview"):
            clean = stem[len("Image_Overview"):]
            index[clean.lower()] = vsi
        else:
            index[stem.lower()] = vsi
    return index


def build_pyramid(img: np.ndarray) -> list:
    levels = [img]
    while min(levels[-1].shape[0], levels[-1].shape[1]) >= 256:
        prev = levels[-1]
        levels.append(prev[::2, ::2])
    return levels


def convert_one(name: str, vsi_path: Path) -> str:
    out_path = OUTPUT_DIR / f"{name}.tiff"
    tmp_path = OUTPUT_DIR / f"{name}.tiff.converting"

    if out_path.exists():
        return f"[SKIP] {name} already exists"

    slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    mpp = scene.resolution[0] * 1e6  # metres/px -> µm/px
    if mpp <= 0:
        raise ValueError(f"Invalid mpp={mpp} from VSI metadata")

    img = scene.read_block()
    if img is None or img.size == 0:
        raise ValueError("Empty image returned by slideio")

    pyramid = build_pyramid(img)
    res_cm = 1e4 / mpp

    with tifffile.TiffWriter(str(tmp_path), bigtiff=True) as tif:
        opts = dict(
            tile=(TILE_SIZE, TILE_SIZE),
            compression="lzw",
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=(res_cm, res_cm),
            photometric="rgb",
        )
        tif.write(pyramid[0], subfiletype=0, **opts)
        for level_img in pyramid[1:]:
            tif.write(level_img, subfiletype=1, **opts)

    shutil.move(str(tmp_path), str(out_path))
    return f"[OK] {name}  mpp={mpp:.6f}  size={scene.size}  levels={len(pyramid)}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected = set(p.stem for p in OUTPUT_DIR.glob("*.tif*"))
    all_wsi   = set(p.stem for p in ALL_WSI.glob("*.tif*"))
    missing   = sorted(all_wsi - corrected)
    print(f"Slides to convert: {len(missing)}")

    vsi_index = build_vsi_index()

    work = []
    for name in missing:
        vsi = vsi_index.get(name.lower())
        if vsi:
            work.append((name, vsi))
        else:
            print(f"[WARN] No VSI found for {name}")

    print(f"VSI found for: {len(work)}")

    errors = []
    for name, vsi_path in tqdm(work, unit="slide"):
        tmp = OUTPUT_DIR / f"{name}.tiff.converting"
        try:
            msg = convert_one(name, vsi_path)
            tqdm.write(msg)
        except Exception as e:
            tqdm.write(f"[ERROR] {name}: {e}")
            errors.append((name, str(e)))
            if tmp.exists():
                tmp.unlink()

    print(f"\nDone. Errors: {len(errors)}")
    for name, err in errors:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
