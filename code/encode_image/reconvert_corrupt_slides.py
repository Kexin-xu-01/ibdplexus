"""
Re-convert specific VSI slides to pyramidal TIFF, matching the exact format of
the existing tiff_mpp_corrected files:
  - LZW compression, 256x256 tiles
  - Pyramid: halve until min(w,h) < 256
  - ResolutionUnit=centimeter, XResolution=YResolution=1e4/mpp (pixels/cm)
  - Overwrites the corrupt output TIFF in-place (via temp file)

Usage:
    python reconvert_corrupt_slides.py
"""

import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import slideio
import tifffile

TILE_SIZE = 256
OUTPUT_DIR = "/home/jovyan/kgbk271-ibd-datavol-1/data/raw/tiff_mpp_corrected"

SLIDES = [
    {
        "name": "10816615HE1",
        "vsi": "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/VSI_MAY_2022/10816615HE1.vsi",
    },
    {
        "name": "10955306HE1",
        "vsi": "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/VSI_MAY_2022/10955306HE1.vsi",
    },
    {
        "name": "10989128HE1",
        "vsi": "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/VSI_MAY_2022/10989128HE1.vsi",
    },
    {
        "name": "10656524HE1",
        "vsi": "/home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021/10656524HE1.vsi",
    },
]


def build_pyramid(img: np.ndarray) -> list:
    """Halve until min(w,h) < 256; matches existing TIFF pyramid structure."""
    levels = [img]
    while min(levels[-1].shape[0], levels[-1].shape[1]) >= 256:
        prev = levels[-1]
        levels.append(prev[::2, ::2])
    return levels


def convert_one(name: str, vsi_path: str) -> None:
    out_path = os.path.join(OUTPUT_DIR, f"{name}.tiff")
    tmp_path = out_path + ".converting"

    print(f"\n[{name}] Reading VSI: {vsi_path}")
    slide = slideio.open_slide(vsi_path, "VSI")
    scene = slide.get_scene(0)
    mpp = scene.resolution[0] * 1e6  # metres/px -> µm/px
    w, h = scene.size
    print(f"[{name}]   size={w}x{h}  mpp={mpp:.6f}")

    print(f"[{name}] Reading full-resolution pixels...")
    img = scene.read_block()  # (H, W, C) uint8
    if img is None or img.size == 0:
        raise ValueError("Empty image returned by slideio")
    print(f"[{name}]   array shape={img.shape} dtype={img.dtype}")

    pyramid = build_pyramid(img)
    print(f"[{name}]   pyramid levels: {len(pyramid)}  "
          f"({' -> '.join(f'{p.shape[1]}x{p.shape[0]}' for p in pyramid)})")

    # pixels/cm for TIFF resolution tags — OpenSlide derives mpp from this
    res_cm = 1e4 / mpp

    print(f"[{name}] Writing {tmp_path} ...")
    with tifffile.TiffWriter(tmp_path, bigtiff=True) as tif:
        opts = dict(
            tile=(TILE_SIZE, TILE_SIZE),
            compression="lzw",
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=(res_cm, res_cm),
            photometric="rgb",
        )
        # Full-res page (subfiletype=0) followed by reduced-resolution pages
        # (subfiletype=1) as flat multi-page — the format OpenSlide recognises
        # as a generic-tiff pyramid.
        tif.write(pyramid[0], subfiletype=0, **opts)
        for level_img in pyramid[1:]:
            tif.write(level_img, subfiletype=1, **opts)

    # Atomic replace
    print(f"[{name}] Replacing {out_path}")
    shutil.move(tmp_path, out_path)
    print(f"[{name}] Done. Output: {out_path}")


def main():
    import openslide

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for s in SLIDES:
        name, vsi = s["name"], s["vsi"]
        try:
            convert_one(name, vsi)

            # Verify with OpenSlide
            out_path = os.path.join(OUTPUT_DIR, f"{name}.tiff")
            slide = openslide.OpenSlide(out_path)
            mpp_x = slide.properties.get("openslide.mpp-x", "?")
            mpp_y = slide.properties.get("openslide.mpp-y", "?")
            dims = slide.dimensions
            levels = slide.level_count
            slide.close()
            print(f"[{name}] Verify OK: dims={dims} levels={levels} mpp_x={mpp_x} mpp_y={mpp_y}")

        except Exception:
            print(f"[{name}] ERROR:")
            traceback.print_exc()
            # Clean up partial temp file if present
            tmp = os.path.join(OUTPUT_DIR, f"{name}.tiff.converting")
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    main()
