"""
Convert Olympus VSI files to pyramidal TIFF using slideio.
MPP is read directly from the VSI file metadata.

Usage:
    python convert_vsi_to_tiff.py \
        --input_dir /path/to/vsi \
        --output_dir /path/to/tiff_converted \
        --num_workers 4

    # Test on a single slide:
    python convert_vsi_to_tiff.py \
        --input_dir /path/to/vsi \
        --output_dir /path/to/tiff_converted \
        --limit 1
"""

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


TILE_SIZE = 256
PYRAMID_LEVELS = 4  # full res + 3 downsampled levels


def read_vsi(vsi_path: str):
    """Read full-resolution image and mpp from a VSI file via slideio."""
    import slideio
    slide = slideio.open_slide(vsi_path, 'VSI')
    scene = slide.get_scene(0)
    mpp = scene.resolution[0] * 1e6  # meters/px -> microns/px
    w, h = scene.size
    img = scene.read_block()  # (H, W, C) uint8
    return img, mpp, w, h


def build_pyramid(img: np.ndarray, levels: int):
    """Return list of downsampled images: [full, half, quarter, ...]."""
    pyramid = [img]
    for _ in range(levels - 1):
        prev = pyramid[-1]
        h, w = prev.shape[:2]
        # simple 2x downsample by slicing (fast, no scipy needed)
        downsampled = prev[::2, ::2]
        pyramid.append(downsampled)
    return pyramid


def convert_one(vsi_path: str, output_dir: str) -> str:
    """Convert a single VSI to pyramidal TIFF. Returns slide name on success."""
    name = Path(vsi_path).stem
    out_path = os.path.join(output_dir, f"{name}.tiff")

    if os.path.exists(out_path):
        return f"[SKIP] {name} already exists"

    img, mpp, w, h = read_vsi(vsi_path)

    if img is None or img.size == 0:
        raise ValueError(f"Empty image returned for {vsi_path}")

    # resolution in pixels/cm for TIFF tag (mpp is um/px -> px/cm = 1e4/mpp)
    res_cm = 1e4 / mpp

    pyramid = build_pyramid(img, PYRAMID_LEVELS)

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        options = dict(
            tile=(TILE_SIZE, TILE_SIZE),
            compression="jpeg",
            compressionargs={"level": 90},
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=(res_cm, res_cm),
            photometric="rgb",
            metadata={"mpp": mpp},
        )
        # Write full resolution with subifds for pyramid levels
        tif.write(pyramid[0], subifds=len(pyramid) - 1, **options)
        for level in pyramid[1:]:
            tif.write(level, subfiletype=1, **options)

    return f"[OK] {name}  mpp={mpp:.6f}"


def convert_batch(input_dir: str, output_dir: str, limit: int = None, num_workers: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    vsi_files = sorted(Path(input_dir).glob("*.vsi"))
    # exclude overview files
    vsi_files = [f for f in vsi_files if not f.stem.endswith("_Overview")]

    if limit:
        vsi_files = vsi_files[:limit]

    print(f"Found {len(vsi_files)} VSI files to convert -> {output_dir}")

    if num_workers <= 1:
        for vsi in tqdm(vsi_files):
            try:
                msg = convert_one(str(vsi), output_dir)
                tqdm.write(msg)
            except Exception as e:
                tqdm.write(f"[ERROR] {vsi.name}: {e}")
                tqdm.write(traceback.format_exc())
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            futures = {exe.submit(convert_one, str(v), output_dir): v for v in vsi_files}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                vsi = futures[fut]
                try:
                    tqdm.write(fut.result())
                except Exception as e:
                    tqdm.write(f"[ERROR] {vsi.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Convert only N slides (for testing)")
    args = parser.parse_args()

    convert_batch(args.input_dir, args.output_dir, limit=args.limit, num_workers=args.num_workers)
