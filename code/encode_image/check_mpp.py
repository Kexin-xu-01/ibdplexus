import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def rational_to_float(value):
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0] / value[1]
        return float(value)
    except Exception:
        return None

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <tiff_path>")
        sys.exit(1)

    tiff_path = sys.argv[1]

    with Image.open(tiff_path) as img:
        tags = img.tag_v2

        description = tags.get(270)   # ImageDescription
        x_res = tags.get(282)         # XResolution
        y_res = tags.get(283)         # YResolution
        res_unit = tags.get(296)      # ResolutionUnit

        print("ImageDescription:")
        print(description)
        print()

        print("XResolution:", x_res)
        print("YResolution:", y_res)
        print("ResolutionUnit:", res_unit)
        print()

        x_res_f = rational_to_float(x_res) if x_res else None
        y_res_f = rational_to_float(y_res) if y_res else None

        if x_res_f and y_res_f and res_unit:
            if res_unit == 2:
                mpp_x = 25400 / x_res_f
                mpp_y = 25400 / y_res_f
                print(f"MPP X: {mpp_x:.4f} um/pixel")
                print(f"MPP Y: {mpp_y:.4f} um/pixel")
            elif res_unit == 3:
                mpp_x = 10000 / x_res_f
                mpp_y = 10000 / y_res_f
                print(f"MPP X: {mpp_x:.4f} um/pixel")
                print(f"MPP Y: {mpp_y:.4f} um/pixel")
            else:
                print("Unknown ResolutionUnit, cannot compute MPP.")
        else:
            print("Standard TIFF resolution metadata not found.")

if __name__ == "__main__":
    main()