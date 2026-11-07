"""Command-line entry point.

Example:
    python cli.py images/sample.png --width 150 --height 120 --algo dp --output out/dp.png
"""

import argparse
import time
from pathlib import Path

from seam_carving.core import carve, load_image, save_image
from seam_carving.finders import FINDERS, get_finder


def parse_args():
    parser = argparse.ArgumentParser(description="Content-aware image resizing by seam carving")
    parser.add_argument("input", help="path to the input image")
    parser.add_argument("--width", type=int, help="target width (default: keep)")
    parser.add_argument("--height", type=int, help="target height (default: keep)")
    parser.add_argument("--algo", choices=list(FINDERS), default="dp", help="seam finder to use")
    parser.add_argument("--output", default="out/result.png", help="where to save the result")
    return parser.parse_args()


def main():
    args = parse_args()
    img = load_image(args.input)
    height, width = img.shape[:2]

    new_width = args.width or width
    new_height = args.height or height

    start = time.perf_counter()
    result = carve(img, new_width, new_height, get_finder(args.algo))
    elapsed = time.perf_counter() - start

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_image(result, args.output)

    seams = (width - new_width) + (height - new_height)
    print(f"{args.algo}: {width}x{height} -> {new_width}x{new_height}, "
          f"{seams} seams in {elapsed:.2f}s, saved to {args.output}")


if __name__ == "__main__":
    main()
