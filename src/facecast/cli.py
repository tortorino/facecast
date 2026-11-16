"""Command line interface: `facecast upscale photo.jpg -m gan -o face.png [--roi LEFT TOP RIGHT BOTTOM]`."""

import argparse
import sys
from pathlib import Path

from .weights import DEFAULT_MODEL, MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="facecast", description="Face super-resolution, 16x16 -> 128x128.")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upscale", help="upscale a square picture of a face")
    up.add_argument("input", help="a square picture (it is resized to 16x16 first)")
    up.add_argument("-o", "--output", help="where to save the result (default: <input>_facecast.png)")
    up.add_argument("-m", "--model", default=DEFAULT_MODEL, help="one of: %s, or a .safetensors file (default: %s)"
                    % (", ".join(MODELS), DEFAULT_MODEL))
    up.add_argument("--roi", nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    help="only upscale this region of the picture (in pixels), e.g. a small face in a large photo")
    up.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")

    sub.add_parser("models", help="list the bundled models")

    args = parser.parse_args(argv)

    if args.command == "models":
        for name, description in MODELS.items():
            print("%-14s %s" % (name, description))
        return 0

    # Imported here, so `facecast models` and `--help` stay fast
    from .upscaler import FaceUpscaler

    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name(source.stem + "_facecast.png")
    try:
        upscaler = FaceUpscaler.from_pretrained(args.model, device=args.device)
        upscaler.upscale(source, roi=args.roi).save(output)
    except (ValueError, OSError) as e:
        print("facecast: %s" % e, file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
