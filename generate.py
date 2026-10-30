#!/usr/bin/env python3
"""Add wooden acoustic slat panels to a living-room photo with the OpenAI image edit API.

Usage:
    python generate.py original-images/3.jpg
    python generate.py original-images/3.jpg --wood walnut --coverage tv
    python generate.py original-images/3.jpg --reference panel.jpg --n 2
    python generate.py original-images/3.jpg --all-variants
    python generate.py original-images/3.jpg --dry-run   # only print the prompt

Reads OPENAI_API_KEY from the environment or from a .env file next to this script.
Results are written to output/.
"""

import argparse
import base64
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

WOODS = {
    "oak": "light natural oak with a matte finish",
    "walnut": "warm dark walnut with a matte finish",
    "black": "black-stained oak with a matte finish, wood grain still visible",
}

COVERAGES = {
    "wall": (
        "Cover the entire back wall with the panels: from the curtain on the left "
        "to the inside corner on the right, and from the floor up to the ceiling."
    ),
    "tv": (
        "Cover only a section of the back wall, centred behind the TV: about one and a "
        "half times as wide as the white TV cabinet, running from the floor up to the "
        "ceiling. The rest of the back wall keeps its current wallpaper."
    ),
}

PROMPT = """\
Edit this photo of a living room. Add wooden acoustic slat panels to the back wall \
(the wall behind the TV) and change nothing else.

Panels: vertical wooden slats in {wood}, each slat about 2.7 cm wide with gaps of \
about 1.3 cm, mounted on black acoustic felt that is visible in the gaps. The slats \
run perfectly vertical and follow the perspective of the wall. {coverage}
{reference}
Keep everything in front of the wall exactly as it is and in front of the panels: \
the TV and its picture, the white TV cabinet, both speakers, the electric piano and \
its stand, the globe, the guitar, the white chest of drawers with the photo frames, \
plants and decorations, the cables and wall sockets, the curtains, the sofa, the rug, \
the coffee table and all toys and objects on the floor. Do not tidy up, move, add or \
remove anything.

Keep the side wall on the right, the door, the ceiling, the ceiling light and the \
floor unchanged. Keep the same camera angle, framing, lens distortion, lighting, \
colour temperature and shadows. The panels must be lit by the existing light in the \
room, with soft shadows between the slats, so the result looks like a real \
photograph and not a render.
"""

REFERENCE = (
    "The second image shows an example of the panel style; match its slat shape and "
    "spacing, but use the wood colour described above.\n"
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_prompt(wood: str, coverage: str, has_reference: bool) -> str:
    return PROMPT.format(
        wood=WOODS[wood],
        coverage=COVERAGES[coverage],
        reference=REFERENCE if has_reference else "",
    )


def generate(client, args, photo: Path, wood: str, coverage: str) -> None:
    prompt = build_prompt(wood, coverage, args.reference is not None)
    if args.dry_run:
        print(f"--- {photo.name} / {wood} / {coverage} ---\n{prompt}")
        return

    images = [photo] + ([args.reference] if args.reference else [])
    files = [p.open("rb") for p in images]
    try:
        print(f"Generating {photo.name} / {wood} / {coverage} ({args.n}x)...", flush=True)
        result = client.images.edit(
            model=args.model,
            image=files,
            prompt=prompt,
            size=args.size,
            quality=args.quality,
            n=args.n,
        )
    finally:
        for f in files:
            f.close()

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i, item in enumerate(result.data, start=1):
        out = OUTPUT_DIR / f"{photo.stem}-{wood}-{coverage}-{stamp}-{i}.png"
        out.write_bytes(base64.b64decode(item.b64_json))
        print(f"  saved {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", type=Path, help="photo of the room to edit")
    parser.add_argument("--wood", choices=WOODS, default="oak")
    parser.add_argument("--coverage", choices=COVERAGES, default="wall")
    parser.add_argument("--all-variants", action="store_true", help="generate every wood/coverage combination")
    parser.add_argument("--reference", type=Path, help="optional example photo of the panels")
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
    parser.add_argument("--size", default="2048x1536", help="WIDTHxHEIGHT, both divisible by 16 (default keeps 4:3)")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--n", type=int, default=1, help="images per variant")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt(s) without calling the API")
    args = parser.parse_args()

    for path in [args.photo] + ([args.reference] if args.reference else []):
        if not path.exists():
            sys.exit(f"File not found: {path}")

    client = None
    if not args.dry_run:
        load_dotenv(ROOT / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set (export it or put it in .env)")
        from openai import OpenAI

        client = OpenAI()

    variants = (
        itertools.product(WOODS, COVERAGES) if args.all_variants else [(args.wood, args.coverage)]
    )
    for wood, coverage in variants:
        generate(client, args, args.photo, wood, coverage)


if __name__ == "__main__":
    main()
