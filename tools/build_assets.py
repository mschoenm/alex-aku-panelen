#!/usr/bin/env python3
"""Build the demo assets for one photo: wall mask, wall lighting and scene.json.

    .venv/bin/python tools/build_assets.py

The wall corners and the outlines of everything standing in front of the wall were
measured by hand on original-images/3.jpg (GrabCut was tried, but the busy wallpaper
made it leak). The result is written to docs/assets/ and used by the WebGL demo in docs/.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / "original-images" / "3.jpg"
OUT = ROOT / "docs" / "assets"

# Back wall corners: top-left, top-right, bottom-right, bottom-left (pixels in the photo).
WALL = [(388, 426), (1500, 441), (1497, 1010), (378, 1000)]
# Real size of the wall in metres (width, height), used to scale the slats.
WALL_SIZE_M = (4.6, 2.6)

# Objects in front of the wall, as polygons.
OBJECTS = {
    "curtain": [(0, 300), (392, 300), (398, 430), (392, 700), (384, 900), (378, 1100), (0, 1100)],
    "piano": [(401, 838), (650, 834), (652, 893), (365, 893), (363, 878)],
    "piano_leg_l": [(375, 890), (387, 890), (387, 1010), (375, 1010)],
    "piano_leg_rl": [(445, 890), (455, 890), (455, 1000), (445, 1000)],
    "piano_leg_fr": [(583, 890), (594, 890), (594, 1010), (583, 1010)],
    "piano_leg_rr": [(634, 890), (641, 890), (641, 1000), (634, 1000)],
    "globe": [(593, 772), (606, 764), (627, 761), (640, 778), (645, 800), (636, 820),
              (618, 822), (616, 834), (630, 836), (630, 842), (592, 842), (596, 834),
              (607, 830), (590, 815), (587, 790)],
    "sockets": [(496, 935), (560, 935), (560, 956), (496, 956)],
    "guitar": [(653, 846), (658, 840), (668, 842), (674, 855), (679, 935), (698, 946),
               (703, 962), (708, 1010), (622, 1010), (625, 975), (632, 955), (651, 935)],
    "speaker_l": [(752, 868), (772, 868), (776, 858), (781, 868), (796, 868), (796, 935), (752, 935)],
    "tv": [(821, 792), (1048, 792), (1048, 932), (821, 932)],
    "cabinet": [(718, 925), (1148, 934), (1148, 1040), (718, 1040)],
    "speaker_r": [(1046, 872), (1084, 872), (1084, 938), (1046, 938)],
    "toy_coffee": [(1083, 875), (1131, 875), (1131, 941), (1083, 941)],
    "bowl": [(1110, 934), (1159, 934), (1159, 958), (1110, 958)],
    "crate_lego": [(963, 938), (1000, 936), (1004, 975), (1062, 975), (1062, 1040),
                   (890, 1040), (890, 975), (960, 975)],
    "bucket_house": [(1146, 955), (1216, 955), (1216, 1070), (1146, 1070)],
    "boards": [(1198, 895), (1242, 895), (1242, 1070), (1198, 1070)],
    "dresser": [(1222, 757), (1525, 776), (1517, 1030), (1224, 1090)],
    "frog": [(1212, 718), (1228, 716), (1240, 724), (1236, 740), (1222, 760), (1214, 790),
             (1204, 790), (1206, 750)],
    "plush": [(1226, 730), (1250, 728), (1266, 740), (1270, 770), (1226, 770)],
    "orchid_stem_l": [(1281, 604), (1284, 604), (1287, 692), (1284, 692)],
    "orchid_stem_r": [(1301, 598), (1305, 598), (1302, 692), (1298, 692)],
    "orchid_pot": [(1276, 690), (1300, 690), (1305, 725), (1302, 770), (1268, 770), (1266, 712)],
    "frames": [(1296, 625), (1394, 625), (1395, 768), (1296, 768)],
    "flowers": [(1436, 690), (1442, 672), (1460, 672), (1480, 690), (1500, 697), (1512, 700),
                (1512, 770), (1445, 770), (1440, 740), (1432, 715)],
}

# Below the piano the wall is full of thin cables; dark pixels there are cables.
CABLE_AREA = [(388, 893), (720, 893), (720, 1005), (388, 1005)]


def poly(points):
    return np.array(points, dtype=np.int32)


def main():
    img = cv2.imread(str(PHOTO))
    h, w = img.shape[:2]

    wall = np.zeros((h, w), np.uint8)
    cv2.fillPoly(wall, [poly(WALL)], 255)

    objects = np.zeros((h, w), np.uint8)
    for points in OBJECTS.values():
        cv2.fillPoly(objects, [poly(points)], 255)

    # Thin cables: clearly darker than the wallpaper around them.
    cable_area = np.zeros((h, w), np.uint8)
    cv2.fillPoly(cable_area, [poly(CABLE_AREA)], 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 12)
    cables = ((gray < 75) & (gray < local - 30) & (cable_area > 0)).astype(np.uint8) * 255
    cables = cv2.dilate(cables, np.ones((3, 3), np.uint8))

    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.bitwise_or(objects, cables)

    # Panels are visible where the wall is and no object is in front of it.
    visible = cv2.bitwise_and(cv2.dilate(wall, k(3)), cv2.bitwise_not(fg))
    visible = cv2.GaussianBlur(visible, (0, 0), 0.8)

    # Wall lighting: wallpaper luminance with the pattern blurred away. Normalised
    # convolution over wall pixels only, so objects do not bleed into it.
    wall_px = ((wall > 0) & (fg == 0)).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32) / 255.0
    sigma = 18
    num = cv2.GaussianBlur(lum * wall_px, (0, 0), sigma)
    den = cv2.GaussianBlur(wall_px, (0, 0), sigma)
    light = num / np.maximum(den, 1e-4)
    # Fill the far gaps (behind big objects) from a wider blur.
    num2 = cv2.GaussianBlur(lum * wall_px, (0, 0), sigma * 4)
    den2 = cv2.GaussianBlur(wall_px, (0, 0), sigma * 4)
    light2 = num2 / np.maximum(den2, 1e-4)
    t = np.clip(den * 4, 0, 1)
    light = light * t + light2 * (1 - t)
    # Light colour (warm tint of the room light), from the same wall pixels.
    tint = cv2.GaussianBlur(img.astype(np.float32) * wall_px[..., None], (0, 0), sigma * 4)
    tint = tint / np.maximum(den2[..., None], 1e-4)
    tint = tint / np.maximum(tint.mean(axis=2, keepdims=True), 1e-4)
    # The wallpaper is yellowish itself, so keep only a hint of its colour as light tint.
    tint = 1 + (tint - 1) * 0.3

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "photo.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "mask.png"), visible)
    # Stored relative to the average wall brightness: 128 = average.
    light = light / light[wall_px > 0].mean()
    light_rgb = np.clip(tint * light[..., None] * 0.5 * 255, 0, 255).astype(np.uint8)
    # The light map is smooth, so half resolution is plenty.
    light_rgb = cv2.resize(light_rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "light.png"), light_rgb)

    scene = {
        "photo": "assets/photo.jpg",
        "mask": "assets/mask.png",
        "light": "assets/light.png",
        "width": w,
        "height": h,
        "wall": WALL,
        "wallSizeM": WALL_SIZE_M,
        "tvCenterX": 935,
        "wallLuminance": round(float(lum[wall_px > 0].mean()), 3),
    }
    (OUT / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    print("wrote", ", ".join(p.name for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
