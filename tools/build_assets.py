#!/usr/bin/env python3
"""Build the demo assets for one photo: wall mask, wall lighting and scene.json.

    .venv/bin/python tools/build_assets.py

The wall corners and the outlines of everything standing in front of the wall were
measured by hand on original-images/3.jpg (GrabCut was tried, but the busy wallpaper
made it leak). The result is written to docs/assets/ and used by the WebGL demo in docs/.
"""

import hashlib
import json
import urllib.request
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
    "lego": [(963, 938), (1000, 936), (1004, 975), (960, 975)],
    "crate": [(890, 975), (1062, 975), (1062, 1040), (890, 1040)],
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

# Objects above that are toys: with "speelgoed opruimen" the wall behind them shows.
TOY_OBJECTS = {"lego", "toy_coffee", "bowl", "bucket_house", "boards", "frog", "plush"}

# Toys and other clutter to paint out of the photo for the tidy version (a few pixels
# generous). Large areas come last, so they can use the already cleaned-up surroundings.
TOYS = {
    "books": [(790, 927), (842, 927), (842, 952), (790, 952)],
    "papers": [(998, 919), (1092, 919), (1094, 957), (998, 957)],
    "lego": [(962, 928), (1034, 928), (1034, 1012), (943, 1014), (948, 966), (960, 966)],
    "cardboard": [(880, 968), (944, 968), (944, 1005), (880, 1005)],
    "toy_coffee": [(1082, 872), (1133, 872), (1133, 943), (1082, 943)],
    "bowl": [(1107, 934), (1158, 934), (1158, 962), (1107, 962)],
    "bucket_house": [(1143, 954), (1214, 954), (1216, 1082), (1143, 1082)],
    "boards": [(1196, 893), (1244, 893), (1246, 1088), (1196, 1088)],
    "yellow_toy": [(1090, 1028), (1166, 1028), (1222, 1090), (1222, 1136), (1118, 1138),
                   (1090, 1092)],
    "yellow_toy_2": [(1000, 1090), (1074, 1090), (1074, 1144), (1000, 1144)],
    "yellow_toy_3": [(1050, 1044), (1118, 1044), (1118, 1086), (1050, 1086)],
    "scooter": [(1298, 1036), (1330, 1036), (1378, 1001), (1446, 1001), (1446, 1034), (1428, 1040), (1428, 1100),
                (1422, 1146), (1338, 1152), (1296, 1132), (1290, 1088)],
    "floor_plush": [(1248, 1158), (1320, 1158), (1394, 1192), (1394, 1264), (1310, 1264),
                    (1248, 1210)],
    "frog": OBJECTS["frog"],
    "dresser_plush": OBJECTS["plush"],
    "yellow_block": [(308, 1233), (376, 1233), (376, 1272), (308, 1272)],
    "under_bench": [(52, 1176), (140, 1156), (200, 1146), (246, 1158), (250, 1215),
                    (205, 1262), (120, 1292), (52, 1292)],
    "toy_bucket": [(0, 1232), (160, 1232), (160, 1392), (0, 1392)],
    "plastic_bag": [(1494, 1016), (1634, 1016), (1634, 1102), (1494, 1102)],
    "bags_box_jacket": [(1900, 975), (2000, 975), (2000, 1500), (1668, 1500), (1668, 1340),
                        (1716, 1340), (1716, 1196), (1728, 1114), (1900, 1114)],
    # The rolled-up part of the rug.
    "rolled_mat": [(1049, 1059), (1095, 1056), (1183, 1126), (1250, 1126), (1255, 1188),
                   (1392, 1280), (1504, 1392), (1568, 1470), (1568, 1500), (1224, 1500),
                   (1168, 1364), (1105, 1224), (1056, 1126)],
}

# Items on the plain floor, filled first from the surrounding floor colour: LaMa read the
# floor under the rolled-up rug as a dark shadow and kept redrawing it as a dark band.
FLOOR_FILL = {"rolled_mat", "floor_plush"}

# The floor corner between crate, cabinet and rug is filled in one go (filling the pieces
# one by one made the model redraw bits of the toy and the rolled-up rug). Furniture is
# cut out of these outlines, so the crate stays.
CORNER = ["yellow_toy", "yellow_toy_2", "yellow_toy_3", "rolled_mat"]

# Floor items next to the rug: the rug is hidden from the model while filling these,
# otherwise it paints orange-yellow blobs onto the floor.
NO_RUG_CONTEXT = {"corner"}

# The rolled-up part of the rug is "unrolled": up to this outline the rug continues (its
# top-right corner was still visible), beyond it is floor.
RUG_TOP = ((380, 1112), (1250, 1128))     # far edge of the rug
RUG_AREA = [(150, 1095), (1270, 1095), (1700, 1500), (150, 1500)]  # where rug colours count as rug
RUG_RIGHT = ((1250, 1128), (1600, 1500))  # assumed right edge once unrolled

# LaMa inpainting model (Apache-2.0), downloaded once into .cache/ (about 200 MB).
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_PATH = ROOT / ".cache" / "lama_fp32.onnx"

# Wallpapered surfaces that can be painted another colour. Each gets its own lighting,
# so the brightness step at the wall corners stays sharp.
PAINT_SURFACES = {
    "tv_wall": WALL,
    # Side wall next to the dresser, up to the outer wall corner.
    "side_wall": [(1500, 441), (1600, 385), (1716, 334), (1720, 340), (1710, 370), (1708, 430),
                  (1705, 460), (1701, 490), (1695, 520), (1691, 550), (1688, 610), (1681, 640),
                  (1678, 670), (1675, 700), (1672, 730), (1670, 760), (1660, 835), (1649, 910),
                  (1650, 940), (1642, 970), (1640, 1000), (1636, 1030), (1632, 1086),
                  (1560, 1038), (1497, 1012)],
    # The narrow wall face beside and above the door.
    "door_wall": [(1716, 334), (2000, 330), (2000, 441), (1745, 441), (1740, 460), (1734, 490),
                  (1728, 520), (1725, 550), (1721, 580), (1711, 640), (1697, 730), (1686, 800),
                  (1677, 850), (1673, 910), (1670, 940), (1664, 970), (1661, 1000), (1658, 1030),
                  (1652, 1095), (1632, 1086), (1636, 1030), (1640, 1000), (1642, 970),
                  (1650, 940), (1649, 910), (1660, 835), (1670, 760), (1672, 730), (1675, 700),
                  (1678, 670), (1681, 640), (1688, 610), (1691, 550), (1695, 520), (1701, 490),
                  (1705, 460), (1708, 430), (1710, 370), (1720, 340)],
}

# Things on those surfaces that stay when the wall is painted.
PAINT_OBJECTS = {
    "owl": [(1560, 764), (1592, 764), (1606, 790), (1604, 830), (1594, 880), (1580, 903),
            (1552, 893), (1550, 840), (1548, 790)],
    "switch": [(1567, 720), (1600, 720), (1600, 757), (1567, 757)],
    "thermostat": [(1613, 729), (1641, 729), (1641, 758), (1613, 758)],
    "basket": [(1415, 1005), (1497, 1012), (1560, 1038), (1632, 1086), (1700, 1110),
               (1700, 1160), (1415, 1160)],
}

# Below the piano the wall is full of thin cables; dark pixels there are cables.
CABLE_AREA = [(388, 893), (720, 893), (720, 1005), (388, 1005)]


def poly(points):
    return np.array(points, dtype=np.int32)


def fill_floor(out, hole, remaining, blockers, scale=4, iterations=1500):
    """Fill holes on the plain floor by smoothly interpolating the floor around them.

    A harmonic fill (repeated neighbour averaging) at a quarter of the resolution, so the
    fill meets the real floor at every edge. The rug and other strongly coloured things
    are not used as edge values, so nothing orange bleeds in. Then floor-like mottling
    and grain are added on top.
    """
    h, w = hole.shape
    sat = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 1]
    usable = (remaining == 0) & (sat < 170) & (blockers == 0)
    small = (w // scale, h // scale)
    img = cv2.resize(out.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    unknown = cv2.resize((hole > 0).astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0
    known = cv2.resize(usable.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0.99
    known &= ~unknown
    active = unknown | known

    # Start from the mean floor colour around the holes, then relax.
    ring = cv2.dilate(unknown.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    field = img.copy()
    field[unknown] = img[ring & known].mean(axis=0)
    weight = active.astype(np.float32)
    for _ in range(iterations):
        num = (np.roll(field * weight[..., None], 1, 0) + np.roll(field * weight[..., None], -1, 0)
               + np.roll(field * weight[..., None], 1, 1) + np.roll(field * weight[..., None], -1, 1))
        den = np.roll(weight, 1, 0) + np.roll(weight, -1, 0) + np.roll(weight, 1, 1) + np.roll(weight, -1, 1)
        field[unknown] = (num / np.maximum(den, 1e-4)[..., None])[unknown]
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

    # Concrete-like mottling plus fine grain, instead of a perfectly smooth fill.
    rng = np.random.default_rng(1)
    mottle = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 5) * 10
    field += mottle[..., None]

    alpha = cv2.GaussianBlur((hole > 0).astype(np.float32), (0, 0), 2)[..., None]
    alpha[hole > 0] = 1
    out[:] = np.clip(out * (1 - alpha) + field * alpha, 0, 255).astype(np.uint8)


def fill_rug(out, hole, remaining):
    """Fill part of a hole with rug: copy real rug from the same row a bit to the left.

    Pixels in one row are at the same distance from the camera, so the weave there has
    the same size; a sideways copy keeps it sharp.
    """
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    rug = (hsv[..., 0] < 20) & (hsv[..., 1] > 180) & (hsv[..., 2] > 70) & (remaining == 0) & (hole == 0)
    src = out.astype(np.float32)
    done = np.zeros(hole.shape, bool)
    ys, xs = np.nonzero(hole)
    for y in np.unique(ys):
        row = xs[ys == y]
        width = row.max() - row.min() + 1
        for shift in (width + 8, width + 40, width + 80, 2 * width + 8):
            sx = row - shift
            if sx.min() >= 0 and rug[y, sx].mean() > 0.9:
                # Match the brightness of the rug right next to the hole.
                edge = row.min() - np.arange(1, 21)
                edge = edge[rug[y, edge]] if edge.min() >= 0 else edge[:0]
                gain = src[y, edge].mean(0) / np.maximum(src[y, sx[:20]].mean(0), 1) if len(edge) else 1
                out[y, row] = np.clip(src[y, sx] * gain, 0, 255).astype(np.uint8)
                done[y, row] = True
                break
    return done


def wallpaper_detail(tidy, img, holes, wall_ok, src_ok, sigma=8):
    """Give inpainted wallpaper its pattern back.

    LaMa gets the colour and light right but blurs the busy wallpaper. Keep its low
    frequencies and take the pattern from real wallpaper a little to the left or right
    (the back wall faces the camera, so a sideways copy fits).
    """
    target = (holes > 0) & wall_ok
    count, labels, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8))
    img_f, tidy_f = img.astype(np.float32), tidy.astype(np.float32)
    detail = img_f - cv2.GaussianBlur(img_f, (0, 0), sigma)
    base = cv2.GaussianBlur(tidy_f, (0, 0), sigma)
    w = img.shape[1]
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        ys, xs = np.nonzero(labels == i)
        for step in (1, -1, 2, -2, 3, -3, 4, -4):
            dx = step * (bw + 20)
            sx = xs + dx
            if sx.min() < 0 or sx.max() >= w or src_ok[ys, sx].mean() < 0.97:
                continue
            tidy_f[ys, xs] = base[ys, xs] + detail[ys, sx]
            break
    tidy[:] = np.clip(tidy_f, 0, 255).astype(np.uint8)


def add_grain(out, img, filled, known):
    """Add photo grain to filled areas, as strong as in the real photo around them."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    k = known.astype(np.float32)
    local = np.sqrt(cv2.GaussianBlur(hf * hf * k, (0, 0), 30) / np.maximum(cv2.GaussianBlur(k, (0, 0), 30), 1e-4))
    local = np.minimum(local, 3.0) * 0.8  # edges nearby would otherwise make it too coarse
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, gray.shape).astype(np.float32)
    noise = noise - cv2.GaussianBlur(noise, (0, 0), 1.5)
    noise /= noise.std()
    alpha = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 2)
    grain = (noise * local * alpha)[..., None]
    out[:] = np.clip(out.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def inpaint_toys(img, furniture):
    """Paint the clutter out with LaMa, one crop around each item scaled to 512x512.

    The crop is 2.5 times the item (capped for the big ones, which then come out a bit
    softer); that leaves enough room around it for the model to see what belongs there.
    """
    import onnxruntime as ort

    if not LAMA_PATH.exists():
        print("downloading", LAMA_URL)
        LAMA_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(LAMA_URL, LAMA_PATH)
    session = ort.InferenceSession(str(LAMA_PATH), providers=["CPUExecutionProvider"])

    h, w = img.shape[:2]
    # Generous: a few leftover pixels of an item are enough for the model to redraw it.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    def item_mask(name):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly(TOYS[name])], 255)
        m = cv2.dilate(m, grow)
        if name in CORNER:
            m[furniture > 0] = 0
        return m

    # Everything not yet filled stays a hole, so clutter is never copied in as context.
    remaining = np.zeros((h, w), np.uint8)
    for name in TOYS:
        remaining |= item_mask(name)

    out = img.copy()
    floor = np.zeros((h, w), np.uint8)
    for name in FLOOR_FILL:
        cv2.fillPoly(floor, [poly(TOYS[name])], 255)
    floor = cv2.dilate(floor, grow)
    (ax, ay), (bx, by) = RUG_TOP
    (cx, cy), (dx, dy) = RUG_RIGHT
    yy, xx = np.mgrid[0:h, 0:w]
    below_top = yy >= ay + (xx - ax) * (by - ay) / (bx - ax)
    on_rug = below_top & (xx <= cx + (yy - cy) * (dx - cx) / (dy - cy))
    rug = ((floor > 0) & on_rug).astype(np.uint8) * 255
    # Also redo a strip of the real rug along the old edge, which was shaded by the roll.
    rug = (cv2.dilate(rug, np.ones((1, 25), np.uint8)) > 0) & on_rug
    rug = rug.astype(np.uint8) * 255
    # Leave other items (the toy's tail lies across the rug edge) to their own pass.
    others = np.zeros((h, w), np.uint8)
    for name in TOYS:
        if name not in FLOOR_FILL:
            others |= item_mask(name)
    rug[others > 0] = 0
    floor[others > 0] = 0
    # Open floor only; the strip above the rug, between the furniture, is left to LaMa.
    floor[(rug > 0) | ~below_top] = 0
    # Furniture and rug are no edge values for the floor.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    rugish = (hsv[..., 0] < 13) & (hsv[..., 1] > 170)
    blockers = (cv2.dilate(furniture, np.ones((9, 9), np.uint8)) > 0) | rugish
    fill_floor(out, floor, remaining, blockers.astype(np.uint8))
    remaining[floor > 0] = 0
    remaining[fill_rug(out, rug, remaining)] = 0

    # The rug as it is now (unrolled), to hide from the model for floor items next to it.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    rug_area = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rug_area, [poly(RUG_AREA)], 255)
    rug_core = (hsv[..., 0] < 13) & (hsv[..., 1] > 170) & (hsv[..., 2] > 70)  # not the yellow toy
    # A dozen pixels below the edge: the toy's tail lies over the edge and, in the shade,
    # has the rug's colour.
    rug_core &= (rug_area > 0) & (yy >= ay + 12 + (xx - ax) * (by - ay) / (bx - ax))
    rug_hide = cv2.dilate(rug_core.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))

    def corner_mask():
        m = np.zeros((h, w), np.uint8)
        for name in CORNER:
            m |= item_mask(name)
        return m

    for name in ["corner"] + [n for n in TOYS if n not in CORNER]:
        own = cv2.bitwise_and(corner_mask() if name == "corner" else item_mask(name), remaining)
        if name in NO_RUG_CONTEXT:  # keep the real rug where an outline overlaps it
            remaining[(own > 0) & rug_core] = 0
            own[rug_core] = 0
        rug_done = np.zeros((h, w), bool)
        if name == "corner":  # where the toy lay across the rug's edge, put rug back
            rug_done = fill_rug(out, ((own > 0) & on_rug).astype(np.uint8) * 255, remaining)
        x, y, bw, bh = cv2.boundingRect(own)
        if bw == 0:  # already filled above
            continue
        side = min(max(640 if name in NO_RUG_CONTEXT else 256, int(max(bw, bh) * 2.5)), 960, h)
        x0 = int(np.clip(x + bw // 2 - side // 2, 0, w - side))
        y0 = int(np.clip(y + bh // 2 - side // 2, 0, h - side))
        win = (slice(y0, y0 + side), slice(x0, x0 + side))

        rgb = cv2.cvtColor(out[win], cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        hide = remaining[win] if name not in NO_RUG_CONTEXT else cv2.bitwise_or(remaining[win], rug_hide[win])
        mask = (cv2.resize(hide, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        image *= 1 - mask[..., None]  # the exported model does not blank the hole itself
        result = session.run(None, {"image": image.transpose(2, 0, 1)[None], "mask": mask[None, None]})[0]
        result = np.clip(result[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)  # output is 0-255
        result = cv2.resize(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), (side, side), interpolation=cv2.INTER_CUBIC)

        alpha = cv2.GaussianBlur(own[win], (0, 0), 2.5).astype(np.float32)[..., None] / 255
        alpha[rug_done[win]] = 0  # keep floor colour off the rug that was just put back
        out[win] = (out[win] * (1 - alpha) + result * alpha).astype(np.uint8)
        remaining[win][own[win] > 0] = 0
    return out


def main():
    img = cv2.imread(str(PHOTO))
    h, w = img.shape[:2]

    wall = np.zeros((h, w), np.uint8)
    cv2.fillPoly(wall, [poly(WALL)], 255)

    def fill(polygons):
        out = np.zeros((h, w), np.uint8)
        for points in polygons:
            cv2.fillPoly(out, [poly(points)], 255)
        return out

    objects = fill(OBJECTS.values())
    furniture = fill(v for key, v in OBJECTS.items() if key not in TOY_OBJECTS)

    # Thin cables: clearly darker than the wallpaper around them.
    cable_area = np.zeros((h, w), np.uint8)
    cv2.fillPoly(cable_area, [poly(CABLE_AREA)], 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 12)
    cables = ((gray < 75) & (gray < local - 30) & (cable_area > 0)).astype(np.uint8) * 255
    cables = cv2.dilate(cables, np.ones((3, 3), np.uint8))

    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.bitwise_or(objects, cables)

    # Paint goes where wallpaper shows: on any wallpapered surface, not behind objects.
    surfaces = {name: fill([points]) for name, points in PAINT_SURFACES.items()}
    papered = cv2.dilate(fill(PAINT_SURFACES.values()), k(2))
    # Beside the door the orange wallpaper is told apart by colour from the door frame and
    # the drawings, magnets and cards stuck over it.
    hsv = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 0), 1.5)
    orange = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 18) & (hsv[..., 1] > 115) & (hsv[..., 2] > 70)
    orange = cv2.morphologyEx(orange.astype(np.uint8) * 255, cv2.MORPH_OPEN, k(2))
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, k(3))
    door_wall = cv2.dilate(surfaces["door_wall"], k(2))
    papered[(door_wall > 0) & (cv2.dilate(surfaces["side_wall"], k(2)) == 0) & (orange == 0)] = 0
    on_wall = fill(PAINT_OBJECTS.values())
    paint_fg = objects | fill(TOYS.values()) | cables | on_wall
    paint_fg_tidy = furniture | cables | on_wall

    # The object outlines are a little generous. Wallpaper in that margin (colour within
    # the local spread of the wallpaper next to it) counts as wall again. Measured on the
    # photo as shot, with everything in place, and used for the tidy masks as well.
    lab_f = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    known = ((papered > 0) & (paint_fg == 0)).astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(known, (0, 0), 20), 1e-4)[..., None]
    mean = cv2.GaussianBlur(lab_f * known[..., None], (0, 0), 20) / den
    var = cv2.GaussianBlur(lab_f ** 2 * known[..., None], (0, 0), 20) / den - mean ** 2
    z = np.abs(lab_f - mean) / np.sqrt(np.maximum(var, 0) + 4)
    near = cv2.dilate(known.astype(np.uint8), k(10)) > 0
    # Brightness may differ a bit more (contact shadows), colour hardly.
    like = (z[..., 0] < 2.8) & (z[..., 1:].max(axis=2) < 2.0)
    margin = ((paint_fg > 0) & like & near).astype(np.uint8) * 255
    margin = cv2.morphologyEx(margin, cv2.MORPH_OPEN, k(1))

    def trim(fg):
        return cv2.bitwise_and(fg, cv2.bitwise_not(margin))

    # Panels are visible where the wall is and no object is in front of it.
    def visible_mask(fg):
        visible = cv2.bitwise_and(cv2.dilate(wall, k(3)), cv2.bitwise_not(trim(fg)))
        return cv2.GaussianBlur(visible, (0, 0), 0.8)

    visible = visible_mask(fg)
    visible_tidy = visible_mask(cv2.bitwise_or(furniture, cables))

    def paint_mask(fg):
        return cv2.GaussianBlur(cv2.bitwise_and(papered, cv2.bitwise_not(trim(fg))), (0, 0), 0.8)

    # Mask images: red = where panels show, green = where paint shows.
    def mask_image(visible, fg):
        return cv2.merge([np.zeros_like(visible), paint_mask(fg), visible])

    tidy = inpaint_toys(img, furniture)
    holes = fill(TOYS.values())
    wall_ok = (wall > 0) & (furniture == 0)
    src_ok = (wall > 0) & (objects == 0) & (holes == 0) & (cables == 0)
    holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    wallpaper_detail(tidy, img, holes, wall_ok, src_ok)
    filled = holes > 0
    add_grain(tidy, img, filled & ~wall_ok, ~filled)

    # Wall lighting: wallpaper luminance with the pattern blurred away. Normalised
    # convolution over wall pixels only, so objects do not bleed into it.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32) / 255.0

    def lighting(px, sigma=18):
        num = cv2.GaussianBlur(lum * px, (0, 0), sigma)
        den = cv2.GaussianBlur(px, (0, 0), sigma)
        light = num / np.maximum(den, 1e-4)
        # Fill the far gaps (behind big objects) from a wider blur.
        num2 = cv2.GaussianBlur(lum * px, (0, 0), sigma * 4)
        den2 = cv2.GaussianBlur(px, (0, 0), sigma * 4)
        light2 = num2 / np.maximum(den2, 1e-4)
        t = np.clip(den * 4, 0, 1)
        light = light * t + light2 * (1 - t)
        # Light colour (warm tint of the room light), from the same wall pixels.
        tint = cv2.GaussianBlur(img.astype(np.float32) * px[..., None], (0, 0), sigma * 4)
        tint = tint / np.maximum(den2[..., None], 1e-4)
        tint = tint / np.maximum(tint.mean(axis=2, keepdims=True), 1e-4)
        # The wallpaper is yellowish itself, so keep only a hint of its colour as light tint.
        return light, 1 + (tint - 1) * 0.3

    # Leave out a margin around objects: their contact shadows would otherwise show up as
    # dark smudges where the toys are painted out.
    shadowed = cv2.dilate(paint_fg, k(8))
    wall_px = ((wall > 0) & (shadowed == 0)).astype(np.float32)
    light, tint = lighting(wall_px)
    # The other wallpapered surfaces get their own lighting, used by the paint only.
    for name, surface in surfaces.items():
        if name == "tv_wall":
            continue
        inside = cv2.dilate(surface, k(3)) > 0
        light_s, tint_s = lighting(((surface > 0) & (shadowed == 0)).astype(np.float32))
        light[inside] = light_s[inside]
        tint[inside] = tint_s[inside]

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "photo.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "mask.png"), mask_image(visible, paint_fg))
    cv2.imwrite(str(OUT / "photo-tidy.jpg"), tidy, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "mask-tidy.png"), mask_image(visible_tidy, paint_fg_tidy))
    # Stored relative to the average wall brightness: 128 = average.
    light = light / light[wall_px > 0].mean()
    light_rgb = np.clip(tint * light[..., None] * 0.5 * 255, 0, 255).astype(np.uint8)
    # The light map is smooth, so half resolution is plenty.
    light_rgb = cv2.resize(light_rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "light.png"), light_rgb)

    # Content hash in the URL, so browsers never keep an old image after a rebuild.
    def url(name):
        return f"assets/{name}?v={hashlib.md5((OUT / name).read_bytes()).hexdigest()[:8]}"

    scene = {
        "photo": url("photo.jpg"),
        "mask": url("mask.png"),
        "light": url("light.png"),
        "tidyPhoto": url("photo-tidy.jpg"),
        "tidyMask": url("mask-tidy.png"),
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
