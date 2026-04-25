import cv2
import numpy as np

# HSV ranges for each colour.
# red wraps around 180 so needs two entries.
COLOR_RANGES = {
    "red":    [((0,  80,  60), (10,  255, 255)),
               ((165, 80, 60), (180, 255, 255))],
    "blue":   [((95,  55,  35), (135, 255, 255))],
    "white":  [((0,   0, 175), (180,  35, 255))],
    "black":  [((0,   0,   0), (180, 255,  55))],
    "silver": [((0,   0, 120), (180,  28, 200))],
    "yellow": [((18, 80, 100), (35,  255, 255))],
    "green":  [((36, 50,  50), (90,  255, 255))],
}


def get_dominant_color(roi):
    if roi is None or roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # skip very dark pixels (tyres, shadows) and very bright ones (sky glare)
    useful = cv2.inRange(v, 25, 245)

    scores = {}
    for name, ranges in COLOR_RANGES.items():
        total = 0
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.bitwise_and(m, useful)
            total += cv2.countNonZero(m)
        scores[name] = total

    best = max(scores, key=scores.get)
    if scores[best] < 80:
        return "unknown"
    return best


def color_to_bgr(name):
    palette = {
        "red":     (0,   0,   220),
        "blue":    (220, 80,   0),
        "white":   (230, 230, 230),
        "black":   (30,  30,   30),
        "silver":  (180, 180, 180),
        "yellow":  (0,   210, 230),
        "green":   (0,   180,   0),
        "unknown": (128, 128, 128),
    }
    return palette.get(name, (128, 128, 128))
