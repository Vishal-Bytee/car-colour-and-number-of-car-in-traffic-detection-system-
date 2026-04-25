import cv2
import numpy as np

# hsv ranges for each colour
# red is annoying because it wraps around 180 in hsv so needs two ranges
COLOUR_RANGES = {
    "red":    [((0,  80,  60), (10,  255, 255)),
               ((165, 80, 60), (180, 255, 255))],
    "blue":   [((95,  55, 35), (135, 255, 255))],
    "white":  [((0,   0, 175), (180,  30, 255))],
    "black":  [((0,   0,   0), (180, 255,  50))],
    "silver": [((0,   0, 120), (180,  25, 200))],
    "yellow": [((18, 80, 100), (35,  255, 255))],
    "green":  [((36, 50,  50), (90,  255, 255))],
}


def get_car_colour(roi):
    if roi is None or roi.size == 0:
        return "unknown"

    # resize to speed things up a bit
    if roi.shape[1] > 100 or roi.shape[0] > 100:
        roi = cv2.resize(roi, (100, 100))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ignore pixels that are basically pure shadow or pure glare
    v = hsv[:, :, 2]
    good_pixels = cv2.inRange(v, 30, 240)

    scores = {}
    for name, ranges in COLOUR_RANGES.items():
        total = 0
        for lo, hi in ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.bitwise_and(mask, good_pixels)
            total += cv2.countNonZero(mask)
        scores[name] = total

    best = max(scores, key=scores.get)
    if scores[best] < 60:
        return "unknown"
    return best
