import cv2
import numpy as np
import random


def make_sample_image(width=640, height=400):
    """
    draws a fake overhead highway scene with coloured cars.
    good enough to test detection + colour classification without needing
    a real photo.
    """
    img = np.full((height, width, 3), (110, 115, 105), dtype=np.uint8)

    # road
    road_top = int(height * 0.15)
    road_bot = int(height * 0.88)
    cv2.rectangle(img, (0, road_top), (width, road_bot), (90, 95, 88), -1)

    # lane markings
    lane_xs = [width//4, width//2, 3*width//4]
    for lx in lane_xs:
        for y in range(road_top + 20, road_bot - 20, 50):
            cv2.line(img, (lx, y), (lx, y+28), (220, 220, 200), 2)

    # cars: (x, y, w, h, BGR colour, label)
    cars = [
        (60,  200, 100, 55, (180, 60, 40),   "blue"),
        (200, 210,  90, 50, (50,  50, 50),   "black"),
        (340, 205, 105, 58, (200, 200, 200), "silver"),
        (490, 200,  95, 52, (30,  30, 200),  "red"),
        (80,  300, 100, 55, (200, 200, 230), "white"),
        (240, 295,  90, 50, (30, 160,  30),  "green"),
        (400, 300, 100, 55, (20, 200, 220),  "yellow"),
        (150, 145,  70, 38, (160, 90, 40),   "blue2"),
        (370, 148,  75, 40, (80, 80, 80),    "dark"),
    ]

    for (x, y, w, h, colour, _) in cars:
        # body
        cv2.rectangle(img, (x, y), (x+w, y+h), colour, -1)
        # windscreen (slightly lighter)
        ws_x1 = x + int(w*0.18)
        ws_x2 = x + int(w*0.82)
        ws_y1 = y + int(h*0.12)
        ws_y2 = y + int(h*0.45)
        ws_col = tuple(min(255, c+60) for c in colour)
        cv2.rectangle(img, (ws_x1, ws_y1), (ws_x2, ws_y2), ws_col, -1)
        # tyres
        for tx in [x + int(w*0.15), x + int(w*0.75)]:
            cv2.circle(img, (tx, y+h), 7, (20, 20, 20), -1)

    # add a tiny person on the shoulder
    px, py = 30, 280
    cv2.ellipse(img, (px, py-22), (6, 8), 0, 0, 360, (120, 100, 80), -1)   # head
    cv2.rectangle(img, (px-5, py-14), (px+5, py+10), (60, 80, 160), -1)    # body
    cv2.line(img, (px, py+10), (px-4, py+24), (60, 80, 160), 3)            # legs
    cv2.line(img, (px, py+10), (px+4, py+24), (60, 80, 160), 3)

    # slight noise so it doesn't look completely flat
    noise = np.random.randint(-12, 12, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
