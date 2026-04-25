import cv2
import numpy as np
from utils.color_detector import get_dominant_color, color_to_bgr
from utils.object_detector import detect_cars, detect_people


def process_image(img):
    """
    runs car + person detection, draws boxes, returns annotated image + stats.
    returns: (annotated_img, stats_dict)
    """
    output = img.copy()
    ih, iw = img.shape[:2]

    car_boxes   = detect_cars(img)
    person_boxes = detect_people(img)

    color_counts = {}
    blue_car_count = 0
    other_car_count = 0

    for (x, y, bw, bh) in car_boxes:
        # crop with small inset so we don't include the road around the car
        pad = 4
        cx1 = max(0, x + pad)
        cy1 = max(0, y + pad)
        cx2 = min(iw, x + bw - pad)
        cy2 = min(ih, y + bh - pad)
        roi = img[cy1:cy2, cx1:cx2]

        car_color = get_dominant_color(roi)
        color_counts[car_color] = color_counts.get(car_color, 0) + 1

        # task requirement: red box for blue cars, blue box for others
        if car_color == "blue":
            box_color = (0, 0, 220)   # red in BGR
            blue_car_count += 1
        else:
            box_color = (220, 80, 0)  # blue in BGR
            other_car_count += 1

        cv2.rectangle(output, (x, y), (x+bw, y+bh), box_color, 2)

        label = car_color
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(output, (x, y-th-6), (x+tw+4, y), box_color, -1)
        cv2.putText(output, label, (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    for (x, y, bw, bh) in person_boxes:
        cv2.rectangle(output, (x, y), (x+bw, y+bh), (0, 200, 0), 2)
        cv2.putText(output, "person", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1, cv2.LINE_AA)

    stats = {
        "total_cars":   len(car_boxes),
        "blue_cars":    blue_car_count,
        "other_cars":   other_car_count,
        "people":       len(person_boxes),
        "color_counts": color_counts,
    }
    return output, stats
