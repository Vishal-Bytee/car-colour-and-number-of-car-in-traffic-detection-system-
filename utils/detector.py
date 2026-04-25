import cv2
from utils.color_classifier import get_car_colour
from utils.model_loader import CAR_CLASSES, PERSON_CLASS


def detect_and_annotate(model, img):
    out = img.copy()
    ih, iw = img.shape[:2]

    # run yolo on the image, confidence 0.35 works well for traffic scenes
    results = model(img, conf=0.35, verbose=False)
    boxes = results[0].boxes

    total_cars = 0
    blue_cars = 0
    other_cars = 0
    people = 0
    color_counts = {}

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # clamp coords to image size
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(iw, x2); y2 = min(ih, y2)

        if cls == PERSON_CLASS:
            people += 1
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 2)
            draw_label(out, f"person {conf:.0%}", x1, y1, (0, 200, 80))
            continue

        if cls not in CAR_CLASSES:
            continue

        total_cars += 1

        # small inset so we dont include road pixels at the edge of the box
        pad = 6
        roi = img[y1+pad:y2-pad, x1+pad:x2-pad]
        colour = get_car_colour(roi)

        color_counts[colour] = color_counts.get(colour, 0) + 1

        # task says: red box for blue cars, blue box for everything else
        if colour == "blue":
            box_col = (0, 0, 220)
            blue_cars += 1
        else:
            box_col = (220, 100, 0)
            other_cars += 1

        cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 2)
        draw_label(out, f"{colour} {conf:.0%}", x1, y1, box_col)

    # small info box in the corner
    draw_summary(out, total_cars, blue_cars, other_cars, people)

    stats = {
        "total_cars":   total_cars,
        "blue_cars":    blue_cars,
        "other_cars":   other_cars,
        "people":       people,
        "color_counts": color_counts,
    }
    return out, stats


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_summary(img, cars, blue, other, people):
    lines = [
        f"Cars   : {cars}",
        f"Blue   : {blue}  (red box)",
        f"Other  : {other}  (blue box)",
        f"People : {people}",
    ]
    x, y = 10, 22
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (215, y + len(lines) * 20 + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for line in lines:
        cv2.putText(img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        y += 20
