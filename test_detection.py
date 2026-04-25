import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.model_loader import load_model
from utils.detector import detect_and_annotate


def test(path=None):
    print("loading yolo model...")
    model = load_model()
    if model is None:
        print("model load failed. make sure ultralytics is installed.")
        return

    if path:
        img = cv2.imread(path)
        if img is None:
            print(f"cannot read image: {path}")
            return
        name = os.path.basename(path)
    else:
        print("no image path given. usage: python test_detection.py <image.jpg>")
        return

    print(f"running detection on {name} ({img.shape[1]}x{img.shape[0]})...")
    result, stats = detect_and_annotate(model, img)

    print(f"\nresults:")
    print(f"  total cars  : {stats['total_cars']}")
    print(f"  blue cars   : {stats['blue_cars']}  <- red box drawn")
    print(f"  other cars  : {stats['other_cars']}  <- blue box drawn")
    print(f"  people      : {stats['people']}")
    print(f"  colour breakdown: {stats['color_counts']}")

    out = "test_result.jpg"
    cv2.imwrite(out, result)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    test(path)
