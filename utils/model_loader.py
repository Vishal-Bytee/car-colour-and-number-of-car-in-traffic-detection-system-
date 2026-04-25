from ultralytics import YOLO

# yolov8n is the smallest/fastest version, only ~6mb
# ultralytics downloads it automatically on first run
MODEL_FILE = "yolov8n.pt"

# these are the coco dataset class ids we care about
CAR_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PERSON_CLASS = 0


def load_model():
    try:
        model = YOLO(MODEL_FILE)
        return model
    except Exception as e:
        print(f"couldn't load model: {e}")
        return None
