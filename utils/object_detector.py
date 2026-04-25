import cv2
import numpy as np

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def _nms(boxes, scores, thresh=0.25):
    if not boxes:
        return []
    boxes  = np.array([[x, y, x+bw, y+bh] for x, y, bw, bh in boxes], float)
    scores = np.array(scores, float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    kept  = []
    while order.size:
        i = order[0]
        kept.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < thresh]
    return kept


def detect_cars(img):
    """
    Segment vehicles from road by comparing pixel brightness/saturation
    to an estimated road baseline.  Works on real traffic photos without
    any external model files.
    """
    ih, iw = img.shape[:2]

    # Only scan the part of the frame that has road (not sky / far horizon)
    roi_top    = int(ih * 0.10)
    roi_bottom = int(ih * 0.90)
    roi        = img[roi_top:roi_bottom, :]
    rh, rw     = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1].astype(int)

    # estimate road colour from the centre strip
    road_sample = gray[rh//3 : 2*rh//3, rw//5 : 4*rw//5]
    road_med    = float(np.median(road_sample))
    road_std    = float(np.std(road_sample))

    # pixels that clearly differ from road = likely vehicle
    diff      = np.abs(gray.astype(float) - road_med)
    non_road  = (diff  > max(road_std * 1.8, 20)).astype(np.uint8) * 255
    colored   = (sat   > 38).astype(np.uint8) * 255    # vivid car paint
    dark      = (gray  < 58).astype(np.uint8) * 255    # black cars

    combined  = cv2.bitwise_or(non_road, colored)
    combined  = cv2.bitwise_or(combined, dark)

    # strip edge columns (trees, guardrails)
    margin = int(rw * 0.06)
    combined[:, :margin]   = 0
    combined[:, rw-margin:] = 0
    combined[:int(rh*0.04), :] = 0   # top few rows

    # small open (kill noise), small close (fill gaps inside car body)
    k_open  = np.ones((4, 4), np.uint8)
    k_close = np.ones((9, 6), np.uint8)
    mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = rh * rw
    boxes, scores = [], []
    for c in contours:
        area = cv2.contourArea(c)
        pct  = area / roi_area * 100
        x, y, bw, bh = cv2.boundingRect(c)
        asp = bw / (bh + 1e-6)

        if 0.08 < pct < 40 and 0.30 < asp < 8.0 and bw > 25 and bh > 14:
            # map y back to full image coordinates
            boxes.append((x, y + roi_top, bw, bh))
            scores.append(area)

    if not boxes:
        return _fallback(img)

    keep = _nms(boxes, scores, thresh=0.25)
    return [boxes[i] for i in keep]


def _fallback(img):
    """
    Last resort when road segmentation finds nothing
    (e.g. a single car on a plain background).
    Uses colour-blob detection.
    """
    ih, iw = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat  = hsv[:, :, 1]

    mask = cv2.inRange(sat, 20, 255) & cv2.inRange(gray, 35, 240)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((8,  8),  np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    img_area = ih * iw
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.03:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        asp = bw / (bh + 1e-6)
        if 0.4 < asp < 6:
            results.append((x, y, bw, bh))

    if not results:
        m = 6
        return [(m, m, iw-2*m, ih-2*m)]
    return results


def detect_people(img):
    ih, iw = img.shape[:2]
    scale  = min(1.0, 640 / max(ih, iw))
    small  = cv2.resize(img, (int(iw*scale), int(ih*scale)))

    rects, weights = _hog.detectMultiScale(
        small, winStride=(8,8), padding=(4,4), scale=1.05
    )
    if len(rects) == 0:
        return []

    rects = [(int(x/scale), int(y/scale), int(bw/scale), int(bh/scale))
             for x, y, bw, bh in rects]
    keep = _nms(rects, list(weights.flatten()), thresh=0.4)
    return [rects[i] for i in keep]
