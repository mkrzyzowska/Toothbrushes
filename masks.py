import cv2
import numpy as np
import os

def get_defect_mask(img):

    # --- 1. Preprocess ---
    blur = cv2.GaussianBlur(img, (7,7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # --- 2. Toothbrush segmentation (white plastic) ---
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, mask_body = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # keep biggest object (toothbrush)
    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    mask_body = np.zeros_like(mask_body)
    cv2.drawContours(mask_body, [c], -1, 255, -1)

    # --- 3. Extract bristles region ---
    # remove smooth plastic -> keep only high texture areas
    edges = cv2.Canny(gray, 50, 150)
    mask_bristles = cv2.bitwise_and(edges, mask_body)

    # dilate to fill clusters
    kernel = np.ones((5,5), np.uint8)
    mask_bristles = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    np.ones((3,3), np.uint8))

    mask_bristles = cv2.morphologyEx(mask_bristles, cv2.MORPH_OPEN,
                                    np.ones((3,3), np.uint8))
    # --- 4. Color anomaly detection ---
    # GOOD toothbrush → yellow/white clusters
    # BAD → strange colors (red, dark, etc.)

    # mask expected colors (white + yellow)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])

    lower_yellow = np.array([15, 80, 100])
    upper_yellow = np.array([40, 255, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_good_color = cv2.bitwise_or(mask_white, mask_yellow)

    # defects = NOT expected colors but inside bristles

    # --- 5. Structural defects (broken / sparse clusters) ---
    # remove small noise first
    clean = cv2.morphologyEx(mask_bristles, cv2.MORPH_OPEN,
                             np.ones((3,3), np.uint8))

    # --- 4. Detect bristle clusters properly ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bristles)

    mask_defects = np.zeros_like(mask_bristles)

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    mean_area = np.mean(areas)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # detect abnormal clusters
        if area < 0.4 * mean_area or area > 2.5 * mean_area:
            mask_defects[labels == i] = 255


    # --- 7. Final cleanup ---
    mask_defects = cv2.morphologyEx(mask_defects,
                                   cv2.MORPH_CLOSE,
                                   np.ones((5,5), np.uint8))

    return mask_defects

def process_folder(folder):

    for fname in os.listdir(folder):

        if not fname.endswith(".png"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, None, fx=0.5, fy=0.5,
                 interpolation=cv2.INTER_AREA)

        mask = get_defect_mask(img)

        # visualization
        vis = img.copy()
        vis[mask > 0] = [0, 0, 255]  # mark defects in red

        combined = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis])

        cv2.imshow("image | mask | result", combined)
        key = cv2.waitKey(0)

        if key == 27:  # ESC to quit
            return
        
BASE_PATH = "original_dataset/train"
CLASSES = ["good", "defective"]

if __name__ == "__main__":


    for cls in CLASSES:
        folder = os.path.join(BASE_PATH, cls)
        print(f"Processing: {cls}")

        process_folder(folder)

    cv2.destroyAllWindows()