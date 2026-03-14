import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def analyze_image(image_path, mask_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    eq = cv2.equalizeHist(blurred)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # assume bristles are darker -> invert to make bristles foreground
    fg = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = (clean > 0).astype(np.uint8)

    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = np.ones_like(bw)

    total_area = int(mask.sum()) if mask.sum() > 0 else bw.size

    # foreground connected components (bristle clusters)
    labeled = (bw * mask).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(labeled, connectivity=8)
    comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    num_components = len(comp_areas)
    mean_comp_area = float(np.mean(comp_areas)) if comp_areas else 0.0

    # background holes inside mask (missing/bristle gaps)
    bg = ((1 - bw) * mask).astype(np.uint8)
    nb_labels, nlabels, nstats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    hole_areas = [int(nstats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_labels)]
    largest_hole = max(hole_areas) if hole_areas else 0
    largest_hole_frac = float(largest_hole) / float(total_area)

    return {
        'file': image_path,
        'num_components': num_components,
        'mean_comp_area': mean_comp_area,
        'largest_hole_frac': largest_hole_frac,
        'total_area': total_area,
    }


def gather_features(data_root, gt_root=None):
    rows = []
    for cls in ['good', 'defective']:
        dirpath = os.path.join(data_root, cls)
        if not os.path.isdir(dirpath):
            continue
        for f in sorted(os.listdir(dirpath)):
            if not f.lower().endswith('.png'):
                continue
            img_path = os.path.join(dirpath, f)
            mask = None
            if gt_root is not None and cls == 'defective':
                name = os.path.splitext(f)[0]
                mask_candidate = os.path.join(gt_root, f'{name}_mask.png')
                if os.path.exists(mask_candidate):
                    mask = mask_candidate
            feats = analyze_image(img_path, mask_path=mask)
            feats['label'] = 1 if cls == 'defective' else 0
            rows.append(feats)
    return pd.DataFrame(rows)


def train_thresholds(df, k_num=3.0, k_hole=1.5):
    """Compute thresholds from `good` images.

    Defaults tuned to be more conservative (reduce false positives):
    - `k_num` larger -> lower `thr_num` (fewer goods flagged for low component count)
    - `k_hole` larger -> higher `thr_hole` (fewer goods flagged for hole size)
    """
    goods = df[df['label'] == 0]
    num_mean = goods['num_components'].mean()
    num_std = goods['num_components'].std()
    hole_mean = goods['largest_hole_frac'].mean()
    hole_std = goods['largest_hole_frac'].std()
    thr_num = float(num_mean - k_num * (num_std if not np.isnan(num_std) else 0.0))
    thr_hole = float(hole_mean + k_hole * (hole_std if not np.isnan(hole_std) else 0.0))
    return thr_num, thr_hole


def evaluate(df, thr_num, thr_hole, mode='or'):
    """Evaluate using either 'or' (default) or 'and' combination rule.

    - 'or': predict defective when num_components < thr_num OR largest_hole_frac > thr_hole
    - 'and': predict defective when BOTH conditions hold (stricter)
    """
    if mode == 'and':
        preds = ((df['num_components'] < thr_num) & (df['largest_hole_frac'] > thr_hole)).astype(int)
    else:
        preds = ((df['num_components'] < thr_num) | (df['largest_hole_frac'] > thr_hole)).astype(int)
    y_true = df['label'].astype(int)
    report = classification_report(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    return report, cm


if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'data', 'train')
    gt = os.path.join(base, 'ground_truth', 'defective')
    df = gather_features(data_train, gt_root=gt)
    thr_n, thr_h = train_thresholds(df)
    print('thresholds: num_components=', thr_n, ' largest_hole_frac=', thr_h)
    report, cm = evaluate(df, thr_n, thr_h)
    print(report)
    print('confusion matrix:\n', cm)
