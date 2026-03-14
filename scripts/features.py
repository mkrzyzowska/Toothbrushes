import cv2
import numpy as np
import os


def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _clean_binary(bw, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw


def compute_bristle_density(image_path, mask_path=None, method='adaptive'):
    img = _read_gray(image_path)
    if method == 'adaptive':
        bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    elif method == 'otsu':
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    bw = _clean_binary((bw > 0).astype(np.uint8) * 255)
    bw_bin = (bw > 0).astype(np.uint8)
    if mask_path is not None and os.path.exists(mask_path):
        mask = _read_gray(mask_path)
        mask = (mask > 0).astype(np.uint8)
        area = mask.sum()
        if area == 0:
            return 0.0
        bristle_pixels = (bw_bin * mask).sum()
        return float(bristle_pixels) / float(area)
    else:
        area = bw_bin.size
        bristle_pixels = bw_bin.sum()
        return float(bristle_pixels) / float(area)


def compute_void_fraction(image_path, mask_path=None):
    # fraction of background connected-component area inside mask (large gaps)
    img = _read_gray(image_path)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)
    if mask_path is not None and os.path.exists(mask_path):
        mask = _read_gray(mask_path)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = np.ones_like(bw)
    # background within mask
    bg = (1 - bw) * mask
    # find connected components of background and compute large-hole area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg.astype(np.uint8), connectivity=8)
    # ignore the background label 0
    hole_area = 0
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        hole_area += a
    total_area = mask.sum()
    if total_area == 0:
        return 0.0
    return float(hole_area) / float(total_area)


def edge_density(image_path, mask_path=None):
    img = _read_gray(image_path)
    edges = cv2.Canny(img, 50, 150)
    edges_bin = (edges > 0).astype(np.uint8)
    if mask_path is not None and os.path.exists(mask_path):
        mask = _read_gray(mask_path)
        mask = (mask > 0).astype(np.uint8)
        area = mask.sum()
        if area == 0:
            return 0.0
        return float((edges_bin * mask).sum()) / float(area)
    else:
        return float(edges_bin.sum()) / float(edges_bin.size)


def mean_intensity(image_path, mask_path=None):
    img = _read_gray(image_path)
    if mask_path is not None and os.path.exists(mask_path):
        mask = _read_gray(mask_path)
        mask = (mask > 0).astype(np.uint8)
        area = mask.sum()
        if area == 0:
            return float(img.mean())
        return float((img * mask).sum()) / float(area)
    else:
        return float(img.mean())


def extract_features(image_path, mask_path=None):
    return {
        'density': compute_bristle_density(image_path, mask_path=mask_path),
        'void_frac': compute_void_fraction(image_path, mask_path=mask_path),
        'edge_density': edge_density(image_path, mask_path=mask_path),
        'mean_intensity': mean_intensity(image_path, mask_path=mask_path)
    }


if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_img = os.path.join(base, 'data', 'train', 'good', '000.png')
    print('features', extract_features(test_img))
