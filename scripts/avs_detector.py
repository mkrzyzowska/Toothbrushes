import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def compute_bristle_area(img_path, threshold_value=35, crop_x=200):
    """Binarizuje i liczy białe piksele (powierzchnię włosia)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Bezpieczne cięcie (sanity check)
    h, w = img.shape[:2]
    if crop_x == 0 or w <= 2 * crop_x:
        img_cropped = img
    else:
        img_cropped = img[:, crop_x:-crop_x]
        
    if img_cropped.size == 0:
        return 0.0
        
    _, img_bin = cv2.threshold(img_cropped, threshold_value, 255, cv2.THRESH_BINARY)
    
    if img_bin.size == 0:
        return 0.0
        
    img_med = cv2.medianBlur(img_bin, 5)
    kernel = np.ones((5, 5), np.uint8)
    img_ero = cv2.erode(img_med, kernel, iterations=1)
    img_processed = cv2.dilate(img_ero, kernel, iterations=2)
    
    return float(np.count_nonzero(img_processed))


def analyze_image(img_path):
    """Zwraca słownik z wyliczonymi cechami dla jednego obrazu."""
    return {
        'bristle_area': compute_bristle_area(img_path)
    }


def gather_features(data_root):
    """Przechodzi przez foldery good/defective i buduje tabelę (DataFrame)."""
    rows = []
    for cls in ['good', 'defective']:
        dirpath = os.path.join(data_root, cls)
        if not os.path.isdir(dirpath):
            continue
        for f in sorted(os.listdir(dirpath)):
            if not f.lower().endswith('.png'):
                continue
            
            img_path = os.path.join(dirpath, f)
            feats = analyze_image(img_path)
            
            label = 1 if cls == 'defective' else 0
            row = {'file': img_path, 'label': label}
            row.update(feats)
            rows.append(row)
            
    return pd.DataFrame(rows)


def train_thresholds(df, k_area=2.0):
    """Wylicza próg tolerancji na podstawie DOBRYCH szczoteczek."""
    goods = df[df['label'] == 0]
    
    area_mean = goods['bristle_area'].mean()
    area_std = goods['bristle_area'].std()
    
    # Próg odcięcia: średnia + margines błędu
    thr_area = float(area_mean + k_area * (area_std if not pd.isna(area_std) else 0.0))
    
    return thr_area


def evaluate(df, thr_area):
    """Ocenia wszystkie szczoteczki na podstawie wyliczonego progu."""
    # Szczoteczka jest ZEPSUTA (1), jeśli pikseli jest za dużo
    preds = (df['bristle_area'] > thr_area).astype(int)
    
    y_true = df['label'].astype(int)
    report = classification_report(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    
    return report, cm