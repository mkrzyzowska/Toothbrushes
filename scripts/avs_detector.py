import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def compute_bristle_area(img_path, threshold_value=35, crop_x=200):
    """Zlicza powierzchnię samego włosia."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    
    h, w = img.shape[:2]
    if crop_x == 0 or w <= 2 * crop_x: img_cropped = img
    else: img_cropped = img[:, crop_x:-crop_x]
        
    if img_cropped.size == 0: return 0.0
        
    _, img_bin = cv2.threshold(img_cropped, threshold_value, 255, cv2.THRESH_BINARY)
    if img_bin.size == 0: return 0.0
        
    img_med = cv2.medianBlur(img_bin, 5)
    kernel = np.ones((5, 5), np.uint8)
    img_ero = cv2.erode(img_med, kernel, iterations=1)
    img_processed = cv2.dilate(img_ero, kernel, iterations=2)
    
    return float(np.count_nonzero(img_processed))


def compute_holes_area(img_path, threshold_value=35, crop_x=200):
    """Wyłapuje i zlicza powierzchnię brakujących kępek (dziur)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    
    # Bezpieczne cięcie
    h, w = img.shape[:2]
    if crop_x == 0 or w <= 2 * crop_x: img_cropped = img
    else: img_cropped = img[:, crop_x:-crop_x]
        
    if img_cropped.size == 0: return 0.0
        
    # 1. Binaryzacja i czyszczenie szumów
    _, img_bin = cv2.threshold(img_cropped, threshold_value, 255, cv2.THRESH_BINARY)
    img_med = cv2.medianBlur(img_bin, 5)
    
    # 2. Morfologiczne zamykanie - zakleja dziury wewnątrz włosia
    # Używamy owalnego kształtu jądra o rozmiarze 25x25 (wystarczająco duże by pokryć brak kępki)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img_closed = cv2.morphologyEx(img_med, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Magia: odejmujemy oryginał od zaklejonego. Zostają SAME dziury!
    holes_only = cv2.subtract(img_closed, img_med)
    
    # 4. Zwracamy pole powierzchni wykrytych dziur
    return float(np.count_nonzero(holes_only))


def analyze_image(img_path):
    """Zwraca słownik z dwiema cechami."""
    return {
        'bristle_area': compute_bristle_area(img_path),
        'holes_area': compute_holes_area(img_path)
    }


def gather_features(data_root):
    rows = []
    for cls in ['good', 'defective']:
        dirpath = os.path.join(data_root, cls)
        if not os.path.isdir(dirpath): continue
        for f in sorted(os.listdir(dirpath)):
            if not f.lower().endswith('.png'): continue
            
            img_path = os.path.join(dirpath, f)
            feats = analyze_image(img_path)
            
            label = 1 if cls == 'defective' else 0
            row = {'file': img_path, 'label': label}
            row.update(feats)
            rows.append(row)
            
    return pd.DataFrame(rows)


def train_thresholds(df, k_area=2.0, k_holes=3.0):
    """Wylicza progi dla obu defektów na podstawie dobrych szczoteczek."""
    goods = df[df['label'] == 0]
    
    # Próg 1: Wystające włoski (powierzchnia)
    area_mean = goods['bristle_area'].mean()
    area_std = goods['bristle_area'].std()
    thr_area = float(area_mean + k_area * (area_std if not pd.isna(area_std) else 0.0))
    
    # Próg 2: Dziury (idealna szczoteczka powinna mieć pole dziur bliskie 0)
    holes_mean = goods['holes_area'].mean()
    holes_std = goods['holes_area'].std()
    thr_holes = float(holes_mean + k_holes * (holes_std if not pd.isna(holes_std) else 0.0))
    
    return thr_area, thr_holes


def evaluate(df, thr_area, thr_holes):
    """
    Szczoteczka jest ZEPSUTA (1), jeśli:
    - ma za dużo białych pikseli (wystające włoski) LUB
    - ma za duże pole wewnętrznych dziur (brak kępki)
    """
    preds = (
        (df['bristle_area'] > thr_area) | 
        (df['holes_area'] > thr_holes)
    ).astype(int)
    
    print("\n--- Analiza poszczególnych zdjęć ---")
    # Pętla wypisująca decyzję dla każdego zdjęcia
    for i in range(len(df)):
        file_name = os.path.basename(df['file'].iloc[i])
        
        # Formatowanie etykiet dla lepszej czytelności w konsoli
        true_label = "ZEPSUTA" if df['label'].iloc[i] == 1 else "DOBRA  "
        pred_label = "ZEPSUTA" if preds.iloc[i] == 1 else "DOBRA  "
        
        b_area = df['bristle_area'].iloc[i]
        h_area = df['holes_area'].iloc[i]
        
        # Znacznik, czy algorytm ocenił poprawnie
        marker = "✅" if true_label == pred_label else "❌"
        
        print(f"{marker} {file_name:<20} | Rzeczywistość: {true_label} | Decyzja algorytmu: {pred_label} | (Włosie: {b_area:.0f}, Dziury: {h_area:.0f})")
    
    y_true = df['label'].astype(int)
    report = classification_report(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    
    return report, cm