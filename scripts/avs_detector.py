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


def compute_holes_area(img_path, threshold_value=130, crop_x=200):
    """Wyłapuje i zlicza powierzchnię brakujących kępek (dziur)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    
    h, w = img.shape[:2]
    if crop_x == 0 or w <= 2 * crop_x: img_cropped = img
    else: img_cropped = img[:, crop_x:-crop_x]
        
    if img_cropped.size == 0: return 0.0
        
    _, img_bin = cv2.threshold(img_cropped, threshold_value, 255, cv2.THRESH_BINARY)
    img_med = cv2.medianBlur(img_bin, 5)
    
    # 1. Zaklejamy dziury (duże jądro)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    img_closed = cv2.morphologyEx(img_med, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Wyciągamy same dziury
    holes_only = cv2.subtract(img_closed, img_med)
    
    # ==========================================
    # 3. NOWOŚĆ: CZYSZCZENIE SZUMU (Otwarcie)
    # Usuwamy naturalne, małe przerwy między włoskami.
    # Używamy jądra np. 9x9, żeby zniszczyć cienkie linie, a zostawić dziury po kępkach.
    # ==========================================
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    holes_clean = cv2.morphologyEx(holes_only, cv2.MORPH_OPEN, kernel_clean)
    
    return float(np.count_nonzero(holes_clean))


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


def debug_misclassified(img_path, true_lbl, pred_lbl, b_area, h_area, thr_a, thr_h):
    """Funkcja pomocnicza: wyświetla przetworzony obraz w przypadku błędu algorytmu."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    # Cięcie tak jak w głównych funkcjach
    h, w = img.shape[:2]
    crop_x = 200
    if crop_x == 0 or w <= 2 * crop_x: img_cropped = img
    else: img_cropped = img[:, crop_x:-crop_x]
    
    # --- Symulacja kroków dla Włosia ---
    _, img_bin = cv2.threshold(img_cropped, 35, 255, cv2.THRESH_BINARY)
    img_med = cv2.medianBlur(img_bin, 5)
    kernel = np.ones((5, 5), np.uint8)
    img_ero = cv2.erode(img_med, kernel, iterations=1)
    bristle_mask = cv2.dilate(img_ero, kernel, iterations=2)
    
    # --- Symulacja kroków dla Dziur ---
    kernel_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_sealed = cv2.morphologyEx(img_med, cv2.MORPH_CLOSE, kernel_seal)
    
    # Tworzymy kolorowy obraz, żeby móc rysować na czerwono
    holes_vis = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(img_sealed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1: # Szukamy wewnetrznych dziur
                area = cv2.contourArea(contour)
                if area > 250:
                    # Rysujemy wyłapane dziury grubą czerwoną linią
                    cv2.drawContours(holes_vis, contours, i, (0, 0, 255), 3)
                    
    # Wypisywanie informacji prosto na obrazku
    cv2.putText(holes_vis, f"BLEDNA DECYZJA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(holes_vis, f"Rzeczywistosc: {true_lbl} | Algorytm: {pred_lbl}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(holes_vis, f"Wlosie: {b_area:.0f} (prog: {thr_a:.0f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(holes_vis, f"Dziury: {h_area:.0f} (prog: {thr_h:.0f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Powiększamy obrazki, żeby były czytelne na ekranie
    scale = 800 / holes_vis.shape[1] if holes_vis.shape[1] > 0 else 1
    holes_vis_disp = cv2.resize(holes_vis, None, fx=scale, fy=scale)
    bristle_mask_disp = cv2.resize(bristle_mask, None, fx=scale, fy=scale)
    
    # Wyświetlamy
    cv2.imshow("DETEKCJA DZIUR (Na czerwono)", holes_vis_disp)
    cv2.imshow("MASKA WLOSIA", bristle_mask_disp)
    
    print(f" -> Wyświetlam analizę dla {os.path.basename(img_path)}. Wciśnij DOWOLNY KLAWISZ w okienku obrazu, aby kontynuować...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate(df, thr_area, thr_holes):
    preds = (
        (df['bristle_area'] > thr_area) | 
        (df['holes_area'] > thr_holes)
    ).astype(int)
    
    print("\n--- Analiza poszczególnych zdjęć ---")
    for i in range(len(df)):
        file_path = df['file'].iloc[i]
        file_name = os.path.basename(file_path)
        
        true_label = "ZEPSUTA" if df['label'].iloc[i] == 1 else "DOBRA  "
        pred_label = "ZEPSUTA" if preds.iloc[i] == 1 else "DOBRA  "
        
        b_area = df['bristle_area'].iloc[i]
        h_area = df['holes_area'].iloc[i]
        
        marker = "✅" if true_label == pred_label else "❌"
        
        print(f"{marker} {file_name:<20} | Rzeczywistość: {true_label} | Decyzja algorytmu: {pred_label} | (Włosie: {b_area:.0f}, Dziury: {h_area:.0f})")
        
        # Jeśli algorytm się pomylił, uruchamiamy debuger wizualny
        #if true_label != pred_label:
            #debug_misclassified(file_path, true_label, pred_label, b_area, h_area, thr_area, thr_holes)
    
    y_true = df['label'].astype(int)
    report = classification_report(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    
    return report, cm