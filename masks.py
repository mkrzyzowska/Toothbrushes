import cv2
import numpy as np
import os
import math

# --- USTAWIENIA ŚCIEŻEK ---
TRAIN_GOOD_PATH = "original_dataset/train/good"
TRAIN_DEFECTIVE_PATH = "original_dataset/train/defective"
TEST_FOLDER = TRAIN_DEFECTIVE_PATH 

def train_average_shape(train_folder):
    """Tworzy bazową maskę uśrednionego kształtu główki szczoteczki."""
    print(f"Tworzenie średniego kształtu z: {train_folder}...")
    
    if not os.path.exists(train_folder):
        print(f"BŁĄD: Nie znaleziono folderu {train_folder}!")
        return None

    sum_shape = None
    img_count = 0
    filenames = [f for f in os.listdir(train_folder) if f.endswith(".png")]

    for fname in filenames:
        path = os.path.join(train_folder, fname)
        img = cv2.imread(path)
        if img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_thresh = np.zeros_like(thresh)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(clean_thresh, [c], -1, 255, -1)

        if sum_shape is None:
            sum_shape = np.zeros_like(clean_thresh, dtype=np.float32)
        
        sum_shape += (clean_thresh / 255.0)
        img_count += 1

    if img_count == 0:
        return None

    base_mask = np.zeros_like(sum_shape, dtype=np.uint8)
    base_mask[sum_shape > 0] = 255 

    kernel = np.ones((5,5), np.uint8)
    base_mask = cv2.dilate(base_mask, kernel, iterations=1)

    print(f"Gotowe! Baza utworzona z {img_count} zdjęć.")
    return base_mask

def get_defect_mask(img, base_mask):
    """Wykrywa wady wystające poza obrys szczoteczki (włoski)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    defect_mask = cv2.subtract(thresh, base_mask)
    
    kernel_clean = np.ones((3,3), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    is_defective = cv2.countNonZero(defect_mask) > 0
    if is_defective:
        defect_mask = cv2.dilate(defect_mask, np.ones((5,5), np.uint8), iterations=1)
        
    return defect_mask, is_defective

def detect_holes(img, base_mask, current_defect_mask, is_defective, crop_x=70, MIN_AREA=400, MIN_CIRCULARITY=0.50):
    """
    Wykrywa dziury analizując 3 różne kanały jasności: L (z HLS), V (z HSV) oraz Y (z YCrCb).
    Jeśli jakikolwiek kanał znajdzie dziurę, zostaje ona dodana do wspólnej maski.
    """
    # 1. Konwersje do 3 przestrzeni barw
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 2. Wyciągnięcie odpowiednich kanałów jasności
    l_channel = hls[:, :, 1]   # Lightness z HLS
    v_channel = hsv[:, :, 2]   # Value z HSV
    y_channel = ycbcr[:, :, 0] # Luminance z YCrCb
    
    # Lista kanałów do przeanalizowania
    channels_to_check = [l_channel, v_channel, y_channel]
    
    # Wspólna maska, na której będziemy zbierać dziury ze wszystkich 3 kanałów
    combined_hole_mask = np.zeros_like(l_channel)

    # 3. Pętla analizująca każdy kanał osobno
    for channel in channels_to_check:
        # Przycięcie boków do lepszego obliczenia progu Otsu
        if crop_x > 0 and crop_x * 2 < channel.shape[1]:
            cropped = channel[:, crop_x:-crop_x]
        else:
            cropped = channel

        # Otsu na przyciętym fragmencie (szukamy ciemnych dziur -> THRESH_BINARY_INV)
        thresh_val, _ = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Nałożenie wyliczonego progu na cały kanał
        _, thresh_full = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Szukanie dziur na zbinaryzowanym kanale
        cnts, _ = cv2.findContours(thresh_full, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            # Jeśli kształt jest zbliżony do koła, dodajemy go do naszej wspólnej maski ubytków
            if circularity > MIN_CIRCULARITY:
                cv2.drawContours(combined_hole_mask, [c], -1, 255, -1)
                
    # 4. Wszystkie znalezione dziury (z 3 kanałów) ograniczamy tylko do obszaru główki szczoteczki
    combined_hole_mask = cv2.bitwise_and(combined_hole_mask, base_mask)

    # 5. Złączenie nowo znalezionych dziur z dotychczasową maską defektów (np. wystającymi włoskami)
    if cv2.countNonZero(combined_hole_mask) > 0:
        is_defective = True
        current_defect_mask = cv2.bitwise_or(current_defect_mask, combined_hole_mask)

    return current_defect_mask, is_defective

def process_folder(folder, base_mask):
    print(f"\n---> Sprawdzam folder: {folder}")
    if not os.path.exists(folder):
        print(f"Brak folderu: {folder}")
        return False

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None: continue

        # KROK 1: Wykrywanie wystających włosków
        mask, is_defective = get_defect_mask(img, base_mask)
        
        # KROK 2: Wykrywanie dziur w 3 przestrzeniach barw (połączenie HLS, HSV, YCrCb)
        mask, is_defective = detect_holes(img, base_mask, mask, is_defective, crop_x=200)

        status_str = "[ X ] WADA WYKRYTA" if is_defective else "[ OK ] PRAWIDŁOWA"
        print(f"{fname:<20} | WYNIK: {status_str}")

        # WIZUALIZACJA
        vis = img.copy()
        vis[mask > 0] = [0, 0, 255]

        img_disp = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        mask_disp = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        vis_disp = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        mask_bgr = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img_disp, mask_bgr, vis_disp])

        cv2.imshow("Original | Potrojna Maska (Wloski + Dziury) | Wynik", combined)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC przerywa
            return True 

    return False

if __name__ == "__main__":
    print("--- ETAP 1: UCZENIE WZORCA ---")
    base_mask = train_average_shape(TRAIN_GOOD_PATH)

    if base_mask is not None:
        print("\n--- ETAP 2: TESTOWANIE HYBRYDOWE (WŁOSKI + POTRÓJNE DZIURY) ---")
        process_folder(TEST_FOLDER, base_mask)

    cv2.destroyAllWindows()