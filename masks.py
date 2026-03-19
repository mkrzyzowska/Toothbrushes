import cv2
import numpy as np
import os

# --- USTAWIENIA ŚCIEŻEK ---
TRAIN_GOOD_PATH = "split_dataset/train/good"
VAL_PATH = "split_dataset/val"
CLASSES = ["good", "defective"]

def train_baseline(train_folder):
    """
    Tworzy "strefę bezpieczeństwa" na podstawie poprawnych szczoteczek.
    """
    print(f"Rozpoczynam trening na danych z: {train_folder}...")
    
    if not os.path.exists(train_folder):
        print(f"BŁĄD: Nie znaleziono folderu {train_folder}!")
        return None

    sum_shape = None
    img_count = 0

    for fname in os.listdir(train_folder):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(train_folder, fname)
        img = cv2.imread(path)
        if img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)

        if sum_shape is None:
            sum_shape = np.zeros_like(thresh, dtype=np.float32)
        
        sum_shape += (thresh / 255.0)
        img_count += 1

    if img_count == 0:
        return None

    # Tworzymy powłokę (strefę bezpieczeństwa). 
    # Jeśli piksel był biały chociażby na 2% poprawnych zdjęć, uznajemy go za bezpieczny twardy margines.
    safe_zone = np.zeros_like(sum_shape, dtype=np.uint8)
    safe_zone[sum_shape > (0.02 * img_count)] = 255

    # Dylatacja (pogrubienie) strefy bezpieczeństwa.
    # Dajemy szczoteczce 3 piksele "luzu" na wypadek drobnego przesunięcia pod kamerą.
    kernel_dilate = np.ones((5,5), np.uint8)
    safe_zone = cv2.dilate(safe_zone, kernel_dilate, iterations=1)

    print(f"Trening zakończony! Przeanalizowano {img_count} zdjęć.")
    return safe_zone

def get_defect_mask(img, safe_zone):
    """
    Szuka pikseli, które wychodzą poza wyuczoną strefę bezpieczeństwa.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    
    # Wyciągamy tylko to, co wystaje POZA strefę bezpieczeństwa
    not_safe_zone = cv2.bitwise_not(safe_zone)
    out_of_bounds = cv2.bitwise_and(thresh, not_safe_zone)
    
    # Usuwamy pojedyncze kropeczki (szum z matrycy), żeby nie było fałszywych alarmów
    kernel_clean = np.ones((3,3), np.uint8)
    defect_mask = cv2.morphologyEx(out_of_bounds, cv2.MORPH_OPEN, kernel_clean)
    
    # Zliczamy, ile pikseli faktycznie wystaje
    defect_area = cv2.countNonZero(defect_mask)
    
    # Jeśli wystaje więcej niż np. 15 pikseli, uznajemy to za wadliwy włosek
    MIN_DEFECT_PIXELS = 15 
    
    if defect_area > MIN_DEFECT_PIXELS:
        # Możemy lekko pogrubić maskę defektu, żeby czerwona plama była lepiej widoczna na ekranie
        vis_mask = cv2.dilate(defect_mask, np.ones((5,5), np.uint8), iterations=1)
        return vis_mask, True
    else:
        # Jeśli nic nie wystaje (lub to tylko np. 5 pikseli szumu), szczoteczka jest OK
        empty_mask = np.zeros_like(thresh)
        return empty_mask, False

def process_folder_val(folder, safe_zone):
    if not os.path.exists(folder):
        print(f"Brak folderu: {folder}")
        return False

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None: continue

        # Pobieramy maskę i werdykt (True = zepsuta, False = dobra)
        mask, is_defective = get_defect_mask(img, safe_zone)

        # Wypisywanie werdyktu w terminalu z odpowiednim oznaczeniem
        verdict_text = "[!!! DEFECTIVE !!!]" if is_defective else "[OK - GOOD]"
        print(f"Plik: {fname:<20} | Werdykt: {verdict_text}")

        # Rysowanie defektów
        vis = img.copy()
        vis[mask > 0] = [0, 0, 255]

        # Wyświetlanie (skalowanie w dół dla wygody)
        img_disp = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        mask_disp = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        vis_disp = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        mask_bgr = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img_disp, mask_bgr, vis_disp])

        cv2.imshow("Original | Defect Mask | Result", combined)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            return True 

    return False

if __name__ == "__main__":
    print("--- ETAP 1: UCZENIE (TWORZENIE STREFY BEZPIECZEŃSTWA) ---")
    safe_zone = train_baseline(TRAIN_GOOD_PATH)

    if safe_zone is not None:
        print("\n--- ETAP 2: TESTOWANIE ---")
        
        for cls in CLASSES:
            val_folder = os.path.join(VAL_PATH, cls)
            print(f"\n---> Otwieram folder: {cls.upper()}")
            
            stop_program = process_folder_val(val_folder, safe_zone)
            if stop_program:
                break

    cv2.destroyAllWindows()