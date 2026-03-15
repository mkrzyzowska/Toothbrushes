"""Run the AVS-lab-based detector (Area & Holes) with Grid Search."""
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from avs_detector import gather_features, train_thresholds, evaluate

def main():
    parser = argparse.ArgumentParser(description='Run AVS detector (Area and Holes)')
    parser.add_argument('--k-area', type=float, default=2.0, help='Mnożnik dla wystających włosków')
    parser.add_argument('--k-holes', type=float, default=3.0, help='Mnożnik tolerancji dla dziur')
    parser.add_argument('--thr-area', type=float, default=None, help='Ręczny próg dla włosków')
    parser.add_argument('--thr-holes', type=float, default=None, help='Ręczny próg dla dziur')
    
    # NOWA FLAGA: Uruchamia tryb szukania optymalnych parametrów
    parser.add_argument('--search', action='store_true', help='Uruchom automatyczne szukanie najlepszych mnożników k')
    
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'split_dataset', 'train')
    data_val = os.path.join(base, 'split_dataset', 'val')
    
    print(f"Rozpoczynam ekstrakcję cech z folderu TRENINGOWEGO: {data_train}")
    df_train = gather_features(data_train)
    
    print(f"Rozpoczynam ekstrakcję cech z folderu WALIDACYJNEGO: {data_val}")
    df_val = gather_features(data_val)
    
    # ==========================================
    # TRYB POSZUKIWAŃ (GRID SEARCH)
    # ==========================================
    if args.search:
        print("\n=== ROZPOCZYNAM AUTOMATYCZNE SZUKANIE PARAMETRÓW ===")
        print("Sprawdzam kombinacje k_area i k_holes (od 0.0 do 5.0 ze skokiem 0.5)...")
        
        best_f1 = -1.0
        best_acc = -1.0
        best_k_area = 0
        best_k_holes = 0
        best_thr_a = 0
        best_thr_h = 0
        
        # Tworzymy listę parametrów: [0.0, 0.5, 1.0, 1.5 ... 5.0]
        k_values = np.arange(0.0, 5.5, 0.5)
        y_true_val = df_val['label'].astype(int)
        
        # Szybka podwójna pętla po parametrach
        for k_a in k_values:
            for k_h in k_values:
                # 1. Wyliczamy progi dla aktualnego zestawu 'k'
                thr_a, thr_h = train_thresholds(df_train, k_area=k_a, k_holes=k_h)
                
                # 2. Cicha ewaluacja (bez printowania i wyskakujących okienek)
                preds = ((df_val['bristle_area'] > thr_a) | (df_val['holes_area'] > thr_h)).astype(int)
                
                # 3. Wyliczamy wynik: skupiamy się na F1 (najlepszy balans między błędami)
                current_f1 = f1_score(y_true_val, preds, average='macro', zero_division=0)
                current_acc = accuracy_score(y_true_val, preds)
                
                # 4. Sprawdzamy, czy ten zestaw jest lepszy niż dotychczasowy rekordzista
                if current_f1 > best_f1 or (current_f1 == best_f1 and current_acc > best_acc):
                    best_f1 = current_f1
                    best_acc = current_acc
                    best_k_area = k_a
                    best_k_holes = k_h
                    best_thr_a = thr_a
                    best_thr_h = thr_h
                    
        print(f"\n✅ ZNALEZIONO NAJLEPSZE PARAMETRY!")
        print(f" -> Optymalne k_area  = {best_k_area}")
        print(f" -> Optymalne k_holes = {best_k_holes}")
        print(f" -> Osiągnięte wyniki: Macro F1 = {best_f1:.2f} | Accuracy = {best_acc:.2f}")
        print("\n=== ODPALAM PEŁNĄ EWALUACJĘ DLA NAJLEPSZEGO WYNIKU ===")
        
        # Przypisujemy najlepsze progi do ostatecznej ewaluacji
        thr_a = best_thr_a
        thr_h = best_thr_h
        
    # ==========================================
    # TRYB STANDARDOWY (Pojedyncze uruchomienie)
    # ==========================================
    else:
        if args.thr_area is not None and args.thr_holes is not None:
            print("Używam narzuconych z góry progów (pomijam uczenie).")
            thr_a = args.thr_area
            thr_h = args.thr_holes
        else:
            dobre_szczoteczki_train = df_train[df_train['label'] == 0]
            if len(dobre_szczoteczki_train) == 0:
                print("UWAGA: Brak dobrych szczoteczek do nauki! Ustawiam wartości domyślne.")
                thr_a = 315500.0  
                thr_h = 250.0    
            else:
                print(f"Wyliczam progi dla k_area={args.k_area} i k_holes={args.k_holes}...")
                thr_a, thr_h = train_thresholds(df_train, k_area=args.k_area, k_holes=args.k_holes)

    # Końcowe wypisanie błędów (i otwieranie okienek debugera)
    report, cm = evaluate(df_val, thr_a, thr_h)
    
    print('\n--- Podsumowanie ---')
    print(f'thresholds: bristle_area= {thr_a:.2f} | holes_area= {thr_h:.2f}')
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()