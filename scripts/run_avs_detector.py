"""Run the AVS-lab-based detector (Area & Holes) with Train/Val split."""
import argparse
import os
from avs_detector import gather_features, train_thresholds, evaluate

def main():
    parser = argparse.ArgumentParser(description='Run AVS detector (Area and Holes)')
    parser.add_argument('--k-area', type=float, default=2.0, help='Mnożnik dla wystających włosków')
    parser.add_argument('--k-holes', type=float, default=3.0, help='Mnożnik tolerancji dla dziur')
    
    # Ręczne ustawianie progów
    parser.add_argument('--thr-area', type=float, default=None, help='Ręczny próg dla włosków')
    parser.add_argument('--thr-holes', type=float, default=None, help='Ręczny próg dla dziur')
    
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. ZMIANA ŚCIEŻEK - Teraz używamy split_dataset
    data_train = os.path.join(base, 'split_dataset', 'train')
    data_val = os.path.join(base, 'split_dataset', 'val')
    
    print(f"Rozpoczynam ekstrakcję cech z folderu TRENINGOWEGO: {data_train}")
    df_train = gather_features(data_train)
    
    print(f"Rozpoczynam ekstrakcję cech z folderu WALIDACYJNEGO: {data_val}")
    df_val = gather_features(data_val)
    
    # ==========================================
    # LOGIKA PROGÓW: Uczymy się TYLKO na zbiorze TRAIN
    # ==========================================
    if args.thr_area is not None and args.thr_holes is not None:
        print("Używam narzuconych z góry progów (pomijam uczenie).")
        thr_a = args.thr_area
        thr_h = args.thr_holes
        
    else:
        dobre_szczoteczki_train = df_train[df_train['label'] == 0]
        
        if len(dobre_szczoteczki_train) == 0:
            print("UWAGA: Brak dobrych szczoteczek do nauki w Train! Ustawiam wartości domyślne.")
            thr_a = 315500.0  
            thr_h = 1000.0    
        else:
            print("Wyliczam progi na podstawie dobrych szczoteczek ze zbioru Train...")
            thr_a, thr_h = train_thresholds(df_train, k_area=args.k_area, k_holes=args.k_holes)
    # ==========================================

    # 2. EWALUACJA: Sprawdzamy wyniki TYLKO na zbiorze VAL
    print("\n=== WYNIKI DLA ZBIORU WALIDACYJNEGO (VAL) ===")
    report, cm = evaluate(df_val, thr_a, thr_h)
    
    print('\n--- Podsumowanie ---')
    print(f'thresholds: bristle_area= {thr_a:.2f} | holes_area= {thr_h:.2f}')
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()