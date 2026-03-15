"""Run the AVS-lab-based detector (Area & Holes) and print evaluation."""
import argparse
import os
from avs_detector import gather_features, train_thresholds, evaluate

def main():
    parser = argparse.ArgumentParser(description='Run AVS detector (Area and Holes)')
    parser.add_argument('--k-area', type=float, default=2.0, help='Mnożnik dla wystających włosków')
    parser.add_argument('--k-holes', type=float, default=3.0, help='Mnożnik tolerancji dla dziur')
    
    # NOWE PARAMETRY: Ręczne ustawianie progów (pomijają statystykę)
    parser.add_argument('--thr-area', type=float, default=None, help='Ręczny próg dla włosków')
    parser.add_argument('--thr-holes', type=float, default=None, help='Ręczny próg dla dziur')
    
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'original_dataset', 'train')
    
    print(f"Rozpoczynam ekstrakcję cech z: {data_train}")
    df = gather_features(data_train)
    
    # ==========================================
    # LOGIKA PROGÓW: Uczenie vs Wartości startowe
    # ==========================================
    # 1. Sprawdzamy, czy użytkownik wymusił własne progi z konsoli
    if args.thr_area is not None and args.thr_holes is not None:
        print("Używam narzuconych z góry progów (pomijam uczenie).")
        thr_a = args.thr_area
        thr_h = args.thr_holes
        
    else:
        # 2. Jeśli nie ma podanych progów, sprawdzamy czy mamy dane do nauki (dobre szczoteczki)
        dobre_szczoteczki = df[df['label'] == 0]
        
        if len(dobre_szczoteczki) == 0:
            print("UWAGA: Brak dobrych szczoteczek do nauki! Ustawiam wartości domyślne.")
            # Tutaj wpisujesz to, co wcześniej wychodziło Ci z logów jako dobre progi
            thr_a = 315500.0  
            thr_h = 1000.0    
        else:
            print("Wyliczam progi na podstawie dobrych szczoteczek...")
            thr_a, thr_h = train_thresholds(df, k_area=args.k_area, k_holes=args.k_holes)
    # ==========================================

    report, cm = evaluate(df, thr_a, thr_h)
    
    print('\n--- AVS-lab-based detector results ---')
    print(f'thresholds: bristle_area= {thr_a:.2f} | holes_area= {thr_h:.2f}')
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()