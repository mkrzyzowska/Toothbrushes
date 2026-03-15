"""Run the AVS-lab-based detector (Bristle Area Only) and print evaluation."""
import argparse
import os
from avs_detector import gather_features, train_thresholds, evaluate

def main():
    parser = argparse.ArgumentParser(description='Run AVS detector (Bristle Area Only)')
    # Teraz z konsoli sterujesz TYLKO tolerancją Twojego algorytmu
    parser.add_argument('--k-area', type=float, default=2.0, help='Mnożnik tolerancji dla pikseli włosia')
    args = parser.parse_args()

    # Ustalenie ścieżek
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'original_dataset', 'train')
    
    print(f"Wczytywanie obrazów z: {data_train}")
    
    # 1. Oblicz piksele dla każdego zdjęcia
    df = gather_features(data_train)
    
    # 2. Wylicz próg
    thr_a = train_thresholds(df, k_area=args.k_area)
    
    # 3. Oceń wyniki
    report, cm = evaluate(df, thr_a)
    
    # Wypisz podsumowanie
    print('\n--- AVS-lab-based detector results (Area Only) ---')
    print('parameters: k_area=', args.k_area)
    print('thresholds: bristle_area=', thr_a)
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()