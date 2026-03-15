"""Run the AVS-lab-based detector (Area & Holes) and print evaluation."""
import argparse
import os
from avs_detector import gather_features, train_thresholds, evaluate

def main():
    parser = argparse.ArgumentParser(description='Run AVS detector (Area and Holes)')
    parser.add_argument('--k-area', type=float, default=2.0, help='Mnożnik dla wystających włosków')
    parser.add_argument('--k-holes', type=float, default=3.0, help='Mnożnik tolerancji dla dziur')
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'original_dataset', 'train')
    
    print(f"Rozpoczynam ekstrakcję cech z: {data_train}")
    print("To może potrwać kilka-kilkanaście sekund...")
    
    df = gather_features(data_train)
    thr_a, thr_h = train_thresholds(df, k_area=args.k_area, k_holes=args.k_holes)
    report, cm = evaluate(df, thr_a, thr_h)
    
    print('\n--- AVS-lab-based detector results ---')
    print(f'parameters: k_area= {args.k_area} | k_holes= {args.k_holes}')
    print(f'thresholds: bristle_area= {thr_a:.2f} | holes_area= {thr_h:.2f}')
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()