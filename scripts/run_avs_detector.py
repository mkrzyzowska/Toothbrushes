"""Run the AVS-lab-based detector on the dataset and print evaluation."""
import argparse
import os
from scripts.avs_detector import gather_features, train_thresholds, evaluate


def main():
    parser = argparse.ArgumentParser(description='Run AVS-lab-based detector')
    parser.add_argument('--k-num', type=float, default=3.0, help='k multiplier for num_components threshold')
    parser.add_argument('--k-hole', type=float, default=1.5, help='k multiplier for largest_hole_frac threshold')
    parser.add_argument('--mode', choices=['or', 'and'], default='or', help="Combination rule: 'or' (default) or 'and')")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'data', 'train')
    gt = os.path.join(base, 'ground_truth', 'defective')
    df = gather_features(data_train, gt_root=gt)
    thr_n, thr_h = train_thresholds(df, k_num=args.k_num, k_hole=args.k_hole)
    report, cm = evaluate(df, thr_n, thr_h, mode=args.mode)
    print('--- AVS-lab-based detector results ---')
    print('parameters: k_num=', args.k_num, ' k_hole=', args.k_hole, ' mode=', args.mode)
    print('thresholds: num_components=', thr_n, ' largest_hole_frac=', thr_h)
    print(report)
    print('confusion matrix:\n', cm)


if __name__ == '__main__':
    main()
