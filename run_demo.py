"""Run the pipeline demo:
- create splits (data/)
- preprocess images (processed/)
- extract density features and run rule-based detector
"""
from scripts.data_split import split_dataset
from scripts.preprocess import batch_preprocess
from scripts.detector import gather_features, train_rule_thresholds, evaluate
import os

def main():
    base = os.path.abspath(os.path.dirname(__file__))
    # 1) create splits
    src_train = os.path.join(base, 'train')
    dst = os.path.join(base, 'data')
    split_dataset(src_train, dst, val_frac=0.2)
    # 2) preprocess train images
    proc_train = os.path.join(base, 'processed', 'train')
    batch_preprocess(os.path.join(dst, 'train'), proc_train)
    # 3) extract features and evaluate (use processed images if present)
    data_train = os.path.join(base, 'data', 'train')
    processed_train = os.path.join(base, 'processed', 'train')
    if os.path.isdir(processed_train):
        # processed structure mirrors data/train/<class>/*
        df = gather_features(processed_train, gt_root=os.path.join(base, 'ground_truth', 'defective'))
    else:
        df = gather_features(data_train, gt_root=os.path.join(base, 'ground_truth', 'defective'))
    thr_d, thr_v , thr_a= train_rule_thresholds(df)
    report, cm = evaluate(df, thr_d, thr_v, thr_a)
    print('--- Rule-based detector results ---')
    print('thresholds: density=', thr_d, ' void=', thr_v)
    print(report)
    print('confusion matrix:\n', cm)

if __name__ == '__main__':
    main()
