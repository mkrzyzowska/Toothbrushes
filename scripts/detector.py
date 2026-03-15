import os
import pandas as pd
from .features import extract_features
from sklearn.metrics import classification_report, confusion_matrix


def _mask_for_filename(f, gt_root):
    name = os.path.splitext(f)[0]
    candidate = os.path.join(gt_root, f'{name}_mask.png')
    return candidate if gt_root is not None and os.path.exists(candidate) else None


def gather_features(data_root, gt_root=None):
    rows = []
    for cls in ['good', 'defective']:
        dirpath = os.path.join(data_root, cls)
        if not os.path.isdir(dirpath):
            continue
        for f in sorted(os.listdir(dirpath)):
            if not f.lower().endswith('.png'):
                continue
            img_path = os.path.join(dirpath, f)
            mask_path = None
            if gt_root is not None and cls == 'defective':
                mask_path = _mask_for_filename(f, gt_root)
            feats = extract_features(img_path, mask_path=mask_path)
            label = 1 if cls == 'defective' else 0
            row = {'file': img_path, 'label': label}
            row.update(feats)
            rows.append(row)
    return pd.DataFrame(rows)


def train_rule_thresholds(df, k_density=1.5, k_void=1.5, k_area=2.0):
    goods = df[df['label'] == 0]
    d_mean = goods['density'].mean()
    d_std = goods['density'].std()
    v_mean = goods['void_frac'].mean()
    v_std = goods['void_frac'].std()

    a_mean = goods['bristle_area'].mean()
    a_std = goods['bristle_area'].std()

    thr_density = float(d_mean - k_density * (d_std if not pd.isna(d_std) else 0.0))
    thr_void = float(v_mean + k_void * (v_std if not pd.isna(v_std) else 0.0))

    thr_area = float(a_mean + k_area * (a_std if not pd.isna(a_std) else 0.0))
    return thr_density, thr_void


def evaluate(df, thr_density, thr_void, thr_area):
    preds = ((df['density'] < thr_density) | (df['void_frac'] > thr_void) | (df['bristle_area'] > thr_area)).astype(int)
    y_true = df['label'].astype(int)
    report = classification_report(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    return report, cm


if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_train = os.path.join(base, 'data', 'train')
    gt = os.path.join(base, 'ground_truth', 'defective')
    df = gather_features(data_train, gt_root=gt)
    thr_d, thr_v , thr_a= train_rule_thresholds(df)
    print('thresholds: density=', thr_d, ' void=', thr_v)
    report, cm = evaluate(df, thr_d, thr_v, thr_a)
    print(report)
    print('confusion matrix:\n', cm)
