"""Run the pipeline demo:
- create splits (data/)
- preprocess images (processed/)
- extract density features and run rule-based detector
"""
from scripts.data_split import split_dataset
from scripts.preprocess import preprocess_image
import os

def main():
    base = os.path.abspath(os.path.dirname(__file__))

    # 1) create splits
    src = os.path.join(base, 'original_dataset/train')
    dst = os.path.join(base, 'split_dataset')
    split_dataset(src, dst, val_frac=0.2)

    # 2) preprocess train images
    for root, _, files in os.walk(dst):
        for f in files:
            if not f.lower().endswith('.png'):
                continue
            in_path = os.path.join(root, f)
            preprocess_image(in_path, in_path)
   
if __name__ == '__main__':
    main()
