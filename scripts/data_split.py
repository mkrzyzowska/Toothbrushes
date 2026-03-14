import os
import random
import shutil

def split_dataset(source_root, dest_root, val_frac=0.2, seed=42):
    '''
    Takes data and splits into train (80%) and val (20%) folders
    '''
    random.seed(seed)
    classes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    for cls in classes:
        src_dir = os.path.join(source_root, cls)
        files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith('.png')])
        random.shuffle(files)
        n_val = int(len(files) * val_frac)
        val_files = files[:n_val]
        train_files = files[n_val:]
        for split_name, split_files in [('train', train_files), ('val', val_files)]:
            out_dir = os.path.join(dest_root, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for fname in split_files:
                shutil.copyfile(os.path.join(src_dir, fname), os.path.join(out_dir, fname))

if __name__ == '__main__':
    src = os.path.join(os.path.dirname(__file__), '..', 'train')
    src = os.path.abspath(src)
    dst = os.path.join(os.path.dirname(__file__), '..', 'data')
    dst = os.path.abspath(dst)
    print('Splitting', src, '->', dst)
    split_dataset(src, dst, val_frac=0.2)
