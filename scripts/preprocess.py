import cv2
import os

def preprocess_image(in_path, out_path, clahe_clip=2.0, clahe_tile=(8,8)):
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(blurred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, enhanced)
    return out_path

def batch_preprocess(src_dir, dst_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith('.png'):
                continue
            rel = os.path.relpath(root, src_dir)
            out_sub = os.path.join(dst_dir, rel)
            os.makedirs(out_sub, exist_ok=True)
            in_path = os.path.join(root, f)
            out_path = os.path.join(out_sub, f)
            preprocess_image(in_path, out_path)

if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src = os.path.join(base, 'data', 'train')
    dst = os.path.join(base, 'processed', 'train')
    batch_preprocess(src, dst)
