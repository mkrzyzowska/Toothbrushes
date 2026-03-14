import os
import cv2
import numpy as np

def check_for_holes(in_path):
    ref = cv2.imread("check_for_holes_reference.png")

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

    _, mask = cv2.threshold(img.astype(np.uint8) ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    scale = 800 / mask.shape[1]

    cv2.imshow("mask", cv2.resize(mask,None,fx=scale,fy=scale))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    img_po_morph= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    img_po_morph= cv2.morphologyEx(img_po_morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imshow("img_po_morph", cv2.resize(img_po_morph,None,fx=scale,fy=scale))


    return True




def main():
    imgs = []
    
    for root, _, files in os.walk('split_dataset/train/good'):
        for f in files:
            if not f.lower().endswith('.png'):
                continue

            path = os.path.join(root, f)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            imgs.append(img.astype(np.float32))

    if len(imgs) == 0:
        print("No images found")
        return

    avg = np.mean(imgs, axis=0)

    avg = np.clip(avg, 0, 255).astype(np.uint8)

    _, mask = cv2.threshold(avg ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    img_po_morph= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    img_po_morph= cv2.morphologyEx(img_po_morph, cv2.MORPH_CLOSE, kernel, iterations=1)


    cv2.imwrite("check_for_holes_reference.png", img_po_morph)


if __name__ == '__main__':
    main()