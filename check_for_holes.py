import os
import cv2
import numpy as np


import cv2
import numpy as np


def check_for_holes(path):

    img = cv2.imread(path)
    if img is None:
        return False, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- 1. wykrycie główki szczoteczki ----------
    blur = cv2.GaussianBlur(gray,(21,21),0)

    _, head = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    head = cv2.morphologyEx(head,cv2.MORPH_CLOSE,kernel)

    contours,_ = cv2.findContours(head,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False, head, img

    head_contour = max(contours, key=cv2.contourArea)

    mask_head = np.zeros_like(gray)
    cv2.drawContours(mask_head,[head_contour],-1,255,-1)

    # ograniczamy analizę tylko do główki
    roi = cv2.bitwise_and(gray,gray,mask=mask_head)

    # ---------- 2. znajdowanie centrów pęczków ----------
    blur = cv2.GaussianBlur(roi,(11,11),0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    local_max = cv2.dilate(blur,kernel)

    peaks = (blur == local_max) & (blur > 120)
    peaks = peaks.astype(np.uint8)*255

    peaks = cv2.morphologyEx(
        peaks,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    )

    contours,_ = cv2.findContours(peaks,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:

        area = cv2.contourArea(c)
        if area < 50 or area > 2000:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        centers.append((cx,cy))

    # ---------- 3. sprawdzanie dziur ----------
    debug = img.copy()
    mask = peaks.copy()

    holes = 0

    for (x,y) in centers:

        r = 12
        patch = gray[y-r:y+r, x-r:x+r]

        if patch.size == 0:
            continue

        mean_val = np.mean(patch)

        if mean_val < 120:
            holes += 1
            cv2.circle(debug,(x,y),18,(0,0,255),2)
        else:
            cv2.circle(debug,(x,y),18,(0,255,0),1)

    defect = holes > 0

    return defect, mask, debug



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