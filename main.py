import cv2
import numpy as np
import os
from check_for_holes import check_for_holes
#from scripts.avs_detector import show_hole_detection

def main():
    '''
    Bierzemy zbiory walidacyjne

    iterujemy po zdjęciach
        1. Czy są włoski
        2. Czy są dziury
        3. Czy są obiekty dziwne
    
    pokazujemy każde zdjęcie z napisem czy jest OK czy NOT OK (ewentualnie błędy)

    liczymy confusion matrix i wskaźnik F1

    '''


    # for root, _, files in os.walk('split_dataset/val'):
    for root, _, files in os.walk('split_dataset/train/defective'):

        for f in files:

            if not f.lower().endswith('.png'):
                continue
            in_path = os.path.join(root, f)

            #show_hole_detection(in_path)

            defect = False

            defect, mask, debug = check_for_holes(in_path)

            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            text = "DEFECT" if defect else "OK"

            cv2.putText(img, text,(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,
                        (0,0,255) if defect else (0,255,0),2)

            scale = 800 / img.shape[1]
            img = cv2.resize(img,None,fx=scale,fy=scale)
            mask = cv2.resize(mask,None,fx=scale,fy=scale)
            debug = cv2.resize(debug,None,fx=scale,fy=scale)

            cv2.imshow("result", img)
            cv2.imshow("mask", mask)
            cv2.imshow("debug contours", debug)

            if cv2.waitKey(0) == 27:
                break

            cv2.destroyAllWindows()
                

if __name__ == '__main__':
    main()
