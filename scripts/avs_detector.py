import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def fiber_score(img_path, crop_x=200):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    if w > 2 * crop_x:
        img = img[:, crop_x:-crop_x]

    img = cv2.GaussianBlur(img, (5,5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    _, th = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th)

    score = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 < area < 400:
            score += area

    return score


def gather_features(dataset_path):

    rows = []

    for label_name in os.listdir(dataset_path):

        label_path = os.path.join(dataset_path, label_name)

        if not os.path.isdir(label_path):
            continue

        label = 0 if label_name.lower() == "good" else 1

        for f in os.listdir(label_path):

            if not f.lower().endswith(".png"):
                continue

            img_path = os.path.join(label_path, f)

            score = fiber_score(img_path)

            rows.append({
                "file": f,
                "fiber_score": score,
                "label": label
            })

    return pd.DataFrame(rows)


def train_thresholds(df_train, k_area=2.0, k_holes=3.0):

    good = df_train[df_train["label"] == 0]

    mean = good["fiber_score"].mean()
    std = good["fiber_score"].std()

    thr = mean + k_area * std

    return thr, 0


def evaluate(df, thr_area, thr_holes):

    preds = (df["fiber_score"] > thr_area).astype(int)

    report = classification_report(df["label"], preds)
    cm = confusion_matrix(df["label"], preds)

    return report, cm