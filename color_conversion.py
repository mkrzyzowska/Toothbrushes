import cv2
import os

src_root = "original_dataset/train"

# color_spaces = {
#     "HSV": cv2.COLOR_BGR2HSV,
#     "HLS": cv2.COLOR_BGR2HLS,
#     "LAB": cv2.COLOR_BGR2LAB,
#     "YCrCb": cv2.COLOR_BGR2YCrCb
# }

classes = ["good", "defective"]

# for space, code in color_spaces.items():

#     dst_root = f"dataset_{space}/train"

#     for cls in classes:

#         src_dir = os.path.join(src_root, cls)
#         dst_dir = os.path.join(dst_root, cls)

#         os.makedirs(dst_dir, exist_ok=True)

#         for file in os.listdir(src_dir):

#             if not file.endswith(".png"):
#                 continue

#             path = os.path.join(src_dir, file)

#             img = cv2.imread(path)

#             converted = cv2.cvtColor(img, code)

#             out_path = os.path.join(dst_dir, file)

#             cv2.imwrite(out_path, converted)


dst_root = "dataset_HLS_GRAY/train"

for cls in classes:

    src_dir = os.path.join(src_root, cls)
    dst_dir = os.path.join(dst_root, cls)

    os.makedirs(dst_dir, exist_ok=True)

    for file in os.listdir(src_dir):

        if not file.endswith(".png"):
            continue

        path = os.path.join(src_dir, file)

        img = cv2.imread(path)

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(hls, cv2.COLOR_BGR2GRAY)

        out_path = os.path.join(dst_dir, file)
        cv2.imwrite(out_path, gray)