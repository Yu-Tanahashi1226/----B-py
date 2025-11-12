import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt") 

IMAGE_PATH = "ex1.jpg"
results = model(IMAGE_PATH)

if not results or results[0].boxes.data.shape[0] == 0:
    print("Error: 画像から人物を検出できませんでした。")
    exit()

img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Error: 画像ファイルが見つからないか、読み込めません。")
    exit()

nodes = results[0].keypoints.xy[0] 

FACE_KEYPOINT_INDICES = [0, 1, 2, 3, 4]
NON_FACE_KEYPOINT_INDICES = [i for i in range(len(nodes)) if i not in FACE_KEYPOINT_INDICES]

POSE_PAIRS = [
    [5, 7],   # 左肩 - 左肘
    [6, 8],   # 右肩 - 右肘
    [7, 9],   # 左肘 - 左手首
    [8, 10],  # 右肘 - 右手首
    [11, 13], # 左腰 - 左膝
    [12, 14], # 右腰 - 右膝
    [13, 15], # 左膝 - 左足首
    [14, 16], # 右膝 - 右足首
    [5, 11],  # 左肩 - 左腰
    [6, 12],  # 右肩 - 右腰
    [5, 6],   # 左肩 - 右肩
    [11, 12], # 左腰 - 右腰
]

for n1_idx, n2_idx in POSE_PAIRS:
    pt1 = nodes[n1_idx]
    pt2 = nodes[n2_idx]

    if pt1[0] * pt1[1] * pt2[0] * pt2[1] == 0:
        continue

    start_point = pt1[:2].to(torch.int).tolist()
    end_point = pt2[:2].to(torch.int).tolist()

    cv2.line(
        img,
        start_point,
        end_point,
        (0, 0, 255),
        thickness=3,
    )

for i in NON_FACE_KEYPOINT_INDICES:
    point = nodes[i]
    x, y = point[:2]

    if x * y == 0:
        continue

    center = (int(x.item()), int(y.item()))    
    cv2.circle(img, center, 5, (0, 255, 255), thickness=-1)


cv2.imshow("Task 1: Pose Estimation Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()