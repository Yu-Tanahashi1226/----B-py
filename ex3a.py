import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt") 
VIDEO_PATH = "ex3a.mp4"

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

FACE_KEYPOINT_INDICES = [0, 1, 2, 3, 4]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH}' を開けませんでした。")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)

        if results and results[0].boxes.data.shape[0] > 0:
            
            nodes = results[0].keypoints.xy[0] 

            NON_FACE_KEYPOINT_INDICES = [i for i in range(len(nodes)) if i not in FACE_KEYPOINT_INDICES]

            cv2.rectangle(frame,
              pt1=(0, 0),
              pt2=(1000, 1000),
              color=(127, 127, 127),
              thickness=1000,
              lineType=cv2.LINE_4,
              shift=0)

            for n1_idx, n2_idx in POSE_PAIRS:
                pt1 = nodes[n1_idx]
                pt2 = nodes[n2_idx]

                if pt1[0] * pt1[1] * pt2[0] * pt2[1] == 0:
                    continue

                start_point = tuple(pt1[:2].to(torch.int).tolist())
                end_point = tuple(pt2[:2].to(torch.int).tolist())

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 0, 255),
                    thickness=3,
                )

            for i in NON_FACE_KEYPOINT_INDICES:
                point = nodes[i]
                x, y = point[:2]

                if x.item() * y.item() == 0:
                    continue

                center = (int(x.item()), int(y.item()))
                
                cv2.circle(frame, center, 5, (0, 255, 255), thickness=-1)

        cv2.imshow("Task 2: Video Pose Estimation", frame)
        
        if cv2.waitKey(20) & 0xFF == 27: 
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()