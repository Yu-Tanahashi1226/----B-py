import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt") 
VIDEO_PATH_3A = "ex3a.mp4"
VIDEO_PATH_3B = "ex3b.mp4"

LEFT_HIP_IDX = 11  # 左腰
RIGHT_HIP_IDX = 12 # 右腰

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


def draw_skeleton(frame, keypoints, pose_pairs, keypoint_color, bone_color, keypoint_radius=5, bone_thickness=3, exclude_keypoints=None):
    if exclude_keypoints is None:
        exclude_keypoints = []

    if keypoints.ndim == 2:
        keypoints = keypoints.reshape(-1, 17, 3)
    elif keypoints.ndim == 3 and keypoints.shape[1] != 17:
        keypoints = np.transpose(keypoints, (0, 2, 1))
        
    for person_kpts in keypoints:
        for pair in pose_pairs:
            idx1, idx2 = pair
            
            if idx1 in exclude_keypoints or idx2 in exclude_keypoints:
                continue

            pt1 = person_kpts[idx1]
            pt2 = person_kpts[idx2]
            
            x1, y1, score1 = pt1
            x2, y2, score2 = pt2

            if score1 < 0.5 or score2 < 0.5:
                continue
            
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            
            cv2.line(frame, p1, p2, bone_color, bone_thickness)

        for i, (x, y, score) in enumerate(person_kpts):
            if i in exclude_keypoints or score < 0.5:
                continue
            
            center = (int(x), int(y))
            cv2.circle(frame, center, keypoint_radius, keypoint_color, -1)


cap1 = cv2.VideoCapture(VIDEO_PATH_3A)
cap2 = cv2.VideoCapture(VIDEO_PATH_3B)

if not cap1.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH_3A}' を開けませんでした。")
if not cap2.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH_3B}' を開けませんでした。")
    exit()

WINDOW_NAME = "Combined Skeleton Task 3c (Aligned)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cap1_active = cap1.isOpened()
target_size = (1000, 1000)

while True:
    
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    if cap1_active:
        ret1, frame1 = cap1.read()
        if not ret1:
            print(f"'{VIDEO_PATH_3A}' の終端に達しました。以降、3aの骨格検出はスキップします。")
            cap1_active = False
        else:
            frame1 = cv2.resize(frame1, target_size)
    
    frame2 = cv2.resize(frame2, target_size)
    output_frame = frame2.copy()
    output_frame[:] = (127, 127, 127)

    keypoints1 = None
    if cap1_active:
        results1 = model.predict(frame1, verbose=False)
        if results1 and results1[0].keypoints is not None and results1[0].keypoints.data.shape[0] > 0:
            keypoints1 = results1[0].keypoints.data.cpu().numpy()
    
    keypoints2 = None
    results2 = model.predict(frame2, verbose=False)
    if results2 and results2[0].keypoints is not None and results2[0].keypoints.data.shape[0] > 0:
        keypoints2 = results2[0].keypoints.data.cpu().numpy()
    
    
    if keypoints1 is not None and keypoints2 is not None:
        
        hip_1_left = keypoints1[0, LEFT_HIP_IDX, :2]
        hip_1_right = keypoints1[0, RIGHT_HIP_IDX, :2]
        center_hip_1 = (hip_1_left + hip_1_right) / 2
        
        hip_2_left = keypoints2[0, LEFT_HIP_IDX, :2]
        hip_2_right = keypoints2[0, RIGHT_HIP_IDX, :2]
        center_hip_2 = (hip_2_left + hip_2_right) / 2
        
        translation_vector = center_hip_1 - center_hip_2
        
        keypoints2[0, :, 0:2] = keypoints2[0, :, 0:2] + translation_vector
        
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints1, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255),
            bone_color=(0, 0, 255),
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )
        
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints2, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255),
            bone_color=(255, 0, 0),
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )
    
    elif keypoints2 is not None:

        keypoints2[0, :, 0:2] = keypoints2[0, :, 0:2] + translation_vector
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints2, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255), 
            bone_color=(255, 0, 0),       
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )

    
    cv2.imshow(WINDOW_NAME, output_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("処理を終了しました。")