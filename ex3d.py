import cv2
import torch
from ultralytics import YOLO
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

model = YOLO("yolov8n-pose.pt")
VIDEO_PATH_3A = "ex3a.mp4"
VIDEO_PATH_3B = "ex3b.mp4"

# キーポイントのインデックス
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# 描画するボーンのペア
POSE_PAIRS = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]

# 除外する顔のパーツ
FACE_KEYPOINT_INDICES = [0, 1, 2, 3, 4]
target_size = (1000, 1000)
CENTER_X = target_size[0] // 2
CENTER_Y = target_size[1] // 2
REFERENCE_HIP_CENTER = np.array([CENTER_X, CENTER_Y], dtype=np.float32)


def draw_skeleton(frame, keypoints, pose_pairs, keypoint_color, bone_color,
                  keypoint_radius=5, bone_thickness=3, exclude_keypoints=None):
    if exclude_keypoints is None:
        exclude_keypoints = []

    if keypoints.ndim == 2:
        keypoints = keypoints.reshape(-1, 17, 3)
    elif keypoints.ndim == 3 and keypoints.shape[1] != 17:
        keypoints = np.transpose(keypoints, (0, 2, 1))

    for person_kpts in keypoints:
        for idx1, idx2 in pose_pairs:
            if idx1 in exclude_keypoints or idx2 in exclude_keypoints:
                continue

            x1, y1, s1 = person_kpts[idx1]
            x2, y2, s2 = person_kpts[idx2]

            if s1 < 0.5 or s2 < 0.5:
                continue

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     bone_color, bone_thickness)

        for i, (x, y, score) in enumerate(person_kpts):
            if i in exclude_keypoints or score < 0.5:
                continue
            cv2.circle(frame, (int(x), int(y)), keypoint_radius,
                       keypoint_color, -1)


def extract_and_normalize_keypoints(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: 動画 '{video_path}' を開けません。")
        return []

    data_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_size)
        results = model.predict(frame, verbose=False)

        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.data.cpu().numpy()

            if kpts.ndim == 3:
                kpts = kpts[0]
            elif kpts.ndim == 2:
                pass
            else:
                continue

            hip_left = kpts[LEFT_HIP_IDX, :2]
            hip_right = kpts[RIGHT_HIP_IDX, :2]
            center_hip = (hip_left + hip_right) / 2

            normalized = kpts[:, :2] - center_hip
            data_sequence.append(normalized.flatten())

    cap.release()
    return np.array(data_sequence)


print("骨格データ抽出中...")
sequence_a = extract_and_normalize_keypoints(VIDEO_PATH_3A, model)
sequence_b = extract_and_normalize_keypoints(VIDEO_PATH_3B, model)

if len(sequence_a) == 0 or len(sequence_b) == 0:
    print("Error: 骨格抽出に失敗。")
    exit()

print("動画Aフレーム数:", len(sequence_a))
print("動画Bフレーム数:", len(sequence_b))

def vector_distance(a, b):
    return np.linalg.norm(a - b)

print("DTW 計算中...")
distance, path = fastdtw(sequence_a, sequence_b, dist=vector_distance)
print("DTW 距離:", distance)
print("パス長:", len(path))

path_a = [p[0] for p in path]
path_b = [p[1] for p in path]


cap1 = cv2.VideoCapture(VIDEO_PATH_3A)
cap2 = cv2.VideoCapture(VIDEO_PATH_3B)
WINDOW_NAME = "DTW Aligned Skeleton"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

dtw_path_index = 0
total_dtw_steps = len(path)

print("描画開始...")

while dtw_path_index < total_dtw_steps:

    idx_a = path_a[dtw_path_index]
    idx_b = path_b[dtw_path_index]

    cap1.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, idx_b)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        dtw_path_index += 1
        continue

    frame1 = cv2.resize(frame1, target_size)
    frame2 = cv2.resize(frame2, target_size)

    r1 = model.predict(frame1, verbose=False)
    r2 = model.predict(frame2, verbose=False)

    output_frame = np.zeros_like(frame1) + 127

    if r1 and r1[0].keypoints is not None:
        keypoints1 = r1[0].keypoints.data.cpu().numpy()
        if keypoints1.ndim == 3:
            keypoints1 = keypoints1[0]

        hip_l = keypoints1[LEFT_HIP_IDX, :2]
        hip_r = keypoints1[RIGHT_HIP_IDX, :2]
        center = (hip_l + hip_r) / 2
        translation = REFERENCE_HIP_CENTER - center

        k1 = keypoints1.copy()
        k1[:, :2] += translation
        draw_skeleton(output_frame, k1.reshape(1, 17, 3),
                      POSE_PAIRS, (0, 255, 255), (0, 0, 255),
                      exclude_keypoints=FACE_KEYPOINT_INDICES)

    if r2 and r2[0].keypoints is not None:
        keypoints2 = r2[0].keypoints.data.cpu().numpy()
        if keypoints2.ndim == 3:
            keypoints2 = keypoints2[0]

        hip_l = keypoints2[LEFT_HIP_IDX, :2]
        hip_r = keypoints2[RIGHT_HIP_IDX, :2]
        center = (hip_l + hip_r) / 2
        translation = REFERENCE_HIP_CENTER - center

        k2 = keypoints2.copy()
        k2[:, :2] += translation
        draw_skeleton(output_frame, k2.reshape(1, 17, 3),
                      POSE_PAIRS, (0, 255, 255), (255, 0, 0),
                      exclude_keypoints=FACE_KEYPOINT_INDICES)

    cv2.imshow(WINDOW_NAME, output_frame)

    if cv2.waitKey(30) == 27:
        break

    dtw_path_index += 1


cap1.release()
cap2.release()
cv2.destroyAllWindows()

print("終了しました。")