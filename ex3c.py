import cv2
import torch
from ultralytics import YOLO
import numpy as np

# --- 1. 定数とモデルの準備 ---
model = YOLO("yolov8n-pose.pt") 
VIDEO_PATH_3A = "ex3a.mp4"
VIDEO_PATH_3B = "ex3b.mp4"

# 17キーポイントのインデックス定義
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
    """フレーム上に骨格を描画する関数"""
    if exclude_keypoints is None:
        exclude_keypoints = []

    # キーポイントデータの整形 (1, 17, 3) へ
    if keypoints.ndim == 2:
        keypoints = keypoints.reshape(-1, 17, 3)
    elif keypoints.ndim == 3 and keypoints.shape[1] != 17:
        keypoints = np.transpose(keypoints, (0, 2, 1))
        
    for person_kpts in keypoints:
        # ボーン（骨）の描画
        for pair in pose_pairs:
            idx1, idx2 = pair
            
            if idx1 in exclude_keypoints or idx2 in exclude_keypoints:
                continue

            pt1 = person_kpts[idx1]
            pt2 = person_kpts[idx2]
            
            x1, y1, score1 = pt1
            x2, y2, score2 = pt2

            # スコアが閾値未満の場合は描画しない
            if score1 < 0.5 or score2 < 0.5:
                continue
            
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            
            cv2.line(frame, p1, p2, bone_color, bone_thickness)

        # キーポイント（点）の描画
        for i, (x, y, score) in enumerate(person_kpts):
            if i in exclude_keypoints or score < 0.5:
                continue
            
            center = (int(x), int(y))
            cv2.circle(frame, center, keypoint_radius, keypoint_color, -1)


# --- 2. 動画の読み込み ---
cap1 = cv2.VideoCapture(VIDEO_PATH_3A)
cap2 = cv2.VideoCapture(VIDEO_PATH_3B)

if not cap1.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH_3A}' を開けませんでした。")
    # 3bの処理を続けるためexit()はしない
if not cap2.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH_3B}' を開けませんでした。")
    exit()

WINDOW_NAME = "Combined Skeleton Task 3c (Aligned)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# 処理状態フラグ
cap1_active = cap1.isOpened()
target_size = (1000, 1000)

# --- 3. メイン処理ループ ---
while True:
    
    # 3bのフレーム読み込み (必須)
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # 3aのフレーム読み込み
    if cap1_active:
        ret1, frame1 = cap1.read()
        if not ret1:
            print(f"'{VIDEO_PATH_3A}' の終端に達しました。以降、3aの骨格検出はスキップします。")
            cap1_active = False
        else:
            frame1 = cv2.resize(frame1, target_size)
    
    # 3bのフレームをリサイズし、出力フレームのベースとする
    frame2 = cv2.resize(frame2, target_size)
    output_frame = frame2.copy()
    output_frame[:] = (127, 127, 127) # 背景を灰色で塗りつぶし

    # --- 骨格検出 ---
    # frame1の検出 (アクティブな場合のみ)
    keypoints1 = None
    if cap1_active:
        results1 = model.predict(frame1, verbose=False)
        if results1 and results1[0].keypoints is not None and results1[0].keypoints.data.shape[0] > 0:
            # 最初の人物のキーポイントを取得 (Numpy配列)
            keypoints1 = results1[0].keypoints.data.cpu().numpy() # (1, 17, 3) 形式
    
    # frame2の検出
    keypoints2 = None
    results2 = model.predict(frame2, verbose=False)
    if results2 and results2[0].keypoints is not None and results2[0].keypoints.data.shape[0] > 0:
        keypoints2 = results2[0].keypoints.data.cpu().numpy() # (1, 17, 3) 形式
    
    
    # --- 4. 腰の位置を揃える処理 (アライメント) ---
    if keypoints1 is not None and keypoints2 is not None:
        
        # 1. 骨格1 (keypoints1) の腰の中心を基準点とする
        hip_1_left = keypoints1[0, LEFT_HIP_IDX, :2]
        hip_1_right = keypoints1[0, RIGHT_HIP_IDX, :2]
        center_hip_1 = (hip_1_left + hip_1_right) / 2
        
        # 2. 骨格2 (keypoints2) の腰の中心を計算
        hip_2_left = keypoints2[0, LEFT_HIP_IDX, :2]
        hip_2_right = keypoints2[0, RIGHT_HIP_IDX, :2]
        center_hip_2 = (hip_2_left + hip_2_right) / 2
        
        # 3. 骨格2を骨格1の腰に合わせるための移動ベクトルを計算
        # 移動ベクトル = 基準点 - 対象点の中心
        translation_vector = center_hip_1 - center_hip_2
        
        # 4. 骨格2の全キーポイントのx, y座標に移動ベクトルを適用
        # keypoints2 は (1, 17, 3)
        keypoints2[0, :, 0:2] = keypoints2[0, :, 0:2] + translation_vector
        
        # 描画 (3a: 青, 3b: 赤)
        # 骨格1 (3a) を青色で描画
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints1, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255), # キーポイント: 青
            bone_color=(0, 0, 255),     # ボーン: 青
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )
        
        # 骨格2 (3b, 位置調整済み) を赤色で描画
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints2, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255), # キーポイント: 黄色 (出力例に合わせて)
            bone_color=(255, 0, 0),       # ボーン: 赤色 (出力例に合わせて)
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )
    
    elif keypoints2 is not None:

        keypoints2[0, :, 0:2] = keypoints2[0, :, 0:2] + translation_vector
        # keypoints1が検出されない場合、keypoints2のみを描画 (位置調整なし)
        # 骨格2 (3b) を赤色で描画
        draw_skeleton(
            frame=output_frame, 
            keypoints=keypoints2, 
            pose_pairs=POSE_PAIRS, 
            keypoint_color=(0, 255, 255), 
            bone_color=(255, 0, 0),       
            exclude_keypoints=FACE_KEYPOINT_INDICES
        )

    
    # フレーム表示
    cv2.imshow(WINDOW_NAME, output_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

# --- 5. 終了時処理 ---
cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("処理を終了しました。")