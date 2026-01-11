import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO # YOLOモデルの使用

# 1. 射影変換行列 M の定義 (ex6a.py の対応点を使用)
pts1 = np.array([[959, 67], [693, 140], [1088, 114], [1157, 424]], dtype=np.float32)
pts2 = np.array([[3140, 227], [2723, 575], [3143, 575], [2721, 1625]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts1, pts2)

# 2. 画像とモデルの読み込み
model = YOLO("yolov8n.pt")
img_src = cv2.imread("ex4-25.jpg")
img_dst = cv2.imread("soccer_field.png")

if img_src is None or img_dst is None:
    print("エラー: 画像ファイルが読み込めませんでした。")
    exit()

# 3. YOLOによる人物検出 (閾値を 0.25 に設定)
# conf パラメータで閾値を指定します
results = model(img_src, conf=0.25) 

# 4. 座標変換とプロット
for box in results[0].boxes:
    class_id = int(box.cls[0])
    
    # class_id == 0 (person) のみを処理
    if class_id == 0:
        # バウンディングボックスの座標を取得
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        # 足元の座標 (底辺の中央) を同次座標系で定義
        foot_pt = np.array([(x1 + x2) / 2.0, y2, 1.0], dtype=np.float32)
        
        # 射影変換行列 M を用いた行列演算 (課題4の応用)
        transformed_h = np.dot(M, foot_pt)
        
        # 正規化して2次元座標に変換
        dst_x = int(transformed_h[0] / transformed_h[2])
        dst_y = int(transformed_h[1] / transformed_h[2])
        
        # サッカーコート画像の範囲内であれば描画
        if 0 <= dst_x < img_dst.shape[1] and 0 <= dst_y < img_dst.shape[0]:
            # マゼンタ色の円を描画
            cv2.circle(img_dst, (dst_x, dst_y), 35, (255, 0, 255), 8)

# 5. 結果の表示
img_dst_rgb = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))
plt.imshow(img_dst_rgb)
plt.title("YOLO Detection (conf=0.25) Mapped to Soccer Field")
plt.axis('off')
plt.show()