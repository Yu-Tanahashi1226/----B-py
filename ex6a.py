import cv2
import numpy as np
import matplotlib.pyplot as plt



pts1 = np.array([
    [959, 67],    # ペナルティエリア上角
    [693, 140],  # サイドライン側上
    [1088, 114],  # サイドライン側下
    [1157, 424]    # ペナルティエリア下角
], dtype=np.float32)

# 3. 出力画像(soccer_field.png)上の対応点
# 右ペナルティエリアの同じ位置
pts2 = np.array([
    [3140, 227],   # ペナルティエリア上角
    [2723, 575],   # サイドライン側上
    [3143, 575],   # サイドライン側下
    [2721, 1625]    # ペナルティエリア下角
], dtype=np.float32)

# 4. 射影変換行列
M = cv2.getPerspectiveTransform(pts1, pts2)
np.set_printoptions(precision=5, suppress=True)
print(M)

img1 = cv2.imread("ex4-25.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

w2,h2 = pts2.max(axis=0).astype(int) 

h2+=750

img2 = cv2.warpPerspective(img1, M, (w2, h2))

def transform_pt(pt, M):
    pt = np.append(pt, 1.0)
    pt = np.dot(M, pt)  
    pt = pt / pt[2] 
    pt = pt[:2]  
    return pt

fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1).imshow(img1)
fig.add_subplot(1, 2, 2).imshow(img2)
plt.show()
