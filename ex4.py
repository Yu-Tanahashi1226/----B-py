import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

IMAGE_PATH = "ex4-25.jpg"
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: 画像ファイル '{IMAGE_PATH}' を読み込めませんでした。")
    exit()

results = model(image,conf=0.2)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    
    if class_id == 0:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Detection - Person", image)
cv2.imwrite("ex4_result.jpg", image)

print("描画が完了しました。何かキーを押すと終了します。")
cv2.waitKey(0)
cv2.destroyAllWindows()