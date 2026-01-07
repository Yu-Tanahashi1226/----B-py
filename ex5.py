import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

VIDEO_PATH = "ex5.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: 動画ファイル '{VIDEO_PATH}' を開けませんでした。")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame, conf=0.2)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            
            if class_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow("Task 5 - Person Detection in Video", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

print("動画の処理が完了しました。")