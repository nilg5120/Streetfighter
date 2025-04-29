import cv2
import pandas as pd
import joblib
from ultralytics import YOLO

# モデルをロード
detect_model = YOLO(r'C:\Users\ohkub\runs\detect\train5\weights\best.pt')    # 豪鬼検出モデル
pose_model = YOLO('yolov8n-pose.pt')                                          # 骨格推定モデル
motion_model = joblib.load(r'C:\develop\Streetfighter\gouki_motion_model.pkl') # 動き分類モデル

# 動画ファイル
video_path = r'C:\Users\ohkub\Videos\Street Fighter 6\hadoken.mp4'

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 豪鬼を検出
    results = detect_model(frame)
    boxes = results[0].boxes.xyxy

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # 豪鬼の範囲だけクロップ
            gouki_crop = frame[y1:y2, x1:x2]

            # クロップに対してPose推定
            pose_results = pose_model(gouki_crop)
            keypoints = pose_results[0].keypoints.xy

            if len(keypoints) > 0:
                person = keypoints[0]

                if person.shape[0] > 10:
                    hip = person[8]
                    left_wrist = person[9]
                    right_wrist = person[10]

                    # 特徴量作成
                    features = pd.DataFrame([{
                        'hip_x': hip[0].item(),
                        'hip_y': hip[1].item(),
                        'left_wrist_x': left_wrist[0].item(),
                        'left_wrist_y': left_wrist[0].item(),
                        'right_wrist_x': right_wrist[0].item(),
                        'right_wrist_y': right_wrist[1].item(),
                    }])

                    # 推論
                    prediction = motion_model.predict(features)[0]
                    label_map = {0: '波動拳', 1: '昇竜拳'}
                    motion_name = label_map[prediction]

                    # 表示
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, motion_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gouki Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
