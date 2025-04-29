from ultralytics import YOLO
import cv2
import csv
import os

# モデルロード（Pose推定モデル）
pose_model = YOLO('yolov8n-pose.pt')  # or 最新Poseモデル

# 動画ファイル
video_path = r'C:\Users\ohkub\Videos\Street Fighter 6\hadoken.mp4'

# 保存先ファイル
csv_output_path = r'C:\develop\Streetfighter\gouki_pose_data.csv'

# 動画読み込み
cap = cv2.VideoCapture(video_path)

frame_idx = 0

# CSVに保存
with open(csv_output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # ヘッダー
    header = ['frame', 'hip_x', 'hip_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y']
    writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose推論
        results = pose_model(frame)
        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0:
            person = keypoints[0]

            if person.shape[0] > 10:
                hip = person[8]
                left_wrist = person[9]
                right_wrist = person[10]

                row = [frame_idx, hip[0].item(), hip[1].item(), left_wrist[0].item(), left_wrist[1].item(), right_wrist[0].item(), right_wrist[1].item()]
                writer.writerow(row)

        frame_idx += 1  # ★必ずここでインクリメント


cap.release()

print(f"骨格データ保存完了: {csv_output_path}")
