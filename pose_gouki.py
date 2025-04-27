from ultralytics import YOLO
import cv2
import os

# Pose推定用モデルをロード
model = YOLO('yolov8n-pose.pt')  # "pose"版！

# フレーム画像のフォルダ
frames_folder = r'C:\develop\Streetfighter\frames'

# 骨格推定結果の保存先
pose_output_folder = r'C:\develop\Streetfighter\pose_frames'
os.makedirs(pose_output_folder, exist_ok=True)

# フォルダ内のフレームに対して処理
for filename in sorted(os.listdir(frames_folder)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(frames_folder, filename)
        
        # 画像読み込み
        img = cv2.imread(img_path)

        # 推論（Pose推定）
        results = model(img)

        # 骨格をプロットした画像取得
        pose_img = results[0].plot()

        # 保存
        save_path = os.path.join(pose_output_folder, filename)
        cv2.imwrite(save_path, pose_img)

print("豪鬼の骨格推定完了！！")
