from ultralytics import YOLO
import cv2
import os
import shutil

# モデルをロード（標準の物体検出モデル）
model = YOLO('yolov8n.pt')  # "n"はNano版＝超軽量版（速い）

# フレーム画像のフォルダ
frames_folder = r'C:\develop\Streetfighter\frames'

# 保存先フォルダ
output_folder = r'C:\develop\Streetfighter\detected_frames'
# ★最初に保存先フォルダをまっさらにする
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# フォルダ内の全フレームを読み込んで検出
for filename in sorted(os.listdir(frames_folder)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(frames_folder, filename)
        
        # 画像読み込み
        img = cv2.imread(img_path)

        # 推論（物体検出）
        results = model(img)

        # 結果を画像に描画
        result_img = results[0].plot()

        # 保存
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result_img)

print("検出と保存完了！")
