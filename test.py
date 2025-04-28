from ultralytics import YOLO
import cv2
import os
import shutil

# モデルをロード
model = YOLO(r'C:\Users\ohkub\runs\detect\train5\weights\best.pt')

# 読み込む動画ファイル
video_path = r'C:\Users\ohkub\Videos\Street Fighter 6\Street Fighter 6_04-28-2025_21-25-29-900.mp4'  # ←あなたの動画ファイルに置き換えてね

# 保存用フォルダ
output_folder = r'C:\develop\Streetfighter\detected_video_frames'
# ★最初に保存先フォルダをまっさらにする
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# 動画を読み込み
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 動画が終わったらループ終了

    # 豪鬼検出
    results = model(frame)

    # 検出された結果を描画
    result_img = results[0].plot()

    # 保存する（オプション）
    save_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(save_path, result_img)

    # 表示する（オプション）
    cv2.imshow('Gouki Detection', result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"保存完了！{frame_count}枚のフレームを保存しました。")
