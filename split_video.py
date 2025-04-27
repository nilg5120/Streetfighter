import cv2
import os
import shutil

# 動画ファイルのパス
video_path = r'C:\Users\ohkub\Videos\Street Fighter 6\無題の動画 ‐ Clipchampで作成.mp4'



# 出力するフォルダ
output_folder = r'C:\develop\Streetfighter\frames'
# ★最初にフォルダの中身を全部消す
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # フォルダごと削除
os.makedirs(output_folder, exist_ok=True)  # フォルダ作り直す

# 動画ファイルを開く
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 動画が終わったらループ終了

    # フレームを保存
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()

print(f"保存完了！{frame_count}枚のフレームを保存しました。")
