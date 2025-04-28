from ultralytics import YOLO
import cv2
import os

# Pose推定モデルロード
model = YOLO('yolov8n-pose.pt')

# フレームフォルダ（framesから読む）
frames_folder = r'C:\develop\Streetfighter\frames'

# パラメータ（チューニングできる）
hip_threshold = 300   # 腰の高さ基準
arm_extension_threshold = 100  # 手の横幅の基準
hand_height_diff_threshold = 50  # 手の高さ差の許容範囲

for filename in sorted(os.listdir(frames_folder)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(frames_folder, filename)
        img = cv2.imread(img_path)
        
        results = model(img)

        if len(results[0].keypoints.xy) == 0:
            continue  # 骨格が検出されなかったフレームは飛ばす

        keypoints = results[0].keypoints.xy[0]

        # 関節座標取得（番号に注意！）
        left_hand = keypoints[9]    # 左手首
        right_hand = keypoints[10]  # 右手首
        hip = keypoints[8]          # 腰（Hip）

        left_x, left_y = left_hand.tolist()
        right_x, right_y = right_hand.tolist()
        hip_x, hip_y = hip.tolist()

        # 波動拳っぽい判定
        if (hip_y < hip_threshold and  # 腰の高さがそれなり
            abs(left_x - right_x) > arm_extension_threshold and  # 手が広がってる
            abs(left_y - right_y) < hand_height_diff_threshold):  # 手の高さが揃ってる
            print(f"{filename}: 波動拳！！！！！💥")
        else:
            print(f"{filename}: 波動拳ではない")
