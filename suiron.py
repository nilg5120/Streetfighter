from ultralytics import YOLO
import cv2

# 学習済み豪鬼専用モデルをロード
model = YOLO(r'C:\Users\ohkub\runs\detect\train5\weights\best.pt')

# 検出したい画像をロード
img = cv2.imread(r'C:\develop\Streetfighter\frames\frame_00000.jpg')  # 例

# 推論（検出）
results = model(img)

# 検出結果を描画
result_img = results[0].plot()

# 結果を表示
cv2.imshow('Detection', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
