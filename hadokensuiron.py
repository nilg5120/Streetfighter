import pandas as pd
import joblib

# 1. 学習済みモデルをロード
model = joblib.load(r'C:\develop\Streetfighter\gouki_motion_model.pkl')

# 2. 判定したい1フレーム分のデータを作る
# 例：さっきCSVで使った1フレームの値を適当に仮入力
new_data = pd.DataFrame([{
    'hip_x': 1331.5,
    'hip_y': 534.3,
    'left_wrist_x': 1555.9,
    'left_wrist_y': 525.8,
    'right_wrist_x': 1296.9,
    'right_wrist_y': 619.9,
}])

# 3. モデルにデータを渡して予測
prediction = model.predict(new_data)

# 4. 結果を表示
label_map = {0: '波動拳', 1: '昇竜拳'}
print(f"予測結果: {label_map[prediction[0]]}（ラベル={prediction[0]}）")
