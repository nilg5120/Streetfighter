import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. データ読み込み
df = pd.read_csv(r'C:\develop\Streetfighter\gouki_pose_data.csv')

# 2. ラベルが付いている行だけ使う（空白除外）
df = df.dropna(subset=['label'])

# ラベルを整数型にしておく（もしかするとfloatで入ってるかもなので念のため）
df['label'] = df['label'].astype(int)

# 3. 特徴量（X）とラベル（y）に分ける
X = df[['hip_x', 'hip_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y']]
y = df['label']

# 4. 学習用データとテスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. モデル作成（Random Forest）
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. テストデータで予測して評価
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. モデル保存
model_path = r'C:\develop\Streetfighter\gouki_motion_model.pkl'
joblib.dump(model, model_path)

print(f"豪鬼モーション分類モデル保存完了！ -> {model_path}")
