import os
import shutil

src_img_dir = r"C:\develop\Streetfighter\need_annotation\images"
src_lbl_dir = r"C:\develop\Streetfighter\need_annotation\labels"

dst_img_dir = r"C:\develop\Streetfighter\train\images"
dst_lbl_dir = r"C:\develop\Streetfighter\train\labels"

for filename in os.listdir(src_img_dir):
    if filename.endswith(".jpg"):
        base = os.path.splitext(filename)[0]
        new_base = f"add_{base}"

        # 画像コピー
        shutil.copy(
            os.path.join(src_img_dir, filename),
            os.path.join(dst_img_dir, f"{new_base}.jpg")
        )

        # ラベルも対応してコピー
        label_path = os.path.join(src_lbl_dir, f"{base}.txt")
        if os.path.exists(label_path):
            shutil.copy(
                label_path,
                os.path.join(dst_lbl_dir, f"{new_base}.txt")
            )

print("リネーム＆コピー完了！")
