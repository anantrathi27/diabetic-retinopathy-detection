import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== PATHS =====
ROOT = r"D:/DR_severity_project/dataset"
IMAGE_DIR = os.path.join(ROOT, "raw/images")
CSV_PATH = os.path.join(ROOT, "raw/labels.csv")
OUTPUT_DIR = r"D:/DR_severity_project/data/severity"

IMG_EXT = ".jpeg"

# ===== LABEL MAP =====
label_map = {
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "proliferative"
}

# ===== LOAD CSV =====
df = pd.read_csv(CSV_PATH)

# Remove No-DR (level 0)
df = df[df["level"] != 0]

# Safety check
print(df["level"].value_counts())

# ===== SPLIT =====
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["level"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["level"],
    random_state=42
)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# ===== COPY FILES =====
for split, split_df in splits.items():
    for _, row in split_df.iterrows():
        img_name = row["image"] + IMG_EXT
        level = row["level"]

        if level not in label_map:
            continue

        class_name = label_map[level]

        dest_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        src_path = os.path.join(IMAGE_DIR, img_name)
        dst_path = os.path.join(dest_dir, img_name)

        if not os.path.exists(src_path):
            print("Missing:", src_path)
            continue

        shutil.copy(src_path, dst_path)

print("✅ Severity dataset prepared successfully")
