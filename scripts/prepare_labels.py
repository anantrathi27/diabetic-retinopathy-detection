import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset/raw")
labels_path = DATASET_DIR / "labels.csv"

print("Looking for:", labels_path)

if not labels_path.exists():
    raise FileNotFoundError(f"{labels_path} not found")

df = pd.read_csv(labels_path)

df["binary_label"] = (df["level"] > 0).astype(int)

print("Binary label distribution:")
print(df["binary_label"].value_counts())

out_path = DATASET_DIR / "labels_binary.csv"
df[["image", "binary_label"]].to_csv(out_path, index=False)

print("Saved:", out_path)
