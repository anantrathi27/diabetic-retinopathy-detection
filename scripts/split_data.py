import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

DATASET_DIR = Path("dataset/raw")

df = pd.read_csv(DATASET_DIR / "labels_binary.csv")

# patient id = part before "_"
df["patient_id"] = df["image"].str.split("_").str[0]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    gss.split(df, groups=df["patient_id"])
)

train_df = df.iloc[train_idx]
val_df   = df.iloc[val_idx]

train_df.to_csv(DATASET_DIR / "train.csv", index=False)
val_df.to_csv(DATASET_DIR / "val.csv", index=False)

print("Train size:", len(train_df))
print("Val size:", len(val_df))

print("\nTrain distribution:")
print(train_df["binary_label"].value_counts())

print("\nVal distribution:")
print(val_df["binary_label"].value_counts())
