import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH  = PROJECT_ROOT / "data" / "raw" / "aptos" / "train.csv"
SAVE_DIR  = PROJECT_ROOT / "data" / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# binary label: 0 = No DR, 1 = DR (any grade)
df["binary_label"] = df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

# stratified 70/15/15 split
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["binary_label"], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["binary_label"], random_state=42)

cols = ["id_code", "diagnosis", "binary_label"]
train_df[cols].to_csv(SAVE_DIR / "train_split.csv", index=False)
val_df[cols].to_csv(SAVE_DIR / "val_split.csv",   index=False)
test_df[cols].to_csv(SAVE_DIR / "test_split.csv",  index=False)

# verify no data leakage
train_ids = set(train_df["id_code"])
val_ids   = set(val_df["id_code"])
test_ids  = set(test_df["id_code"])
no_leak   = (len(train_ids & val_ids) == 0 and
             len(train_ids & test_ids) == 0 and
             len(val_ids  & test_ids) == 0)

def print_dist(name, split):
    counts = split["binary_label"].value_counts().sort_index()
    n      = len(split)
    print("  {} ({} samples):".format(name, n))
    for label, count in counts.items():
        tag = "No DR" if label == 0 else "DR"
        print("    {} - {}: {}  ({:.1f}%)".format(label, tag, count, count / n * 100))

print("Total samples:       {}".format(len(df)))
print("Training samples:    {}".format(len(train_df)))
print("Validation samples:  {}".format(len(val_df)))
print("Test samples:        {}".format(len(test_df)))
print("")
print("Binary distribution per split:")
print_dist("Train",      train_df)
print_dist("Validation", val_df)
print_dist("Test",       test_df)
print("")
print("Data leakage between splits: {}".format("NO" if no_leak else "YES — CHECK REQUIRED"))
print("")
print("Saved:")
print("  data/processed/train_split.csv")
print("  data/processed/val_split.csv")
print("  data/processed/test_split.csv")
