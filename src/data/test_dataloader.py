import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data.dataset    import RetinaDataset
from src.data.transforms import train_transform, val_test_transform

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR    = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"
PLOTS_DIR     = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 16
NUM_WORKERS = 0

train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
val_df   = pd.read_csv(PROCESSED_DIR / "val_split.csv")
test_df  = pd.read_csv(PROCESSED_DIR / "test_split.csv")

train_dataset = RetinaDataset(train_df, IMAGES_DIR, transform=train_transform)
val_dataset   = RetinaDataset(val_df,   IMAGES_DIR, transform=val_test_transform)
test_dataset  = RetinaDataset(test_df,  IMAGES_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- DataLoader batch test ---
train_images, train_labels = next(iter(train_loader))
val_images,   val_labels   = next(iter(val_loader))
test_images,  test_labels  = next(iter(test_loader))

print("=== DataLoader Batch Test ===")
print("Train batch  - images: {}  labels: {}".format(train_images.shape, train_labels.shape))
print("Val batch    - images: {}".format(val_images.shape))
print("Test batch   - images: {}".format(test_images.shape))
print("")
print("Train batch pixel  min: {:.4f}  max: {:.4f}".format(train_images.min().item(), train_images.max().item()))
print("Train batch labels present: {}".format(sorted(train_labels.unique().tolist())))
print("")

# --- Visual preprocessing check ---
# Show 4 original images (no transform) alongside their train-transformed versions
import random
from PIL import Image as PILImage
import numpy as np

random.seed(42)
sample_rows = train_df.sample(4, random_state=42)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Preprocessing Check: Original (top) vs Train Transform (bottom)", fontsize=12)

for col, (_, row) in enumerate(sample_rows.iterrows()):
    img_path = IMAGES_DIR / "{}.png".format(row["id_code"])
    orig     = PILImage.open(img_path).convert("RGB")

    # top row: original resized for display
    axes[0, col].imshow(orig.resize((224, 224)))
    axes[0, col].set_title("label={}".format(int(row["binary_label"])), fontsize=9)
    axes[0, col].axis("off")

    # bottom row: train transform (undo normalization for display)
    tensor    = train_transform(orig)
    mean      = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std       = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display   = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    axes[1, col].imshow(display)
    axes[1, col].set_title("augmented", fontsize=9)
    axes[1, col].axis("off")

plt.tight_layout()
fig.savefig(str(PLOTS_DIR / "preprocessing_samples.png"), dpi=150)
plt.close(fig)
print("Preprocessing visualization saved: results/plots/preprocessing_samples.png")
