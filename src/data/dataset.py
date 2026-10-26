import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = self.images_dir / "{}.png".format(row["id_code"])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["binary_label"])
        return image, label
