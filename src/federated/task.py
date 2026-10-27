import sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet import create_model
from src.data.dataset import RetinaDataset
from src.data.transforms import train_transform, val_test_transform

IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"
VAL_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "val_split.csv"


def get_weights(model):
    """Extract model state_dict weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Load a list of NumPy array weights into the model state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def load_client_data(partition_id, batch_size=16):
    """Load training DataLoader for a specific client partition (0 to 4)."""
    client_num = int(partition_id) + 1
    client_csv = PROJECT_ROOT / "data" / "clients" / "iid" / f"client_{client_num}.csv"
    if not client_csv.exists():
        raise FileNotFoundError(f"Client partition file not found: {client_csv}")

    df = pd.read_csv(client_csv)
    dataset = RetinaDataset(df, IMAGES_DIR, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def load_val_data(batch_size=16):
    """Load server-side global validation DataLoader."""
    if not VAL_CSV_PATH.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV_PATH}")

    df = pd.read_csv(VAL_CSV_PATH)
    dataset = RetinaDataset(df, IMAGES_DIR, transform=val_test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def train_local(model, train_loader, epochs=1, lr=1e-4, weight_decay=1e-4, device=None):
    """Train the model locally on client data for the specified epochs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * images.size(0)
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    return avg_loss, accuracy


def evaluate_model(model, val_loader, device=None):
    """Evaluate model on validation data and compute loss and accuracy."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * images.size(0)
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    return avg_loss, accuracy
