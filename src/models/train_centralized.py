import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import RetinaDataset
from src.data.transforms import train_transform, val_test_transform
from src.models.efficientnet import create_model


def plot_training_curves(history, save_path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label="Training Loss", marker="o", color="#2563eb")
    ax1.plot(epochs, val_loss, label="Validation Loss", marker="s", color="#dc2626")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss vs Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curves
    ax2.plot(epochs, train_acc, label="Training Accuracy", marker="o", color="#2563eb")
    ax2.plot(epochs, val_acc, label="Validation Accuracy", marker="s", color="#16a34a")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def train():
    # Paths
    train_csv_path = PROJECT_ROOT / "data" / "processed" / "train_split.csv"
    val_csv_path = PROJECT_ROOT / "data" / "processed" / "val_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"

    model_save_path = PROJECT_ROOT / "saved_models" / "centralized_best.pth"
    history_save_path = PROJECT_ROOT / "results" / "metrics" / "centralized_training_history.csv"
    plot_save_path = PROJECT_ROOT / "results" / "plots" / "centralized_training_curves.png"

    # Configuration
    batch_size = 16
    num_workers = 0
    max_epochs = 15
    patience = 4
    learning_rate = 1e-4
    weight_decay = 1e-4

    # Device configuration
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    # Data loaders
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    train_dataset = RetinaDataset(train_df, images_dir, transform=train_transform)
    val_dataset = RetinaDataset(val_df, images_dir, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Model initialization
    model = create_model(num_classes=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Terminal header
    print("=" * 40)
    print("CENTRALIZED EFFICIENTNET-B0 TRAINING")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"CUDA Available: {cuda_available}")
    print(f"GPU: {gpu_name}")
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Maximum Epochs: {max_epochs}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("=" * 40)

    # Loss, Optimizer, Scheduler, AMP
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    use_amp = cuda_available
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    early_stopping_triggered = False
    history = []
    total_start_time = time.time()

    try:
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} - Training", unit="batch")

            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_size_curr = images.size(0)
                train_loss_total += loss.item() * batch_size_curr
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += batch_size_curr

                running_loss = train_loss_total / train_total
                pbar.set_postfix(loss=f"{running_loss:.4f}")

            train_loss = train_loss_total / train_total
            train_acc = (train_correct / train_total) * 100.0

            # Validation phase
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    batch_size_curr = images.size(0)
                    val_loss_total += loss.item() * batch_size_curr
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += batch_size_curr

            val_loss = val_loss_total / val_total
            val_acc = (val_correct / val_total) * 100.0

            # Learning rate update
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # Epoch summary
            print("-" * 40)
            print(f"Epoch {epoch}/{max_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch Time: {epoch_time:.1f}s")
            print("-" * 40)

            # Record metrics
            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 2),
                "learning_rate": current_lr
            })

            # Checkpoint and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                early_stop_counter = 0

                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params
                }, model_save_path)
                print(f"Best model updated -> {model_save_path.as_posix()}")
            else:
                early_stop_counter += 1
                print(f"Early stopping: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    early_stopping_triggered = True
                    break

    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" in str(error).lower():
            print("\n" + "=" * 40)
            print("CUDA OUT OF MEMORY ERROR")
            print("=" * 40)
            print("The GPU ran out of memory. Please reduce batch_size to 8.")
            print("=" * 40)
            sys.exit(1)
        raise error

    # Save training history
    history_save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_save_path, index=False)

    # Save training curves
    plot_training_curves(history, plot_save_path)

    # Final summary
    total_time = time.time() - total_start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

    print("\n" + "=" * 40)
    print("CENTRALIZED TRAINING SUMMARY")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs Completed: {len(history)}")
    print(f"Final Learning Rate: {current_lr:.6f}")
    print(f"Early Stopping Triggered: {'YES' if early_stopping_triggered else 'NO'}")
    print(f"Training Time: {time_str}")
    print(f"Model Saved: {model_save_path.as_posix()}")
    print(f"Training History: {history_save_path.as_posix()}")
    print(f"Training Curves: {plot_save_path.as_posix()}")
    print("=" * 40)


if __name__ == "__main__":
    train()
