import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import RetinaDataset
from src.data.transforms import val_test_transform
from src.models.efficientnet import create_model


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.
    Hooks into the final convolutional feature layer to compute gradients of the
    predicted class score with respect to feature activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

            # Register backward hook on the output tensor to record gradients
            if out.requires_grad:
                def backward_hook(grad):
                    self.gradients = grad

                out.register_hook(backward_hook)

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, target_class=None):
        self.activations = None
        self.gradients = None
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Backpropagate score of target class
        score = outputs[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients yields channel importance weights (alpha)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Linear combination of feature maps weighted by alpha
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU ensures only features positively influencing the target prediction are kept
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize heatmap to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, probs, target_class


def create_gradcam_overlay(image_rgb, cam_mask, alpha=0.4):
    """
    Resizes Grad-CAM heatmap to original image resolution and overlays it.
    Preserves retinal fundus anatomy while highlighting influential regions.
    """
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(cam_mask, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0

    if image_rgb.max() > 1.0:
        base_img = image_rgb.astype(np.float32) / 255.0
    else:
        base_img = image_rgb.astype(np.float32)

    overlay = alpha * heatmap_rgb + (1.0 - alpha) * base_img
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay, heatmap_resized


def select_test_examples(model, test_df, images_dir, device):
    """
    Identifies representative test images from test_split.csv based on actual model predictions:
    - 2 True Negatives (Correct No-DR)
    - 2 True Positives (Correct DR)
    - 1 False Positive (Actual No-DR, Predicted DR)
    - 1 False Negative (Actual DR, Predicted No-DR)
    """
    dataset = RetinaDataset(test_df, images_dir, transform=val_test_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    df = test_df.copy()
    df["predicted_label"] = all_preds
    df["dr_probability"] = all_probs

    tn_examples = df[(df["binary_label"] == 0) & (df["predicted_label"] == 0)].head(2)
    tp_examples = df[(df["binary_label"] == 1) & (df["predicted_label"] == 1)].head(2)
    fp_examples = df[(df["binary_label"] == 0) & (df["predicted_label"] == 1)].head(1)
    fn_examples = df[(df["binary_label"] == 1) & (df["predicted_label"] == 0)].head(1)

    selected = []
    for _, row in tn_examples.iterrows():
        selected.append((row["id_code"], int(row["binary_label"]), "True Negative"))
    for _, row in tp_examples.iterrows():
        selected.append((row["id_code"], int(row["binary_label"]), "True Positive"))
    for _, row in fp_examples.iterrows():
        selected.append((row["id_code"], int(row["binary_label"]), "False Positive"))
    for _, row in fn_examples.iterrows():
        selected.append((row["id_code"], int(row["binary_label"]), "False Negative"))

    return selected, df


def run_gradcam(check_only=False):
    # Paths
    model_path = PROJECT_ROOT / "saved_models" / "centralized_best.pth"
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"

    output_dir = PROJECT_ROOT / "results" / "plots" / "gradcam"
    summary_plot_path = PROJECT_ROOT / "results" / "plots" / "gradcam_examples.png"
    results_csv_path = PROJECT_ROOT / "results" / "metrics" / "gradcam_examples.csv"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # Load trained model
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test split metadata
    test_df = pd.read_csv(test_csv_path)

    # Identify representative examples from test split based on model predictions
    selected_examples, _ = select_test_examples(model, test_df, images_dir, device)

    # Target layer: last convolutional feature stage before global pooling
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Terminal header
    print("========================================")
    print("GRAD-CAM EXPLAINABILITY")
    print("========================================")
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Model checkpoint: {model_path.as_posix()}")
    print("Target layer: model.features[-1] (Conv2dNormActivation 320 -> 1280)")
    print(f"Test images: {len(test_df)}")
    print("========================================")

    # Verification mode: test forward/backward hooks, CAM properties, and exit cleanly
    if check_only:
        sample_row = test_df.iloc[0]
        sample_path = images_dir / f"{sample_row['id_code']}.png"
        sample_pil = Image.open(sample_path).convert("RGB")
        sample_tensor = val_test_transform(sample_pil).unsqueeze(0).to(device)

        cam_map, probs, pred_cls = gradcam.generate(sample_tensor)

        assert cam_map.shape == (7, 7), f"Expected CAM shape (7, 7), got {cam_map.shape}"
        assert np.isfinite(cam_map).all(), "CAM map contains non-finite values"
        assert cam_map.max() > cam_map.min(), "CAM map is completely uniform"

        sample_np = np.array(sample_pil)
        overlay, _ = create_gradcam_overlay(sample_np, cam_map)
        assert overlay.shape == sample_np.shape, "Overlay shape mismatch"
        assert len(selected_examples) >= 4, "Failed to identify representative test examples"

        print(f"[PASS] Model checkpoint loaded from: {model_path.as_posix()}")
        print(f"[PASS] Target layer verified: {type(target_layer).__name__}")
        print(f"[PASS] Activation maps shape: {list(gradcam.activations.shape)}")
        print(f"[PASS] Gradients shape: {list(gradcam.gradients.shape)}")
        print(f"[PASS] Grad-CAM output shape: {cam_map.shape}")
        print(f"[PASS] Grad-CAM range: [{cam_map.min():.4f}, {cam_map.max():.4f}] (mean: {cam_map.mean():.4f})")
        print(f"[PASS] Test selection identified {len(selected_examples)} representative cases.")
        print("[PASS] Grad-CAM pipeline verified successfully.")
        return

    # Regular execution mode: generate explanations
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_path.parent.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    selected_examples, _ = select_test_examples(model, test_df, images_dir, device)

    class_names = {0: "No DR", 1: "DR"}
    results_records = []
    generated_data = []

    counts = {
        "True Negative": 0,
        "True Positive": 0,
        "False Positive": 0,
        "False Negative": 0,
    }

    for id_code, actual_lbl, category in selected_examples:
        img_path = images_dir / f"{id_code}.png"
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)

        # Preprocess using existing val_test_transform
        input_tensor = val_test_transform(pil_img).unsqueeze(0).to(device)

        # Generate Grad-CAM for the predicted class
        cam_map, probs, pred_cls = gradcam.generate(input_tensor)
        dr_prob = probs[1].item() * 100.0

        actual_str = class_names[actual_lbl]
        pred_str = class_names[pred_cls]
        is_correct = (actual_lbl == pred_cls)

        overlay, _ = create_gradcam_overlay(img_np, cam_map, alpha=0.45)

        # Save individual figure
        save_filename = f"{id_code}_gradcam.png"
        save_filepath = output_dir / save_filename

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Retinal Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM ({pred_str} Class Influence)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        fig.suptitle(
            f"Image ID: {id_code} | Actual: {actual_str} | Predicted: {pred_str} | DR Probability: {dr_prob:.2f}%",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            "Note: Highlighted regions contributed strongly to the model's prediction.",
            ha="center",
            fontsize=9,
            style="italic",
            color="#4b5563",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(save_filepath, dpi=300)
        plt.close(fig)

        counts[category] += 1

        print(f"Image: {id_code}")
        print(f"Actual: {actual_str}")
        print(f"Predicted: {pred_str}")
        print(f"DR Probability: {dr_prob:.2f}%")
        print(f"Grad-CAM saved: {save_filepath.as_posix()}\n")

        results_records.append({
            "id_code": id_code,
            "category": category,
            "actual_label": actual_lbl,
            "actual_class": actual_str,
            "predicted_label": pred_cls,
            "predicted_class": pred_str,
            "dr_probability": round(dr_prob, 2),
            "prediction_correct": is_correct,
            "gradcam_file": save_filepath.as_posix(),
        })

        generated_data.append({
            "id_code": id_code,
            "category": category,
            "actual_str": actual_str,
            "pred_str": pred_str,
            "dr_prob": dr_prob,
            "img_np": img_np,
            "overlay": overlay,
        })

    # Save metrics CSV
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(results_csv_path, index=False)

    # Generate multi-example summary figure
    num_examples = len(generated_data)
    # Arrange in a 3-row x 4-column grid (each pair is Original + Grad-CAM)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))

    for idx, item in enumerate(generated_data):
        row = idx // 2
        col_pair = (idx % 2) * 2

        ax_orig = axes[row, col_pair]
        ax_cam = axes[row, col_pair + 1]

        ax_orig.imshow(item["img_np"])
        ax_orig.set_title(f"Original\n({item['category']})", fontsize=10, fontweight="bold")
        ax_orig.axis("off")

        ax_cam.imshow(item["overlay"])
        ax_cam.set_title(
            f"Grad-CAM\nPred: {item['pred_str']} ({item['dr_prob']:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        ax_cam.axis("off")

    fig.suptitle(
        "Centralized EfficientNet-B0 - Grad-CAM Visual Explanations\n"
        "(Visualizing retinal image regions influencing DR screening predictions)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(summary_plot_path, dpi=300)
    plt.close(fig)

    # Terminal summary
    print("========================================")
    print("GRAD-CAM SUMMARY")
    print("========================================")
    print(f"Examples Generated: {len(results_records)}")
    print(f"Correct No-DR Examples: {counts['True Negative']}")
    print(f"Correct DR Examples: {counts['True Positive']}")
    print(f"False Positive Examples: {counts['False Positive']}")
    print(f"False Negative Examples: {counts['False Negative']}")
    print(f"Summary Figure: {summary_plot_path.as_posix()}")
    print(f"Results CSV: {results_csv_path.as_posix()}")
    print("========================================")


if __name__ == "__main__":
    is_check = "--check" in sys.argv
    run_gradcam(check_only=is_check)
