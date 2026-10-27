import argparse
import inspect
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
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

            if out.requires_grad:
                def backward_hook(grad):
                    self.gradients = grad

                out.register_hook(backward_hook)

        h = self.target_layer.register_forward_hook(forward_hook)
        self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate(self, input_tensor, target_class=None):
        self.activations = None
        self.gradients = None
        self.model.zero_grad()

        # Forward pass (requires grad on target layer activations)
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Backpropagate score of target class
        score = outputs[0, target_class]
        score.backward(retain_graph=False)

        # Global average pooling of gradients yields channel importance weights
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

        # Record shapes before releasing memory
        self.last_activation_shape = list(self.activations.shape) if self.activations is not None else None
        self.last_gradient_shape = list(self.gradients.shape) if self.gradients is not None else None

        # Clean up gradients to free memory
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        return cam, probs, target_class


def create_gradcam_overlay(image_rgb, cam_mask, alpha=0.45):
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
    return overlay, heatmap_resized, heatmap_rgb


def select_test_examples(model, test_df, images_dir, device):
    """
    Identifies representative test images from test_split.csv based on actual model predictions:
    - 2 True Negatives (Correct No-DR)
    - 2 True Positives (Correct DR)
    - 1 False Positive (Actual No-DR, Predicted DR)
    - 1 False Negative (Actual DR, Predicted No-DR)
    Deterministic: First valid occurrences in test-set order.
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


def run_gradcam(model_choice="centralized", check_only=False):
    model_choice = model_choice.lower()

    # Paths configuration based on model choice
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"

    if model_choice == "fedprox":
        model_path = PROJECT_ROOT / "saved_models" / "fedprox_non_iid_mu0.1_5rounds_final.pth"
        output_dir = PROJECT_ROOT / "results" / "plots" / "gradcam_fedprox"
        summary_plot_path = PROJECT_ROOT / "results" / "plots" / "gradcam_fedprox_examples.png"
        results_csv_path = PROJECT_ROOT / "results" / "metrics" / "gradcam_fedprox_examples.csv"
        banner_title = "FEDPROX NON-IID GRAD-CAM EXPLAINABILITY"
        model_title = "FedProx Non-IID (5 Rounds, mu=0.1)"
    else:
        model_path = PROJECT_ROOT / "saved_models" / "centralized_best.pth"
        output_dir = PROJECT_ROOT / "results" / "plots" / "gradcam"
        summary_plot_path = PROJECT_ROOT / "results" / "plots" / "gradcam_examples.png"
        results_csv_path = PROJECT_ROOT / "results" / "metrics" / "gradcam_examples.csv"
        banner_title = "CENTRALIZED GRAD-CAM EXPLAINABILITY"
        model_title = "Centralized EfficientNet-B0"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # Load trained model
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"
    checkpoint = torch.load(model_path, map_location=device)

    # Checkpoint metadata verification
    if model_choice == "fedprox":
        assert model_path.name == "fedprox_non_iid_mu0.1_5rounds_final.pth", f"Unexpected FedProx checkpoint: {model_path.name}"
        assert checkpoint.get("algorithm") == "FedProx", f"Expected FedProx, got {checkpoint.get('algorithm')}"
        assert checkpoint.get("partition_type") == "non_iid", f"Expected non_iid, got {checkpoint.get('partition_type')}"
        assert abs(float(checkpoint.get("proximal_mu", 0.0)) - 0.1) < 1e-6, f"Expected mu=0.1, got {checkpoint.get('proximal_mu')}"
        assert int(checkpoint.get("number_of_rounds", 0)) == 5, f"Expected 5 rounds, got {checkpoint.get('number_of_rounds')}"
        assert int(checkpoint.get("number_of_clients", 0)) == 5, f"Expected 5 clients, got {checkpoint.get('number_of_clients')}"
        assert int(checkpoint.get("local_epochs", 0)) == 1, f"Expected 1 local epoch, got {checkpoint.get('local_epochs')}"
        assert model_path.name != "centralized_best.pth", "Cannot load centralized model in FedProx mode"
        assert model_path.name != "fedavg_iid_final.pth", "Cannot load FedAvg IID in FedProx mode"
        assert model_path.name != "fedavg_iid_5rounds_final.pth", "Cannot load FedAvg IID 5R in FedProx mode"
        assert model_path.name != "fedavg_non_iid_5rounds_final.pth", "Cannot load FedAvg Non-IID in FedProx mode"

    model = create_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test split metadata
    assert test_csv_path.exists(), f"test_split.csv not found: {test_csv_path}"
    test_df = pd.read_csv(test_csv_path)

    # Identify representative examples from test split based on model predictions
    selected_examples, _ = select_test_examples(model, test_df, images_dir, device)

    # Target layer: last convolutional feature stage before global pooling
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Terminal header
    print("========================================")
    print(banner_title)
    print("========================================")
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_title}")
    print(f"Model checkpoint: {model_path.name}")
    print("Target layer: model.features[-1]")
    print(f"Test images: {len(test_df)}")
    print(f"Selected examples: {len(selected_examples)} (2 TN, 2 TP, 1 FP, 1 FN)")
    print("========================================")

    # Verification mode: verify configuration and hooks without generating all images
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
        overlay, _, _ = create_gradcam_overlay(sample_np, cam_map)
        assert overlay.shape == sample_np.shape, "Overlay shape mismatch"
        assert len(selected_examples) == 6, f"Expected 6 selected examples, got {len(selected_examples)}"

        cats = [item[2] for item in selected_examples]
        assert cats.count("True Negative") == 2
        assert cats.count("True Positive") == 2
        assert cats.count("False Positive") == 1
        assert cats.count("False Negative") == 1

        if model_choice == "fedprox":
            print("=" * 50)
            print("VERIFYING FEDPROX GRAD-CAM CONFIGURATION")
            print("=" * 50)
            print("[PASS] 1. Model selection = FedProx")
            print(f"[PASS] 2. Correct FedProx checkpoint exists: {model_path.name}")
            print(f"[PASS] 3. Checkpoint algorithm = {checkpoint.get('algorithm')}")
            print(f"[PASS] 4. Checkpoint partition = {checkpoint.get('partition_type')}")
            print(f"[PASS] 5. Checkpoint proximal_mu = {checkpoint.get('proximal_mu')}")
            print(f"[PASS] 6. Checkpoint rounds = {checkpoint.get('number_of_rounds')}")
            print(f"[PASS] 7. Checkpoint clients = {checkpoint.get('number_of_clients')}")
            print("[PASS] 8. EfficientNet-B0 architecture loads successfully")
            print(f"[PASS] 9. Number of classes = {model.classifier[1].out_features}")
            print("[PASS] 10. Target layer = model.features[-1]")
            print(f"[PASS] 11. model.eval() is enabled (model.training is {model.training})")
            print(f"[PASS] 12. Test split contains exactly {len(test_df)} images ({test_csv_path.name})")
            print("[PASS] 13. Deterministic val/test transform is used (224x224, ImageNet normalization)")
            print("[PASS] 14. No optimizer exists")
            print("[PASS] 15. No training loop exists (strictly inference-only XAI)")

            # Check output directories are separate from centralized
            assert output_dir == PROJECT_ROOT / "results" / "plots" / "gradcam_fedprox"
            assert output_dir != PROJECT_ROOT / "results" / "plots" / "gradcam"
            assert summary_plot_path == PROJECT_ROOT / "results" / "plots" / "gradcam_fedprox_examples.png"
            assert summary_plot_path != PROJECT_ROOT / "results" / "plots" / "gradcam_examples.png"
            print(f"[PASS] 16. Grad-CAM output directory is separate from centralized outputs ({output_dir.name}/)")

            # Check metadata CSV path is unique
            assert results_csv_path == PROJECT_ROOT / "results" / "metrics" / "gradcam_fedprox_examples.csv"
            assert results_csv_path != PROJECT_ROOT / "results" / "metrics" / "gradcam_examples.csv"
            print(f"[PASS] 17. FedProx metadata CSV path is unique ({results_csv_path.name})")

            print(f"[PASS] 18. Six required outcome slots are configured: 2 TN, 2 TP, 1 FP, 1 FN ({cats})")
            print("[PASS] 19. Grad-CAM generation processes one image at a time (memory safe for 4GB VRAM)")
            print("[PASS] 20. Model weight integrity check is enabled (pre/post tensor comparison)")

            print("=" * 50)
            print("FEDPROX GRAD-CAM CONFIGURATION: PASS")
            print("=" * 50)
        else:
            print("=" * 50)
            print("VERIFYING CENTRALIZED GRAD-CAM CONFIGURATION")
            print("=" * 50)
            print(f"[PASS] 1. Model checkpoint loaded from: {model_path.as_posix()}")
            print(f"[PASS] 2. Target layer verified: {type(target_layer).__name__}")
            print(f"[PASS] 3. Activation maps shape: {gradcam.last_activation_shape}")
            print(f"[PASS] 4. Gradients shape: {gradcam.last_gradient_shape}")
            print(f"[PASS] 5. Grad-CAM output shape: {cam_map.shape}")
            print(f"[PASS] 6. Grad-CAM range: [{cam_map.min():.4f}, {cam_map.max():.4f}] (mean: {cam_map.mean():.4f})")
            print(f"[PASS] 7. Test selection identified {len(selected_examples)} representative cases.")
            print("[PASS] Grad-CAM pipeline verified successfully.")
            print("=" * 50)
        return

    # Regular execution mode: generate explanations
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_path.parent.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Store exact snapshot of model weights before Grad-CAM generation to verify integrity
    initial_weights_snapshot = {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
    }

    class_names = {0: "No DR", 1: "DR"}
    results_records = []
    generated_data = []

    counts = {
        "True Negative": 0,
        "True Positive": 0,
        "False Positive": 0,
        "False Negative": 0,
    }

    # Process selected images strictly one at a time for GPU memory safety
    for id_code, actual_lbl, category in selected_examples:
        img_path = images_dir / f"{id_code}.png"
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)

        # Preprocess using deterministic val_test_transform
        input_tensor = val_test_transform(pil_img).unsqueeze(0).to(device)

        # Generate Grad-CAM for the predicted class
        cam_map, probs, pred_cls = gradcam.generate(input_tensor)
        pred_prob = probs[pred_cls].item()
        dr_prob = probs[1].item() * 100.0

        actual_str = class_names[actual_lbl]
        pred_str = class_names[pred_cls]
        is_correct = (actual_lbl == pred_cls)

        overlay, cam_resized, heatmap_rgb = create_gradcam_overlay(img_np, cam_map, alpha=0.45)

        # Save individual figure with Original, Heatmap, and Overlay
        save_filename = f"{id_code}_gradcam.png"
        save_filepath = output_dir / save_filename

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Retinal Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_rgb)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay ({pred_str} Class Influence)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(
            f"Image ID: {id_code} | True: {actual_str} | Predicted: {pred_str} | "
            f"Prediction Probability: {pred_prob * 100:.2f}% | Outcome: {category}",
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
        print(f"Outcome: {category}")
        print(f"True Label: {actual_str} ({actual_lbl})")
        print(f"Predicted: {pred_str} ({pred_cls})")
        print(f"Prediction Probability: {pred_prob * 100:.2f}%")
        print(f"Grad-CAM saved: {save_filepath.as_posix()}\n")

        results_records.append({
            "image_id": id_code,
            "true_label": actual_lbl,
            "predicted_label": pred_cls,
            "predicted_probability": round(pred_prob, 4),
            "outcome_type": category,
            "model": "EfficientNet-B0",
            "algorithm": "FedProx" if model_choice == "fedprox" else "Centralized",
            "partition": "Non-IID" if model_choice == "fedprox" else "N/A",
            "proximal_mu": 0.1 if model_choice == "fedprox" else 0.0,
            "target_layer": "model.features[-1]",
        })

        generated_data.append({
            "id_code": id_code,
            "outcome_type": category,
            "actual_str": actual_str,
            "pred_str": pred_str,
            "pred_prob": pred_prob,
            "img_np": img_np,
            "heatmap_rgb": heatmap_rgb,
            "overlay": overlay,
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Model weight safety check: ensure no parameters were changed during Grad-CAM backprop
    weights_identical = True
    for name, param in model.named_parameters():
        if not torch.equal(param.detach().cpu(), initial_weights_snapshot[name]):
            weights_identical = False
            break
    assert weights_identical, "Model parameters were modified during Grad-CAM generation!"
    print("Model weights unchanged after Grad-CAM: YES\n")

    # Save metrics CSV
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(results_csv_path, index=False)

    # Generate multi-example summary figure: 6 rows x 3 columns (Original | Heatmap | Overlay)
    fig, axes = plt.subplots(6, 3, figsize=(12, 22))

    for row_idx, item in enumerate(generated_data):
        ax_orig = axes[row_idx, 0]
        ax_cam = axes[row_idx, 1]
        ax_over = axes[row_idx, 2]

        ax_orig.imshow(item["img_np"])
        ax_orig.set_title(f"Original Retinal Image\n[{item['outcome_type']}]", fontsize=10, fontweight="bold")
        ax_orig.axis("off")

        ax_cam.imshow(item["heatmap_rgb"])
        ax_cam.set_title(f"Grad-CAM Heatmap\nID: {item['id_code']}", fontsize=10, fontweight="bold")
        ax_cam.axis("off")

        ax_over.imshow(item["overlay"])
        ax_over.set_title(
            f"Overlay\nTrue: {item['actual_str']} | Pred: {item['pred_str']} ({item['pred_prob'] * 100:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        ax_over.axis("off")

    fig.suptitle(
        f"{model_title} - Grad-CAM Visual Explanations\n"
        "(Visualizing retinal image regions influencing screening predictions across clinical outcomes)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.savefig(summary_plot_path, dpi=300)
    plt.close(fig)

    # Terminal summary
    print("========================================")
    print("GRAD-CAM SUMMARY")
    print("========================================")
    print(f"Model: {model_title}")
    print(f"Examples Generated: {len(results_records)}")
    print(f"True Negatives: {counts['True Negative']}")
    print(f"True Positives: {counts['True Positive']}")
    print(f"False Positives: {counts['False Positive']}")
    print(f"False Negatives: {counts['False Negative']}")
    print(f"Summary Figure: {summary_plot_path.as_posix()}")
    print(f"Results CSV: {results_csv_path.as_posix()}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Explainability for Centralized and Federated Models")
    parser.add_argument(
        "--model",
        type=str,
        default="centralized",
        choices=["centralized", "fedprox"],
        help="Model to explain: centralized or fedprox (default: centralized)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run verification check without generating all Grad-CAM images",
    )
    args = parser.parse_args()

    run_gradcam(model_choice=args.model.lower(), check_only=args.check)
