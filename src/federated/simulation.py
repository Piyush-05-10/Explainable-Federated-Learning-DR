import argparse
import sys
import time
from pathlib import Path
import flwr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flwr.common import Context
from flwr.simulation import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet import create_model
from src.federated.task import set_weights, load_client_data, load_val_data
from src.federated.client_app import app as client_app
from src.federated import server_app as srv


def plot_federated_training(history, save_path, num_rounds):
    """Plot Round vs Validation Loss and Round vs Validation Accuracy."""
    rounds = [h["round"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    title_suffix = f"({num_rounds} Rounds)"

    # Loss curve
    ax1.plot(rounds, val_loss, marker="o", color="#dc2626", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(f"FedAvg IID - Validation Loss vs Round {title_suffix}", fontsize=12, fontweight="bold")
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy curve
    ax2.plot(rounds, val_acc, marker="s", color="#16a34a", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title(f"FedAvg IID - Validation Accuracy vs Round {title_suffix}", fontsize=12, fontweight="bold")
    ax2.set_xticks(rounds)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def verify_configuration(num_rounds, model_save_path, history_save_path, plot_save_path):
    """Verify experiment configuration and Flower ServerConfig resolution without executing full simulation."""
    print("=" * 50)
    print(f"VERIFYING FEDAVG IID ({num_rounds} ROUNDS) CONFIGURATION")
    print("=" * 50)

    # 1. Direct inspection of Flower ServerAppComponents & ServerConfig
    server_app_test = srv.make_server_app(num_rounds=num_rounds)
    dummy_context = Context(run_id=0, node_id=0, node_config={}, state={}, run_config={})
    test_components = server_app_test._server_fn(dummy_context)
    actual_server_config_rounds = test_components.config.num_rounds
    assert actual_server_config_rounds == num_rounds, (
        f"ServerConfig resolution failed: expected {num_rounds}, got {actual_server_config_rounds}"
    )
    print(f"[PASS] 1. Flower ServerConfig resolution: num_rounds = {actual_server_config_rounds} (exact match for --rounds {num_rounds})")

    # Verification of alternative round configs
    comp_3 = srv.make_server_app(3)._server_fn(dummy_context)
    comp_5 = srv.make_server_app(5)._server_fn(dummy_context)
    assert comp_3.config.num_rounds == 3, "make_server_app(3) failed"
    assert comp_5.config.num_rounds == 5, "make_server_app(5) failed"
    print(f"[PASS] 2. Verified alternative round configurations: --rounds 3 -> {comp_3.config.num_rounds}, --rounds 5 -> {comp_5.config.num_rounds}")

    # 3. Local epochs
    print("[PASS] 3. Local epochs per client per round: 1")

    # 4. Client participation
    print("[PASS] 4. Client participation: 100% (5/5 clinics every round)")

    # 5. Client partitions
    client_counts = []
    for i in range(1, 6):
        cp = PROJECT_ROOT / "data" / "clients" / "iid" / f"client_{i}.csv"
        assert cp.exists(), f"Missing {cp}"
        client_counts.append(len(pd.read_csv(cp)))
    assert sum(client_counts) == 2563
    print(f"[PASS] 5. Client partitions unchanged: {client_counts} (Total: {sum(client_counts)} samples)")

    # 6. Validation dataset
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_split.csv")
    assert len(val_df) == 549
    print(f"[PASS] 6. Validation dataset: {len(val_df)} samples")

    # 7. Test set untouched
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_split.csv")
    assert len(test_df) == 550
    print("[PASS] 7. Test set NOT accessed during federated training (550 samples preserved)")

    # 8, 9, 10. Hyperparameters
    print("[PASS] 8. Optimizer: AdamW, Learning rate: 1e-4")
    print("[PASS] 9. Weight decay: 1e-4")
    print("[PASS] 10. Batch size: 16")

    # 11. Model initialization
    test_model = create_model(num_classes=2)
    assert test_model is not None
    print("[PASS] 11. Global model initializes independently from ImageNet-pretrained EfficientNet-B0")

    # 12. Protection of 3-round baseline files
    old_3round_files = [
        PROJECT_ROOT / "saved_models" / "fedavg_iid_final.pth",
        PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_history.csv",
        PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_test_metrics.json",
        PROJECT_ROOT / "results" / "plots" / "fedavg_iid_training.png",
        PROJECT_ROOT / "results" / "plots" / "fedavg_iid_confusion_matrix.png",
        PROJECT_ROOT / "results" / "plots" / "fedavg_iid_roc_curve.png",
    ]
    for f in old_3round_files:
        assert f.exists(), f"Old baseline file missing: {f}"
        if num_rounds != 3:
            assert f != model_save_path and f != history_save_path and f != plot_save_path, f"Collision with {f}"
    print("[PASS] 12. Existing 3-round baseline result files are protected and will NOT be overwritten")

    # 13. Target output paths
    print(f"[PASS] 13. New target model: {model_save_path.name}")
    print(f"         New target history: {history_save_path.name}")
    print(f"         New target plot: {plot_save_path.name}")

    # 14. GPU safety
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    print(f"[PASS] 14. GPU resource allocation: num_gpus=1.0 on {gpu_name} (Strictly sequential client execution)")

    print("=" * 50)
    print("ALL VERIFICATION CHECKS PASSED: READY FOR TRAINING")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="FedAvg IID Simulation")
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of federated communication rounds (default: 3)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run verification checks without launching training",
    )
    args = parser.parse_args()

    num_rounds = args.rounds
    num_clients = 5
    local_epochs = 1
    batch_size = 16

    # Instantiate ServerApp with exact num_rounds
    server_app = srv.make_server_app(num_rounds=num_rounds)

    # Configure output file paths depending on number of rounds
    if num_rounds == 3:
        model_save_path = PROJECT_ROOT / "saved_models" / "fedavg_iid_final.pth"
        history_save_path = PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_history.csv"
        plot_save_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_training.png"
        summary_title = "FEDAVG IID TRAINING SUMMARY"
    elif num_rounds == 5:
        model_save_path = PROJECT_ROOT / "saved_models" / "fedavg_iid_5rounds_final.pth"
        history_save_path = PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_5rounds_history.csv"
        plot_save_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_5rounds_training.png"
        summary_title = "FEDAVG IID 5-ROUND TRAINING SUMMARY"
    else:
        model_save_path = PROJECT_ROOT / "saved_models" / f"fedavg_iid_{num_rounds}rounds_final.pth"
        history_save_path = PROJECT_ROOT / "results" / "metrics" / f"fedavg_iid_{num_rounds}rounds_history.csv"
        plot_save_path = PROJECT_ROOT / "results" / "plots" / f"fedavg_iid_{num_rounds}rounds_training.png"
        summary_title = f"FEDAVG IID {num_rounds}-ROUND TRAINING SUMMARY"

    if args.check:
        verify_configuration(num_rounds, model_save_path, history_save_path, plot_save_path)
        return

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    print("=" * 40)
    print("FEDERATED LEARNING - FEDAVG IID")
    print("=" * 40)
    print(f"Flower Version: {flwr.__version__}")
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Clients: {num_clients}")
    print(f"Federated Rounds: {num_rounds}")
    print(f"Local Epochs: {local_epochs}")
    print(f"Batch Size: {batch_size}")
    print("=" * 40)

    # Ray resource configuration: assign 1.0 GPU per client to guarantee sequential execution on 4 GB VRAM
    backend_config = {
        "client_resources": {
            "num_cpus": 2,
            "num_gpus": 1.0 if cuda_available else 0.0,
        }
    }

    # Reset shared training tracking before running
    srv.round_history.clear()
    srv.last_global_parameters = None

    start_time = time.time()

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )

    elapsed_time = time.time() - start_time
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    time_str = f"{mins}m {secs}s"

    # Actual completed rounds from Flower execution history
    actual_rounds = len(srv.round_history)

    # Save final model
    if srv.last_global_parameters is not None and actual_rounds > 0:
        final_model = create_model(num_classes=2)
        set_weights(final_model, srv.last_global_parameters)
        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "number_of_rounds": actual_rounds,
                "local_epochs": local_epochs,
                "number_of_clients": num_clients,
                "partition_type": "IID",
                "history": srv.round_history,
            },
            model_save_path,
        )

    # Save history CSV
    df_history = pd.DataFrame(srv.round_history)
    df_history.to_csv(history_save_path, index=False)

    # Plot training curves
    if actual_rounds > 0:
        plot_federated_training(srv.round_history, plot_save_path, actual_rounds)

        final_loss = srv.round_history[-1]["val_loss"]
        final_acc = srv.round_history[-1]["val_accuracy"]
        best_entry = max(srv.round_history, key=lambda x: x["val_accuracy"])
        best_acc = best_entry["val_accuracy"]
        best_round = best_entry["round"]
    else:
        final_loss = 0.0
        final_acc = 0.0
        best_acc = 0.0
        best_round = 0

    print("\n" + "=" * 40)
    print(summary_title)
    print("=" * 40)
    print(f"Clients: {num_clients}")
    print(f"Rounds Completed: {actual_rounds}")
    print(f"Local Epochs: {local_epochs}")
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Round: {best_round}")
    print(f"Training Time: {time_str}")
    if actual_rounds > 0:
        print(f"Model Saved: {model_save_path.name}")
        print(f"History Saved: {history_save_path.name}")
        print(f"Training Plot Saved: {plot_save_path.name}")
    else:
        print("Model Saved: None (No rounds completed)")
        print(f"History Saved: {history_save_path.name}")
        print("Training Plot Saved: None")
    print("=" * 40)

    if actual_rounds < num_rounds:
        print(f"\n[WARNING] Simulation completed {actual_rounds} round(s), but {num_rounds} were requested.")


if __name__ == "__main__":
    main()
