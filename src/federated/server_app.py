import sys
from pathlib import Path
import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet import create_model
from src.federated.task import get_weights, set_weights, load_val_data, evaluate_model

# Shared state to record metrics and final weights across rounds
round_history = []
last_global_parameters = None
latest_train_metrics = {}


def fit_metrics_aggregation_fn(metrics):
    """Aggregate local client training metrics weighted by number of samples."""
    print("\nAggregating client updates with FedAvg...")

    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_loss = sum(num_samples * m["train_loss"] for num_samples, m in metrics) / total_samples
    weighted_acc = sum(num_samples * m["train_accuracy"] for num_samples, m in metrics) / total_samples

    latest_train_metrics["train_loss"] = float(weighted_loss)
    latest_train_metrics["train_accuracy"] = float(weighted_acc)

    return {
        "train_loss": float(weighted_loss),
        "train_accuracy": float(weighted_acc),
    }


def get_evaluate_fn():
    """Create evaluation callback for server-side global model validation."""
    val_loader = load_val_data(batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_fn(server_round: int, parameters: list[np.ndarray], config: dict):
        global last_global_parameters
        last_global_parameters = parameters

        if server_round == 0:
            # Skip initial evaluation before round 1
            return None

        eval_model = create_model(num_classes=2)
        set_weights(eval_model, parameters)

        val_loss, val_acc = evaluate_model(eval_model, val_loader, device=device)

        print("\nGlobal Validation")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.2f}%")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        round_entry = {
            "round": server_round,
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "train_loss": round(latest_train_metrics.get("train_loss", 0.0), 4),
            "train_accuracy": round(latest_train_metrics.get("train_accuracy", 0.0), 2),
        }
        round_history.append(round_entry)

        return float(val_loss), {"val_accuracy": float(val_acc)}

    return evaluate_fn


def make_server_app(num_rounds: int, partition_type: str = "iid", algorithm: str = "fedavg", mu: float = 0.0):
    """Factory to create a ServerApp with ServerConfig explicitly set to num_rounds."""
    num_rounds = int(num_rounds)
    partition_type = str(partition_type)
    algorithm = str(algorithm).lower()
    mu = float(mu)
    is_fedprox = (algorithm == "fedprox") or (mu > 0.0)
    partition_label = "NON-IID" if partition_type.lower() in ["non_iid", "non-iid"] else "IID"

    def on_fit_config_fn(server_round: int):
        print(f"\n{'=' * 40}")
        if is_fedprox:
            print(f"ROUND {server_round}/{num_rounds} - FEDPROX {partition_label}")
        elif partition_label == "NON-IID":
            print(f"ROUND {server_round}/{num_rounds} - NON-IID")
        else:
            print(f"ROUND {server_round}/{num_rounds}")
        print(f"{'=' * 40}")
        return {
            "server_round": server_round,
            "local_epochs": 1,
            "batch_size": 16,
            "partition_type": partition_type,
            "algorithm": "fedprox" if is_fedprox else "fedavg",
            "proximal_mu": mu if is_fedprox else 0.0,
            "mu": mu if is_fedprox else 0.0,
        }

    def server_fn(context: Context):
        # Fresh ImageNet pretrained EfficientNet-B0 initialization
        initial_model = create_model(num_classes=2)
        initial_params = ndarrays_to_parameters(get_weights(initial_model))

        if is_fedprox:
            strategy = FedProx(
                fraction_fit=1.0,
                min_fit_clients=5,
                min_available_clients=5,
                fraction_evaluate=0.0,
                min_evaluate_clients=0,
                initial_parameters=initial_params,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_fn=get_evaluate_fn(),
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                proximal_mu=mu,
            )
        else:
            strategy = FedAvg(
                fraction_fit=1.0,
                min_fit_clients=5,
                min_available_clients=5,
                fraction_evaluate=0.0,
                min_evaluate_clients=0,
                initial_parameters=initial_params,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_fn=get_evaluate_fn(),
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)


# Default 3-round ServerApp for CLI / static usage
app = make_server_app(3, "iid", "fedavg", 0.0)
