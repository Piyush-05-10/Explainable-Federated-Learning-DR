import sys
from pathlib import Path
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet import create_model
from src.federated.task import get_weights, set_weights, load_client_data, train_local


class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.clinic_name = f"Clinic {partition_id + 1}"
        self.model = create_model(num_classes=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 16))

        train_loader = load_client_data(self.partition_id, batch_size=batch_size)
        num_samples = len(train_loader.dataset)

        print(f"\n{self.clinic_name}")
        print(f"Samples: {num_samples}")
        print("Local training...")

        train_loss, train_acc = train_local(
            self.model,
            train_loader,
            epochs=local_epochs,
            lr=1e-4,
            weight_decay=1e-4,
            device=self.device,
        )

        print(f"Loss: {train_loss:.4f}")
        print(f"Accuracy: {train_acc:.2f}%")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        updated_weights = get_weights(self.model)
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
        }
        return updated_weights, num_samples, metrics

    def evaluate(self, parameters, config):
        # Local client evaluation is not needed for FedAvg server-side validation
        return 0.0, 0, {}


def client_fn(context: Context):
    partition_id = context.node_config.get("partition-id", 0)
    return FlowerClient(partition_id=partition_id).to_client()


app = ClientApp(client_fn=client_fn)
