import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NUM_CLIENTS = 5
RANDOM_STATE = 42


def create_iid_partitions():
    # Paths
    train_csv = PROJECT_ROOT / "data" / "processed" / "train_split.csv"
    val_csv = PROJECT_ROOT / "data" / "processed" / "val_split.csv"
    test_csv = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"

    output_dir = PROJECT_ROOT / "data" / "clients" / "iid"
    plot_path = PROJECT_ROOT / "results" / "plots" / "federated_iid_distribution.png"
    summary_csv_path = PROJECT_ROOT / "results" / "metrics" / "federated_iid_distribution.csv"

    # Load training set
    train_df = pd.read_csv(train_csv)

    print("========================================")
    print("FEDERATED CLIENT PARTITIONING")
    print("========================================")
    print(f"Source Training Samples: {len(train_df)}")
    print(f"Number of Clients: {NUM_CLIENTS}")
    print(f"Partition Type: IID / Stratified")
    print(f"Random State: {RANDOM_STATE}")
    print("========================================\n")

    # Stratified K-Fold gives approximately equal-sized folds with preserved class ratios.
    # Each fold becomes one simulated rural clinic's local dataset.
    skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=RANDOM_STATE)

    client_dfs = []
    for fold_idx, (_, client_indices) in enumerate(skf.split(train_df, train_df["binary_label"])):
        client_df = train_df.iloc[client_indices].reset_index(drop=True)
        client_dfs.append(client_df)

    # Save client CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, client_df in enumerate(client_dfs):
        client_path = output_dir / f"client_{i + 1}.csv"
        client_df.to_csv(client_path, index=False)

    # Print per-client distributions
    summary_records = []
    for i, client_df in enumerate(client_dfs):
        total = len(client_df)
        no_dr = int((client_df["binary_label"] == 0).sum())
        dr = int((client_df["binary_label"] == 1).sum())
        no_dr_pct = no_dr / total * 100
        dr_pct = dr / total * 100

        print(f"Clinic {i + 1}")
        print(f"  Total: {total}")
        print(f"  No DR: {no_dr} ({no_dr_pct:.1f}%)")
        print(f"  DR:    {dr} ({dr_pct:.1f}%)")
        print()

        summary_records.append({
            "client": f"Clinic {i + 1}",
            "total_samples": total,
            "no_dr_samples": no_dr,
            "dr_samples": dr,
            "no_dr_percentage": round(no_dr_pct, 2),
            "dr_percentage": round(dr_pct, 2),
        })

    # ==========================================
    # VERIFICATION
    # ==========================================

    all_client_ids = pd.concat(client_dfs)["id_code"].tolist()
    original_ids = set(train_df["id_code"])
    client_id_set = set(all_client_ids)

    total_client_samples = len(all_client_ids)
    missing_ids = original_ids - client_id_set
    extra_ids = client_id_set - original_ids
    duplicates = total_client_samples - len(client_id_set)

    # Data leakage checks
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    val_overlap = client_id_set & set(val_df["id_code"])
    test_overlap = client_id_set & set(test_df["id_code"])

    # Verify image files exist for a sample from each client
    all_images_exist = True
    for i, client_df in enumerate(client_dfs):
        sample_id = client_df["id_code"].iloc[0]
        img_path = images_dir / f"{sample_id}.png"
        if not img_path.exists():
            print(f"[FAIL] Missing image for Clinic {i + 1}: {img_path}")
            all_images_exist = False

    print("========================================")
    print("PARTITION VERIFICATION")
    print("========================================")
    print(f"Original Training Samples: {len(train_df)}")
    print(f"Total Client Samples: {total_client_samples}")
    print(f"Missing Training IDs: {len(missing_ids)}")
    print(f"Duplicate Client IDs: {duplicates}")
    print(f"Client/Validation Overlap: {len(val_overlap)}")
    print(f"Client/Test Overlap: {len(test_overlap)}")
    print(f"All Referenced Images Exist: {all_images_exist}")
    print("========================================\n")

    # Final pass/fail
    all_pass = (
        total_client_samples == len(train_df)
        and len(missing_ids) == 0
        and duplicates == 0
        and len(val_overlap) == 0
        and len(test_overlap) == 0
        and all_images_exist
    )

    if all_pass:
        print("FEDERATED IID PARTITIONING: PASS")
    else:
        print("FEDERATED IID PARTITIONING: FAIL")
        if len(missing_ids) > 0:
            print(f"  Missing IDs: {missing_ids}")
        if duplicates > 0:
            print(f"  Duplicates found: {duplicates}")
        if len(val_overlap) > 0:
            print(f"  Validation leakage: {val_overlap}")
        if len(test_overlap) > 0:
            print(f"  Test leakage: {test_overlap}")
        return

    # ==========================================
    # SAVE SUMMARY CSV
    # ==========================================
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nDistribution CSV: {summary_csv_path.as_posix()}")

    # ==========================================
    # DISTRIBUTION PLOT
    # ==========================================
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clinic_names = [f"Clinic {i + 1}" for i in range(NUM_CLIENTS)]
    no_dr_counts = [r["no_dr_samples"] for r in summary_records]
    dr_counts = [r["dr_samples"] for r in summary_records]

    x = np.arange(NUM_CLIENTS)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - bar_width / 2, no_dr_counts, bar_width, label="No DR (0)", color="#3b82f6")
    bars2 = ax.bar(x + bar_width / 2, dr_counts, bar_width, label="DR (1)", color="#ef4444")

    # Add count labels on each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Simulated Rural Healthcare Center", fontsize=11)
    ax.set_ylabel("Number of Retinal Images", fontsize=11)
    ax.set_title("Federated IID Client Partitioning - Class Distribution\n"
                 "(5 Simulated Rural Clinics from APTOS 2019 Training Set)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clinic_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(no_dr_counts), max(dr_counts)) + 40)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Distribution Plot: {plot_path.as_posix()}")

    # Print saved client files
    print(f"\nClient CSVs saved to: {output_dir.as_posix()}/")
    for i in range(NUM_CLIENTS):
        client_path = output_dir / f"client_{i + 1}.csv"
        print(f"  {client_path.name} ({len(client_dfs[i])} samples)")


if __name__ == "__main__":
    create_iid_partitions()
