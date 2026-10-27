import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUM_CLIENTS = 5
RANDOM_STATE = 42

# Heterogeneous allocations to simulate prevalence variation across rural health centers:
# Clinic 1 (~20% DR), Clinic 2 (~35% DR), Clinic 3 (~50% DR), Clinic 4 (~65% DR), Clinic 5 (~80% DR)
# Exactly sums to: No DR = 1263, DR = 1300, Total = 2563 samples across 5 balanced clients.
CLIENT_ALLOCATIONS = [
    {"client": "Clinic 1", "total": 513, "no_dr": 406, "dr": 107},
    {"client": "Clinic 2", "total": 513, "no_dr": 330, "dr": 183},
    {"client": "Clinic 3", "total": 513, "no_dr": 253, "dr": 260},
    {"client": "Clinic 4", "total": 512, "no_dr": 175, "dr": 337},
    {"client": "Clinic 5", "total": 512, "no_dr": 99,  "dr": 413},
]


def create_non_iid_partitions():
    # Paths
    train_csv = PROJECT_ROOT / "data" / "processed" / "train_split.csv"
    val_csv = PROJECT_ROOT / "data" / "processed" / "val_split.csv"
    test_csv = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"

    output_dir = PROJECT_ROOT / "data" / "clients" / "non_iid"
    plot_path = PROJECT_ROOT / "results" / "plots" / "federated_non_iid_distribution.png"
    summary_csv_path = PROJECT_ROOT / "results" / "metrics" / "federated_non_iid_distribution.csv"

    # Load original training split
    train_df = pd.read_csv(train_csv)
    orig_total = len(train_df)
    orig_no_dr = int((train_df["binary_label"] == 0).sum())
    orig_dr = int((train_df["binary_label"] == 1).sum())

    print("========================================")
    print("NON-IID FEDERATED CLIENT PARTITIONING")
    print("========================================")
    print(f"Source Training Samples: {orig_total}")
    print(f"Number of Clients: {NUM_CLIENTS}")
    print("Partition Type: Non-IID / Label Skew")
    print(f"Random State: {RANDOM_STATE}")
    print("========================================\n")

    # Deterministically shuffle No-DR and DR pools
    no_dr_pool = train_df[train_df["binary_label"] == 0].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    dr_pool = train_df[train_df["binary_label"] == 1].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    client_dfs = []
    summary_records = []
    no_dr_start = 0
    dr_start = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, alloc in enumerate(CLIENT_ALLOCATIONS):
        c_name = alloc["client"]
        c_no_dr_count = alloc["no_dr"]
        c_dr_count = alloc["dr"]

        # Slice disjoint subsets from each class pool
        c_no_dr_df = no_dr_pool.iloc[no_dr_start : no_dr_start + c_no_dr_count]
        c_dr_df = dr_pool.iloc[dr_start : dr_start + c_dr_count]

        no_dr_start += c_no_dr_count
        dr_start += c_dr_count

        # Combine and shuffle local clinic dataset
        combined_df = pd.concat([c_no_dr_df, c_dr_df]).sample(frac=1.0, random_state=RANDOM_STATE + i).reset_index(drop=True)
        client_dfs.append(combined_df)

        # Save local client CSV
        client_path = output_dir / f"client_{i + 1}.csv"
        combined_df.to_csv(client_path, index=False)

        total_samples = len(combined_df)
        no_dr_pct = (c_no_dr_count / total_samples) * 100
        dr_pct = (c_dr_count / total_samples) * 100

        print(c_name)
        print(f"  Total: {total_samples}")
        print(f"  No DR: {c_no_dr_count} ({no_dr_pct:.1f}%)")
        print(f"  DR:    {c_dr_count} ({dr_pct:.1f}%)")
        print()

        summary_records.append({
            "client": c_name,
            "total_samples": total_samples,
            "no_dr_samples": c_no_dr_count,
            "dr_samples": c_dr_count,
            "no_dr_percentage": round(no_dr_pct, 2),
            "dr_percentage": round(dr_pct, 2),
        })

    # ==========================================
    # VERIFICATION
    # ==========================================
    all_client_df = pd.concat(client_dfs, ignore_index=True)
    all_client_ids = all_client_df["id_code"].tolist()
    original_ids = set(train_df["id_code"])
    client_id_set = set(all_client_ids)

    total_client_samples = len(all_client_ids)
    total_client_no_dr = int((all_client_df["binary_label"] == 0).sum())
    total_client_dr = int((all_client_df["binary_label"] == 1).sum())

    missing_ids = original_ids - client_id_set
    duplicates = total_client_samples - len(client_id_set)

    # Overlap / Leakage checks
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    val_overlap = client_id_set & set(val_df["id_code"])
    test_overlap = client_id_set & set(test_df["id_code"])

    # Check that image files exist for all client partitions
    all_images_exist = all((images_dir / f"{img_id}.png").exists() for img_id in all_client_ids)

    print("========================================")
    print("PARTITION VERIFICATION")
    print("========================================")
    print(f"Original Training Samples: {orig_total}")
    print(f"Total Samples Across Clients: {total_client_samples}")
    print(f"Original No DR: {orig_no_dr}")
    print(f"Original DR: {orig_dr}")
    print(f"Total Client No DR: {total_client_no_dr}")
    print(f"Total Client DR: {total_client_dr}")
    print(f"Missing Training IDs: {len(missing_ids)}")
    print(f"Duplicate Client IDs: {duplicates}")
    print(f"Client/Validation Overlap: {len(val_overlap)}")
    print(f"Client/Test Overlap: {len(test_overlap)}")
    print(f"All Referenced Images Exist: {all_images_exist}")
    print(f"Number of Clients: {len(client_dfs)}")
    for r in summary_records:
        print(f"  {r['client']}: No DR = {r['no_dr_samples']}, DR = {r['dr_samples']} ({r['dr_percentage']}%)")
    print("========================================\n")

    all_pass = (
        orig_total == 2563
        and total_client_samples == 2563
        and orig_no_dr == 1263
        and orig_dr == 1300
        and total_client_no_dr == 1263
        and total_client_dr == 1300
        and len(missing_ids) == 0
        and duplicates == 0
        and len(val_overlap) == 0
        and len(test_overlap) == 0
        and all_images_exist
        and len(client_dfs) == 5
    )

    if all_pass:
        print("NON-IID FEDERATED PARTITIONING: PASS")
    else:
        print("NON-IID FEDERATED PARTITIONING: FAIL")
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

    clinic_names = [r["client"] for r in summary_records]
    no_dr_counts = [r["no_dr_samples"] for r in summary_records]
    dr_counts = [r["dr_samples"] for r in summary_records]

    x = np.arange(NUM_CLIENTS)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - bar_width / 2, no_dr_counts, bar_width, label="No DR (0)", color="#3b82f6")
    bars2 = ax.bar(x + bar_width / 2, dr_counts, bar_width, label="DR (1)", color="#ef4444")

    # Add count and percentage labels on bars
    for i, bar in enumerate(bars1):
        h = bar.get_height()
        pct = (h / summary_records[i]["total_samples"]) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 6,
                f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for i, bar in enumerate(bars2):
        h = bar.get_height()
        pct = (h / summary_records[i]["total_samples"]) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 6,
                f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Simulated Rural Healthcare Center", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Retinal Images", fontsize=11)
    ax.set_title("Federated Non-IID Client Partitioning - Label Distribution Skew\n"
                 "(5 Heterogeneous Simulated Rural Clinics from APTOS 2019 Training Set)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clinic_names, fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, max(max(no_dr_counts), max(dr_counts)) + 55)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Distribution Plot: {plot_path.as_posix()}")

    # Print saved files
    print(f"\nClient CSVs saved to: {output_dir.as_posix()}/")
    for i in range(NUM_CLIENTS):
        c_path = output_dir / f"client_{i + 1}.csv"
        print(f"  {c_path.name} ({len(client_dfs[i])} samples)")


if __name__ == "__main__":
    create_non_iid_partitions()
