import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH   = PROJECT_ROOT / "data" / "raw" / "aptos" / "train.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"
PLOTS_DIR  = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
BINARY_NAMES = {0: "No DR", 1: "DR"}
SAMPLE_SIZE = 10


def save_bar_chart(dist, names, title, filepath, colors, figsize=(9, 5)):
    labels = ["{} - {}".format(i, names[i]) for i in dist.index]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, dist.values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, dist.max() * 1.15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(str(filepath), dpi=150)
    plt.close(fig)


def main():
    random.seed(42)

    if not CSV_PATH.exists():
        raise FileNotFoundError("CSV not found: {}".format(CSV_PATH))
    if not IMAGES_DIR.exists():
        raise FileNotFoundError("Images directory not found: {}".format(IMAGES_DIR))

    print("Loading CSV ...")
    df = pd.read_csv(CSV_PATH)

    required = {"id_code", "diagnosis"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError("CSV is missing columns: {}".format(missing_cols))

    print("Scanning image directory ...")
    image_stems = {p.stem for p in IMAGES_DIR.glob("*.png")}

    csv_ids    = set(df["id_code"].astype(str))
    duplicates = int(df["id_code"].duplicated().sum())
    missing    = csv_ids - image_stems   # in CSV but no .png found
    extra      = image_stems - csv_ids   # .png exists but not in CSV

    dist5  = df["diagnosis"].value_counts().sort_index()
    # binary: 0 = No DR, 1 = any DR grade
    distb  = df["diagnosis"].apply(lambda x: 0 if x == 0 else 1).value_counts().sort_index()

    print("Sampling {} images for property check ...".format(SAMPLE_SIZE))
    sample_ids = random.sample(sorted(image_stems), min(SAMPLE_SIZE, len(image_stems)))
    sample_results = []
    for stem in sample_ids:
        path = IMAGES_DIR / "{}.png".format(stem)
        try:
            with Image.open(path) as img:
                sample_results.append({"id": stem, "readable": True,
                                       "size": img.size, "mode": img.mode})
        except Exception as e:
            sample_results.append({"id": stem, "readable": False, "error": str(e)})

    print("Generating plots ...")
    save_bar_chart(
        dist5, CLASS_NAMES,
        "APTOS 2019 -- 5-Class DR Severity Distribution",
        PLOTS_DIR / "aptos_class_distribution.png",
        ["#4CAF50", "#2196F3", "#FFC107", "#FF5722", "#9C27B0"],
    )
    save_bar_chart(
        distb, BINARY_NAMES,
        "APTOS 2019 -- Binary DR Distribution",
        PLOTS_DIR / "aptos_binary_distribution.png",
        ["#4CAF50", "#F44336"],
        figsize=(6, 5),
    )

    all_readable = all(r["readable"] for r in sample_results)
    integrity    = len(missing) == 0 and duplicates == 0 and all_readable
    total        = len(df)
    sep          = "=" * 40

    print("")
    print(sep)
    print("APTOS 2019 DATASET VERIFICATION")
    print(sep)
    print("CSV rows:          {}".format(total))
    print("PNG images:        {}".format(len(image_stems)))
    print("Missing images:    {}".format(len(missing)))
    print("Extra images:      {}".format(len(extra)))
    print("Duplicate IDs:     {}".format(duplicates))

    print("")
    print("5-CLASS DISTRIBUTION")
    for cls_id, count in dist5.items():
        print("  {} - {:<18}: {:>5}  ({:.1f}%)".format(
            cls_id, CLASS_NAMES[cls_id], count, count / total * 100))

    print("")
    print("BINARY DISTRIBUTION")
    for bin_id, count in distb.items():
        print("  {} - {:<10}: {:>5}  ({:.1f}%)".format(
            bin_id, BINARY_NAMES[bin_id], count, count / total * 100))

    print("")
    print("SAMPLED IMAGE PROPERTIES (n={})".format(len(sample_results)))
    for r in sample_results:
        if r["readable"]:
            w, h = r["size"]
            print("  {}: {}x{} px | mode={}".format(r["id"], w, h, r["mode"]))
        else:
            print("  {}: UNREADABLE -- {}".format(r["id"], r.get("error", "")))

    print("")
    print("Sample images readable: {}".format("YES" if all_readable else "NO"))
    print("Dataset integrity:      {}".format("PASS" if integrity else "FAIL"))
    print(sep)

    if missing:
        print("\n[WARNING] {} id_code(s) in CSV have no matching .png:".format(len(missing)))
        for m in sorted(missing)[:10]:
            print("  -", m)
        if len(missing) > 10:
            print("  ... and {} more.".format(len(missing) - 10))

    if extra:
        print("\n[INFO] {} .png file(s) not referenced in CSV.".format(len(extra)))

    print("\nPlots saved:")
    print("  {}".format(PLOTS_DIR / "aptos_class_distribution.png"))
    print("  {}".format(PLOTS_DIR / "aptos_binary_distribution.png"))


if __name__ == "__main__":
    main()
