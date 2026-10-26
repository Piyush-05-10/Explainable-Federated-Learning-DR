# Explainable Federated Learning for Diabetic Retinopathy Screening

A B.Tech Major Project implementing a privacy-preserving diabetic retinopathy screening framework across simulated rural healthcare centers using Federated Learning (FL) and Explainable Artificial Intelligence (XAI).

---

## Project Overview

Diabetic Retinopathy (DR) is a severe microvascular complication of diabetes and one of the leading causes of preventable blindness worldwide. Early screening is critical to preserving patient vision, but rural health centers frequently suffer from a shortage of expert ophthalmologists, specialized diagnostic infrastructure, and high-bandwidth network connectivity.

Centralizing sensitive medical records (such as retinal fundus images) into a single cloud repository raises critical patient privacy concerns, compliance hurdles with healthcare data regulations, and significant risk of data leakage.

This project addresses these fundamental bottlenecks by developing a **Privacy-Preserving, Decentralized Screening Framework**:
- **Federated Learning (FL):** Enables distributed health centers (clients) to collaboratively train robust deep learning models locally without exchanging raw retinal imagery. Only encrypted or aggregated model weight updates are communicated with a central server.
- **Explainable AI (XAI):** Generates interpretable visual heatmaps (e.g., Grad-CAM) over retinal fundus photographs, highlighting lesions such as microaneurysms, exudates, and hemorrhages to provide actionable, trustworthy clinical decision support to rural health workers.

---

## Objectives

- **Privacy-Preserving Collaborative Training:** Implement a federated learning architecture that trains deep neural networks across distributed, non-IID client datasets without raw data transmission.
- **Automated Diabetic Retinopathy Grading:** Accurately classify retinal fundus images into standard disease severity stages (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
- **Clinical Interpretability & Trust (XAI):** Integrate explainability techniques to generate visual attribution maps, validating that models attend to true pathological features rather than image artifacts.
- **Robustness Against Client Heterogeneity:** Handle non-IID (non-independent and identically distributed) data splits that reflect real-world variations in equipment, image quality, and patient demographics across rural centers.
- **Clinician Dashboard Interface:** Design an intuitive screening interface allowing frontline rural healthcare personnel to upload fundus images, obtain severity grading predictions, inspect XAI overlays, and monitor federated training metrics.

---

## System Architecture

The framework consists of four core tiers:

```
+-----------------------------------------------------------------------------------+
|                            Central Aggregator Server                              |
|  - Federated Orchestration & Round Management                                     |
|  - Global Model Initialization & Aggregation (FedAvg / FedProx)                   |
|  - Global Performance Validation & Checkpoint Management                          |
+-----------------------------------------------------------------------------------+
                                   ▲          │
                 Model Updates     │          │  Global Model
             (Weights / Gradients) │          │  Parameters
                                   │          ▼
+-----------------------------------------------------------------------------------+
|                        Distributed Client Nodes (Rural Centers)                   |
|                                                                                   |
|  +---------------------------+  +---------------------------+  +---------------+  |
|  |     Client Node 1         |  |     Client Node 2         |  | Client Node N |  |
|  |  (Primary Health Center)  |  |  (Community Health Ctr)   |  | (Rural Clinic)|  |
|  |                           |  |                           |  |               |  |
|  | - Local Private Dataset   |  | - Local Private Dataset   |  | - Local Data  |  |
|  | - Preprocessing Pipeline  |  | - Preprocessing Pipeline  |  | - Local Train |  |
|  | - Local Model Training    |  | - Local Model Training    |  | - XAI Heatmap |  |
|  | - XAI (Grad-CAM Engine)   |  | - XAI (Grad-CAM Engine)   |  | - Local Eval  |  |
|  +---------------------------+  +---------------------------+  +---------------+  |
+-----------------------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------------------+
|                               Clinician Dashboard                                 |
|  - Fundus Image Upload & Screening Support                                        |
|  - Multi-Class DR Severity Prediction & Confidence Scores                         |
|  - Visual Diagnostic Overlay (Grad-CAM Heatmaps)                                  |
|  - Federated Aggregation & Convergence Monitoring                                 |
+-----------------------------------------------------------------------------------+
```

---

## Technology Stack

- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Computer Vision & Image Processing:** OpenCV, PIL, NumPy, Albumentations
- **Federated Learning:** PyTorch Distributed Simulation / Flower (flwr)
- **Explainable AI (XAI):** PyTorch-Grad-CAM, Captum
- **Evaluation & Metrics:** Scikit-learn (Quadratic Weighted Kappa, ROC-AUC, Sensitivity, Specificity), SciPy
- **Dashboard & Visualization:** Streamlit, Matplotlib, Seaborn
- **Version Control:** Git, GitHub

---

## Project Structure

```text
MAJOR PROJECT/
├── dashboard/              # Interactive clinician & FL monitoring dashboard
├── data/
│   ├── raw/                # Raw APTOS retinal fundus images (excluded from git)
│   ├── processed/          # Preprocessed & normalized images (excluded from git)
│   └── clients/            # Federated non-IID/IID client partitions (excluded from git)
├── notebooks/              # Jupyter notebooks for EDA and experimental prototyping
├── results/
│   ├── metrics/            # Evaluation logs, confusion matrices, and test reports
│   └── plots/              # Training loss curves, ROC curves, and XAI visualizations
├── saved_models/           # Saved model checkpoints and global weights (excluded from git)
├── src/
│   ├── data/               # Preprocessing pipelines, data loaders, and client splitters
│   ├── models/             # Deep CNN / Vision Transformer model architectures
│   ├── federated/          # Federated server, client trainer, and aggregation logic
│   ├── xai/                # Explainable AI modules (Grad-CAM and saliency visualizers)
│   ├── evaluation/         # Performance evaluation metrics (QWK, F1, Kappa)
│   └── utils/              # Configuration loaders, seed management, and logger helpers
├── .gitignore              # Git exclusions for datasets, checkpoints, and environments
├── README.md               # Comprehensive project documentation
├── requirements.txt        # Python dependency manifest
└── train.csv               # Dataset annotations metadata
```

---

## Dataset

This project utilizes the **APTOS 2019 Blindness Detection dataset** (sourced from the Asia Pacific Tele-Ophthalmology Society symposium competition).

- **Modality:** Retinal Fundus Photography taken under varied clinic conditions.
- **Labels:** 5-class clinical grading of Diabetic Retinopathy severity:
  - `0` - No DR
  - `1` - Mild
  - `2` - Moderate
  - `3` - Severe
  - `4` - Proliferative DR

### Dataset Privacy & Repository Exclusion Notice
> **IMPORTANT:** In strict compliance with healthcare research best practices, data protection policies, and Git storage constraints, **the dataset images and client splits are NOT included in this GitHub repository**. 
>
> Users wishing to reproduce the experiments must download the dataset directly from Kaggle and place the images in `data/raw/` in accordance with the setup instructions below.

---

## Current Progress

- [x] Initialized modular project architecture and folder taxonomy
- [x] Established Git version control with clean `.gitignore` and `.gitkeep` directory tracking
- [x] Configured project documentation, architecture specifications, and requirements manifest
- [ ] Exploratory Data Analysis (EDA) and Ben Graham circular crop preprocessing
- [ ] Non-IID Dirichlet distribution simulation across virtual rural healthcare clients
- [ ] Baseline centralized deep learning model training (ResNet/EfficientNet)
- [ ] Federated Learning training pipeline (FedAvg / FedProx simulation)
- [ ] Explainable AI (Grad-CAM) module for lesion attribution
- [ ] Quantitative comparative evaluation (Centralized vs. Federated performance)
- [ ] Interactive Streamlit screening and inspection dashboard

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "MAJOR PROJECT"
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup the Dataset
1. Download the APTOS 2019 Blindness Detection dataset images from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection).
2. Place the raw training images into `data/raw/`.
3. Verify that `train.csv` is present in the workspace root.

---

## Usage

*(Modules under active development)*

### Preprocess Data and Partition for Federated Clients
```bash
python -m src.data.preprocess
python -m src.data.partition_clients --num_clients 3 --alpha 0.5
```

### Run Federated Training
```bash
python -m src.federated.train --rounds 20 --local_epochs 3 --batch_size 16
```

### Generate XAI Visual Explanations
```bash
python -m src.xai.generate_heatmaps --model_path saved_models/global_best.pth
```

### Launch the Screening Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Results

Experimental training and benchmark evaluations are currently in progress. Benchmark results across centralized and decentralized configurations will be populated once model training is finalized.

| Configuration | Accuracy | Quadratic Weighted Kappa (QWK) | Macro Sensitivity | Macro Specificity |
|---|---|---|---|---|
| Centralized Baseline | To be updated after training | To be updated after training | To be updated after training | To be updated after training |
| Federated (FedAvg - IID) | To be updated after training | To be updated after training | To be updated after training | To be updated after training |
| Federated (FedAvg - Non-IID) | To be updated after training | To be updated after training | To be updated after training | To be updated after training |
| Client 1 (Rural Health Center A) | To be updated after training | To be updated after training | To be updated after training | To be updated after training |
| Client 2 (Rural Health Center B) | To be updated after training | To be updated after training | To be updated after training | To be updated after training |
| Client 3 (Rural Health Center C) | To be updated after training | To be updated after training | To be updated after training | To be updated after training |

---

## Future Work

- **Differential Privacy (DP):** Integrate Renyi Differential Privacy (DP-SGD) to mathematically bound privacy leakage against reconstruction attacks.
- **Secure Aggregation:** Implement cryptographic Secure Multi-Party Computation (SMPC) or Homomorphic Encryption for secure parameter aggregation.
- **Edge Device Optimization:** Quantize models (INT8/FP16) using TensorRT / ONNX for deployment on low-cost edge accelerators (e.g., NVIDIA Jetson, Raspberry Pi) in rural clinics.
- **Multimodal Health Integration:** Extend the classification framework to combine fundus imagery with tabular patient indicators (HbA1c levels, patient age, systolic/diastolic blood pressure).

---

## Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE ONLY**
>
> This project is developed strictly as an academic and engineering research prototype for a B.Tech Major Project. It is **not** certified, accredited, or approved as a medical diagnostic software or medical device by any regulatory authority (e.g., FDA, CE, CDSCO). It is **not intended for clinical diagnosis, patient screening, disease staging, or medical treatment planning in real-world clinical environments**. Any diagnostic decisions must always be made by a qualified healthcare professional and ophthalmologist.
