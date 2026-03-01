# 🌿 Tree Leaf Lifecycle Classifier

A deep learning system that classifies tree leaf images by **species** and **lifecycle stage**, combined with a **C++ 3D depth algorithm** for spatial leaf identification using z-axis positioning.

---

## 📁 Repository Structure

```
├── tree_lifecycle_classifier.py   # ML training pipeline (MobileNetV2)
├── download_dataset.py            # iNaturalist dataset downloader
└── algo.cpp                       # 3D leaf z-depth identification algorithm
```

---

## 🧠 ML Model — `tree_lifecycle_classifier.py`

### Architecture
- **Base**: MobileNetV2 pretrained on ImageNet
- **Heads**: Two softmax output heads — species and lifecycle stage
- **Regularization**: Dropout (0.6), L2 weight decay, label smoothing (0.1)

### Training Strategy
| Phase | Description | LR |
|---|---|---|
| Phase 1 | Frozen base, train heads only | 1e-3 |
| Phase 2 | Unfreeze top 30 layers (BatchNorm frozen) | 5e-5 |

### Results
| Metric | Score |
|---|---|
| Species Accuracy | ~75% |
| Lifecycle Accuracy | ~82% |

### Classes
- **Species**: `birch`, `maple`, `oak`
- **Lifecycle Stages**: `bud_emergence`, `expansion_maturity`, `senescence`, `abscission`

---

## 📦 Dataset — `download_dataset.py`

Automatically downloads research-grade leaf images from [iNaturalist](https://www.inaturalist.org/) API.

- **200 images** per species × stage combination
- **2,400 total images** (3 species × 4 stages × 200)
- Auto-validates and filters corrupt/unsupported files

**Dataset structure:**
```
data/
  oak/
    bud_emergence/
    expansion_maturity/
    senescence/
    abscission/
  maple/  ...
  birch/  ...
```

---

## 🔬 3D Algorithm — `algo.cpp`

A standalone C++ implementation for identifying and classifying leaves in 3D space using depth (z-axis) data.

### Mathematical Model

| Concept | Formula |
|---|---|
| Point representation | `p = (x, y, z) ∈ ℝ³` |
| Leaf plane | `â · p = d` (â = unit normal) |
| Z-depth ordering | `Δz = μ_A.z − μ_B.z` |
| Occlusion condition | `π_xy(A) ∩ π_xy(B) ≠ ∅ AND z_A > z_B` |
| Normal estimation | PCA — eigenvector of min eigenvalue of covariance matrix |
| Canopy centroid | `μ_tree = (1/N) Σ μᵢ` |

### Pipeline
1. **DBSCAN Clustering** — groups 3D point cloud into leaf clusters
2. **Z-Depth Ordering** — sorts leaves by depth (nearest first)
3. **Occlusion Detection** — flags which leaves are hidden behind others
4. **Tree Canopy Model** — computes per-layer radii and trunk estimate

### Example Output
```
[Step 2] Z-Depth ordering
  L1  z = +0.004   expansion_maturity   (nearest)
  L2  z = -1.002   senescence           (1 unit behind)

[Step 3] Pairwise delta-z analysis
  L1 -> L2 :  dz = +1.006   [L1 OCCLUDES L2]
```

---

## 🚀 How to Run

### ML Training (Google Colab)
```bash
# Install dependencies
pip install tensorflow pandas numpy pillow requests scikit-learn seaborn

# Download dataset
python download_dataset.py

# Train model
python tree_lifecycle_classifier.py
```

### 3D Algorithm (C++)
```bash
# Compile
g++ -O2 -std=c++17 algo.cpp -o algo

# Run
./algo
```

**On Google Colab:**
```bash
!g++ -O2 -std=c++17 algo.cpp -o /content/algo
!/content/algo
```

---

## 📊 Output Files

| File | Description |
|---|---|
| `best_tree_lifecycle_model.weights.h5` | Phase 1 best weights |
| `best_tree_lifecycle_finetuned.weights.h5` | Fine-tuned weights |
| `confusion_matrices.png` | Species + lifecycle confusion matrices |

---

## 🛠 Dependencies

```
tensorflow
pandas
numpy
pillow
requests
scikit-learn
seaborn
matplotlib
```
