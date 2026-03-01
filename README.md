# Tree Leaf Lifecycle Classifier

A deep learning system that classifies tree leaf images by **species** and **lifecycle stage**, combined with a **C++ 3D depth algorithm** for spatial leaf identification using z-axis positioning.

---

## Repository Structure

```
├── tree_lifecycle_classifier.py   # ML training pipeline (MobileNetV2)
├── download_dataset.py            # iNaturalist dataset downloader
└── algo.cpp                       # 3D leaf z-depth identification algorithm
```

---

## ML Model — `tree_lifecycle_classifier.py`

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

##  Dataset — `download_dataset.py`

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
*  MATHEMATICAL MODEL
 *  ==================
 *
 *  1. COORDINATE SYSTEM
 *     Each point in the scene is represented as a 3-vector:
 *
 *         p = (x, y, z)  ∈ ℝ³
 *
 *     where z is the depth axis (positive = towards viewer):
 *         z =  0   → leaf on the near plane (closest to camera)
 *         z = -1   → leaf one unit behind it
 *         z = -d   → leaf d units into the scene
 *
 *  2. LEAF AS AN ORIENTED PLANE
 *     Each leaf L_i is modelled as a planar patch:
 *
 *         π_i : â_i · p = d_i
 *
 *     where â_i = (a, b, c) is the unit normal of the leaf,
 *     and d_i is the signed distance from the origin.
 *
 *     For a leaf lying roughly parallel to the image plane:
 *         â_i ≈ (0, 0, 1)  →  the leaf faces the camera head-on.
 *
 *  3. Z-DEPTH ORDERING
 *     Given two leaves whose centroids are μ_A and μ_B:
 *
 *         Δz = μ_A.z − μ_B.z
 *
 *         Δz > 0  → A is in front of B  (closer to camera)
 *         Δz < 0  → A is behind B
 *         Δz = 0  → coplanar (same depth layer)
 *
 *     For the specific case z_A = 0, z_B = -1:
 *         Δz = 0 − (−1) = 1   →  leaf A is 1 unit in FRONT of leaf B.
 *
 *  4. OCCLUSION TEST
 *     Leaf B (z = -1) is occluded by leaf A (z = 0) at pixel (u,v)
 *     if and only if the XY-projections of their boundaries overlap:
 *
 *         occluded(B,A) = (π_xy(B) ∩ π_xy(A)) ≠ ∅  AND  z_A > z_B
 *
 *     where π_xy projects the 3D convex hull onto the image plane.
 *
 *  5. LEAF SEGMENTATION VIA DEPTH SLICING
 *     Given a depth map D(u,v), we slice into K layers:
 *
 *         layer_k = { p : z_k ≤ p.z < z_{k+1} }
 *
 *     Within each layer we apply 2D connected-component labelling
 *     (4-connectivity on the XY image grid) seeded by colour and
 *     position proximity using DBSCAN with ε-ball radius ε in ℝ³.
 *
 *  6. LEAF NORMAL ESTIMATION (PCA)
 *     For a neighbourhood N(p) ⊂ ℝ³ around centroid μ:
 *
 *         C = (1/|N|) Σ_{q∈N} (q−μ)(q−μ)ᵀ      (3×3 covariance)
 *
 *     The leaf normal â is the eigenvector of C corresponding to
 *     the SMALLEST eigenvalue λ_min (the direction of least spread).
 *
 *  7. LIFECYCLE COLOUR SCORE
 *     Using (R,G,B) ∈ [0,1]³ measured from the leaf patch:
 *
 *         greenness  G_score = G / (R + G + B + ε)
 *         redness    R_score = R / (R + G + B + ε)
 *         brightness V_score = max(R, G, B)
 *
 *     Stage decision rule:
 *         G_score > 0.50                → expansion_maturity  (rich green)
 *         G_score ∈ [0.30, 0.50]        → senescence         (yellowing)
 *         R_score > 0.40                → senescence/absciss. (red-brown)
 *         V_score < 0.20                → abscission         (bare/dark)
 *         else                          → bud_emergence      (pale green)
 *
 *  8. TREE STRUCTURE MODEL
 *     The tree canopy is approximated as a collection of convex
 *     depth-layer hulls stacked along z.  The trunk is modelled
 *     as a cylinder Cyl(axis, r) where axis passes through the
 *     canopy centroid downward.
 *
 *     Given N leaf centroids {μ_i}, the canopy centroid is:
 *         μ_tree = (1/N) Σ μ_i
 *
 *     Canopy radius at depth z_k:
 *         R_k = max_{i: layer(i)=k} ||μ_i.xy − μ_tree.xy||₂


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

##  How to Run

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
